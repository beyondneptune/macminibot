"""End-to-end pipeline smoke test on the scheduler stub.

Cycle-7 migration: fixtures use shared_core canonical ``Signal``,
``NavSnapshot``, ``Symbol`` (via the strategy_engine.contracts re-exports).
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

import numpy as np
import pandas as pd

from shared_core.contracts import Venue
from strategy_engine.contracts import (
    ChainMode,
    NavSnapshot,
    Signal,
    SolverStatus,
    Symbol,
    normalize_solver_status,
)
from strategy_engine.scheduler import RebalanceInputs, run_rebalance


def _snapshot_version() -> str:
    """A throwaway but well-formed sha256 hex string for test snapshots."""
    return hashlib.sha256(b"test-snapshot-v1").hexdigest()


def _mk_symbol(ticker: str) -> Symbol:
    return Symbol(ticker=ticker, venue=Venue.NASDAQ, currency="USD")


def _build_inputs(n: int = 5, seed: int = 42) -> RebalanceInputs:
    rng = np.random.default_rng(seed=seed)
    syms = [_mk_symbol(f"S{i}") for i in range(n)]
    returns = pd.DataFrame(
        rng.standard_normal(size=(252, n)) * 0.01,
        columns=syms,
    )
    signals = [
        Signal(
            symbol=s,
            r_hat_daily=Decimal(str(round(float(rng.normal(0, 0.002)), 6))),
            confidence=0.7,
            model_version="m1",
            as_of_ns=1_700_000_000_000_000_000,
        )
        for s in syms
    ]
    snap = NavSnapshot(
        snapshot_ns=1_700_000_000_000_000_000,
        net_liq=Decimal("100000"),
        buying_power=Decimal("100000"),
        positions={},
        snapshot_version=_snapshot_version(),
        account_id="DU0001",
    )
    return RebalanceInputs(
        signals=signals,
        prev_positions={s: 0.0 for s in syms},
        returns=returns,
        nav_snapshot=snap,
        mode=ChainMode.WHATIF,
        as_of_ns=1_700_000_000_000_000_000,
        session_close_ns=1_700_023_400_000_000_000,
        valid_until_ns=1_700_023_100_000_000_000,
        model_version="m1",
    )


def test_run_rebalance_smoke():
    inputs = _build_inputs(n=5)
    batch = run_rebalance(inputs)
    assert batch.targets_count == 5
    assert batch.mode == ChainMode.WHATIF
    assert batch.decision_id  # non-empty hash
    assert all(t.hold_reason is None for t in batch.targets)


def test_run_rebalance_emits_held_names_with_hold_reason():
    """Improvement #6: a held name without a signal still gets an explicit target."""
    inputs = _build_inputs(n=5)
    held_sym = _mk_symbol("ZZZZ")
    # Inject a held name not in the signal universe
    inputs = RebalanceInputs(
        signals=inputs.signals,
        prev_positions={
            **{s: 0.0 for s in inputs.returns.columns},
            held_sym: 0.02,
        },
        returns=inputs.returns,
        nav_snapshot=inputs.nav_snapshot,
        mode=inputs.mode,
        as_of_ns=inputs.as_of_ns,
        session_close_ns=inputs.session_close_ns,
        valid_until_ns=inputs.valid_until_ns,
        model_version=inputs.model_version,
    )
    batch = run_rebalance(inputs)
    held = [t for t in batch.targets if t.symbol == held_sym]
    assert len(held) == 1
    assert held[0].hold_reason == "held.no_signal_in_universe"
    assert held[0].target_weight == 0.02


def test_target_position_to_shared_core_lowering():
    """Cycle-7 migration: the local TargetPosition lowers to the shared_core canonical."""
    inputs = _build_inputs(n=3)
    batch = run_rebalance(inputs)
    canonical = batch.to_shared_core(
        nav_snapshot_version=inputs.nav_snapshot.snapshot_version
    )
    assert canonical.batch_id == batch.decision_id
    assert canonical.targets_count == batch.targets_count
    assert canonical.nav_snapshot_version == inputs.nav_snapshot.snapshot_version
    # Every target lowers, and the reasoning trace survives in reason_trace.
    assert len(canonical.targets) == len(batch.targets)
    for c, s in zip(canonical.targets, batch.targets):
        assert c.symbol == s.symbol
        assert "target_weight" in c.reason_trace
        assert "r_hat" in c.reason_trace
        assert "confidence" in c.reason_trace
        assert "w_prev" in c.reason_trace
        assert "sigma_contribution" in c.reason_trace


def test_to_shared_core_forwards_cycle8_required_fields():
    """Cycle-8: ``decision_id`` / ``mode`` / ``solver_status`` are REQUIRED.

    Verify the lowering forwards them bit-for-bit so the canonical envelope
    construction does not raise and the values survive the round-trip.
    """
    inputs = _build_inputs(n=4)
    batch = run_rebalance(inputs)
    canonical = batch.to_shared_core(
        nav_snapshot_version=inputs.nav_snapshot.snapshot_version
    )
    assert canonical.decision_id == batch.decision_id
    assert canonical.mode is batch.mode
    assert canonical.mode is ChainMode.WHATIF
    # solver_status string is non-empty and belongs to the standardized vocab
    assert canonical.solver_status == batch.solver_status
    assert canonical.solver_status in {s.value for s in SolverStatus}


def test_solver_status_is_standardized_vocab():
    """Cycle-9: scheduler emissions must use the SolverStatus enum value set."""
    inputs = _build_inputs(n=5)
    batch = run_rebalance(inputs)
    valid = {s.value for s in SolverStatus}
    assert batch.solver_status in valid, (
        f"solver_status={batch.solver_status!r} not in standardized vocabulary {valid}"
    )


# ---------------------------------------------------------------------------
# Cycle-11: VIX-regime soft-dampening wiring through run_rebalance
# ---------------------------------------------------------------------------


def _build_inputs_with_vix(
    *,
    vix_value: float | None,
    vix_history: np.ndarray | None,
    seed: int = 42,
    n: int = 5,
) -> RebalanceInputs:
    """Like ``_build_inputs`` but with optional regime-dampening fields."""
    base = _build_inputs(n=n, seed=seed)
    return RebalanceInputs(
        signals=base.signals,
        prev_positions=base.prev_positions,
        returns=base.returns,
        nav_snapshot=base.nav_snapshot,
        mode=base.mode,
        as_of_ns=base.as_of_ns,
        session_close_ns=base.session_close_ns,
        valid_until_ns=base.valid_until_ns,
        model_version=base.model_version,
        vix_history=vix_history,
        vix_proxy_value=vix_value,
    )


def test_run_rebalance_omits_regime_rules_when_vix_inputs_absent():
    """Without vix_history/vix_proxy_value the vanilla scaler runs (no regime rule)."""
    inputs = _build_inputs(n=5)
    batch = run_rebalance(inputs)
    assert not any(
        r.startswith("vol_target.regime_dampening") for r in batch.edge_case_rules_fired
    )


def test_run_rebalance_normal_regime_emits_rule_and_keeps_target_vol():
    """In the normal band, the regime rule fires but effective target == base."""
    vix_dist = np.linspace(10.0, 50.0, 1000)
    inputs = _build_inputs_with_vix(vix_value=15.0, vix_history=vix_dist)
    batch = run_rebalance(inputs)
    regime_rules = [
        r for r in batch.edge_case_rules_fired
        if r.startswith("vol_target.regime_dampening")
    ]
    assert len(regime_rules) == 1
    rule = regime_rules[0]
    assert "regime=normal" in rule
    assert "effective_target_vol=0.120000" in rule
    assert not any(
        r in ("vol_target.regime_halt_zeroed", "vol_target.regime_intermediate_dampened")
        for r in batch.edge_case_rules_fired
    )


def test_run_rebalance_intermediate_regime_emits_dampened_rule():
    """70th-pct VIX reading triggers the intermediate dampening rule + 6% target."""
    vix_dist = np.linspace(10.0, 50.0, 1000)
    inputs = _build_inputs_with_vix(vix_value=38.0, vix_history=vix_dist)
    batch = run_rebalance(inputs)
    rules = batch.edge_case_rules_fired
    regime_rules = [r for r in rules if r.startswith("vol_target.regime_dampening")]
    assert len(regime_rules) == 1
    assert "regime=intermediate" in regime_rules[0]
    assert "effective_target_vol=0.060000" in regime_rules[0]
    assert "vol_target.regime_intermediate_dampened" in rules
    # Realized portfolio vol on the envelope should track the 6% target.
    assert batch.sigma_portfolio_annualized < 0.12


def test_run_rebalance_halt_regime_zeroes_weights_and_emits_halt_rule():
    """At/above 80th-pct, all target weights collapse to 0 + halt rule fires."""
    vix_dist = np.linspace(10.0, 50.0, 1000)
    inputs = _build_inputs_with_vix(vix_value=46.0, vix_history=vix_dist)
    batch = run_rebalance(inputs)
    rules = batch.edge_case_rules_fired
    regime_rules = [r for r in rules if r.startswith("vol_target.regime_dampening")]
    assert len(regime_rules) == 1
    assert "regime=halt" in regime_rules[0]
    assert "effective_target_vol=0.000000" in regime_rules[0]
    assert "vol_target.regime_halt_zeroed" in rules
    assert batch.sigma_portfolio_annualized == 0.0
    for t in batch.targets:
        assert t.target_weight == 0.0


def test_run_rebalance_regime_rule_contains_percentile():
    """The emitted rule must carry the vix_percentile so replay sees the input."""
    vix_dist = np.linspace(10.0, 50.0, 1000)
    inputs = _build_inputs_with_vix(vix_value=38.0, vix_history=vix_dist)
    batch = run_rebalance(inputs)
    regime_rules = [
        r for r in batch.edge_case_rules_fired
        if r.startswith("vol_target.regime_dampening")
    ]
    assert len(regime_rules) == 1
    assert "vix_percentile=" in regime_rules[0]
    # Extract the percentile and sanity-check it's in the intermediate band.
    pct_token = [tok for tok in regime_rules[0].split(",") if tok.startswith("vix_percentile=")][0]
    pct = float(pct_token.split("=")[1])
    assert 0.60 <= pct < 0.80


def test_run_rebalance_partial_vix_inputs_falls_back_to_vanilla():
    """vix_value present but vix_history None → vanilla scaler (no regime rule)."""
    inputs = _build_inputs_with_vix(vix_value=38.0, vix_history=None)
    batch = run_rebalance(inputs)
    assert not any(
        r.startswith("vol_target.regime_dampening") for r in batch.edge_case_rules_fired
    )


def test_normalize_solver_status_mapping():
    """Cycle-9: the cvxpy / pipeline status strings collapse to the right enum."""
    assert normalize_solver_status("optimal") is SolverStatus.OPTIMAL
    assert normalize_solver_status("optimal_inaccurate") is SolverStatus.OPTIMAL
    assert normalize_solver_status("hrp_ok") is SolverStatus.HRP_OK
    assert normalize_solver_status("zero_signal_decay") is SolverStatus.ZERO_SIGNAL_DECAY
    assert (
        normalize_solver_status("halted_no_universe") is SolverStatus.HALTED_NO_UNIVERSE
    )
    assert (
        normalize_solver_status("fallback_hrp_from_infeasible")
        is SolverStatus.FALLBACK_HRP
    )
    assert normalize_solver_status("infeasible") is SolverStatus.INFEASIBLE
    assert normalize_solver_status("unbounded") is SolverStatus.INFEASIBLE
    # Unknown statuses degrade safely to FALLBACK_HRP rather than raise.
    assert (
        normalize_solver_status("some_future_unknown_status")
        is SolverStatus.FALLBACK_HRP
    )
