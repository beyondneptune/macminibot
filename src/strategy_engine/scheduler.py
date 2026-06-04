"""Rebalance orchestration — stub (not yet wired to APScheduler).

``run_rebalance`` is the synchronous pipeline that the daily 09:35 ET wake
will call. It does no I/O; the wake-time wiring (APScheduler, control-plane
reads, ledger writes) lands in a follow-on PR.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from strategy_engine.contracts import (
    ChainMode,
    NavSnapshot,
    Signal,
    SolverStatus,
    Symbol,
    TargetPosition,
    TargetPositionBatch,
    normalize_solver_status,
)
from strategy_engine.covariance import ledoit_wolf_shrunk
from strategy_engine.edge_cases import (
    apply_no_trade_band,
    apply_turnover_ceiling,
    is_zero_signal_day,
    validate_signal_scale,
)
from strategy_engine.optimizer import mean_variance_optimize
from strategy_engine.sizing import (
    DEFAULT_TARGET_VOL,
    RegimeDampenedVolTargetResult,
    signal_winsorize,
    vol_target_scale,
    vol_target_with_regime_dampening,
)


@dataclass(frozen=True)
class RebalanceInputs:
    """All the inputs ``run_rebalance`` needs in one envelope (test-friendly).

    Cycle-7 migration: ``prev_positions`` and ``returns.columns`` are now
    keyed by ``Symbol`` (shared_core canonical) rather than bare ticker
    strings. The scheduler pipeline still uses each ``Symbol`` as a dict key
    directly (it is hashable by-value) — pandas accepts ``Symbol`` as a
    column label because it inherits ``__hash__`` from the frozen dataclass.
    """

    signals: Sequence[Signal]
    prev_positions: Dict[Symbol, float]  # symbol -> w_prev (risk-units)
    returns: pd.DataFrame  # T-1 dollar-bar returns, wide; cols == universe (Symbol-typed)
    nav_snapshot: NavSnapshot
    mode: ChainMode
    as_of_ns: int
    session_close_ns: int
    valid_until_ns: int
    model_version: str = "unknown"
    target_vol_annualized: float = DEFAULT_TARGET_VOL
    lambda_risk: float = 5.0
    gamma_turnover: float = 0.002
    # L2 ridge weight on (w - w_prev). None → optimizer default of 0.1 * gamma_turnover
    # (hybrid penalty, Improvement #1). When r_hat comes from a ridge/linear ML model
    # the shrinkage is already in r_hat; the sizing path can dial this to 0.0 to
    # avoid double-shrinking. Kept as an input so a future per-model-family hook
    # can set it from the model registry without touching the scheduler.
    gamma_turnover_l2: Optional[float] = None
    L_gross: float = 2.0
    c_max: float = 0.05
    contract_spec_version: str = "v0"
    # Cycle-11: optional VIX-regime soft-dampening inputs.  When BOTH
    # ``vix_history`` and ``vix_proxy_value`` are supplied, the scheduler
    # invokes :func:`vol_target_with_regime_dampening` instead of the plain
    # :func:`vol_target_scale`.  The decision (regime label + effective
    # target vol + percentile) is emitted into ``edge_case_rules_fired`` so
    # downstream replay can see why sizing was dampened.  Either field being
    # ``None`` means "no regime dampening" — we fall back to the vanilla
    # vol-target scaler (this is the safe default; the hard
    # ``VIX_HIGH_REGIME_HALT`` gate in risk-management still fires
    # independently).
    vix_history: Optional[np.ndarray] = None
    vix_proxy_value: Optional[float] = None


def _decision_id(
    inputs: RebalanceInputs,
    signal_payload_hash: str,
    w_prev_hash: str,
    solver_status: str,
) -> str:
    """``hash(as_of || model_version || signal_payload || w_prev || contract_spec_version || solver_status)``."""
    blob = json.dumps(
        {
            "as_of_ns": inputs.as_of_ns,
            "model_version": inputs.model_version,
            "signal_payload": signal_payload_hash,
            "w_prev": w_prev_hash,
            "contract_spec_version": inputs.contract_spec_version,
            "solver_status": solver_status,
            "snapshot_version": inputs.nav_snapshot.snapshot_version,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _hash_obj(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_rebalance(
    inputs: RebalanceInputs,
) -> TargetPositionBatch:
    """Synchronous rebalance pipeline.

    Steps (per CONTEXT.md):
      1. Validate signals (reject blowups, NaN).
      2. Build the optimization universe (intersection of signals and returns
         columns; held names without signals are union'd back per Improvement #6).
      3. Ledoit-Wolf Σ.
      4. Winsorize signals.
      5. MV optimize.
      6. Vol-target scale.
      7. No-trade band + turnover ceiling.
      8. Assemble TargetPositionBatch with hold_reason on frozen names.
    """
    rules_fired: List[str] = []

    # --- (1) validate signals -------------------------------------------------
    accepted_signals: Dict[Symbol, Signal] = {}
    for sig in inputs.signals:
        v = validate_signal_scale(sig.r_hat_daily)
        if v.accepted:
            accepted_signals[sig.symbol] = sig
        else:
            rules_fired.append(f"{v.reason}:{sig.symbol}")

    # --- (2) universe ---------------------------------------------------------
    returns_cols = list(inputs.returns.columns)
    opt_universe = [s for s in returns_cols if s in accepted_signals]
    if not opt_universe:
        rules_fired.append("universe.empty_after_validation")
        return _empty_batch(inputs, rules_fired)

    held_names = [s for s, w in inputs.prev_positions.items() if abs(w) > 0.0]
    union_universe = list(dict.fromkeys(opt_universe + held_names))  # preserve order, dedupe

    # --- (3) Σ ---------------------------------------------------------------
    sub_returns = inputs.returns[opt_universe]
    shrink = ledoit_wolf_shrunk(sub_returns)
    sigma_degraded = bool(shrink.diagonal_fallback)
    if sigma_degraded:
        rules_fired.append("sigma.ill_conditioned_diag_fallback")

    # --- (4) winsorize r_hat -------------------------------------------------
    r_hat_arr = np.array(
        [float(accepted_signals[s].r_hat_daily) for s in opt_universe],
        dtype=np.float64,
    )
    conf_arr = np.array(
        [float(accepted_signals[s].confidence) for s in opt_universe],
        dtype=np.float64,
    )
    r_hat_clipped, clip_mask = signal_winsorize(r_hat_arr)
    if clip_mask.any():
        clipped_names = [opt_universe[i] for i, c in enumerate(clip_mask) if c]
        rules_fired.append(
            "signal.winsorized:" + ",".join(str(s) for s in clipped_names)
        )

    # multiplicative confidence tilt (kept simple per the contract; tilt scaler)
    r_hat_tilted = r_hat_clipped * conf_arr

    # zero-signal-day → decay rather than optimize
    if is_zero_signal_day(r_hat_tilted):
        rules_fired.append("signal.zero_day_decay")
        targets = _decay_targets(inputs, union_universe, opt_universe, rules_fired)
        return _assemble_batch(
            inputs, targets, solver_status=SolverStatus.ZERO_SIGNAL_DECAY,
            sigma_p=0.0, rules_fired=rules_fired,
        )

    # --- (5) optimize --------------------------------------------------------
    w_prev_arr = np.array(
        [inputs.prev_positions.get(s, 0.0) for s in opt_universe],
        dtype=np.float64,
    )
    opt_res = mean_variance_optimize(
        r_hat_tilted,
        shrink.sigma,
        w_prev_arr,
        lambda_risk=inputs.lambda_risk,
        gamma_turnover=inputs.gamma_turnover,
        gamma_turnover_l2=inputs.gamma_turnover_l2,
        L_gross=inputs.L_gross,
        c_max=inputs.c_max,
    )
    rules_fired.append(f"solver.{opt_res.solver_tier}.{opt_res.solver_status}")

    # Cycle-9: standardize the raw optimizer status to the SolverStatus
    # vocabulary. If Σ degraded to the diagonal fallback this session, that
    # diagnostic supersedes a clean OPTIMAL — replay tooling looks at one
    # status string per batch to decide whether to flag for review.
    canonical_status = normalize_solver_status(opt_res.solver_status)
    if sigma_degraded and canonical_status is SolverStatus.OPTIMAL:
        canonical_status = SolverStatus.FALLBACK_DIAGONAL

    # --- (6) vol-target scale ------------------------------------------------
    # Cycle-11: if the caller passed both ``vix_history`` and
    # ``vix_proxy_value`` we route through the regime-dampening variant; the
    # regime decision is logged into ``rules_fired`` so replay tooling can see
    # the dampened target vol without re-deriving it.
    use_regime_dampening = (
        inputs.vix_history is not None and inputs.vix_proxy_value is not None
    )
    if use_regime_dampening:
        vol_res = vol_target_with_regime_dampening(
            opt_res.w,
            shrink.sigma,
            float(inputs.vix_proxy_value),
            inputs.vix_history,
            base_target_vol=inputs.target_vol_annualized,
        )
        _emit_regime_dampening_rules(vol_res, rules_fired)
    else:
        vol_res = vol_target_scale(
            opt_res.w,
            shrink.sigma,
            target_vol_annualized=inputs.target_vol_annualized,
        )
    if vol_res.clamped:
        rules_fired.append("vol_target.clamped_at_leverage_cap")

    # --- (7) no-trade band + turnover ceiling --------------------------------
    w_band = apply_no_trade_band(vol_res.w_scaled, w_prev_arr)
    w_final, capped = apply_turnover_ceiling(w_band, w_prev_arr)
    if capped:
        rules_fired.append("turnover.ceiling_l1_applied")

    # --- (8) assemble targets ------------------------------------------------
    sigma_contrib = shrink.sigma @ w_final  # marginal contribution to variance
    targets: List[TargetPosition] = []
    for i, sym in enumerate(opt_universe):
        targets.append(
            TargetPosition(
                symbol=sym,
                target_weight=float(w_final[i]),
                r_hat=float(r_hat_clipped[i]),
                confidence=float(conf_arr[i]),
                w_prev=float(w_prev_arr[i]),
                sigma_contribution=float(sigma_contrib[i]),
                hold_reason=None,
            )
        )

    # held names without a signal: emit explicit hold (Improvement #6)
    optimized_set = set(opt_universe)
    for sym in held_names:
        if sym in optimized_set:
            continue
        w_prev_i = float(inputs.prev_positions[sym])
        targets.append(
            TargetPosition(
                symbol=sym,
                target_weight=w_prev_i,
                r_hat=0.0,
                confidence=0.0,
                w_prev=w_prev_i,
                sigma_contribution=0.0,
                hold_reason="held.no_signal_in_universe",
            )
        )

    return _assemble_batch(
        inputs,
        targets,
        solver_status=canonical_status,
        sigma_p=vol_res.realized_vol_post,
        rules_fired=rules_fired,
    )


def _emit_regime_dampening_rules(
    vol_res: RegimeDampenedVolTargetResult,
    rules_fired: List[str],
) -> None:
    """Mirror the regime decision into ``edge_case_rules_fired``.

    Always emits the regime label + effective target vol + VIX percentile so
    replay can see *why* sizing was dampened (or that it wasn't). When the
    regime is ``halt`` we also emit a dedicated rule so downstream filters
    can short-circuit on it cheaply.
    """
    pct = vol_res.vix_percentile
    pct_str = f"{pct:.4f}" if np.isfinite(pct) else "nan"
    rules_fired.append(
        f"vol_target.regime_dampening:"
        f"regime={vol_res.regime_label},"
        f"effective_target_vol={vol_res.effective_target_vol:.6f},"
        f"vix_percentile={pct_str}"
    )
    if vol_res.regime_label == "halt":
        rules_fired.append("vol_target.regime_halt_zeroed")
    elif vol_res.regime_label == "intermediate":
        rules_fired.append("vol_target.regime_intermediate_dampened")


def _decay_targets(
    inputs: RebalanceInputs,
    union_universe: List[str],
    opt_universe: List[str],
    rules_fired: List[str],
) -> List[TargetPosition]:
    """Build per-name targets for a zero-signal session: decay w_prev by 20%."""
    targets: List[TargetPosition] = []
    optimized_set = set(opt_universe)
    for sym in union_universe:
        w_prev_i = float(inputs.prev_positions.get(sym, 0.0))
        w_new = w_prev_i * (1.0 - 0.2)
        targets.append(
            TargetPosition(
                symbol=sym,
                target_weight=w_new,
                r_hat=0.0,
                confidence=0.0,
                w_prev=w_prev_i,
                sigma_contribution=0.0,
                hold_reason="zero_signal.decay_to_cash" if sym not in optimized_set else None,
            )
        )
    return targets


def _empty_batch(inputs: RebalanceInputs, rules_fired: List[str]) -> TargetPositionBatch:
    """Used when validation rejects the entire signal payload."""
    return _assemble_batch(
        inputs, targets=[], solver_status=SolverStatus.HALTED_NO_UNIVERSE,
        sigma_p=0.0, rules_fired=rules_fired,
    )


def _assemble_batch(
    inputs: RebalanceInputs,
    targets: List[TargetPosition],
    *,
    solver_status: SolverStatus | str,
    sigma_p: float,
    rules_fired: List[str],
) -> TargetPositionBatch:
    # Cycle-9: accept either the enum or a raw str (back-compat); always
    # store the canonical ``.value`` on the envelope so the wire / ledger
    # round-trip is stable.
    status_str = (
        solver_status.value
        if isinstance(solver_status, SolverStatus)
        else solver_status
    )
    signal_hash = _hash_obj(
        [
            (str(s.symbol), str(s.r_hat_daily), str(s.confidence), s.as_of_ns)
            for s in inputs.signals
        ]
    )
    # Symbol is not natively orderable; sort by str() for deterministic hash.
    w_prev_hash = _hash_obj(
        sorted(((str(k), v) for k, v in inputs.prev_positions.items()))
    )
    did = _decision_id(inputs, signal_hash, w_prev_hash, status_str)
    return TargetPositionBatch(
        decision_id=did,
        as_of_ns=inputs.as_of_ns,
        session_close_ns=inputs.session_close_ns,
        valid_until_ns=inputs.valid_until_ns,
        nav_snapshot=inputs.nav_snapshot.net_liq,
        nav_snapshot_source=f"risk-management.snapshot@{inputs.nav_snapshot.snapshot_version}",
        targets=targets,
        targets_count=len(targets),
        mode=inputs.mode,
        supersedes=None,
        solver_status=status_str,
        sigma_portfolio_annualized=float(sigma_p),
        edge_case_rules_fired=rules_fired,
    )


__all__ = [
    "RebalanceInputs",
    "run_rebalance",
]
