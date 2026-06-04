"""Tests for the 10-check pre-trade chain (CONTEXT.md framework table).

Each of the 10 rules has a passing AND a rejecting case (golden-file style).
Final test: BreachEvents carry the correct ``mode`` field.

Epic 26 (cycle-7): ``make_intent`` constructs a shared_core ``OrderIntent``
and auto-installs the ``IntentContext`` sidecar on the supplied
``ChainContext`` so the 10 checks can read ``current_qty``/``target_qty``/
``ticker`` (now sidecarred off the contract type).
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from risk_management.breaches import risk_dir
from risk_management.chain import (
    Approved,
    ChainContext,
    ChainMode,
    Rejected,
    run_check_chain,
)
from risk_management.kill_switch import halt, heartbeat_path
from risk_management.snapshot import NavSnapshot

from tests.risk_management.conftest import make_intent


MARK = Decimal("150")  # AAPL mark; well within 15% band of prev_close=150


# ---------------------------------------------------------------------------
# 1. KILL_SWITCH_ACTIVE
# ---------------------------------------------------------------------------


def test_kill_switch_passes_when_no_halt_and_fresh_heartbeat(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved), out


def test_kill_switch_rejects_when_halt_exists(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    halt("test halt")
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "KILL_SWITCH_ACTIVE"
    assert out.would_halt is True


def test_kill_switch_rejects_on_stale_heartbeat(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    import time

    past = time.time() - 1000.0
    heartbeat_path().write_text(
        json.dumps({"pid": 1, "ts_unix": past, "version": "x"}), encoding="utf-8"
    )
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "KILL_SWITCH_ACTIVE"


# ---------------------------------------------------------------------------
# 2. SNAPSHOT_FRESH
# ---------------------------------------------------------------------------


def test_snapshot_fresh_passes_when_within_window(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    ctx_live.snapshot_window_start_ns = snapshot.as_of_ns - 10_000_000_000
    ctx_live.snapshot_window_end_ns = snapshot.as_of_ns + 10_000_000_000
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_snapshot_fresh_rejects_when_outside_window(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # Window in the far future — snapshot is "stale" relative to it.
    ctx_live.snapshot_window_start_ns = snapshot.as_of_ns + 10_000_000_000
    ctx_live.snapshot_window_end_ns = snapshot.as_of_ns + 20_000_000_000
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "SNAPSHOT_FRESH"


# ---------------------------------------------------------------------------
# 3. INTENT_NONZERO
# ---------------------------------------------------------------------------


def test_intent_nonzero_passes_above_threshold(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # delta=1 share * mark=150 = $150 > $50 threshold
    intent = make_intent(target_qty="81", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_intent_nonzero_rejects_below_threshold(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # delta=0 in IntentContext -> notional=$0 < $50 threshold; the contract
    # OrderIntent.delta_qty is bumped to a tiny positive by make_intent so
    # the shared_core invariant (delta_qty > 0) is satisfied, but the
    # chain's INTENT_NONZERO check reads from IntentContext which keeps
    # the chain-local "delta=0" semantics.
    intent = make_intent(target_qty="80", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "INTENT_NONZERO"


# ---------------------------------------------------------------------------
# 4. PRICE_BAND_FATFINGER
# ---------------------------------------------------------------------------


def test_price_band_passes_within_15pct(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # prev_close=150; mark=160 is ~6.7% above; within ±15%.
    # target_qty=80 keeps notional ($12,800) under MAX_SINGLE_NAME ($15k) so
    # we isolate the price-band check.
    intent = make_intent(target_qty="80", current_qty="0", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, Decimal("160"), mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_price_band_rejects_outside_15pct(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(
        intent, snapshot, Decimal("300"), mode=ChainMode.LIVE, ctx=ctx_live
    )  # 100% jump
    assert isinstance(out, Rejected)
    assert out.rule_id == "PRICE_BAND_FATFINGER"
    assert out.would_halt is True


# ---------------------------------------------------------------------------
# 5. MAX_SINGLE_NAME_NOTIONAL
# ---------------------------------------------------------------------------


def test_max_single_name_passes_under_15pct(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # 50 shares * $150 = $7,500 < 15% * $100k = $15k
    intent = make_intent(target_qty="50", current_qty="0", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_max_single_name_rejects_over_15pct(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # 200 shares * $150 = $30,000 > 15% * $100k = $15k
    intent = make_intent(target_qty="200", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "MAX_SINGLE_NAME_NOTIONAL"


# ---------------------------------------------------------------------------
# 6. MAX_GROSS_EXPOSURE
# ---------------------------------------------------------------------------


def test_gross_exposure_passes_under_100pct(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    ctx_live.planned_gross_notional = Decimal("80000")  # 80% of 100k
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_gross_exposure_rejects_over_100pct(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    ctx_live.planned_gross_notional = Decimal("110000")  # 110% — leverage
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "MAX_GROSS_EXPOSURE"


# ---------------------------------------------------------------------------
# 7. KELLY_CAP
# ---------------------------------------------------------------------------


def test_kelly_cap_passes_within_quarter_kelly(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # kelly_optimal=$100k -> 25% cap = $25k. Target=50 sh * $150 = $7,500 < $25k.
    ctx_live.kelly_optimal_notional = {"AAPL": Decimal("100000")}
    intent = make_intent(target_qty="50", current_qty="0", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_kelly_cap_rejects_over_quarter_kelly(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # kelly_optimal=$10k -> 25% cap = $2.5k. Target=50 sh * $150 = $7,500 > $2.5k.
    # But MAX_SINGLE_NAME ($15k threshold) would pass first since $7,500 < $15k.
    ctx_live.kelly_optimal_notional = {"AAPL": Decimal("10000")}
    intent = make_intent(target_qty="50", current_qty="0", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "KELLY_CAP"


# ---------------------------------------------------------------------------
# 8. DAILY_TURNOVER_CAP
# ---------------------------------------------------------------------------


def test_turnover_passes_under_50pct(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    ctx_live.planned_turnover_notional = Decimal("40000")  # 40% of 100k
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_turnover_rejects_over_50pct(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    ctx_live.planned_turnover_notional = Decimal("60000")  # 60% of 100k
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "DAILY_TURNOVER_CAP"


# ---------------------------------------------------------------------------
# 9. POSITION_RECONCILES
# ---------------------------------------------------------------------------


def test_position_reconciles_passes_within_1_share(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # broker = 81, internal = 80 -> diff = 1, allowed.
    snapshot.broker_positions["AAPL"] = Decimal("81")
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_position_reconciles_rejects_over_1_share(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # broker = 100, internal = 80 -> diff = 20, exceeds.
    snapshot.broker_positions["AAPL"] = Decimal("100")
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "POSITION_RECONCILES"
    assert out.would_halt is True


# ---------------------------------------------------------------------------
# 10. FEATURE_SANITY
# ---------------------------------------------------------------------------


def test_feature_sanity_passes_when_finite_and_flag_true(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)


def test_feature_sanity_rejects_nan_mark(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    nan = Decimal("NaN")
    out = run_check_chain(intent, snapshot, nan, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "FEATURE_SANITY"
    assert out.would_halt is True


def test_feature_sanity_rejects_when_flag_false(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    snapshot.feature_ok["AAPL"] = False
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    assert out.rule_id == "FEATURE_SANITY"


# ---------------------------------------------------------------------------
# Breach events carry the correct mode field
# ---------------------------------------------------------------------------


def test_breach_event_records_live_mode(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # Force a MAX_SINGLE_NAME reject and inspect the JSONL sink.
    intent = make_intent(target_qty="200", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Rejected)
    sink = risk_dir() / "breaches.jsonl"
    assert sink.exists()
    rec = json.loads(sink.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["mode"] == "live"
    assert rec["rule_id"] == "MAX_SINGLE_NAME_NOTIONAL"
    assert rec["config_id"] == "test.001"
    # And the in-memory ctx.breaches list also got the event.
    assert ctx_live.breaches[-1].mode is ChainMode.LIVE


def test_breach_event_records_whatif_mode_no_filesystem(
    snapshot: NavSnapshot, ctx_whatif: ChainContext
) -> None:
    intent = make_intent(target_qty="200", current_qty="80", ctx=ctx_whatif)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.WHATIF, ctx=ctx_whatif)
    assert isinstance(out, Rejected)
    # In-memory event has the right mode...
    assert ctx_whatif.breaches[-1].mode is ChainMode.WHATIF
    # ...and the JSONL sink was NOT written.
    sink = risk_dir() / "breaches.jsonl"
    assert not sink.exists()


def test_breach_event_paper_mode_writes_to_paper_subdir(
    snapshot: NavSnapshot, default_limits
) -> None:
    paper_ctx = ChainContext(limits=default_limits, mode=ChainMode.PAPER)
    intent = make_intent(target_qty="200", current_qty="80", ctx=paper_ctx)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.PAPER, ctx=paper_ctx)
    assert isinstance(out, Rejected)
    paper_sink = risk_dir() / "paper" / "breaches.jsonl"
    live_sink = risk_dir() / "breaches.jsonl"
    assert paper_sink.exists()
    assert not live_sink.exists()
    rec = json.loads(paper_sink.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["mode"] == "paper"


# ---------------------------------------------------------------------------
# Mode misuse contract
# ---------------------------------------------------------------------------


def test_chain_requires_mode_match_ctx(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    with pytest.raises(ValueError, match="ctx.mode"):
        run_check_chain(intent, snapshot, MARK, mode=ChainMode.WHATIF, ctx=ctx_live)


def test_chain_requires_decimal_mark(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    with pytest.raises(TypeError):
        run_check_chain(intent, snapshot, 150.0, mode=ChainMode.LIVE, ctx=ctx_live)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Epic 21a follow-up: chain emits metrics when a sampler is wired
# ---------------------------------------------------------------------------


def test_chain_emits_approved_metric_when_sampler_supplied(
    snapshot: NavSnapshot, default_limits
) -> None:
    from risk_management.metrics_emitter import new_sampler

    sampler = new_sampler(ChainMode.LIVE, config_id=default_limits.config_id)
    ctx = ChainContext(
        limits=default_limits, mode=ChainMode.LIVE, metrics_sampler=sampler
    )
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx)
    assert isinstance(out, Approved)
    names = [s.name for s in sampler.buffer]
    assert "bot_risk_chain_runs_total" in names
    approved_sample = next(
        s for s in sampler.buffer if s.name == "bot_risk_chain_runs_total"
    )
    assert approved_sample.labels["outcome"] == "approved"
    # No rejection counter on a passing intent.
    assert "bot_risk_check_rejections_total" not in names


def test_chain_emits_rejection_metric_with_halt_severity(
    snapshot: NavSnapshot, default_limits
) -> None:
    """PRICE_BAND_FATFINGER is in HALT_INDUCING_RULES → severity=halt."""

    from risk_management.metrics_emitter import new_sampler

    sampler = new_sampler(ChainMode.LIVE, config_id=default_limits.config_id)
    ctx = ChainContext(
        limits=default_limits, mode=ChainMode.LIVE, metrics_sampler=sampler
    )
    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx)
    # mark 200 vs prev_close 150 = +33% → trips the 15% PRICE_BAND_FATFINGER.
    out = run_check_chain(
        intent, snapshot, Decimal("200"), mode=ChainMode.LIVE, ctx=ctx
    )
    assert isinstance(out, Rejected)
    assert out.rule_id == "PRICE_BAND_FATFINGER"
    rej = next(
        s for s in sampler.buffer if s.name == "bot_risk_check_rejections_total"
    )
    assert rej.labels["rule_id"] == "PRICE_BAND_FATFINGER"
    assert rej.labels["severity"] == "halt"
    assert rej.labels["symbol"] == "AAPL"
    # Chain-runs counter also bumped with outcome=rejected.
    runs = next(
        s for s in sampler.buffer if s.name == "bot_risk_chain_runs_total"
    )
    assert runs.labels["outcome"] == "rejected"


def test_chain_emits_warn_severity_for_non_halt_rule(
    snapshot: NavSnapshot, default_limits
) -> None:
    """INTENT_NONZERO is NOT in HALT_INDUCING_RULES → severity=warn."""

    from risk_management.metrics_emitter import new_sampler

    sampler = new_sampler(ChainMode.LIVE, config_id=default_limits.config_id)
    ctx = ChainContext(
        limits=default_limits, mode=ChainMode.LIVE, metrics_sampler=sampler
    )
    # A 0.1-share delta at mark $150 = $15 notional, below the $50 min.
    intent = make_intent(
        target_qty="80.1", current_qty="80", ctx=ctx
    )
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx)
    assert isinstance(out, Rejected)
    assert out.rule_id == "INTENT_NONZERO"
    rej = next(
        s for s in sampler.buffer if s.name == "bot_risk_check_rejections_total"
    )
    assert rej.labels["severity"] == "warn"


def test_chain_no_sampler_is_silent(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    """The Epic 21a wiring is opt-in: ctx without a sampler emits nothing."""

    intent = make_intent(target_qty="100", current_qty="80", ctx=ctx_live)
    out = run_check_chain(intent, snapshot, MARK, mode=ChainMode.LIVE, ctx=ctx_live)
    assert isinstance(out, Approved)
    assert ctx_live.metrics_sampler is None
