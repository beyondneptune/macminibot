"""Tests for VixHighGate (cycle-10 follow-up).

Covers:
  * High-vol tick at/above the 80th percentile triggers HALT in LIVE.
  * Normal-vol tick passes (no breach, no halt) in every mode.
  * Mode-dispatched behavior: LIVE halts; PAPER warns + allows; WHATIF
    warns in-memory (no filesystem writes).
  * Constructor validation + degenerate-input handling.
  * Breach record carries percentile / sample_size / would_halt metadata.

The ``_isolate_fs`` autouse fixture in conftest.py redirects
``MACMINIBOT_RISK_DIR`` / ``MACMINIBOT_STATE_DIR`` to a per-test tmpdir
so breach JSONLs land in a sandbox.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from risk_management.breaches import risk_dir
from risk_management.chain import ChainContext, ChainMode
from risk_management.regime_gate import (
    VIX_HIGH_RULE_ID,
    VIX_OUT_OF_BAND_RULE_ID,
    VixBandGateDecision,
    VixGateDecision,
    VixHighGate,
    VixOutOfBandGate,
)
from risk_management.snapshot import NavSnapshot


# ---------------------------------------------------------------------------
# Helpers — synthetic VIX histories
# ---------------------------------------------------------------------------


def _calm_history(n: int = 252, base: float = 15.0) -> list[float]:
    """Calm-regime series: gentle ramp 13..17 then a calm latest tick."""

    series = [base + (i % 5) * 0.5 for i in range(n - 1)]
    series.append(base)  # latest = base, deep below 80th percentile
    return series


def _spike_history(n: int = 252, base: float = 15.0, spike: float = 45.0) -> list[float]:
    """Trailing distribution centred around ``base``; latest tick = ``spike``."""

    series = [base + (i % 7) * 0.4 for i in range(n - 1)]
    series.append(spike)
    return series


def _low_history(n: int = 252) -> list[float]:
    """Trailing distribution 20..40 with a calm-tail latest tick below p40.

    Build a roughly uniform 20..40 series; latest tick = 21.0 sits well
    below the 40th percentile of that distribution (~28).
    """

    series = [20.0 + (i % 21) for i in range(n - 1)]
    series.append(21.0)
    return series


def _midband_history(n: int = 252) -> list[float]:
    """Distribution 10..30 with latest in the middle (around p50)."""

    series = [10.0 + (i % 21) for i in range(n - 1)]
    series.append(20.0)  # roughly p50
    return series


# ---------------------------------------------------------------------------
# 1. Construction / validation
# ---------------------------------------------------------------------------


def test_constructor_defaults_match_cycle9_finding() -> None:
    gate = VixHighGate()
    assert gate.threshold_pct == 80.0
    assert gate.lookback_days == 252


@pytest.mark.parametrize("bad_pct", [-1.0, 0.0, 100.0, 150.0])
def test_constructor_rejects_out_of_range_percentile(bad_pct: float) -> None:
    with pytest.raises(ValueError, match="threshold_pct"):
        VixHighGate(threshold_pct=bad_pct)


@pytest.mark.parametrize("bad_lb", [0, 1, -5])
def test_constructor_rejects_short_lookback(bad_lb: int) -> None:
    with pytest.raises(ValueError, match="lookback_days"):
        VixHighGate(lookback_days=bad_lb)


def test_evaluate_rejects_non_chainmode_mode() -> None:
    gate = VixHighGate()
    with pytest.raises(TypeError, match="mode must be ChainMode"):
        gate.evaluate(vix_history=[15.0, 16.0], mode="LIVE")  # type: ignore[arg-type]


def test_evaluate_rejects_empty_history() -> None:
    gate = VixHighGate()
    with pytest.raises(ValueError, match="non-empty"):
        gate.evaluate(vix_history=[], mode=ChainMode.WHATIF)


# ---------------------------------------------------------------------------
# 2. Trigger semantics
# ---------------------------------------------------------------------------


def test_high_vol_day_triggers_and_halts_in_live(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    assert decision.triggered is True
    assert decision.halt is True
    assert decision.current_value > decision.threshold_value
    # Breach was emitted into ctx.breaches and to the LIVE sink.
    assert len(ctx_live.breaches) == 1
    ev = ctx_live.breaches[0]
    assert ev.rule_id == VIX_HIGH_RULE_ID
    assert ev.would_halt is True
    assert ev.mode is ChainMode.LIVE


def test_normal_day_does_not_trigger(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_calm_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    assert decision.triggered is False
    assert decision.halt is False
    assert decision.current_value < decision.threshold_value
    assert ctx_live.breaches == []  # no breach emitted on pass


def test_boundary_tick_at_threshold_triggers(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    # Construct a series where the latest equals the 80th percentile exactly.
    series = list(range(1, 11))  # 1..10
    # numpy.percentile(range(1,11), 80) == 8.2 with linear interp; using 8.2
    # as the latest tick gives current == threshold over the full window.
    series.append(8.2)
    gate = VixHighGate(threshold_pct=80.0, lookback_days=11)
    decision = gate.evaluate(
        vix_history=series,
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    assert decision.triggered is True
    assert decision.current_value == decision.threshold_value


# ---------------------------------------------------------------------------
# 3. Mode-dispatched behavior
# ---------------------------------------------------------------------------


def test_paper_mode_triggers_but_does_not_halt(
    snapshot: NavSnapshot, default_limits
) -> None:
    ctx = ChainContext(limits=default_limits, mode=ChainMode.PAPER)
    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.PAPER,
        ctx=ctx,
        snapshot=snapshot,
    )
    assert decision.triggered is True
    assert decision.halt is False  # paper trades continue
    # Breach was written to the paper sink.
    paper_sink = risk_dir() / "paper" / "breaches.jsonl"
    assert paper_sink.exists()
    rec = json.loads(paper_sink.read_text().splitlines()[0])
    assert rec["rule_id"] == VIX_HIGH_RULE_ID
    assert rec["mode"] == "paper"
    assert rec["would_halt"] is False


def test_whatif_mode_triggers_in_memory_only(
    snapshot: NavSnapshot, ctx_whatif: ChainContext
) -> None:
    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.WHATIF,
        ctx=ctx_whatif,
        snapshot=snapshot,
    )
    assert decision.triggered is True
    assert decision.halt is False
    # In-memory record present, no filesystem writes.
    assert len(ctx_whatif.breaches) == 1
    live_sink = risk_dir() / "breaches.jsonl"
    paper_sink = risk_dir() / "paper" / "breaches.jsonl"
    assert not live_sink.exists()
    assert not paper_sink.exists()


def test_live_mode_writes_to_live_sink_on_trigger(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    gate = VixHighGate()
    gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    live_sink = risk_dir() / "breaches.jsonl"
    assert live_sink.exists()
    rec = json.loads(live_sink.read_text().splitlines()[0])
    assert rec["rule_id"] == VIX_HIGH_RULE_ID
    assert rec["mode"] == "live"
    assert rec["would_halt"] is True
    assert "percentile" in rec["extra"]
    assert rec["extra"]["percentile"] == "80.0"
    assert rec["extra"]["lookback_days"] == "252"


# ---------------------------------------------------------------------------
# 4. Robustness — bad ticks shouldn't crash the pre-trade path
# ---------------------------------------------------------------------------


def test_non_finite_ticks_are_dropped(
    snapshot: NavSnapshot, ctx_whatif: ChainContext
) -> None:
    series: list = ["NaN", "Inf", "not-a-number"] + _calm_history(50)
    gate = VixHighGate(lookback_days=50)
    decision = gate.evaluate(
        vix_history=series, mode=ChainMode.WHATIF, ctx=ctx_whatif, snapshot=snapshot
    )
    # Sample size reflects post-coercion length, capped at lookback_days.
    assert decision.sample_size == 50
    assert decision.triggered is False  # calm latest


def test_decision_is_pure_when_ctx_omitted() -> None:
    # ctx + snapshot are optional — gate still returns a decision; it just
    # can't write a breach (which is fine for offline / research callers).
    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_spike_history(), mode=ChainMode.WHATIF
    )
    assert isinstance(decision, VixGateDecision)
    assert decision.triggered is True
    assert decision.halt is False


# ---------------------------------------------------------------------------
# 5. Cycle-11 — soft/hard split coordination (CONTEXT.md I14)
# ---------------------------------------------------------------------------


def test_zero_weights_with_gate_trigger_records_audit_without_double_block(
    snapshot: NavSnapshot, ctx_live: ChainContext, default_limits
) -> None:
    """Cycle-11 contract test for the strategy-engine + risk soft/hard split.

    Scenario:
      * strategy-engine's vol-target-with-regime-dampening returns zero
        weights (vix pct >= 80 -> halt label -> w_scaled all zero).
      * The diff step therefore produces an EMPTY OrderIntent list — no
        intent is ever proposed to the chain.
      * VixHighGate fires in parallel and records a BreachEvent.

    Expected:
      * Breach is in ctx.breaches AND on disk (audit trail intact).
      * decision.halt is True (LIVE mode; the scheduler skips the day).
      * NO chain-level breach is recorded (no intents = no rejections),
        confirming the "no double-blocking" contract: zero weights don't
        produce rule-by-rule chain rejects, the gate breach stands alone.
    """

    from risk_management.breaches import risk_dir

    # Simulate strategy-engine returning an empty intent batch (the
    # 80+-pct soft-dampening path zeroes weights -> diff step emits []).
    empty_intents: list = []

    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )

    # Hard-gate side: audit + halt signal set.
    assert decision.triggered is True
    assert decision.halt is True
    assert len(ctx_live.breaches) == 1
    ev = ctx_live.breaches[0]
    assert ev.rule_id == VIX_HIGH_RULE_ID
    assert ev.would_halt is True

    # Audit trail on disk.
    live_sink = risk_dir() / "breaches.jsonl"
    assert live_sink.exists()
    lines = [json.loads(ln) for ln in live_sink.read_text().splitlines()]
    halt_lines = [ln for ln in lines if ln["rule_id"] == VIX_HIGH_RULE_ID]
    assert len(halt_lines) == 1, "exactly one VIX halt breach should land"

    # No double-blocking: with zero intents from the soft layer, the
    # chain is never invoked -- there are no per-intent rejections to
    # spam the breach stream. Only the gate's single breach is present.
    chain_rule_breaches = [
        ln for ln in lines if ln["rule_id"] != VIX_HIGH_RULE_ID
    ]
    assert chain_rule_breaches == [], (
        "zero-weights path must NOT produce per-intent chain rejections "
        "(would indicate double-blocking)"
    )
    # And of course there's nothing for execution to submit.
    assert empty_intents == []


def test_paper_mode_emits_halt_metric_for_oos_analysis(
    snapshot: NavSnapshot, default_limits
) -> None:
    """Cycle-11: PAPER soak must capture VIX gate firings as metrics so
    OOS validation of the cycle-9 finding has first-class data.

    Without this, the I7e dashboard shows zero VIX-halt activity during
    paper soak (paper never flips the live HALT flag) and an analyst
    has to cross-join breaches.jsonl against equity_curve.parquet by
    hand.
    """

    from risk_management.breaches import risk_dir
    from risk_management.metrics_emitter import new_sampler

    ctx = ChainContext(limits=default_limits, mode=ChainMode.PAPER)
    sampler = new_sampler(ChainMode.PAPER, default_limits.config_id)

    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.PAPER,
        ctx=ctx,
        snapshot=snapshot,
        sampler=sampler,
    )

    assert decision.triggered is True
    assert decision.halt is False  # paper continues
    # In-memory metric was buffered.
    halt_samples = [
        s for s in sampler.buffer if s.name == "bot_risk_halt_total"
    ]
    assert len(halt_samples) == 1
    sample = halt_samples[0]
    assert sample.labels["rule_id"] == VIX_HIGH_RULE_ID
    assert sample.labels["mode"] == "paper"
    assert sample.labels["config_id"] == default_limits.config_id
    assert sample.value == 1.0

    # Flushing in PAPER lands on the paper sink (parallel filesystem
    # per I7a — never collides with live metrics).
    n = sampler.flush()
    assert n == 1
    paper_metrics = risk_dir() / "paper" / "metrics.jsonl"
    assert paper_metrics.exists()
    rec = json.loads(paper_metrics.read_text().splitlines()[0])
    assert rec["name"] == "bot_risk_halt_total"
    assert rec["labels"]["rule_id"] == VIX_HIGH_RULE_ID
    assert rec["labels"]["mode"] == "paper"


def test_whatif_metric_emission_stays_in_memory(
    snapshot: NavSnapshot, ctx_whatif: ChainContext, default_limits
) -> None:
    """Sampler in WHATIF buffers samples but flush is a no-op (per I7a)."""

    from risk_management.breaches import risk_dir
    from risk_management.metrics_emitter import new_sampler

    sampler = new_sampler(ChainMode.WHATIF, default_limits.config_id)
    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.WHATIF,
        ctx=ctx_whatif,
        snapshot=snapshot,
        sampler=sampler,
    )
    assert decision.triggered is True
    assert any(s.name == "bot_risk_halt_total" for s in sampler.buffer)
    n = sampler.flush()
    assert n == 0  # whatif never writes
    # Confirm: no metrics.jsonl on either sink.
    assert not (risk_dir() / "metrics.jsonl").exists()
    assert not (risk_dir() / "paper" / "metrics.jsonl").exists()


def test_no_trigger_means_no_metric_emission(
    snapshot: NavSnapshot, ctx_live: ChainContext, default_limits
) -> None:
    """Calm-regime day: gate doesn't fire, so no halt-counter sample."""

    from risk_management.metrics_emitter import new_sampler

    sampler = new_sampler(ChainMode.LIVE, default_limits.config_id)
    gate = VixHighGate()
    decision = gate.evaluate(
        vix_history=_calm_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
        sampler=sampler,
    )
    assert decision.triggered is False
    assert sampler.buffer == []


# ---------------------------------------------------------------------------
# 6. Cycle-13 — VixOutOfBandGate (two-sided U-shape gate)
# ---------------------------------------------------------------------------


def test_band_gate_constructor_defaults_match_cycle12_finding() -> None:
    gate = VixOutOfBandGate()
    assert gate.low_pct == 40.0
    assert gate.high_pct == 80.0
    assert gate.lookback_days == 252


@pytest.mark.parametrize("bad_pct", [-1.0, 0.0, 100.0, 150.0])
def test_band_gate_rejects_out_of_range_low_pct(bad_pct: float) -> None:
    with pytest.raises(ValueError, match="low_pct"):
        VixOutOfBandGate(low_pct=bad_pct, high_pct=80.0)


@pytest.mark.parametrize("bad_pct", [-1.0, 0.0, 100.0, 150.0])
def test_band_gate_rejects_out_of_range_high_pct(bad_pct: float) -> None:
    with pytest.raises(ValueError, match="high_pct"):
        VixOutOfBandGate(low_pct=40.0, high_pct=bad_pct)


def test_band_gate_rejects_inverted_band() -> None:
    with pytest.raises(ValueError, match="must be <"):
        VixOutOfBandGate(low_pct=80.0, high_pct=40.0)
    with pytest.raises(ValueError, match="must be <"):
        VixOutOfBandGate(low_pct=50.0, high_pct=50.0)


@pytest.mark.parametrize("bad_lb", [0, 1, -5])
def test_band_gate_rejects_short_lookback(bad_lb: int) -> None:
    with pytest.raises(ValueError, match="lookback_days"):
        VixOutOfBandGate(lookback_days=bad_lb)


def test_band_gate_evaluate_rejects_non_chainmode() -> None:
    gate = VixOutOfBandGate()
    with pytest.raises(TypeError, match="mode must be ChainMode"):
        gate.evaluate(vix_history=[15.0, 16.0], mode="LIVE")  # type: ignore[arg-type]


def test_band_gate_evaluate_rejects_empty_history() -> None:
    gate = VixOutOfBandGate()
    with pytest.raises(ValueError, match="non-empty"):
        gate.evaluate(vix_history=[], mode=ChainMode.WHATIF)


# --- trigger semantics ----------------------------------------------------


def test_band_gate_high_tail_triggers_and_halts_in_live(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    assert isinstance(decision, VixBandGateDecision)
    assert decision.triggered is True
    assert decision.halt is True
    assert decision.side == "high"
    assert decision.current_value >= decision.high_threshold_value
    assert len(ctx_live.breaches) == 1
    ev = ctx_live.breaches[0]
    assert ev.rule_id == VIX_OUT_OF_BAND_RULE_ID
    assert ev.would_halt is True
    assert ev.extra["side"] == "high"


def test_band_gate_low_tail_triggers_and_halts_in_live(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_low_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    assert decision.triggered is True
    assert decision.halt is True
    assert decision.side == "low"
    assert decision.current_value < decision.low_threshold_value
    ev = ctx_live.breaches[0]
    assert ev.rule_id == VIX_OUT_OF_BAND_RULE_ID
    assert ev.extra["side"] == "low"
    # The breach's `threshold` field carries the violated boundary.
    assert ev.threshold == decision.low_threshold_value


def test_band_gate_mid_band_does_not_trigger(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_midband_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    assert decision.triggered is False
    assert decision.halt is False
    assert decision.side == ""
    assert ctx_live.breaches == []


def test_band_gate_boundary_high_tick_triggers(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    """Exact-equal upper boundary triggers (matches VixHighGate semantics)."""

    series = list(range(1, 11))  # 1..10, p80 == 8.2
    series.append(8.2)
    gate = VixOutOfBandGate(low_pct=20.0, high_pct=80.0, lookback_days=11)
    decision = gate.evaluate(
        vix_history=series,
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    assert decision.triggered is True
    assert decision.side == "high"


def test_band_gate_boundary_low_tick_does_not_trigger(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    """Exact-equal lower boundary PASSES (strict `<` on the low side)."""

    # range 1..10, p20 == 2.8 (numpy linear); latest = 2.8 → not triggered.
    series = list(range(1, 11))
    series.append(2.8)
    gate = VixOutOfBandGate(low_pct=20.0, high_pct=80.0, lookback_days=11)
    decision = gate.evaluate(
        vix_history=series,
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    assert decision.triggered is False


# --- mode dispatch --------------------------------------------------------


def test_band_gate_paper_records_but_does_not_halt(
    snapshot: NavSnapshot, default_limits
) -> None:
    ctx = ChainContext(limits=default_limits, mode=ChainMode.PAPER)
    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_low_history(),
        mode=ChainMode.PAPER,
        ctx=ctx,
        snapshot=snapshot,
    )
    assert decision.triggered is True
    assert decision.halt is False
    paper_sink = risk_dir() / "paper" / "breaches.jsonl"
    assert paper_sink.exists()
    rec = json.loads(paper_sink.read_text().splitlines()[0])
    assert rec["rule_id"] == VIX_OUT_OF_BAND_RULE_ID
    assert rec["mode"] == "paper"
    assert rec["would_halt"] is False
    assert rec["extra"]["side"] == "low"


def test_band_gate_whatif_stays_in_memory(
    snapshot: NavSnapshot, ctx_whatif: ChainContext
) -> None:
    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.WHATIF,
        ctx=ctx_whatif,
        snapshot=snapshot,
    )
    assert decision.triggered is True
    assert decision.halt is False
    assert len(ctx_whatif.breaches) == 1
    assert ctx_whatif.breaches[0].rule_id == VIX_OUT_OF_BAND_RULE_ID
    # No filesystem writes in WHATIF.
    assert not (risk_dir() / "breaches.jsonl").exists()
    assert not (risk_dir() / "paper" / "breaches.jsonl").exists()


def test_band_gate_live_writes_to_live_sink(
    snapshot: NavSnapshot, ctx_live: ChainContext
) -> None:
    gate = VixOutOfBandGate()
    gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
    )
    live_sink = risk_dir() / "breaches.jsonl"
    assert live_sink.exists()
    rec = json.loads(live_sink.read_text().splitlines()[0])
    assert rec["rule_id"] == VIX_OUT_OF_BAND_RULE_ID
    assert rec["mode"] == "live"
    assert rec["would_halt"] is True
    assert rec["extra"]["low_pct"] == "40.0"
    assert rec["extra"]["high_pct"] == "80.0"
    assert rec["extra"]["lookback_days"] == "252"


# --- sampler / metrics ----------------------------------------------------


def test_band_gate_paper_emits_halt_metric_with_side_label(
    snapshot: NavSnapshot, default_limits
) -> None:
    from risk_management.metrics_emitter import new_sampler

    ctx = ChainContext(limits=default_limits, mode=ChainMode.PAPER)
    sampler = new_sampler(ChainMode.PAPER, default_limits.config_id)

    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_low_history(),
        mode=ChainMode.PAPER,
        ctx=ctx,
        snapshot=snapshot,
        sampler=sampler,
    )
    assert decision.triggered is True
    assert decision.halt is False
    halt_samples = [
        s for s in sampler.buffer if s.name == "bot_risk_halt_total"
    ]
    assert len(halt_samples) == 1
    sample = halt_samples[0]
    assert sample.labels["rule_id"] == VIX_OUT_OF_BAND_RULE_ID
    assert sample.labels["side"] == "low"
    assert sample.labels["mode"] == "paper"
    assert sample.value == 1.0
    n = sampler.flush()
    assert n == 1


def test_band_gate_whatif_sampler_stays_in_memory(
    snapshot: NavSnapshot, ctx_whatif: ChainContext, default_limits
) -> None:
    from risk_management.metrics_emitter import new_sampler

    sampler = new_sampler(ChainMode.WHATIF, default_limits.config_id)
    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_spike_history(),
        mode=ChainMode.WHATIF,
        ctx=ctx_whatif,
        snapshot=snapshot,
        sampler=sampler,
    )
    assert decision.triggered is True
    assert any(s.name == "bot_risk_halt_total" for s in sampler.buffer)
    n = sampler.flush()
    assert n == 0
    assert not (risk_dir() / "metrics.jsonl").exists()
    assert not (risk_dir() / "paper" / "metrics.jsonl").exists()


def test_band_gate_no_trigger_means_no_metric_emission(
    snapshot: NavSnapshot, ctx_live: ChainContext, default_limits
) -> None:
    from risk_management.metrics_emitter import new_sampler

    sampler = new_sampler(ChainMode.LIVE, default_limits.config_id)
    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_midband_history(),
        mode=ChainMode.LIVE,
        ctx=ctx_live,
        snapshot=snapshot,
        sampler=sampler,
    )
    assert decision.triggered is False
    assert sampler.buffer == []


def test_band_gate_decision_pure_when_ctx_omitted() -> None:
    gate = VixOutOfBandGate()
    decision = gate.evaluate(
        vix_history=_low_history(), mode=ChainMode.WHATIF
    )
    assert isinstance(decision, VixBandGateDecision)
    assert decision.triggered is True
    assert decision.side == "low"
    assert decision.halt is False


def test_vix_high_gate_still_available_alongside_band_gate() -> None:
    """Cycle-13 contract: keep the single-sided gate exported."""

    assert VixHighGate is not None
    assert VIX_HIGH_RULE_ID == "VIX_HIGH_REGIME_HALT"
    assert VIX_OUT_OF_BAND_RULE_ID == "VIX_OUT_OF_BAND_HALT"
    assert VIX_HIGH_RULE_ID != VIX_OUT_OF_BAND_RULE_ID
