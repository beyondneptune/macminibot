"""regime_gate — VIX-high circuit breaker (cycle-10 follow-up).

Backtest cycle-9 finding: the ridge model bleeds Sharpe -0.252 in the
``vix_high`` regime (top quintile of trailing-252d VIX-proxy
distribution). Until the model is retrained with regime-conditioned
weights, we gate trading off entirely while VIX sits in that bucket.

This module is intentionally OUT of the 10-check ``run_check_chain``
sequence — it's a sibling pre-gate the strategy_engine calls before
diffing targets into intents. Reasons:

  1. The 10-check chain is per-intent; VIX is per-batch / per-regime.
     Running it 10x per batch wastes cycles and clutters the breach
     stream with N copies of the same rejection.
  2. ``run_check_chain`` is fail-fast per intent; the regime gate is
     an *all-or-nothing* halt and reads cleaner as a top-level guard.
  3. Cycle-9 finding is model-specific; if a future model handles
     vix_high cleanly we'll drop this gate without touching the
     core chain ordering.

Behavior matrix (matches ``ChainMode``):

  +--------+-----------------------------------------------------------+
  | mode   | high-vol decision                                         |
  +========+===========================================================+
  | LIVE   | HALT — emit BreachEvent (would_halt=True), no orders      |
  | PAPER  | WARN — emit BreachEvent to paper sink, allow paper trades |
  | WHATIF | WARN — collect in ctx.breaches, allow (research path)     |
  +--------+-----------------------------------------------------------+

The PAPER/WHATIF "allow" branches are deliberate: we want paper-trade
P&L data on vix_high days to validate (or invalidate) the cycle-9
finding on out-of-sample windows. Halting paper trading would create
a survivorship-style hole in the dataset.

Cycle 11 add: ``evaluate`` accepts an optional ``sampler`` arg that
emits ``bot_risk_halt_total{rule_id="VIX_HIGH_REGIME_HALT"}`` per
trigger. PAPER-mode wiring preserves OOS gate-firing data alongside
the breach record so the I7e metric stream and the cycle-9 OOS
analysis line up.

Public API:

    gate = VixHighGate(threshold_pct=80, lookback_days=252)
    decision = gate.evaluate(
        vix_history=[...],   # trailing series including the latest tick
        mode=ChainMode.LIVE,
        ctx=chain_context,
        snapshot=nav_snapshot,
    )
    if decision.halt:
        # caller skips order submission
        ...

The ``decision.halt`` flag is True only in LIVE mode on a trigger;
PAPER/WHATIF triggers produce ``halt=False`` but ``triggered=True``
so caller can tag the paper trade with the regime label.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

from shared_core.contracts import ChainMode

from risk_management.breaches import emit_breach, make_breach
from risk_management.chain import ChainContext
from risk_management.metrics_emitter import MetricsSampler
from risk_management.snapshot import NavSnapshot


VIX_HIGH_RULE_ID = "VIX_HIGH_REGIME_HALT"
VIX_OUT_OF_BAND_RULE_ID = "VIX_OUT_OF_BAND_HALT"


@dataclass(frozen=True)
class VixGateDecision:
    """Result of a VixHighGate evaluation.

    Attributes
    ----------
    triggered:
        True when the current VIX-proxy is at/above the percentile
        threshold of the trailing window. Independent of mode.
    halt:
        True only when ``triggered`` and ``mode == LIVE``. Caller MUST
        skip order submission when this is True.
    current_value:
        Latest VIX-proxy tick the gate inspected (Decimal for parity
        with the rest of the risk_management surface).
    threshold_value:
        The computed percentile cutoff over the lookback window. The
        gate triggers when ``current_value >= threshold_value``.
    percentile:
        The percentile setting actually applied (e.g. 80.0).
    sample_size:
        Number of observations used to compute the threshold. Useful
        for the audit log when the trailing window is short.
    reason:
        Short human-readable rationale; surfaces in BreachEvent.extra.
    """

    triggered: bool
    halt: bool
    current_value: Decimal
    threshold_value: Decimal
    percentile: float
    sample_size: int
    reason: str = ""


class VixHighGate:
    """Per-batch VIX regime circuit breaker.

    Parameters
    ----------
    threshold_pct:
        Percentile cutoff (0-100). The cycle-9 backtest split into
        quintiles; ``80`` is the floor of the top quintile.
    lookback_days:
        Trailing window size in observations. The gate uses the LAST
        ``lookback_days`` entries of the supplied series, including
        the latest tick. If fewer observations are present, the gate
        runs on what it has but records ``sample_size`` for audit.

    The gate does NOT fetch VIX data itself — the caller (typically
    strategy_engine) supplies the trailing series. This keeps the gate
    pure / testable and avoids a data-fetch dependency in
    risk_management.
    """

    __slots__ = ("threshold_pct", "lookback_days")

    def __init__(self, *, threshold_pct: float = 80.0, lookback_days: int = 252):
        if not (0.0 < threshold_pct < 100.0):
            raise ValueError(
                f"threshold_pct must be in (0, 100); got {threshold_pct}"
            )
        if lookback_days < 2:
            raise ValueError(
                f"lookback_days must be >= 2; got {lookback_days}"
            )
        self.threshold_pct = float(threshold_pct)
        self.lookback_days = int(lookback_days)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        vix_history: Sequence[float | Decimal | int | str],
        mode: ChainMode,
        ctx: ChainContext | None = None,
        snapshot: NavSnapshot | None = None,
        sampler: MetricsSampler | None = None,
    ) -> VixGateDecision:
        """Evaluate the regime gate.

        ``vix_history`` must be a non-empty sequence whose LAST element
        is the current VIX-proxy value. Earlier elements form the
        trailing distribution.

        ``ctx`` and ``snapshot`` are optional — when supplied, a
        triggered evaluation emits a BreachEvent through the same sink
        as the 10-check chain so the breach stream is a single source
        of truth. WHATIF mode never writes to disk regardless.

        ``sampler`` is optional — when supplied, a triggered evaluation
        emits ``bot_risk_halt_total{rule_id="VIX_HIGH_REGIME_HALT"}``
        regardless of mode. PAPER-mode emission is the primary use case:
        we want the OOS metric stream to record every paper-day the gate
        WOULD have halted so analysts can validate the cycle-9 finding
        against actual paper-trade P&L. WHATIF samples stay in-memory
        only (sampler.flush is a no-op there). Emission failures are
        swallowed — observability must never tank the gate decision.
        """

        if not isinstance(mode, ChainMode):
            raise TypeError(
                f"mode must be ChainMode (got {type(mode).__name__})"
            )

        values = self._coerce_series(vix_history)
        if not values:
            raise ValueError("vix_history must be non-empty")

        # The latest tick is the rightmost element. Trailing distribution
        # is the last ``lookback_days`` entries (which by convention
        # *include* the latest tick — matches pandas rolling-window
        # semantics that downstream backtests already use).
        window = values[-self.lookback_days :]
        current = values[-1]
        threshold = _percentile(window, self.threshold_pct)

        triggered = current >= threshold
        # HALT only in LIVE. PAPER/WHATIF want the paper-trade P&L
        # signal preserved (see module docstring).
        halt = bool(triggered and mode is ChainMode.LIVE)

        reason = (
            f"current={current} >= p{self.threshold_pct:g}({threshold}) "
            f"over n={len(window)} obs"
            if triggered
            else f"current={current} < p{self.threshold_pct:g}({threshold})"
        )

        decision = VixGateDecision(
            triggered=triggered,
            halt=halt,
            current_value=current,
            threshold_value=threshold,
            percentile=self.threshold_pct,
            sample_size=len(window),
            reason=reason,
        )

        if triggered and ctx is not None and snapshot is not None:
            self._emit_breach(decision, mode=mode, ctx=ctx, snapshot=snapshot)

        if triggered and sampler is not None:
            self._emit_metric(sampler)

        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_series(
        series: Iterable[float | Decimal | int | str],
    ) -> list[Decimal]:
        """Coerce arbitrary numeric inputs to a list of Decimals.

        Non-finite values (NaN / Inf) and unparseable strings are
        dropped — the regime gate is a coarse classifier and a single
        bad tick shouldn't crash the pre-trade path. The caller's data
        pipeline is responsible for upstream quality; here we degrade
        gracefully.
        """

        out: list[Decimal] = []
        for raw in series:
            try:
                d = Decimal(str(raw))
            except (InvalidOperation, ValueError):
                continue
            if not d.is_finite():
                continue
            out.append(d)
        return out

    def _emit_breach(
        self,
        decision: VixGateDecision,
        *,
        mode: ChainMode,
        ctx: ChainContext,
        snapshot: NavSnapshot,
    ) -> None:
        """Emit a BreachEvent for a triggered evaluation.

        ``would_halt`` mirrors the LIVE-only halt semantics; PAPER and
        WHATIF triggers still emit a breach record (severity=warn) so
        the regime label is part of the audit trail even when trading
        continues.
        """

        ev = make_breach(
            rule_id=VIX_HIGH_RULE_ID,
            observed=decision.current_value,
            threshold=decision.threshold_value,
            intent_id="",  # gate is per-batch, not per-intent
            snapshot_hash=snapshot.snapshot_version,
            mode=mode,
            ts_ns=ctx.now_ns(),
            would_halt=decision.halt,
            config_id=ctx.limits.config_id if ctx.limits else "",
            symbol="",  # market-wide gate; no per-symbol context
            extra={
                "percentile": str(decision.percentile),
                "sample_size": str(decision.sample_size),
                "lookback_days": str(self.lookback_days),
                "reason": decision.reason,
            },
        )
        ctx.breaches.append(ev)
        emit_breach(ev)  # no-op in WHATIF (see breaches._sink_path_for)

    @staticmethod
    def _emit_metric(sampler: MetricsSampler) -> None:
        """Emit a ``bot_risk_halt_total`` sample for a triggered gate.

        Best-effort: a SCHEMA mismatch or sink failure must NEVER bubble
        out of the gate. Mirrors ``chain._emit_chain_metrics`` semantics —
        observability is downstream of the trading decision.
        """

        try:
            sampler.emit(
                "bot_risk_halt_total",
                1,
                labels={"rule_id": VIX_HIGH_RULE_ID},
            )
        except Exception:
            return


# ---------------------------------------------------------------------------
# Percentile helper — small enough to keep numpy out of the import graph
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[Decimal], pct: float) -> Decimal:
    """Compute the linear-interpolation percentile of a Decimal sequence.

    Matches numpy's default ``percentile`` (linear / type-7) so audit
    comparisons against backtest tooling agree to the last cent.

    For a single-element series, returns that element (the gate then
    trivially triggers on equality — acceptable, since one-tick
    histories are a degenerate case the strategy_engine should
    backfill before invoking).
    """

    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = Decimal(str(rank - lo))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


# ---------------------------------------------------------------------------
# Cycle-13: VixOutOfBandGate — two-sided percentile band gate
# ---------------------------------------------------------------------------
#
# Cycle-12 ml-research found a U-shape in regime conditional Sharpe: BOTH
# ``vix_high`` (top quintile) AND ``vix_low`` (bottom quintile) erode the
# current ridge model's edge. The single-sided VixHighGate only catches the
# top end; this gate halts whenever the latest tick is OUTSIDE the
# [low_pct, high_pct] percentile band of the trailing window.
#
# Same mode dispatch as VixHighGate (LIVE halts, PAPER records, WHATIF
# in-memory) and same audit-trail wiring (BreachEvent + optional
# MetricsSampler). New ``rule_id`` (``VIX_OUT_OF_BAND_HALT``) so analysts
# can disaggregate the two gates in breaches.jsonl / metrics streams.
#
# VixHighGate is INTENTIONALLY kept available — callers may still want
# the one-sided cycle-9 gate during shadow runs / experiments. Production
# strategy_engine wiring should switch to the band gate once cycle-13
# OOS data corroborates the U-shape finding.


@dataclass(frozen=True)
class VixBandGateDecision:
    """Result of a VixOutOfBandGate evaluation.

    Attributes
    ----------
    triggered:
        True when the current tick is BELOW ``low_threshold_value`` or
        AT/ABOVE ``high_threshold_value``. ``side`` carries which.
    halt:
        True only when ``triggered`` and ``mode == LIVE``.
    current_value:
        Latest VIX-proxy tick.
    low_threshold_value, high_threshold_value:
        Computed percentile cutoffs over the lookback window.
    low_pct, high_pct:
        Percentile settings actually applied.
    side:
        ``"low"`` when triggered by the lower band, ``"high"`` for the
        upper, ``""`` when not triggered. Surfaces in BreachEvent.extra
        so analysts can split metric streams by which tail fired.
    sample_size:
        Observations used to compute the thresholds.
    reason:
        Short human-readable rationale; surfaces in BreachEvent.extra.
    """

    triggered: bool
    halt: bool
    current_value: Decimal
    low_threshold_value: Decimal
    high_threshold_value: Decimal
    low_pct: float
    high_pct: float
    side: str
    sample_size: int
    reason: str = ""


class VixOutOfBandGate:
    """Two-sided VIX regime circuit breaker (cycle-13).

    Halts when the latest VIX-proxy tick is OUTSIDE the
    ``[low_pct, high_pct]`` percentile band of the trailing window.
    Trigger semantics:

        triggered = (current < p_low) OR (current >= p_high)

    The upper bound uses ``>=`` to match :class:`VixHighGate`'s boundary
    semantics (so a switch from the one-sided gate to this one doesn't
    change the upper-tail behavior at the cutoff). The lower bound uses
    strict ``<`` so an exact-low-percentile tick passes — symmetric to
    "at/below the high threshold passes" would have been ambiguous;
    we err on the side of *more* trading at the boundary.

    Parameters
    ----------
    low_pct:
        Lower percentile cutoff (0-100). Cycle-12 used 40 as the floor
        of the second-from-bottom quintile.
    high_pct:
        Upper percentile cutoff (0-100). Must exceed ``low_pct``. Cycle-12
        ml-research used 80 as the floor of the top quintile (matches
        cycle-9 finding so the upper-tail semantics line up with
        VixHighGate).
    lookback_days:
        Trailing window size in observations.
    """

    __slots__ = ("low_pct", "high_pct", "lookback_days")

    def __init__(
        self,
        *,
        low_pct: float = 40.0,
        high_pct: float = 80.0,
        lookback_days: int = 252,
    ):
        if not (0.0 < low_pct < 100.0):
            raise ValueError(f"low_pct must be in (0, 100); got {low_pct}")
        if not (0.0 < high_pct < 100.0):
            raise ValueError(f"high_pct must be in (0, 100); got {high_pct}")
        if low_pct >= high_pct:
            raise ValueError(
                f"low_pct ({low_pct}) must be < high_pct ({high_pct})"
            )
        if lookback_days < 2:
            raise ValueError(f"lookback_days must be >= 2; got {lookback_days}")
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)
        self.lookback_days = int(lookback_days)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        vix_history: Sequence[float | Decimal | int | str],
        mode: ChainMode,
        ctx: ChainContext | None = None,
        snapshot: NavSnapshot | None = None,
        sampler: MetricsSampler | None = None,
    ) -> VixBandGateDecision:
        """Evaluate the two-sided regime gate.

        Behavior mirrors :meth:`VixHighGate.evaluate` — see that method's
        docstring for the audit-trail / sampler semantics. The only
        difference is the trigger rule (band exit vs. one-sided
        threshold) and the ``rule_id`` on emitted records.
        """

        if not isinstance(mode, ChainMode):
            raise TypeError(
                f"mode must be ChainMode (got {type(mode).__name__})"
            )

        values = VixHighGate._coerce_series(vix_history)
        if not values:
            raise ValueError("vix_history must be non-empty")

        window = values[-self.lookback_days :]
        current = values[-1]
        low_threshold = _percentile(window, self.low_pct)
        high_threshold = _percentile(window, self.high_pct)

        if current >= high_threshold:
            triggered = True
            side = "high"
        elif current < low_threshold:
            triggered = True
            side = "low"
        else:
            triggered = False
            side = ""

        halt = bool(triggered and mode is ChainMode.LIVE)

        if triggered and side == "high":
            reason = (
                f"current={current} >= p{self.high_pct:g}({high_threshold}) "
                f"over n={len(window)} obs"
            )
        elif triggered and side == "low":
            reason = (
                f"current={current} < p{self.low_pct:g}({low_threshold}) "
                f"over n={len(window)} obs"
            )
        else:
            reason = (
                f"current={current} in [p{self.low_pct:g}({low_threshold}), "
                f"p{self.high_pct:g}({high_threshold}))"
            )

        decision = VixBandGateDecision(
            triggered=triggered,
            halt=halt,
            current_value=current,
            low_threshold_value=low_threshold,
            high_threshold_value=high_threshold,
            low_pct=self.low_pct,
            high_pct=self.high_pct,
            side=side,
            sample_size=len(window),
            reason=reason,
        )

        if triggered and ctx is not None and snapshot is not None:
            self._emit_breach(decision, mode=mode, ctx=ctx, snapshot=snapshot)

        if triggered and sampler is not None:
            self._emit_metric(sampler, side=side)

        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_breach(
        self,
        decision: VixBandGateDecision,
        *,
        mode: ChainMode,
        ctx: ChainContext,
        snapshot: NavSnapshot,
    ) -> None:
        """Emit a BreachEvent for a band-exit trigger.

        The breach's ``threshold`` field is set to the side actually
        violated (low or high) so the audit record matches the
        ``observed`` value's relevant boundary. The OTHER threshold is
        preserved in ``extra`` for completeness.
        """

        if decision.side == "high":
            threshold_value = decision.high_threshold_value
        else:
            threshold_value = decision.low_threshold_value

        ev = make_breach(
            rule_id=VIX_OUT_OF_BAND_RULE_ID,
            observed=decision.current_value,
            threshold=threshold_value,
            intent_id="",
            snapshot_hash=snapshot.snapshot_version,
            mode=mode,
            ts_ns=ctx.now_ns(),
            would_halt=decision.halt,
            config_id=ctx.limits.config_id if ctx.limits else "",
            symbol="",
            extra={
                "side": decision.side,
                "low_pct": str(decision.low_pct),
                "high_pct": str(decision.high_pct),
                "low_threshold": str(decision.low_threshold_value),
                "high_threshold": str(decision.high_threshold_value),
                "sample_size": str(decision.sample_size),
                "lookback_days": str(self.lookback_days),
                "reason": decision.reason,
            },
        )
        ctx.breaches.append(ev)
        emit_breach(ev)

    @staticmethod
    def _emit_metric(sampler: MetricsSampler, *, side: str) -> None:
        """Emit ``bot_risk_halt_total`` with side label for the band gate.

        Best-effort; observability never tanks the gate decision.
        """

        try:
            sampler.emit(
                "bot_risk_halt_total",
                1,
                labels={"rule_id": VIX_OUT_OF_BAND_RULE_ID, "side": side},
            )
        except Exception:
            return


__all__ = [
    "VIX_HIGH_RULE_ID",
    "VIX_OUT_OF_BAND_RULE_ID",
    "VixBandGateDecision",
    "VixGateDecision",
    "VixHighGate",
    "VixOutOfBandGate",
]
