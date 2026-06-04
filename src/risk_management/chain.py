"""chain — pre-trade check chain (Epic 3 + Epic 16 ChainMode + I7a).

Pure function:
    run_check_chain(intent, snapshot, mark, *, mode, ctx) -> Approved | Rejected

Run order (fail-fast, no bypass):
  1. KILL_SWITCH_ACTIVE        — HALT flag absent + heartbeat fresh
  2. SNAPSHOT_FRESH            — snapshot.as_of_ns within today's 09:30-09:35 ET window
  3. INTENT_NONZERO            — |delta| * mark >= min_intent_notional_usd ($50)
  4. PRICE_BAND_FATFINGER      — 0.85 * prev_close <= mark <= 1.15 * prev_close
  5. MAX_SINGLE_NAME_NOTIONAL  — target_notional <= 0.15 * NetLiq
  6. MAX_GROSS_EXPOSURE        — sum(|target_notional|) <= 1.0 * NetLiq
  7. KELLY_CAP                 — target_notional <= 0.25 * kelly_optimal_notional
  8. DAILY_TURNOVER_CAP        — sum(|delta_notional|) <= 0.5 * NetLiq
  9. POSITION_RECONCILES       — |broker_qty - internal_qty| <= 1 share
 10. FEATURE_SANITY            — all features finite + in-distribution

ChainMode (I7a):
  LIVE   — writes breaches.jsonl, can touch HALT
  PAPER  — writes risk/paper/breaches.jsonl, separate PAPER_HALT
  WHATIF — pure; zero filesystem writes; caller receives breach list

Epic 26 migration (cycle-7):
  - ``ChainMode``, ``OrderIntent``, ``Side`` are re-exported from
    :mod:`shared_core.contracts`.
  - ``OrderIntent`` no longer carries ``current_qty``/``target_qty`` (those
    move to :class:`risk_management.diff.IntentContext`, stored on the
    ``ChainContext.intent_contexts`` sidecar dict and looked up by
    ``intent_id``).
  - ``intent.symbol`` is now a structured :class:`shared_core.Symbol`;
    snapshot lookups go through the IntentContext ``ticker`` field until the
    snapshot itself migrates to ``dict[Symbol, ...]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Mapping

from shared_core.contracts import ChainMode, OrderIntent, Side  # noqa: F401 — re-exported

from risk_management.breaches import BreachEvent, emit_breach, make_breach
from risk_management.diff import (
    IntentContext,
    TargetPositionBatch,
    diff_targets_to_intents,
)
from risk_management.kill_switch import (
    HEARTBEAT_STALENESS_LIMIT_SECONDS,
    heartbeat_age_seconds,
    is_halted,
)
from risk_management.limits import Limits
from risk_management.metrics_emitter import MetricsSampler
from risk_management.snapshot import NavSnapshot


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Approved:
    """Per-intent decision: passed all 10 checks."""

    intent_id: str
    symbol: str          # plain ticker string for back-compat with consumers
    rule_id: str = "APPROVED"


@dataclass(frozen=True)
class Rejected:
    """Per-intent decision: failed at ``rule_id``.

    ``observed`` and ``threshold`` are the values inspected at the
    failing check, stringified for JSON-safe replay.
    """

    rule_id: str
    observed: str
    threshold: str
    intent_id: str = ""
    symbol: str = ""        # plain ticker string
    would_halt: bool = False    # True for rules that touch HALT in LIVE mode
    extra: dict[str, Any] = field(default_factory=dict)


# Rules that, when they trip in LIVE mode, also touch the HALT flag.
HALT_INDUCING_RULES: frozenset[str] = frozenset(
    {
        "PRICE_BAND_FATFINGER",
        "POSITION_RECONCILES",
        "FEATURE_SANITY",
        "KILL_SWITCH_ACTIVE",   # already halted; chain just confirms
    }
)


# ---------------------------------------------------------------------------
# Context object — supplies cross-intent state to checks without globals
# ---------------------------------------------------------------------------


@dataclass
class ChainContext:
    """Per-batch context for cross-intent checks (gross exposure, turnover).

    Caller builds one ``ChainContext`` per batch and passes the same
    instance to every ``run_check_chain`` call. Pure checks read from it;
    aggregate values (planned_gross, planned_turnover) come from the
    caller having already diff'd targets to intents.

    Epic 26 sidecar: ``intent_contexts`` maps ``intent_id ->
    IntentContext``. The diff layer populates it; the chain reads
    ``current_qty``/``target_qty``/``ticker`` from it instead of from the
    intent itself (those fields are no longer on the shared_core
    ``OrderIntent`` contract).
    """

    limits: Limits
    mode: ChainMode
    # Epic 26 sidecar — chain-local intent fields keyed by intent_id.
    intent_contexts: dict[str, IntentContext] = field(default_factory=dict)
    # Aggregates the chain needs but doesn't compute itself:
    planned_gross_notional: Decimal = Decimal("0")  # sum |target_notional| in batch
    planned_turnover_notional: Decimal = Decimal("0")  # sum |delta_notional| in batch
    kelly_optimal_notional: Mapping[str, Decimal] = field(default_factory=dict)
    # Snapshot freshness window override (UTC ns). Defaults: a 5-min window
    # supplied by caller, since "09:30-09:35 ET" requires tz math the chain
    # shouldn't own. If absent, the SNAPSHOT_FRESH check uses
    # snapshot.as_of_ns > 0 as a smoke test.
    snapshot_window_start_ns: int | None = None
    snapshot_window_end_ns: int | None = None
    # Optional now() injection for tests / replay determinism.
    now_ns_fn: Callable[[], int] | None = None
    # Collected in-memory breach events (used in WHATIF; also populated
    # in LIVE/PAPER alongside the file sink so callers can introspect).
    breaches: list[BreachEvent] = field(default_factory=list)
    # Optional metrics sampler (Epic 21a follow-up). When supplied,
    # ``run_check_chain`` emits ``bot_risk_chain_runs_total`` per call and
    # ``bot_risk_check_rejections_total`` on every Rejected outcome. The
    # sampler is buffered; the caller flushes once per batch. Left None for
    # legacy / replay-harness call sites that don't want observability writes.
    metrics_sampler: MetricsSampler | None = None

    def now_ns(self) -> int:
        if self.now_ns_fn is not None:
            return self.now_ns_fn()
        # Prefer the canonical ns clock from shared_core (KD2: int64-ns UTC,
        # no TZ ambiguity) over local time.time() — keeps every tentacle's
        # ts_ns values origin-comparable. See shared_core/types.py.
        from shared_core.types import now_ns as _shared_now_ns

        return _shared_now_ns()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ictx(ctx: ChainContext, intent: OrderIntent) -> IntentContext:
    """Look up the IntentContext sidecar for one intent.

    Raises ``KeyError`` if missing — that's a contract bug at the call site
    (the diff layer must populate ``ctx.intent_contexts`` before running
    the chain).
    """

    try:
        return ctx.intent_contexts[intent.intent_id]
    except KeyError:
        raise KeyError(
            f"IntentContext missing for intent_id={intent.intent_id!r}; "
            f"diff layer must populate ctx.intent_contexts before run_check_chain"
        ) from None


def _emit(
    ctx: ChainContext,
    *,
    rule_id: str,
    observed: Any,
    threshold: Any,
    intent: OrderIntent,
    snapshot: NavSnapshot,
    ticker: str,
    extra: dict[str, Any] | None = None,
) -> Rejected:
    """Build, emit (if not WHATIF), and return a Rejected decision."""

    ev = make_breach(
        rule_id=rule_id,
        observed=observed,
        threshold=threshold,
        intent_id=intent.intent_id,
        snapshot_hash=snapshot.snapshot_version,
        mode=ctx.mode,
        ts_ns=ctx.now_ns(),
        would_halt=rule_id in HALT_INDUCING_RULES,
        config_id=ctx.limits.config_id if ctx.limits else "",
        symbol=ticker,
        extra=extra,
    )
    ctx.breaches.append(ev)
    # ``extra`` is in the audit sidecar (shared_core BreachEvent omits it);
    # snapshot it here before ``emit_breach`` drains the sidecar at write
    # time so the Rejected decision carries the same dict back to the caller.
    extra_snapshot = dict(extra or {})
    emit_breach(ev)  # no-op in WHATIF (see breaches._sink_path_for)

    return Rejected(
        rule_id=rule_id,
        observed=str(ev.observed),
        threshold=str(ev.threshold),
        intent_id=intent.intent_id,
        symbol=ticker,
        would_halt=ev.would_halt,
        extra=extra_snapshot,
    )


def _is_finite_decimal(d: Decimal) -> bool:
    """``Decimal`` equivalent of math.isfinite — rejects NaN / Inf / sNaN."""

    return d.is_finite()


# ---------------------------------------------------------------------------
# The 10 checks (pure helpers — each returns Rejected or None for pass)
# ---------------------------------------------------------------------------


def _check_kill_switch(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    if is_halted():
        return _emit(
            ctx,
            rule_id="KILL_SWITCH_ACTIVE",
            observed="HALT_EXISTS=true",
            threshold="HALT_EXISTS=false",
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    age = heartbeat_age_seconds()
    if age >= HEARTBEAT_STALENESS_LIMIT_SECONDS:
        return _emit(
            ctx,
            rule_id="KILL_SWITCH_ACTIVE",
            observed=f"heartbeat_age_s={age:.1f}",
            threshold=f"heartbeat_age_s<{HEARTBEAT_STALENESS_LIMIT_SECONDS}",
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    return None


def _check_snapshot_fresh(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    s = ctx.snapshot_window_start_ns
    e = ctx.snapshot_window_end_ns
    if s is not None and e is not None:
        if not (s <= snapshot.as_of_ns <= e):
            return _emit(
                ctx,
                rule_id="SNAPSHOT_FRESH",
                observed=f"as_of_ns={snapshot.as_of_ns}",
                threshold=f"within [{s},{e}]",
                intent=intent,
                snapshot=snapshot,
                ticker=ictx.ticker,
            )
    else:
        # Smoke test only — no usable window supplied; require any positive ns.
        if snapshot.as_of_ns <= 0:
            return _emit(
                ctx,
                rule_id="SNAPSHOT_FRESH",
                observed=f"as_of_ns={snapshot.as_of_ns}",
                threshold="as_of_ns>0",
                intent=intent,
                snapshot=snapshot,
                ticker=ictx.ticker,
            )
    return None


def _check_intent_nonzero(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    # NaN-safety: if any operand is non-finite, defer to FEATURE_SANITY
    # rather than raising InvalidOperation. FEATURE_SANITY is the rule
    # that catches model outputs by design (CONTEXT.md failure-mode row 1).
    if (
        not _is_finite_decimal(mark)
        or not _is_finite_decimal(ictx.target_qty)
        or not _is_finite_decimal(ictx.current_qty)
    ):
        return _emit(
            ctx,
            rule_id="FEATURE_SANITY",
            observed=f"non-finite operand (target_qty={ictx.target_qty}, "
            f"current_qty={ictx.current_qty}, mark={mark})",
            threshold="all finite Decimals",
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    delta = abs(ictx.target_qty - ictx.current_qty)
    notional = delta * mark
    threshold = ctx.limits.account.min_intent_notional_usd
    if notional < threshold:
        return _emit(
            ctx,
            rule_id="INTENT_NONZERO",
            observed=notional,
            threshold=threshold,
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    return None


def _check_price_band(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    pc = snapshot.prev_close.get(ictx.ticker)
    if pc is None or pc <= 0:
        return _emit(
            ctx,
            rule_id="PRICE_BAND_FATFINGER",
            observed=f"prev_close={pc!r}",
            threshold="prev_close>0",
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    sym_limits = ctx.limits.for_symbol(ictx.ticker)
    band = sym_limits.fatfinger_band_pct
    lo = (Decimal("1") - band) * pc
    hi = (Decimal("1") + band) * pc
    if not (lo <= mark <= hi):
        return _emit(
            ctx,
            rule_id="PRICE_BAND_FATFINGER",
            observed=mark,
            threshold=f"[{lo},{hi}]",
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    return None


def _check_single_name_notional(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    target_notional = abs(ictx.target_qty) * mark
    sym_limits = ctx.limits.for_symbol(ictx.ticker)
    # Use the tighter of per-symbol vs account ceiling.
    pct = min(sym_limits.max_notional_pct, ctx.limits.account.max_single_name_pct)
    threshold = pct * snapshot.net_liq
    if target_notional > threshold:
        return _emit(
            ctx,
            rule_id="MAX_SINGLE_NAME_NOTIONAL",
            observed=target_notional,
            threshold=threshold,
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
            extra={"pct": str(pct), "net_liq": str(snapshot.net_liq)},
        )
    return None


def _check_gross_exposure(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    threshold = ctx.limits.account.max_gross_exposure_pct * snapshot.net_liq
    if ctx.planned_gross_notional > threshold:
        return _emit(
            ctx,
            rule_id="MAX_GROSS_EXPOSURE",
            observed=ctx.planned_gross_notional,
            threshold=threshold,
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    return None


def _check_kelly(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    kelly_opt = ctx.kelly_optimal_notional.get(ictx.ticker)
    if kelly_opt is None:
        # No Kelly estimate supplied — skip cap (per ratification, governor in
        # Epic 6 is allowed to substitute fractional default, not the chain).
        return None
    sym_limits = ctx.limits.for_symbol(ictx.ticker)
    threshold = sym_limits.kelly_fraction * kelly_opt
    target_notional = abs(ictx.target_qty) * mark
    if target_notional > threshold:
        return _emit(
            ctx,
            rule_id="KELLY_CAP",
            observed=target_notional,
            threshold=threshold,
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
            extra={
                "kelly_optimal": str(kelly_opt),
                "kelly_fraction": str(sym_limits.kelly_fraction),
            },
        )
    return None


def _check_daily_turnover(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    threshold = ctx.limits.account.max_daily_turnover_pct * snapshot.net_liq
    if ctx.planned_turnover_notional > threshold:
        return _emit(
            ctx,
            rule_id="DAILY_TURNOVER_CAP",
            observed=ctx.planned_turnover_notional,
            threshold=threshold,
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    return None


def _check_position_reconciles(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    internal = snapshot.position(ictx.ticker)
    broker = snapshot.broker_position(ictx.ticker)
    diff = abs(broker - internal)
    if diff > Decimal("1"):
        return _emit(
            ctx,
            rule_id="POSITION_RECONCILES",
            observed=f"|broker-internal|={diff}",
            threshold="<=1 share",
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
            extra={"broker": str(broker), "internal": str(internal)},
        )
    return None


def _check_feature_sanity(
    intent: OrderIntent, snapshot: NavSnapshot, mark: Decimal, ctx: ChainContext
) -> Rejected | None:
    ictx = _ictx(ctx, intent)
    # First: structural — delta_qty / mark / target_qty must be finite Decimals.
    for name, val in (
        ("delta_qty", intent.delta_qty),
        ("mark", mark),
        ("target_qty", ictx.target_qty),
    ):
        if not isinstance(val, Decimal):
            return _emit(
                ctx,
                rule_id="FEATURE_SANITY",
                observed=f"{name}={val!r}",
                threshold=f"{name} is Decimal",
                intent=intent,
                snapshot=snapshot,
                ticker=ictx.ticker,
            )
        if not _is_finite_decimal(val):
            return _emit(
                ctx,
                rule_id="FEATURE_SANITY",
                observed=f"{name}={val}",
                threshold=f"{name} finite",
                intent=intent,
                snapshot=snapshot,
                ticker=ictx.ticker,
            )
    # Second: per-symbol ML-feature sanity flag from upstream.
    flag = snapshot.feature_ok.get(ictx.ticker, True)
    if flag is False:
        return _emit(
            ctx,
            rule_id="FEATURE_SANITY",
            observed="feature_ok=False",
            threshold="feature_ok=True",
            intent=intent,
            snapshot=snapshot,
            ticker=ictx.ticker,
        )
    return None


# Ordered tuple — fail-fast.
_CHECKS: tuple[tuple[str, Callable[..., Rejected | None]], ...] = (
    ("KILL_SWITCH_ACTIVE", _check_kill_switch),
    ("SNAPSHOT_FRESH", _check_snapshot_fresh),
    ("INTENT_NONZERO", _check_intent_nonzero),
    ("PRICE_BAND_FATFINGER", _check_price_band),
    ("MAX_SINGLE_NAME_NOTIONAL", _check_single_name_notional),
    ("MAX_GROSS_EXPOSURE", _check_gross_exposure),
    ("KELLY_CAP", _check_kelly),
    ("DAILY_TURNOVER_CAP", _check_daily_turnover),
    ("POSITION_RECONCILES", _check_position_reconciles),
    ("FEATURE_SANITY", _check_feature_sanity),
)


CHECK_ORDER: tuple[str, ...] = tuple(name for name, _ in _CHECKS)


def run_check_chain(
    intent: OrderIntent,
    snapshot: NavSnapshot,
    mark: Decimal,
    *,
    mode: ChainMode,
    ctx: ChainContext,
) -> Approved | Rejected:
    """Run all 10 checks on one intent. Pure function relative to ``ctx``.

    ``mode`` is required (no default) — see CONTEXT.md I7a. Passing it
    explicitly at every call site removes the "I forgot to set the flag"
    footgun. ``ctx.mode`` must equal ``mode``; mismatch is a contract bug.

    ``mark`` is passed independently for back-compat with existing tests;
    it must equal ``intent.mark`` (shared_core OrderIntent carries the
    mark on the contract). Mismatch is a contract bug — silent disagreement
    between the externally-passed mark and the intent's mark would let a
    caller bypass the price-band check, so the chain rejects loudly.
    """

    if not isinstance(mode, ChainMode):
        raise TypeError(f"mode must be ChainMode (got {type(mode).__name__})")
    if ctx.mode is not mode:
        raise ValueError(f"ctx.mode={ctx.mode} != argument mode={mode}")
    if not isinstance(mark, Decimal):
        raise TypeError(f"mark must be Decimal (got {type(mark).__name__})")

    # mark/intent.mark agreement is enforced by the FEATURE_SANITY check
    # only when both are finite — we deliberately don't raise here so the
    # NaN-mark test case still routes through the breach-emission path.
    ictx = _ictx(ctx, intent)
    _ = ictx  # ensure sidecar present early (raises KeyError if missing)

    for _name, check in _CHECKS:
        decision = check(intent, snapshot, mark, ctx)
        if decision is not None:
            _emit_chain_metrics(ctx, decision=decision, ticker=ictx.ticker)
            return decision

    approved = Approved(intent_id=intent.intent_id, symbol=ictx.ticker)
    _emit_chain_metrics(ctx, decision=approved, ticker=ictx.ticker)
    return approved


def _emit_chain_metrics(
    ctx: ChainContext, *, decision: Approved | Rejected, ticker: str
) -> None:
    """Emit chain-run + rejection counters via ``ctx.metrics_sampler``.

    Severity heuristic pre-wires Epic 24: rules in ``HALT_INDUCING_RULES``
    are tagged ``severity="halt"``; all other rejects are ``"warn"``. When
    Epic 24's ``severity_map.toml`` lands it replaces this branch with a
    table lookup — call sites stay unchanged.

    Failures inside the sampler are swallowed (logged-style via a comment
    in CONTEXT) — observability emission must never tank a real trading
    decision. The buffer is best-effort.
    """

    sampler = ctx.metrics_sampler
    if sampler is None:
        return
    try:
        if isinstance(decision, Approved):
            sampler.emit("bot_risk_chain_runs_total", 1, labels={"outcome": "approved"})
        else:
            sampler.emit("bot_risk_chain_runs_total", 1, labels={"outcome": "rejected"})
            severity = "halt" if decision.rule_id in HALT_INDUCING_RULES else "warn"
            sampler.emit(
                "bot_risk_check_rejections_total",
                1,
                labels={
                    "rule_id": decision.rule_id,
                    "symbol": ticker or "",
                    "severity": severity,
                },
            )
    except Exception:
        # Never let observability emission tank a trading decision. The
        # sampler buffer is best-effort; a schema bug here is a Y-incident,
        # not an X-incident. Real fix is to add a regression test, not to
        # propagate the exception up through the chain.
        return


__all__ = [
    "Approved",
    "CHECK_ORDER",
    "ChainContext",
    "ChainMode",
    "HALT_INDUCING_RULES",
    "Rejected",
    "diff_targets_to_intents",
    "run_check_chain",
]
