"""diff — TargetPosition -> OrderIntent translator (Epic 13 / CONTEXT.md I4).

Semantics (ratified):
  - Per-symbol delta = ``target_qty - snapshot.positions.get(ticker, 0)``.
  - Symbol in snapshot but not in targets => flatten-to-zero (never "hold").
  - Universe mismatch (target symbol with no mark) => HALT-worthy reject of the
    whole batch (atomicity — no partial-batch execution).
  - Mark-mismatch in I4 maps to ``MarkMissing`` / ``MARK_MISMATCH`` here; the
    chain treats this as a HALT condition (caller emits BreachEvent).

This module produces the imperative delta list; per-intent gating happens
in :mod:`risk_management.chain`.

Epic 26 migration (cycle-7):
  - ``OrderIntent``, ``Side``, ``TargetPosition``, ``TargetPositionBatch`` are
    now re-exported from :mod:`shared_core.contracts` — single source of truth
    for cross-tentacle types.
  - The shared_core ``OrderIntent`` has ``delta_qty``/``mark``/``decision_id``
    fields and a ``Symbol``-typed ``symbol`` (not a bare string). The
    chain-local ``current_qty``/``target_qty``/``ticker-key`` fields move to
    :class:`IntentContext`, a sidecar carried on ``ChainContext`` and looked
    up by ``intent_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping

from shared_core.contracts import (
    OrderIntent,
    Side,
    Symbol,
    TargetPosition,
    TargetPositionBatch,
)

from risk_management.snapshot import NavSnapshot


# ---------------------------------------------------------------------------
# IntentContext (chain-local sidecar — Epic 26)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntentContext:
    """Chain-local context for one ``OrderIntent``, keyed by ``intent_id``.

    The shared_core ``OrderIntent`` schema is the contract-wire shape; it
    intentionally excludes ``current_qty``/``target_qty`` (those are
    chain-local — neither execution nor downstream consumers need them) and
    keeps ``symbol`` as a structured :class:`Symbol`. This sidecar carries
    the extra fields the 10-check chain needs:

    - ``ticker`` — plain string ticker, used as a key into the local
      ``NavSnapshot.positions`` / ``prev_close`` / ``feature_ok`` /
      ``broker_positions`` dicts which remain str-keyed for now (a separate
      epic will migrate the snapshot to ``dict[Symbol, ...]`` to fully
      converge with shared_core ``NavSnapshot``).
    - ``current_qty`` — snapshot position at batch-build time (chain reads
      this rather than re-querying the snapshot to keep the chain
      side-effect-free w.r.t. snapshot mutation between intents).
    - ``target_qty`` — full per-symbol target_qty from the
      ``TargetPositionBatch`` (chain uses this for MAX_SINGLE_NAME_NOTIONAL
      and KELLY_CAP, which gate on target notional, not delta).
    """

    ticker: str
    current_qty: Decimal
    target_qty: Decimal


# ---------------------------------------------------------------------------
# Diff errors (kept local — these are diff-layer raises, not contract types)
# ---------------------------------------------------------------------------


class BatchChecksumError(ValueError):
    """Raised when targets_count doesn't match len(targets).

    Kept local rather than promoting to shared_core: the shared_core
    ``TargetPositionBatch`` raises a plain ``ValueError`` with the same
    message, but the diff layer wants a distinct exception type so callers
    can catch checksum failures separately from other batch errors.
    """


class MarkMissing(ValueError):
    """Raised when a target references a symbol with no mark in market-data."""


@dataclass(frozen=True)
class DiffResult:
    """Output of ``diff_targets_to_intents`` (extended form).

    Carries the emitted intents AND any reasons the whole batch failed
    (which is the atomic-batch HALT signal). The ``intent_contexts`` mapping
    is the IntentContext sidecar — caller folds this into the
    ``ChainContext.intent_contexts`` dict before running the chain.
    """

    intents: list[OrderIntent] = field(default_factory=list)
    intent_contexts: dict[str, IntentContext] = field(default_factory=dict)
    halt_reason: str | None = None
    halt_rule_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ticker_of(sym: Symbol | str) -> str:
    """Extract the plain ticker string for snapshot dict-key lookups.

    Accepts either a :class:`Symbol` (the shared_core canonical) or a bare
    string — the latter is for backwards compatibility with call sites that
    haven't yet migrated to constructing ``Symbol`` instances.
    """

    if isinstance(sym, Symbol):
        return sym.ticker
    return str(sym)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def diff_targets_to_intents(
    batch: TargetPositionBatch,
    snapshot: NavSnapshot,
    *,
    marks: Mapping[str, Decimal] | None = None,
    flatten_missing: bool = True,
    decision_id: str | None = None,
) -> tuple[list[OrderIntent], dict[str, IntentContext]]:
    """Translate a ``TargetPositionBatch`` into a list of ``OrderIntent``.

    Returns a pair: ``(intents, intent_contexts)``. The contexts map
    ``intent_id -> IntentContext`` and the caller installs it on
    ``ChainContext.intent_contexts`` before running the chain.

    Parameters
    ----------
    batch
        The atomic-envelope batch from strategy-engine (shared_core type).
    snapshot
        09:30 ET snapshot — source of current positions (NOT a live IBKR call).
        Position dicts are still str-keyed (see :class:`IntentContext`).
    marks
        Per-ticker mark prices. Required: shared_core ``OrderIntent`` carries
        ``mark`` as a non-optional field. If any target has no mark, raise
        :class:`MarkMissing` so the caller can HALT the whole batch.
    flatten_missing
        If True (default), positions in the snapshot but absent from the
        batch are emitted as SELL-to-zero intents.
    decision_id
        Decision identifier propagated onto every emitted ``OrderIntent``
        (shared_core requires a non-empty ``decision_id``). Defaults to
        ``batch.decision_id`` — cycle-8 lifted ``decision_id`` onto the
        shared_core ``TargetPositionBatch`` itself, so the batch is now
        the canonical source. Explicit override remains supported for
        legacy callers that haven't migrated.

    Raises
    ------
    MarkMissing
        Atomic-batch reject: a target references a symbol with no mark, OR
        ``marks`` was not supplied at all (shared_core ``OrderIntent``
        requires ``mark`` so we cannot construct intents without one).
    ValueError
        Duplicate symbol within the batch.
    """

    if marks is None:
        # shared_core OrderIntent requires mark; no longer optional.
        raise MarkMissing("marks mapping is required (shared_core OrderIntent.mark)")

    decision_id_str = decision_id or batch.decision_id

    targets_by_ticker: dict[str, tuple[Symbol, Decimal]] = {}
    for t in batch.targets:
        ticker = _ticker_of(t.symbol)
        if ticker in targets_by_ticker:
            raise ValueError(f"DUPLICATE_SYMBOL_IN_BATCH: {ticker}")
        targets_by_ticker[ticker] = (t.symbol, t.target_qty)

    missing = [tk for tk in targets_by_ticker if tk not in marks]
    if missing:
        raise MarkMissing(
            f"MARK_MISMATCH: targets reference symbols with no mark: {missing}"
        )

    intents: list[OrderIntent] = []
    contexts: dict[str, IntentContext] = {}

    # Targeted positions: emit BUY/SELL deltas; skip exact-zero deltas (those
    # become check #3 INTENT_NONZERO rejects when present in the chain — but
    # at the diff level a true zero-delta is a no-op, no intent emitted).
    for ticker, (sym, target_qty) in targets_by_ticker.items():
        current = snapshot.position(ticker)
        delta = target_qty - current
        if delta == 0:
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        intent_id = f"{batch.batch_id}:{ticker}"
        intents.append(
            OrderIntent(
                intent_id=intent_id,
                symbol=sym,
                side=side,
                delta_qty=abs(delta),
                mark=marks[ticker],
                batch_id=batch.batch_id,
                decision_id=decision_id_str,
            )
        )
        contexts[intent_id] = IntentContext(
            ticker=ticker, current_qty=current, target_qty=target_qty
        )

    # Flatten-on-missing: positions in snapshot but NOT in targets.
    if flatten_missing:
        for ticker, current in snapshot.positions.items():
            if ticker in targets_by_ticker:
                continue
            if current == 0:
                continue
            if ticker not in marks:
                # No mark for a flatten target either — atomic HALT.
                raise MarkMissing(
                    f"MARK_MISMATCH: flatten target has no mark: {ticker}"
                )
            # delta = 0 - current => SELL |current| if long, BUY |current| if short.
            delta = Decimal("0") - current
            side = Side.BUY if delta > 0 else Side.SELL
            intent_id = f"{batch.batch_id}:{ticker}"
            # Synthesize a Symbol from the ticker. We don't have venue/currency
            # for flatten-only positions in the local snapshot (which still
            # uses str keys); use a sentinel venue/currency. The chain only
            # ever uses ``intent.symbol`` via the IntentContext ticker, not
            # by introspecting venue/currency, so this is safe locally — but
            # downstream (execution) MUST receive intents whose symbols carry
            # real venue/currency. The pending NavSnapshot.positions ->
            # ``dict[Symbol, ...]`` migration (separate epic) closes this gap.
            sym = _synth_symbol_from_ticker(ticker)
            intents.append(
                OrderIntent(
                    intent_id=intent_id,
                    symbol=sym,
                    side=side,
                    delta_qty=abs(delta),
                    mark=marks[ticker],
                    batch_id=batch.batch_id,
                    decision_id=decision_id_str,
                )
            )
            contexts[intent_id] = IntentContext(
                ticker=ticker, current_qty=current, target_qty=Decimal("0")
            )

    return intents, contexts


def _synth_symbol_from_ticker(ticker: str) -> Symbol:
    """Construct a placeholder Symbol for str-keyed snapshot positions.

    Local NavSnapshot still uses ``dict[str, Decimal]`` positions; flatten
    emits intents for symbols only known by ticker. Pin to a stable default
    (NASDAQ/USD) — this is a documented bridge until the snapshot migrates
    to ``dict[Symbol, Decimal]``.
    """

    from shared_core.contracts import Venue

    return Symbol(ticker=ticker, venue=Venue.NASDAQ, currency="USD")


__all__ = [
    "BatchChecksumError",
    "DiffResult",
    "IntentContext",
    "MarkMissing",
    "OrderIntent",
    "Side",
    "TargetPosition",
    "TargetPositionBatch",
    "diff_targets_to_intents",
]
