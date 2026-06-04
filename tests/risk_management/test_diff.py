"""Tests for diff_targets_to_intents (CONTEXT.md I4).

Epic 26 (cycle-7): targets and intents use ``shared_core.contracts`` types.
``TargetPosition`` requires ``hold_reason``/``reason_trace``;
``TargetPositionBatch`` requires ``generated_at_ns``/``supersedes``/
``targets_count``/``nav_snapshot_version`` and uses tuple-typed targets.
``OrderIntent`` carries ``mark`` and ``decision_id`` and uses Symbol.
``current_qty``/``target_qty`` move to the ``IntentContext`` sidecar
returned alongside the intents.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from shared_core.contracts import ChainMode, Venue

from risk_management.diff import (
    MarkMissing,
    Side,
    TargetPosition,
    TargetPositionBatch,
    diff_targets_to_intents,
)
from risk_management.snapshot import NavSnapshot

from tests.risk_management.conftest import make_symbol


def _batch(*targets: tuple[str, str]) -> TargetPositionBatch:
    """Build a shared_core ``TargetPositionBatch`` with reasonable defaults."""

    tp = tuple(
        TargetPosition(
            symbol=make_symbol(s),
            target_qty=Decimal(q),
            hold_reason=None,
            reason_trace={},
        )
        for s, q in targets
    )
    return TargetPositionBatch(
        batch_id="test",
        decision_id="test",
        mode=ChainMode.PAPER,
        solver_status="optimal",
        generated_at_ns=1,
        supersedes=None,
        targets_count=len(tp),
        targets=tp,
        nav_snapshot_version="sha256:test",
    )


def _marks(*entries: tuple[str, str]) -> dict[str, Decimal]:
    return {ticker: Decimal(price) for ticker, price in entries}


DEFAULT_MARKS = _marks(
    ("AAPL", "150"), ("MSFT", "300"), ("TSLA", "200"), ("NVDA", "500")
)


def test_target_100_current_80_emits_buy_20(snapshot: NavSnapshot) -> None:
    """Deliverable: target=100 + current=80 → OrderIntent BUY 20."""

    batch = _batch(("AAPL", "100"))
    intents, contexts = diff_targets_to_intents(batch, snapshot, marks=DEFAULT_MARKS)
    assert len(intents) == 1
    intent = intents[0]
    assert intent.symbol.ticker == "AAPL"
    assert intent.symbol.venue is Venue.NASDAQ
    assert intent.side is Side.BUY
    assert intent.delta_qty == Decimal("20")
    assert intent.mark == Decimal("150")
    assert intent.batch_id == "test"
    assert intent.decision_id == "test"
    ictx = contexts[intent.intent_id]
    assert ictx.ticker == "AAPL"
    assert ictx.target_qty == Decimal("100")
    assert ictx.current_qty == Decimal("80")


def test_flatten_on_missing(snapshot: NavSnapshot) -> None:
    """Deliverable: target=missing + current=50 → OrderIntent SELL 50.

    snapshot.positions has AAPL=80; an empty target batch should flatten it.
    """

    # Add another position so we can verify flatten behavior independently.
    snapshot.positions["MSFT"] = Decimal("50")
    batch = _batch()  # no targets
    intents, contexts = diff_targets_to_intents(batch, snapshot, marks=DEFAULT_MARKS)
    by_ticker = {contexts[i.intent_id].ticker: i for i in intents}
    assert by_ticker["AAPL"].side is Side.SELL
    assert by_ticker["AAPL"].delta_qty == Decimal("80")
    assert by_ticker["MSFT"].side is Side.SELL
    assert by_ticker["MSFT"].delta_qty == Decimal("50")


def test_no_op_when_target_equals_current(snapshot: NavSnapshot) -> None:
    batch = _batch(("AAPL", "80"))  # already long 80
    intents, _ = diff_targets_to_intents(batch, snapshot, marks=DEFAULT_MARKS)
    assert intents == []


def test_short_emits_sell(snapshot: NavSnapshot) -> None:
    batch = _batch(("AAPL", "30"))  # decrease from 80 to 30
    intents, _ = diff_targets_to_intents(batch, snapshot, marks=DEFAULT_MARKS)
    assert intents[0].side is Side.SELL
    assert intents[0].delta_qty == Decimal("50")


def test_new_position_emits_buy(snapshot: NavSnapshot) -> None:
    batch = _batch(("TSLA", "10"))  # new position
    intents, contexts = diff_targets_to_intents(batch, snapshot, marks=DEFAULT_MARKS)
    # AAPL not in targets => flatten; TSLA new => buy 10.
    by_ticker = {contexts[i.intent_id].ticker: i for i in intents}
    assert by_ticker["TSLA"].side is Side.BUY
    assert by_ticker["TSLA"].delta_qty == Decimal("10")
    assert by_ticker["AAPL"].side is Side.SELL


def test_mark_missing_raises(snapshot: NavSnapshot) -> None:
    """Atomic-batch: target with no mark => raise (caller HALTs)."""

    batch = _batch(("AAPL", "100"), ("NVDA", "5"))
    with pytest.raises(MarkMissing, match="NVDA"):
        diff_targets_to_intents(batch, snapshot, marks={"AAPL": Decimal("150")})


def test_batch_checksum_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="targets_count"):
        TargetPositionBatch(
            batch_id="x",
            decision_id="x",
            mode=ChainMode.PAPER,
            solver_status="optimal",
            generated_at_ns=1,
            supersedes=None,
            targets_count=99,
            targets=(
                TargetPosition(
                    symbol=make_symbol("AAPL"),
                    target_qty=Decimal("100"),
                    hold_reason=None,
                    reason_trace={},
                ),
            ),
            nav_snapshot_version="sha256:test",
        )


def test_flatten_disabled(snapshot: NavSnapshot) -> None:
    batch = _batch()  # no targets
    intents, _ = diff_targets_to_intents(
        batch, snapshot, marks=DEFAULT_MARKS, flatten_missing=False
    )
    assert intents == []
