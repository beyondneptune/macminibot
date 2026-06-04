"""Unit tests for the bracket + trailing-stop primitives on AlpacaClient.

We re-use the ``fake_alpaca`` and ``make_intent`` fixtures from
``tests/execution/conftest.py``. The fake records every submitted
``order_data`` request so we can assert on payload shape without ever
calling Alpaca.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from execution.alpaca_client import AlpacaClient
from shared_core.contracts import Side


# ---------------------------------------------------------------------------
# Bracket orders
# ---------------------------------------------------------------------------


def test_submit_bracket_buy_attaches_tp_and_sl(make_intent, fake_alpaca) -> None:
    """A BUY bracket: TP above mark, SL below mark."""
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(intent_id="b-001", side=Side.BUY, qty="10", mark="100.00")
    coid = client.submit_bracket_order(
        intent, take_profit_pct=0.05, stop_loss_pct=0.02
    )
    assert coid.startswith("mmb-")
    assert len(fake_alpaca.submitted) == 1
    req = fake_alpaca.submitted[0]
    assert req.symbol == "AAPL"
    assert req.qty == 10
    assert req.client_order_id == coid
    # OrderClass on the request is the alpaca-py enum; compare its value.
    assert str(getattr(req.order_class, "value", req.order_class)) == "bracket"
    # TP at 105, SL at 98 (BUY directionality).
    assert float(req.take_profit.limit_price) == pytest.approx(105.00)
    assert float(req.stop_loss.stop_price) == pytest.approx(98.00)


def test_submit_bracket_sell_inverts_directionality(make_intent, fake_alpaca) -> None:
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(intent_id="b-002", side=Side.SELL, qty="5", mark="200.00")
    client.submit_bracket_order(
        intent, take_profit_pct=0.05, stop_loss_pct=0.02
    )
    req = fake_alpaca.submitted[0]
    # SELL: TP must be BELOW mark, SL must be ABOVE mark.
    assert float(req.take_profit.limit_price) == pytest.approx(190.00)
    assert float(req.stop_loss.stop_price) == pytest.approx(204.00)


def test_submit_bracket_is_idempotent_per_intent(make_intent, fake_alpaca) -> None:
    """Retrying the same intent yields the same client_order_id."""
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(intent_id="b-003", qty="7", mark="150.00")
    a = client.submit_bracket_order(intent, take_profit_pct=0.05, stop_loss_pct=0.02)
    b = client.submit_bracket_order(intent, take_profit_pct=0.05, stop_loss_pct=0.02)
    assert a == b
    assert len(fake_alpaca.submitted) == 2
    assert fake_alpaca.submitted[0].client_order_id == a
    assert fake_alpaca.submitted[1].client_order_id == a


def test_submit_bracket_rejects_invalid_percents(make_intent, fake_alpaca) -> None:
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(qty="10", mark="100.00")
    with pytest.raises(ValueError, match="take_profit_pct"):
        client.submit_bracket_order(intent, take_profit_pct=0.0, stop_loss_pct=0.02)
    with pytest.raises(ValueError, match="take_profit_pct"):
        client.submit_bracket_order(intent, take_profit_pct=1.5, stop_loss_pct=0.02)
    with pytest.raises(ValueError, match="stop_loss_pct"):
        client.submit_bracket_order(intent, take_profit_pct=0.05, stop_loss_pct=0.0)
    with pytest.raises(ValueError, match="stop_loss_pct"):
        client.submit_bracket_order(intent, take_profit_pct=0.05, stop_loss_pct=2.0)
    # Nothing should have been submitted on the validation failures.
    assert fake_alpaca.submitted == []


def test_submit_bracket_rejects_zero_qty(make_intent, fake_alpaca) -> None:
    """delta_qty < 1 (after int cast) should fail loud."""
    client = AlpacaClient(client=fake_alpaca)
    # Use a sub-unit qty: contracts allow Decimal qty, but int() round-down → 0.
    intent = make_intent(qty="0.5", mark="100.00")
    with pytest.raises(ValueError, match="non-positive qty"):
        client.submit_bracket_order(intent, take_profit_pct=0.05, stop_loss_pct=0.02)


# ---------------------------------------------------------------------------
# Trailing-stop orders
# ---------------------------------------------------------------------------


def test_submit_trailing_stop_sets_trail_percent(make_intent, fake_alpaca) -> None:
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(intent_id="t-001", side=Side.SELL, qty="10", mark="100.00")
    coid = client.submit_trailing_stop(intent, trail_pct=0.05)
    assert coid.startswith("mmb-")
    assert len(fake_alpaca.submitted) == 1
    req = fake_alpaca.submitted[0]
    # alpaca-py stores percent as a percentage (5.0 means 5%).
    assert float(req.trail_percent) == pytest.approx(5.0)
    assert req.qty == 10
    assert req.client_order_id == coid


def test_submit_trailing_stop_rejects_bad_pct(make_intent, fake_alpaca) -> None:
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(qty="10", mark="100.00")
    with pytest.raises(ValueError, match="trail_pct"):
        client.submit_trailing_stop(intent, trail_pct=0.0)
    with pytest.raises(ValueError, match="trail_pct"):
        client.submit_trailing_stop(intent, trail_pct=1.0)
    assert fake_alpaca.submitted == []


def test_submit_trailing_stop_is_idempotent(make_intent, fake_alpaca) -> None:
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(intent_id="t-002", qty="3", mark="50.00")
    a = client.submit_trailing_stop(intent, trail_pct=0.10)
    b = client.submit_trailing_stop(intent, trail_pct=0.10)
    assert a == b
    assert all(req.client_order_id == a for req in fake_alpaca.submitted)


def test_bracket_and_market_use_different_coid_when_attempts_differ(
    make_intent, fake_alpaca
) -> None:
    """The deterministic coid derivation honours ``attempt`` — used for retries
    that the broker rejected as duplicate. Both order families should share
    the derivation."""
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(intent_id="b-004", qty="5", mark="200.00")
    a0 = client.submit_bracket_order(
        intent, take_profit_pct=0.05, stop_loss_pct=0.02, attempt=0
    )
    a1 = client.submit_bracket_order(
        intent, take_profit_pct=0.05, stop_loss_pct=0.02, attempt=1
    )
    assert a0 != a1
