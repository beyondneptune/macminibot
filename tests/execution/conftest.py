"""Fixtures for execution tests.

* Isolates ``MACMINIBOT_STATE_DIR`` to a temp dir so HALT/heartbeat tests
  never touch the operator's real ``~/macminibot/state``.
* Provides a tiny ``FakeAlpacaTradingClient`` satisfying
  :class:`execution.alpaca_client.TradingClientProtocol` so we never hit
  the live API in unit tests.
* Disables the heartbeat-staleness gate by default — individual tests
  that care about heartbeat re-enable it explicitly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from shared_core.contracts import OrderIntent, Side, Symbol, Venue


# ---------------------------------------------------------------------------
# Filesystem isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    monkeypatch.setenv("MACMINIBOT_STATE_DIR", str(state_dir))
    # Default: tests don't exercise heartbeat; disable the freshness gate
    # so HALT/runner tests stay focused.
    monkeypatch.setenv("MACMINIBOT_DISABLE_HEARTBEAT_CHECK", "1")
    return state_dir


@pytest.fixture
def journal_root(tmp_path: Path) -> Path:
    root = tmp_path / "journal"
    root.mkdir(parents=True)
    return root


# ---------------------------------------------------------------------------
# Heartbeat helper
# ---------------------------------------------------------------------------


@pytest.fixture
def write_fresh_heartbeat(isolate_state_dir: Path) -> Any:
    def _writer(age_s: float = 0.0) -> None:
        p = isolate_state_dir / "heartbeat.json"
        payload = {"pid": 1, "ts_unix": time.time() - age_s, "version": "test"}
        p.write_text(json.dumps(payload))

    return _writer


# ---------------------------------------------------------------------------
# Fake Alpaca client
# ---------------------------------------------------------------------------


@dataclass
class _FakeOrderReply:
    id: str
    client_order_id: str
    symbol: str
    qty: int
    side: str


@dataclass
class _FakePosition:
    symbol: str
    qty: str
    side: str = "long"


@dataclass
class _FakeAccount:
    equity: str = "100000.00"


@dataclass
class FakeAlpacaTradingClient:
    """Hand-rolled fake satisfying TradingClientProtocol.

    Records calls so tests can assert idempotency, payload shape, etc.
    """

    submitted: list[Any] = field(default_factory=list)
    positions: list[_FakePosition] = field(default_factory=list)
    equity: str = "100000.00"
    cancel_calls: int = 0
    raise_on_submit: Exception | None = None

    def submit_order(self, order_data):  # noqa: ANN001 — opaque request type
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        self.submitted.append(order_data)
        return _FakeOrderReply(
            id=f"server-{len(self.submitted)}",
            client_order_id=getattr(order_data, "client_order_id", "?"),
            symbol=getattr(order_data, "symbol", "?"),
            qty=int(getattr(order_data, "qty", 0)),
            side=str(getattr(order_data, "side", "?")),
        )

    def get_all_positions(self):
        return list(self.positions)

    def get_account(self):
        return _FakeAccount(equity=self.equity)

    def cancel_orders(self):
        self.cancel_calls += 1


@pytest.fixture
def fake_alpaca() -> FakeAlpacaTradingClient:
    return FakeAlpacaTradingClient()


# ---------------------------------------------------------------------------
# OrderIntent factory
# ---------------------------------------------------------------------------


@pytest.fixture
def make_intent():
    def _make(
        intent_id: str = "intent-001",
        ticker: str = "AAPL",
        side: Side = Side.BUY,
        qty: str = "10",
        mark: str = "150.00",
        batch_id: str = "batch-001",
        decision_id: str = "decision-001",
    ) -> OrderIntent:
        return OrderIntent(
            intent_id=intent_id,
            symbol=Symbol(ticker=ticker, venue=Venue.NASDAQ, currency="USD"),
            side=side,
            delta_qty=Decimal(qty),
            mark=Decimal(mark),
            batch_id=batch_id,
            decision_id=decision_id,
        )

    return _make
