"""execution.alpaca_client unit tests (mocked broker)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from execution.alpaca_client import (
    AlpacaClient,
    AlpacaCredentialsMissing,
    derive_client_order_id,
    load_alpaca_config,
)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_load_alpaca_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "K-test")
    monkeypatch.setenv("ALPACA_SECRET", "S-test")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    cfg = load_alpaca_config(dotenv_path=None)
    assert cfg.api_key == "K-test"
    assert cfg.api_secret == "S-test"
    assert cfg.paper is True
    # Redacted form must not leak the secret.
    assert "S-test" not in cfg.redact()


def test_load_alpaca_config_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    # Point dotenv at an empty file so the helper doesn't load the repo .env.
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    with pytest.raises(AlpacaCredentialsMissing):
        load_alpaca_config(dotenv_path=empty_env)


# ---------------------------------------------------------------------------
# client_order_id derivation (Contract 5, minimal form)
# ---------------------------------------------------------------------------


def test_client_order_id_is_deterministic() -> None:
    a = derive_client_order_id("intent-001")
    b = derive_client_order_id("intent-001")
    assert a == b
    assert a.startswith("mmb-")
    assert len(a) <= 48  # Alpaca cap


def test_client_order_id_changes_on_attempt_or_inputs() -> None:
    base = derive_client_order_id("intent-001")
    assert derive_client_order_id("intent-001", attempt=1) != base
    assert derive_client_order_id("intent-002") != base
    assert derive_client_order_id("intent-001", account="LIVE") != base


# ---------------------------------------------------------------------------
# AlpacaClient wrapper — mocked broker
# ---------------------------------------------------------------------------


def test_submit_order_uses_deterministic_coid(make_intent, fake_alpaca) -> None:
    client = AlpacaClient(client=fake_alpaca)
    intent = make_intent(intent_id="intent-XYZ", qty="7")
    coid1 = client.submit_order(intent)
    # Same intent, second submission (e.g. retry path) — same coid
    coid2 = client.submit_order(intent)
    assert coid1 == coid2
    assert len(fake_alpaca.submitted) == 2
    for req in fake_alpaca.submitted:
        assert req.symbol == "AAPL"
        assert req.qty == 7
        assert req.client_order_id == coid1


def test_get_positions_returns_decimal_signed(fake_alpaca) -> None:
    from tests.execution.conftest import _FakePosition

    fake_alpaca.positions = [
        _FakePosition(symbol="AAPL", qty="10", side="long"),
        _FakePosition(symbol="TSLA", qty="5", side="short"),
    ]
    client = AlpacaClient(client=fake_alpaca)
    out = client.get_positions()
    assert out == {"AAPL": Decimal("10"), "TSLA": Decimal("-5")}


def test_get_account_equity_returns_decimal(fake_alpaca) -> None:
    fake_alpaca.equity = "100123.45"
    client = AlpacaClient(client=fake_alpaca)
    assert client.get_account_equity() == Decimal("100123.45")


def test_cancel_all_orders_calls_broker(fake_alpaca) -> None:
    client = AlpacaClient(client=fake_alpaca)
    client.cancel_all_orders()
    client.cancel_all_orders()
    assert fake_alpaca.cancel_calls == 2
