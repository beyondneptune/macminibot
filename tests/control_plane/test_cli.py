"""Tests for the control_plane operator CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from control_plane import cli
from risk_management.kill_switch import (
    audit_log_path,
    halt_path,
    is_halted,
)


class _FakePosition:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeClient:
    def __init__(self, positions: list[_FakePosition] | None = None) -> None:
        self._positions = positions or []
        self.cancel_called = False

    def get_all_positions(self) -> list[_FakePosition]:
        return self._positions

    def cancel_orders(self) -> list[dict[str, Any]]:
        self.cancel_called = True
        return [{"id": "ord-1"}, {"id": "ord-2"}]


# -------------------------------------------------------------------------
# halt / unhalt
# -------------------------------------------------------------------------


def test_halt_sets_flag_and_audits(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["halt", "--reason", "smoke test"])
    assert rc == 0
    assert is_halted() is True
    assert halt_path().exists()
    out = capsys.readouterr().out
    assert "HALT set" in out
    audit = audit_log_path().read_text(encoding="utf-8").strip().splitlines()
    last = json.loads(audit[-1])
    assert last["action"] == "HALT"
    assert last["reason"] == "smoke test"


def test_unhalt_clears_flag_and_audits(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["halt", "--reason", "first"])
    rc = cli.main(["unhalt", "--reason", "all clear"])
    assert rc == 0
    assert is_halted() is False
    assert not halt_path().exists()
    out = capsys.readouterr().out
    assert "HALT cleared" in out
    audit = audit_log_path().read_text(encoding="utf-8").strip().splitlines()
    actions = [json.loads(line)["action"] for line in audit]
    assert "HALT" in actions and "UNHALT" in actions


def test_halt_requires_reason() -> None:
    with pytest.raises(SystemExit):
        cli.main(["halt"])


def test_halt_empty_reason_rejected() -> None:
    # kill_switch.halt raises ValueError on empty/whitespace reason.
    with pytest.raises(ValueError):
        cli.main(["halt", "--reason", "   "])


# -------------------------------------------------------------------------
# status
# -------------------------------------------------------------------------


def test_status_reports_unhalted_with_no_positions(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_trading_client", lambda: _FakeClient())
    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "halted         : False" in out
    assert "0 open" in out


def test_status_reports_halted_with_last_audit(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_trading_client", lambda: _FakeClient())
    cli.main(["halt", "--reason", "audit me"])
    capsys.readouterr()  # drain
    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "halted         : True" in out
    assert "audit me" in out


def test_status_survives_broker_failure(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> Any:
        raise RuntimeError("no creds")

    monkeypatch.setattr(cli, "_trading_client", _boom)
    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "UNAVAILABLE" in out


# -------------------------------------------------------------------------
# positions
# -------------------------------------------------------------------------


def test_positions_lists_open(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient(
        positions=[
            _FakePosition(
                symbol="AAPL",
                qty="10",
                avg_entry_price="150.00",
                market_value="1600.00",
                unrealized_pl="100.00",
                side="long",
            )
        ]
    )
    monkeypatch.setattr(cli, "_trading_client", lambda: fake)
    rc = cli.main(["positions"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "AAPL" in out
    assert "1600.00" in out


def test_positions_empty(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_trading_client", lambda: _FakeClient())
    rc = cli.main(["positions"])
    assert rc == 0
    assert "no open positions" in capsys.readouterr().out


# -------------------------------------------------------------------------
# kill
# -------------------------------------------------------------------------


def test_kill_calls_cancel_all(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(cli, "_trading_client", lambda: fake)
    # Force the fallback path (no execution tentacle) by stubbing the cancel-all
    # helper to use our fake client.
    monkeypatch.setattr(cli, "_cancel_all_orders", lambda: fake.cancel_orders())
    rc = cli.main(["kill"])
    assert rc == 0
    assert fake.cancel_called is True
    assert "cancel-all dispatched" in capsys.readouterr().out
