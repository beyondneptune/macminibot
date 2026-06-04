"""Tests for scripts/41_end_of_day_summary — section builders + status."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

# The EOD script lives outside the ``src/`` package tree, so we import it
# by path (cf. existing scripts/ pattern) rather than via Python import.
_REPO = Path(__file__).resolve().parents[2]
_EOD_SCRIPT = _REPO / "scripts" / "41_end_of_day_summary.py"

_spec = importlib.util.spec_from_file_location("eod_summary", _EOD_SCRIPT)
assert _spec and _spec.loader
eod = importlib.util.module_from_spec(_spec)
sys.modules["eod_summary"] = eod
_spec.loader.exec_module(eod)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Stub broker
# ---------------------------------------------------------------------------


@dataclass
class _Acct:
    equity: str
    last_equity: str
    trading_blocked: bool = False
    status: str = "ACTIVE"


@dataclass
class _Pos:
    symbol: str
    qty: str
    current_price: str
    unrealized_pl: str
    unrealized_plpc: str


class _FakeClient:
    def __init__(
        self,
        *,
        equity="100000",
        last_equity="100000",
        positions: list[_Pos] | None = None,
        trading_blocked=False,
        status="ACTIVE",
        orders: list[Any] | None = None,
    ) -> None:
        self._acct = _Acct(equity, last_equity, trading_blocked, status)
        self._positions = positions or []
        self._orders = orders or []

    def get_account(self):
        return self._acct

    def get_all_positions(self):
        return self._positions

    def get_orders(self, filter=None):  # noqa: A002 — alpaca naming
        return self._orders


# ---------------------------------------------------------------------------
# Section tests
# ---------------------------------------------------------------------------


def test_collect_account_computes_pnl_pct() -> None:
    sec = eod.collect_account(_FakeClient(equity="105000", last_equity="100000"))
    assert sec.day_pnl == Decimal("5000")
    assert sec.day_pnl_pct == Decimal("5000") / Decimal("100000")


def test_collect_positions_sorts_by_abs_unrealized() -> None:
    client = _FakeClient(
        positions=[
            _Pos("AAPL", "10", "150", "100", "0.01"),
            _Pos("MSFT", "5", "300", "-500", "-0.05"),
        ]
    )
    sec = eod.collect_positions(client)
    assert sec.rows[0]["symbol"] == "MSFT"
    assert sec.rows[1]["symbol"] == "AAPL"


def test_collect_breaches_filters_to_day(tmp_path: Path) -> None:
    breach_log = tmp_path / "breaches.jsonl"
    day = "2026-05-18"
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    in_window = {
        "rule_id": "DD_DAILY",
        "would_halt": True,
        "ts_unix": start + 3600,
        "observed": "x",
        "threshold": "y",
    }
    out_of_window = {
        "rule_id": "DD_WEEKLY",
        "would_halt": False,
        "ts_unix": start - 86400,
        "observed": "x",
        "threshold": "y",
    }
    breach_log.write_text(
        json.dumps(in_window) + "\n" + json.dumps(out_of_window) + "\n",
        encoding="utf-8",
    )
    sec = eod.collect_breaches(day, [breach_log])
    assert len(sec.breaches) == 1
    assert sec.halt_count == 1
    assert sec.breaches[0]["rule_id"] == "DD_DAILY"


def test_classify_status_green_on_flat_clean_day() -> None:
    account = eod.AccountSection(
        starting_equity=Decimal("100000"),
        ending_equity=Decimal("100050"),
        day_pnl=Decimal("50"),
        day_pnl_pct=Decimal("0.0005"),
        trading_blocked=False,
        account_status="ACTIVE",
    )
    breaches = eod.BreachSection(breaches=[], halt_count=0)
    status, reason = eod.classify_status(account, breaches)
    assert status == "GREEN"


def test_classify_status_yellow_on_minor_loss() -> None:
    account = eod.AccountSection(
        starting_equity=Decimal("100000"),
        ending_equity=Decimal("99700"),
        day_pnl=Decimal("-300"),
        day_pnl_pct=Decimal("-0.003"),
        trading_blocked=False,
        account_status="ACTIVE",
    )
    breaches = eod.BreachSection(breaches=[], halt_count=0)
    status, _ = eod.classify_status(account, breaches)
    assert status == "YELLOW"


def test_classify_status_red_on_drawdown() -> None:
    account = eod.AccountSection(
        starting_equity=Decimal("100000"),
        ending_equity=Decimal("97000"),
        day_pnl=Decimal("-3000"),
        day_pnl_pct=Decimal("-0.03"),
        trading_blocked=False,
        account_status="ACTIVE",
    )
    breaches = eod.BreachSection(breaches=[], halt_count=0)
    status, reason = eod.classify_status(account, breaches)
    assert status == "RED"
    assert "P&L" in reason


def test_classify_status_red_on_trading_blocked() -> None:
    account = eod.AccountSection(
        starting_equity=Decimal("100000"),
        ending_equity=Decimal("100000"),
        day_pnl=Decimal("0"),
        day_pnl_pct=Decimal("0"),
        trading_blocked=True,
        account_status="ACTIVE",
    )
    breaches = eod.BreachSection(breaches=[], halt_count=0)
    status, reason = eod.classify_status(account, breaches)
    assert status == "RED"
    assert "trading_blocked" in reason


def test_summary_render_includes_status_header() -> None:
    summary = eod.build_summary(
        day_iso="2026-05-18",
        trading_client=_FakeClient(equity="97000", last_equity="100000"),
        breach_paths=[],
    )
    md = summary.render()
    assert "Daily summary — 2026-05-18" in md
    assert "Status: **RED**" in md
    assert "## Account" in md
    assert "## Per-symbol P&L" in md


def test_write_summary_creates_dated_md(tmp_path: Path) -> None:
    summary = eod.build_summary(
        day_iso="2026-05-18",
        trading_client=_FakeClient(),
        breach_paths=[],
    )
    out = eod.write_summary(summary, reports_dir=tmp_path)
    assert out == tmp_path / "2026-05-18.md"
    assert out.exists()
    assert "Daily summary" in out.read_text("utf-8")
