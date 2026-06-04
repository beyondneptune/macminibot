"""treasury.balance_ledger — snapshot persistence and reconciliation tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from treasury.balance_ledger import BalanceLedger, PositionBreak, reconcile


# ---------------------------------------------------------------------------
# Fake Alpaca account / position objects (attribute-access surface)
# ---------------------------------------------------------------------------


@dataclass
class _FakeAccount:
    account_id: str = "PA-TEST-001"
    equity: str = "100000.00"
    cash: str = "55000.50"
    buying_power: str = "200000.00"
    long_market_value: str = "45000.50"
    short_market_value: str = "0.00"
    portfolio_value: str = "100000.00"
    daytrade_count: int = 0
    pattern_day_trader: bool = False
    status: str = "ACTIVE"


# ---------------------------------------------------------------------------
# reconcile (pure function)
# ---------------------------------------------------------------------------


def test_reconcile_clean_returns_empty_list() -> None:
    internal = {"AAPL": Decimal("100"), "MSFT": Decimal("50")}
    venue = {"AAPL": Decimal("100"), "MSFT": Decimal("50")}
    assert reconcile(internal, venue) == []


def test_reconcile_two_symbol_break_case() -> None:
    """The canonical 2-symbol sample from the deliverable spec.

    Internal book thinks we own 100 AAPL + 50 MSFT, but Alpaca says we
    have 95 AAPL (we missed a 5-share sell fill) and 60 MSFT (extra 10
    shares from a fill we didn't record).
    """
    internal = {"AAPL": Decimal("100"), "MSFT": Decimal("50")}
    venue = {"AAPL": Decimal("95"), "MSFT": Decimal("60")}
    breaks = reconcile(internal, venue)
    assert len(breaks) == 2

    by_ticker = {b.ticker: b for b in breaks}
    assert by_ticker["AAPL"].internal_qty == Decimal("100")
    assert by_ticker["AAPL"].venue_qty == Decimal("95")
    assert by_ticker["AAPL"].diff == Decimal("5")

    assert by_ticker["MSFT"].internal_qty == Decimal("50")
    assert by_ticker["MSFT"].venue_qty == Decimal("60")
    assert by_ticker["MSFT"].diff == Decimal("-10")


def test_reconcile_handles_missing_symbol_on_either_side() -> None:
    internal = {"AAPL": Decimal("100")}
    venue = {"MSFT": Decimal("50")}
    breaks = reconcile(internal, venue)
    # AAPL: internal=100 venue=0  → diff +100
    # MSFT: internal=0   venue=50 → diff -50
    assert len(breaks) == 2
    by_ticker = {b.ticker: b for b in breaks}
    assert by_ticker["AAPL"].diff == Decimal("100")
    assert by_ticker["MSFT"].diff == Decimal("-50")


def test_reconcile_tolerance_suppresses_small_breaks() -> None:
    internal = {"AAPL": Decimal("100")}
    venue = {"AAPL": Decimal("99")}
    # tolerance=1 → diff of 1 is exactly at threshold, NOT emitted (uses >)
    assert reconcile(internal, venue, tolerance=Decimal("1")) == []
    # tolerance=0 (default) → emitted
    assert len(reconcile(internal, venue)) == 1


def test_reconcile_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        reconcile({}, {}, tolerance=Decimal("-1"))


def test_reconcile_accepts_str_and_int_inputs() -> None:
    """Caller convenience: ``{ticker: "100"}`` and ``{ticker: 100}`` both work."""
    breaks = reconcile({"AAPL": "100"}, {"AAPL": 95})
    assert len(breaks) == 1
    assert breaks[0].diff == Decimal("5")


def test_position_break_to_jsonable_stringifies_decimals() -> None:
    br = PositionBreak(
        ticker="AAPL",
        internal_qty=Decimal("100"),
        venue_qty=Decimal("95"),
        diff=Decimal("5"),
        tolerance=Decimal("0"),
    )
    j = br.to_jsonable()
    assert j["ticker"] == "AAPL"
    assert j["internal_qty"] == "100"
    assert j["venue_qty"] == "95"
    assert j["diff"] == "5"
    assert j["tolerance"] == "0"
    # Must be JSON-roundtrippable.
    assert json.loads(json.dumps(j)) == j


# ---------------------------------------------------------------------------
# BalanceLedger persistence
# ---------------------------------------------------------------------------


def test_record_snapshot_writes_parquet_row(tmp_path: Path) -> None:
    import pandas as pd

    parquet = tmp_path / "balances.parquet"
    ledger = BalanceLedger(parquet_path=parquet)
    row = ledger.record_snapshot(
        timestamp_ns=1_700_000_000_000_000_000, account_summary=_FakeAccount()
    )

    assert row["account_id"] == "PA-TEST-001"
    assert row["equity"] == "100000.00"
    assert row["cash"] == "55000.50"
    assert parquet.exists()

    df = pd.read_parquet(parquet)
    assert len(df) == 1
    assert int(df.iloc[0]["timestamp_ns"]) == 1_700_000_000_000_000_000


def test_record_snapshot_appends_across_calls(tmp_path: Path) -> None:
    import pandas as pd

    parquet = tmp_path / "balances.parquet"
    ledger = BalanceLedger(parquet_path=parquet)
    ledger.record_snapshot(1_700_000_000_000_000_000, _FakeAccount())
    ledger.record_snapshot(
        1_700_000_900_000_000_000,
        _FakeAccount(equity="101000.00", cash="50000.50"),
    )
    df = pd.read_parquet(parquet)
    assert len(df) == 2
    assert df.iloc[1]["equity"] == "101000.00"


def test_record_snapshot_accepts_dict_account_summary(tmp_path: Path) -> None:
    """The ledger should accept a plain dict as well as an SDK object."""
    parquet = tmp_path / "balances.parquet"
    ledger = BalanceLedger(parquet_path=parquet)
    row = ledger.record_snapshot(
        timestamp_ns=1_700_000_000_000_000_000,
        account_summary={
            "account_id": "PA-DICT",
            "equity": "12345.67",
            "cash": "5000.00",
            "buying_power": "20000.00",
            "long_market_value": "7345.67",
            "short_market_value": "0.00",
            "portfolio_value": "12345.67",
            "status": "ACTIVE",
            "daytrade_count": 0,
            "pattern_day_trader": False,
        },
    )
    assert row["account_id"] == "PA-DICT"
    assert row["equity"] == "12345.67"


def test_record_snapshot_rejects_non_int_timestamp(tmp_path: Path) -> None:
    ledger = BalanceLedger(parquet_path=tmp_path / "balances.parquet")
    with pytest.raises(TypeError, match="timestamp_ns"):
        ledger.record_snapshot(timestamp_ns=1.5, account_summary=_FakeAccount())  # type: ignore[arg-type]


def test_write_breaks_appends_jsonl(tmp_path: Path) -> None:
    parquet = tmp_path / "balances.parquet"
    breaks_path = tmp_path / "breaks.jsonl"
    ledger = BalanceLedger(parquet_path=parquet)

    breaks = reconcile(
        {"AAPL": Decimal("100"), "MSFT": Decimal("50")},
        {"AAPL": Decimal("95"), "MSFT": Decimal("60")},
    )
    ledger.write_breaks(
        1_700_000_000_000_000_000, breaks, breaks_path=breaks_path
    )
    ledger.write_breaks(
        1_700_000_900_000_000_000, breaks, breaks_path=breaks_path
    )

    lines = breaks_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # 2 breaks * 2 calls
    parsed = [json.loads(line) for line in lines]
    assert {p["break"]["ticker"] for p in parsed} == {"AAPL", "MSFT"}
    assert parsed[0]["ts_ns"] == 1_700_000_000_000_000_000
    assert parsed[2]["ts_ns"] == 1_700_000_900_000_000_000


def test_write_breaks_noop_on_empty_list(tmp_path: Path) -> None:
    breaks_path = tmp_path / "breaks.jsonl"
    ledger = BalanceLedger(parquet_path=tmp_path / "balances.parquet")
    ledger.write_breaks(1_700_000_000_000_000_000, [], breaks_path=breaks_path)
    # No file should be created when there's nothing to record.
    assert not breaks_path.exists()
