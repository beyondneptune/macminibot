"""Journal round-trips through parquet with the ratified schema."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backtesting.journal import (
    PRIMARY_KEY_COLUMNS,
    SCHEMA,
    FillRecord,
    write_fill,
    write_fills,
)


def _make_fill(coid: str = "01HRZA0000000000000000COID", seq: int = 0) -> FillRecord:
    ts = 1_770_000_000_000_000_000  # 2026-01-something in ns since epoch UTC
    return FillRecord(
        client_order_id=coid,
        decision_id="01HRZA0000000000000000DECI",
        ts_decision=ts,
        ts_submit=ts + 1_000_000,
        ts_fill=ts + 5_000_000,
        symbol="AAPL",
        asset_class="equity",
        side="BUY",
        qty=100,
        order_type="MOO",
        tif="OPG",
        fill_px=185.25,
        fill_qty=100,
        fill_seq=seq,
        ref_px_close_tm1=184.00,
        ref_px_open_t=185.00,
        commission=0.35,
        exchange_fee=0.20,
        reg_fee=0.05,
        clearing_fee=0.05,
        slippage_bps_realized=13.5,
        gap_pnl=100.0,
        session_pnl=25.0,
        strategy_sha="abc123def456",
        backtest_run_id="01HRZARUN0000000000000000",
        slippage_bps_modeled=14.0,
        venue="NSDQ",
        route="SMART",
    )


def test_schema_has_31_columns() -> None:
    assert len(SCHEMA) == 31


def test_schema_primary_key_present() -> None:
    field_names = SCHEMA.names
    for pk in PRIMARY_KEY_COLUMNS:
        assert pk in field_names, f"PK column {pk} missing from SCHEMA"


def test_schema_timestamp_columns_are_ns_utc() -> None:
    for ts_col in ("ts_decision", "ts_submit", "ts_ack", "ts_fill"):
        field = SCHEMA.field(ts_col)
        assert pa.types.is_timestamp(field.type)
        assert field.type.unit == "ns"
        assert field.type.tz == "UTC"


def test_write_single_fill_round_trip(tmp_path: Path) -> None:
    fill = _make_fill()
    out = write_fill(tmp_path / "single.parquet", fill)
    assert out.exists()
    table = pq.read_table(out)
    assert table.schema.equals(SCHEMA)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["client_order_id"] == fill.client_order_id
    assert row["symbol"] == "AAPL"
    assert row["asset_class"] == "equity"
    assert row["fill_px"] == pytest.approx(185.25)
    assert row["slippage_bps_realized"] == pytest.approx(13.5)
    assert row["account"] == "BACKTEST"
    assert row["backtest_run_id"] == fill.backtest_run_id


def test_write_multiple_fills_round_trip(tmp_path: Path) -> None:
    fills = [_make_fill(seq=i) for i in range(5)]
    out = write_fills(tmp_path / "batch.parquet", fills)
    table = pq.read_table(out)
    assert table.num_rows == 5
    assert [r["fill_seq"] for r in table.to_pylist()] == [0, 1, 2, 3, 4]


def test_write_fills_accepts_dicts(tmp_path: Path) -> None:
    fill_dict = _make_fill().as_dict()
    out = write_fills(tmp_path / "from_dict.parquet", [fill_dict])
    table = pq.read_table(out)
    assert table.num_rows == 1
    assert table.to_pylist()[0]["client_order_id"] == fill_dict["client_order_id"]


def test_write_empty_creates_empty_table(tmp_path: Path) -> None:
    out = write_fills(tmp_path / "empty.parquet", [])
    table = pq.read_table(out)
    assert table.num_rows == 0
    assert table.schema.equals(SCHEMA)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    bad = _make_fill().as_dict()
    del bad["symbol"]
    with pytest.raises(KeyError):
        write_fills(tmp_path / "bad.parquet", [bad])
