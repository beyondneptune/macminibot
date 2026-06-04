"""execution.journal — round-trip + schema tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from shared_core.contracts import ChainMode
from shared_core.types import now_ns

from execution.journal import (
    MINIMAL_SCHEMA,
    fill_record_from_intent,
    journal_path_for,
    read_fills,
    write_fill,
)


def test_minimal_schema_columns_present() -> None:
    expected = {
        "client_order_id",
        "intent_id",
        "symbol",
        "side",
        "qty",
        "submitted_at_ns",
        "filled_at_ns",
        "fill_price",
        "fill_qty",
        "status",
        "mode",
    }
    assert set(MINIMAL_SCHEMA.names) == expected


def test_journal_path_partitions_by_mode_and_date() -> None:
    submitted_at_ns = 1_700_000_000_000_000_000  # 2023-11-14 UTC
    p = journal_path_for(
        ChainMode.PAPER, submitted_at_ns=submitted_at_ns, root=Path("/tmp/j")
    )
    assert "paper" in p.parts
    assert any(part.startswith("dt=2023-11-14") for part in p.parts)
    assert p.suffix == ".parquet"


def test_round_trip_paper_fill(make_intent, journal_root: Path) -> None:
    intent = make_intent(intent_id="intent-RT-1", qty="3")
    submitted_at_ns = now_ns()
    record = fill_record_from_intent(
        intent,
        client_order_id="mmb-test-rt",
        submitted_at_ns=submitted_at_ns,
        status="FILLED",
        mode=ChainMode.PAPER,
        filled_at_ns=submitted_at_ns + 1_000_000,
        fill_price=150.25,
        fill_qty=3,
    )
    path = write_fill(record, root=journal_root)
    assert path.exists()

    table = read_fills([path])
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["client_order_id"] == "mmb-test-rt"
    assert row["intent_id"] == "intent-RT-1"
    assert row["symbol"] == "AAPL"
    assert row["side"] == "BUY"
    assert row["qty"] == 3
    assert row["status"] == "FILLED"
    assert row["mode"] == "paper"
    assert row["fill_price"] == pytest.approx(150.25)
    assert row["fill_qty"] == 3


def test_round_trip_submitted_without_fill(make_intent, journal_root: Path) -> None:
    """A SUBMITTED row may have NULL fill_price/fill_qty/filled_at_ns."""

    intent = make_intent(intent_id="intent-RT-2", qty="5")
    record = fill_record_from_intent(
        intent,
        client_order_id="mmb-test-sub",
        submitted_at_ns=now_ns(),
        status="SUBMITTED",
        mode=ChainMode.PAPER,
    )
    path = write_fill(record, root=journal_root)
    table = read_fills([path])
    row = table.to_pylist()[0]
    assert row["status"] == "SUBMITTED"
    assert row["fill_price"] is None
    assert row["fill_qty"] is None
    assert row["filled_at_ns"] is None


def test_write_fill_rejects_non_record(journal_root: Path) -> None:
    with pytest.raises(TypeError):
        write_fill({"not": "a record"}, root=journal_root)  # type: ignore[arg-type]


def test_read_fills_empty_returns_typed_empty_table() -> None:
    table = read_fills([])
    assert table.num_rows == 0
    assert table.schema.equals(MINIMAL_SCHEMA)
