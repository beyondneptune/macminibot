"""Trade journal — pyarrow schema + writer.

Single source of truth for fills, live + backtest (see CONTEXT.md §Trade-
journal schema). ``execution`` is the OWNER of the schema in production;
this module embeds the same 31-column definition so backtesting can write
parquet files that are byte-compatible with the live trade journal — the
parity report is then a straight
``pyarrow.dataset.dataset([live_path, bt_path]).to_table()``.

The schema constant ``SCHEMA`` is what the cross-tentacle contract test
in ``tests/contracts/test_journal_schema.py`` pins against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "JOURNAL_SCHEMA_VERSION",
    "PRIMARY_KEY_COLUMNS",
    "SCHEMA",
    "FillRecord",
    "write_fill",
    "write_fills",
]

JOURNAL_SCHEMA_VERSION = "v1"

# Asset-class / side / order-type / TIF dictionaries. Pinned here so the
# encoded parquet uses dict-typed columns the live writer also emits.
_ASSET_CLASS = pa.dictionary(pa.int32(), pa.string())
_SIDE = pa.dictionary(pa.int32(), pa.string())
_ORDER_TYPE = pa.dictionary(pa.int32(), pa.string())
_TIF = pa.dictionary(pa.int32(), pa.string())

# Ratified 31-column schema. Order matches CONTEXT.md §Trade-journal schema.
SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("client_order_id", pa.string(), nullable=False),
        pa.field("backtest_run_id", pa.string(), nullable=True),
        pa.field("decision_id", pa.string(), nullable=False),
        pa.field("ts_decision", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("ts_submit", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("ts_ack", pa.timestamp("ns", tz="UTC"), nullable=True),
        pa.field("ts_fill", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("asset_class", _ASSET_CLASS, nullable=False),
        pa.field("side", _SIDE, nullable=False),
        pa.field("qty", pa.int64(), nullable=False),
        pa.field("order_type", _ORDER_TYPE, nullable=False),
        pa.field("tif", _TIF, nullable=False),
        pa.field("limit_px", pa.float64(), nullable=True),
        pa.field("fill_px", pa.float64(), nullable=False),
        pa.field("fill_qty", pa.int64(), nullable=False),
        pa.field("fill_seq", pa.int32(), nullable=False),
        pa.field("ref_px_close_tm1", pa.float64(), nullable=False),
        pa.field("ref_px_open_t", pa.float64(), nullable=False),
        pa.field("commission", pa.float64(), nullable=False),
        pa.field("exchange_fee", pa.float64(), nullable=False),
        pa.field("reg_fee", pa.float64(), nullable=False),
        pa.field("clearing_fee", pa.float64(), nullable=False),
        pa.field("slippage_bps_modeled", pa.float64(), nullable=True),
        pa.field("slippage_bps_realized", pa.float64(), nullable=False),
        pa.field("gap_pnl", pa.float64(), nullable=False),
        pa.field("session_pnl", pa.float64(), nullable=False),
        pa.field("venue", pa.string(), nullable=True),
        pa.field("route", pa.string(), nullable=True),
        pa.field("account", pa.string(), nullable=False),
        pa.field("strategy_sha", pa.string(), nullable=False),
    ]
)

PRIMARY_KEY_COLUMNS: tuple[str, str, str] = (
    "client_order_id",
    "fill_seq",
    "backtest_run_id",
)


@dataclass(slots=True)
class FillRecord:
    """One row of the journal — convenience constructor for the sim.

    Fields mirror :data:`SCHEMA` column names. Optional fields default to
    backtest-appropriate values (``ts_ack=None``, ``backtest_run_id`` is
    typically set, ``account="BACKTEST"``, etc.).
    """

    client_order_id: str
    decision_id: str
    ts_decision: int  # ns since epoch UTC
    ts_submit: int
    ts_fill: int
    symbol: str
    asset_class: str
    side: str
    qty: int
    order_type: str
    tif: str
    fill_px: float
    fill_qty: int
    fill_seq: int
    ref_px_close_tm1: float
    ref_px_open_t: float
    commission: float
    exchange_fee: float
    reg_fee: float
    clearing_fee: float
    slippage_bps_realized: float
    gap_pnl: float
    session_pnl: float
    strategy_sha: str
    backtest_run_id: str | None = None
    ts_ack: int | None = None
    limit_px: float | None = None
    slippage_bps_modeled: float | None = None
    venue: str | None = None
    route: str | None = None
    account: str = "BACKTEST"

    def as_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__slots__}


def _records_to_table(records: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Build a pyarrow.Table from row-dicts, coercing ts columns to ns-UTC."""
    rows = list(records)
    if not rows:
        # Empty table with the right schema — important for the "no trades in
        # this fold" case which the parity join still expects to see.
        empty_arrays = []
        for field in SCHEMA:
            empty_arrays.append(pa.array([], type=field.type))
        return pa.Table.from_arrays(empty_arrays, schema=SCHEMA)

    columns: dict[str, list[Any]] = {f.name: [] for f in SCHEMA}
    for row in rows:
        for field in SCHEMA:
            if field.name not in row:
                raise KeyError(
                    f"row missing required column {field.name!r}; "
                    f"got keys {sorted(row)}"
                )
            columns[field.name].append(row[field.name])

    arrays: list[pa.Array] = []
    for field in SCHEMA:
        data = columns[field.name]
        if pa.types.is_timestamp(field.type):
            # Accept int (ns) or ISO string; coerce via pa.array.
            arr = pa.array(data, type=field.type)
        else:
            arr = pa.array(data, type=field.type)
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


def write_fill(path: str | Path, fill: FillRecord | Mapping[str, Any]) -> Path:
    """Write a single fill to a parquet file (overwrites). Returns the path.

    For high-throughput batched writes, prefer :func:`write_fills`. This is
    the canonical primitive the contract test exercises.
    """
    return write_fills(path, [fill])


def write_fills(
    path: str | Path,
    fills: Iterable[FillRecord | Mapping[str, Any]],
) -> Path:
    """Write N fills to a parquet file with the ratified schema."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _normalise(f: FillRecord | Mapping[str, Any]) -> Mapping[str, Any]:
        return f.as_dict() if isinstance(f, FillRecord) else f

    table = _records_to_table(_normalise(f) for f in fills)
    pq.write_table(
        table,
        out_path,
        compression="snappy",
        use_dictionary=True,
        version="2.6",
    )
    return out_path
