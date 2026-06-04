"""execution.journal — minimal fill-journal parquet writer.

CONTEXT.md pins a 31-column schema. Today's critical path needs just the
columns we will actually populate for the first paper trade:

    client_order_id, intent_id, symbol, side, qty,
    submitted_at_ns, filled_at_ns, fill_price, fill_qty,
    status, mode

We write to ``data/journal/<mode>/dt=YYYY-MM-DD/part-<ts>.parquet`` so
the partition layout already matches the eventual full-schema layout
(LIVE / PAPER / WHATIF roots). The remaining 20 columns will be added
incrementally without breaking this writer — pyarrow tolerates a wider
schema on read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from shared_core.contracts import ChainMode, OrderIntent, Side

__all__ = [
    "DEFAULT_JOURNAL_ROOT",
    "FillRecord",
    "MINIMAL_SCHEMA",
    "fill_record_from_intent",
    "journal_path_for",
    "read_fills",
    "write_fill",
]


# ---------------------------------------------------------------------------
# Schema (minimal — Contract 3's full 31-column shape is forthcoming)
# ---------------------------------------------------------------------------


MINIMAL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("client_order_id", pa.string(), nullable=False),
        pa.field("intent_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("side", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("qty", pa.int64(), nullable=False),
        pa.field("submitted_at_ns", pa.int64(), nullable=False),
        pa.field("filled_at_ns", pa.int64(), nullable=True),
        # fill_price / fill_qty are NULL until the broker reports the fill.
        pa.field("fill_price", pa.float64(), nullable=True),
        pa.field("fill_qty", pa.int64(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),
    ]
)


DEFAULT_JOURNAL_ROOT: Path = Path(__file__).resolve().parents[2] / "data" / "journal"


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FillRecord:
    """One row of the minimal journal schema."""

    client_order_id: str
    intent_id: str
    symbol: str
    side: str  # "BUY" / "SELL"
    qty: int
    submitted_at_ns: int
    status: str  # "SUBMITTED" / "FILLED" / "REJECTED" / "WHATIF" / ...
    mode: str  # ChainMode.value
    filled_at_ns: int | None = None
    fill_price: float | None = None
    fill_qty: int | None = None

    def to_arrow_row(self) -> dict[str, object]:
        return {
            "client_order_id": self.client_order_id,
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "submitted_at_ns": self.submitted_at_ns,
            "filled_at_ns": self.filled_at_ns,
            "fill_price": self.fill_price,
            "fill_qty": self.fill_qty,
            "status": self.status,
            "mode": self.mode,
        }


def fill_record_from_intent(
    intent: OrderIntent,
    *,
    client_order_id: str,
    submitted_at_ns: int,
    status: str,
    mode: ChainMode,
    filled_at_ns: int | None = None,
    fill_price: Decimal | float | None = None,
    fill_qty: int | None = None,
) -> FillRecord:
    """Project an ``OrderIntent`` plus broker reply into a FillRecord.

    ``fill_price`` accepts Decimal or float for ergonomic broker-reply
    handling; we cast to float at the parquet boundary (the full Contract
    3 schema will keep this as float64 for performance).
    """

    return FillRecord(
        client_order_id=client_order_id,
        intent_id=intent.intent_id,
        symbol=intent.symbol.ticker,
        side=Side(intent.side).value,
        qty=int(intent.delta_qty),
        submitted_at_ns=submitted_at_ns,
        filled_at_ns=filled_at_ns,
        fill_price=float(fill_price) if fill_price is not None else None,
        fill_qty=int(fill_qty) if fill_qty is not None else None,
        status=status,
        mode=ChainMode(mode).value,
    )


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------


def journal_path_for(
    mode: ChainMode,
    *,
    submitted_at_ns: int,
    root: Path | None = None,
) -> Path:
    """Compute the parquet file path for a fill.

    Layout: ``<root>/<mode>/dt=YYYY-MM-DD/part-<ns>.parquet``
    """

    root = root if root is not None else DEFAULT_JOURNAL_ROOT
    dt = datetime.fromtimestamp(submitted_at_ns / 1e9, tz=timezone.utc)
    dt_str = dt.strftime("%Y-%m-%d")
    return root / mode.value / f"dt={dt_str}" / f"part-{submitted_at_ns}.parquet"


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------


def write_fill(record: FillRecord, *, root: Path | None = None) -> Path:
    """Write a single FillRecord as a parquet file (one row per file).

    Per-row files are inefficient at scale but keep the writer atomic
    and crash-safe for the first paper trade. A compaction job will fold
    a day's worth of part-files into a single row-group later.
    """

    if not isinstance(record, FillRecord):
        raise TypeError(f"record must be FillRecord (got {type(record).__name__})")
    mode = ChainMode(record.mode)
    path = journal_path_for(mode, submitted_at_ns=record.submitted_at_ns, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([record.to_arrow_row()], schema=MINIMAL_SCHEMA)
    # Write to a .tmp file then rename for atomicity.
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path


def read_fills(paths: Iterable[Path]) -> pa.Table:
    """Read one or more journal part-files and concatenate."""

    tables: list[pa.Table] = []
    for p in paths:
        tables.append(pq.read_table(p, schema=MINIMAL_SCHEMA))
    if not tables:
        return pa.Table.from_pylist([], schema=MINIMAL_SCHEMA)
    return pa.concat_tables(tables)
