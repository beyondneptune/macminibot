"""compliance.journal — append-only regulatory event journal.

Storage:
    data/compliance/journal.parquet

This is a single rolling parquet **file** the operator can grep / open
in DuckDB years from now. We append by reading the existing table,
concatenating the new row, and atomically rewriting via ``.tmp`` +
rename. For the volumes a single retail-grade bot produces (low
thousands of events per year), a rewrite-on-append is well within
budget and keeps the on-disk artefact one file — which is what
regulators and tax software actually want.

The journal is **append-only**: callers cannot mutate prior rows.
A correction is a *new* event referencing the prior one via
``decision_id`` (see CONTEXT.md).

Schema (KD1/KD2 compliant):
    ts_ns:           int64   wall-clock UTC ns when the event happened
    event_type:      string  one of EVENT_TYPES
    symbol:          string  ticker (uppercase)
    qty:             int64   share count (positive integer)
    price:           float64 fill price (NULL for non-fill events)
    decision_id:     string  reproducibility hash from strategy-engine
    model_version:   string  model identifier (or "manual")
    recorded_at_ns:  int64   wall-clock UTC ns when this row was written

``price`` is float64 for parquet-friendliness; callers that need exact
arithmetic should round/quantize at the boundary (tax_lots.py uses
``Decimal`` internally and casts only at I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from shared_core.types import now_ns

__all__ = [
    "EVENT_TYPES",
    "JOURNAL_SCHEMA",
    "ComplianceEvent",
    "default_journal_path",
    "read_journal",
    "record_event",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "ORDER_SUBMITTED",
        "ORDER_FILLED",
        "ORDER_CANCELLED",
        "ORDER_REJECTED",
    }
)


JOURNAL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("ts_ns", pa.int64(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("qty", pa.int64(), nullable=False),
        pa.field("price", pa.float64(), nullable=True),
        pa.field("decision_id", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("recorded_at_ns", pa.int64(), nullable=False),
    ]
)


_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_JOURNAL_DIR: Path = _REPO_ROOT / "data" / "compliance"
_DEFAULT_JOURNAL_FILE: Path = _DEFAULT_JOURNAL_DIR / "journal.parquet"


def default_journal_path() -> Path:
    """Return the canonical journal location (``data/compliance/journal.parquet``)."""

    return _DEFAULT_JOURNAL_FILE


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComplianceEvent:
    """One row of the regulatory journal."""

    ts_ns: int
    event_type: str
    symbol: str
    qty: int
    price: float | None
    decision_id: str
    model_version: str
    recorded_at_ns: int

    def to_arrow_row(self) -> dict[str, object]:
        return {
            "ts_ns": self.ts_ns,
            "event_type": self.event_type,
            "symbol": self.symbol,
            "qty": self.qty,
            "price": self.price,
            "decision_id": self.decision_id,
            "model_version": self.model_version,
            "recorded_at_ns": self.recorded_at_ns,
        }


# ---------------------------------------------------------------------------
# Append (single source of truth)
# ---------------------------------------------------------------------------


def record_event(
    event_type: str,
    symbol: str,
    qty: int,
    price: float | None,
    ts_ns: int,
    decision_id: str,
    model_version: str,
    *,
    journal_path: Path | None = None,
) -> ComplianceEvent:
    """Append one event to the compliance journal.

    Returns the persisted :class:`ComplianceEvent`. The on-disk file is
    rewritten atomically (``.tmp`` + ``replace``) so a crash mid-append
    never leaves a half-written parquet file.

    Raises:
        ValueError: if ``event_type`` is not one of :data:`EVENT_TYPES`,
            or if any required string field is empty, or if ``qty <= 0``.
        TypeError: if ``ts_ns`` is not an int.
    """

    if event_type not in EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(EVENT_TYPES)} (got {event_type!r})"
        )
    if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
        raise TypeError(f"ts_ns must be int (got {type(ts_ns).__name__})")
    if not symbol:
        raise ValueError("symbol must be non-empty")
    if not decision_id:
        raise ValueError("decision_id must be non-empty")
    if not model_version:
        raise ValueError("model_version must be non-empty")
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        raise ValueError(f"qty must be positive int (got {qty!r})")
    if price is not None and not isinstance(price, (int, float)):
        raise TypeError(f"price must be float|None (got {type(price).__name__})")

    event = ComplianceEvent(
        ts_ns=ts_ns,
        event_type=event_type,
        symbol=symbol.upper(),
        qty=qty,
        price=float(price) if price is not None else None,
        decision_id=decision_id,
        model_version=model_version,
        recorded_at_ns=now_ns(),
    )

    path = journal_path if journal_path is not None else default_journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row_table = pa.Table.from_pylist([event.to_arrow_row()], schema=JOURNAL_SCHEMA)
    if path.exists():
        existing = pq.read_table(path, schema=JOURNAL_SCHEMA)
        merged = pa.concat_tables([existing, new_row_table])
    else:
        merged = new_row_table

    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(merged, tmp)
    tmp.replace(path)
    return event


def read_journal(journal_path: Path | None = None) -> pa.Table:
    """Return the full journal as a pyarrow Table (empty if file absent)."""

    path = journal_path if journal_path is not None else default_journal_path()
    if not path.exists():
        return pa.Table.from_pylist([], schema=JOURNAL_SCHEMA)
    return pq.read_table(path, schema=JOURNAL_SCHEMA)
