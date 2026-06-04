"""compliance — regulatory journal, tax-lot accounting, monthly reports.

Greenfield tentacle. See .octogent/tentacles/compliance/CONTEXT.md.

Key conventions (ratified):
- The journal is append-only. Corrections are new entries; never mutate.
- Every entry carries ``decision_id`` so it can be joined to other logs.
- Reports are regenerated from the journal on demand (journal is auth.).
- Tax-lot mutations are deterministic given the journal (FIFO default).

Public surface intentionally narrow — internal helpers are not re-exported.
"""

from __future__ import annotations

from compliance.journal import (
    EVENT_TYPES,
    JOURNAL_SCHEMA,
    ComplianceEvent,
    default_journal_path,
    read_journal,
    record_event,
)
from compliance.tax_lots import (
    FifoTaxLotBook,
    RealizedPnL,
    TaxLot,
    apply_fill,
    realized_pnl_ytd,
)

__all__ = [
    "ComplianceEvent",
    "EVENT_TYPES",
    "FifoTaxLotBook",
    "JOURNAL_SCHEMA",
    "RealizedPnL",
    "TaxLot",
    "apply_fill",
    "default_journal_path",
    "read_journal",
    "realized_pnl_ytd",
    "record_event",
]
