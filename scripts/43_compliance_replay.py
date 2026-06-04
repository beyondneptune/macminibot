#!/usr/bin/env python3
"""43_compliance_replay — replay paper-trade journal into the compliance journal.

Walks ``data/journal/paper/dt=*/part-*.parquet`` in deterministic order,
reads each fill record, and:

    1. Records the appropriate ``ORDER_*`` event(s) into the compliance
       journal at ``data/compliance/journal.parquet``.
    2. Replays the fills through a fresh :class:`FifoTaxLotBook` and
       prints the resulting realized P&L summary to stdout.

Idempotent: dedupes on ``client_order_id`` against the existing
compliance journal so repeated invocations do not double-count. Per
CONTEXT.md, corrections are *new* journal entries — this script never
mutates prior rows.

Run::

    python scripts/43_compliance_replay.py
    python scripts/43_compliance_replay.py --dry-run
    python scripts/43_compliance_replay.py --journal-root /path/to/journal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap src/ onto sys.path so the script runs without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pyarrow.parquet as pq  # noqa: E402

from compliance.journal import default_journal_path, read_journal, record_event  # noqa: E402
from compliance.tax_lots import FifoTaxLotBook  # noqa: E402
from shared_core.types import now_ns  # noqa: E402


_DEFAULT_PAPER_JOURNAL_ROOT = _REPO_ROOT / "data" / "journal" / "paper"


def _existing_client_order_ids(compliance_journal_path: Path) -> set[str]:
    """Return the set of client_order_ids already encoded in the journal.

    We encode the source ``client_order_id`` as the prefix of
    ``decision_id`` in the compliance row (format: ``<coid>::<side>``).
    Reading them back lets repeated runs be idempotent.
    """

    table = read_journal(compliance_journal_path)
    if table.num_rows == 0:
        return set()
    seen: set[str] = set()
    for did in table.column("decision_id").to_pylist():
        if not isinstance(did, str):
            continue
        # decision_id stored as "<client_order_id>::<SIDE>"
        coid = did.split("::", 1)[0]
        if coid:
            seen.add(coid)
    return seen


def _iter_paper_parts(root: Path):
    """Yield part files in deterministic (dt, filename) order."""

    if not root.exists():
        return
    # Sort by dt= directory then by filename so replay is reproducible.
    for dt_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("dt=")):
        for part in sorted(dt_dir.glob("part-*.parquet")):
            yield part


def replay(
    *,
    paper_journal_root: Path,
    compliance_journal_path: Path,
    dry_run: bool = False,
) -> dict[str, int | str]:
    """Run the replay; return a stats dict for the operator."""

    seen = _existing_client_order_ids(compliance_journal_path)
    book = FifoTaxLotBook()

    n_seen = 0
    n_skipped = 0
    n_recorded = 0
    n_filled = 0

    for part_path in _iter_paper_parts(paper_journal_root):
        # The runner writes one row per file (see execution.journal); we
        # still loop the table in case a compactor has merged rows.
        table = pq.read_table(part_path)
        rows = table.to_pylist()
        # Sort within-file by submitted_at_ns to keep FIFO deterministic
        # even if a future compactor reorders rows.
        rows.sort(
            key=lambda r: (
                int(r.get("submitted_at_ns") or 0),
                int(r.get("filled_at_ns") or 0),
            )
        )
        for row in rows:
            n_seen += 1
            coid = row.get("client_order_id")
            if not isinstance(coid, str) or not coid:
                n_skipped += 1
                continue
            if coid in seen:
                n_skipped += 1
                continue

            status = row.get("status")
            side = row.get("side")
            symbol = row.get("symbol")
            qty = row.get("qty")
            submitted_at_ns = row.get("submitted_at_ns")
            filled_at_ns = row.get("filled_at_ns")
            fill_price = row.get("fill_price")
            fill_qty = row.get("fill_qty")

            if not (
                isinstance(symbol, str)
                and isinstance(side, str)
                and isinstance(qty, int)
                and isinstance(submitted_at_ns, int)
                and isinstance(status, str)
            ):
                n_skipped += 1
                continue

            event_type = _status_to_event(status)
            if event_type is None:
                n_skipped += 1
                continue

            # If a fill is present, emit ORDER_FILLED at the fill ts,
            # otherwise emit the lifecycle event at submitted_at_ns.
            is_fill = (
                event_type == "ORDER_FILLED"
                and isinstance(fill_price, float)
                and isinstance(fill_qty, int)
                and isinstance(filled_at_ns, int)
            )

            decision_id = f"{coid}::{side}"

            if is_fill:
                if not dry_run:
                    record_event(
                        event_type="ORDER_FILLED",
                        symbol=symbol,
                        qty=int(fill_qty),
                        price=float(fill_price),
                        ts_ns=int(filled_at_ns),
                        decision_id=decision_id,
                        model_version="paper-replay",
                        journal_path=compliance_journal_path,
                    )
                book.apply_fill(
                    symbol=symbol,
                    side=side,
                    qty=int(fill_qty),
                    price=float(fill_price),
                    ts_ns=int(filled_at_ns),
                )
                n_filled += 1
            else:
                if not dry_run:
                    record_event(
                        event_type=event_type,
                        symbol=symbol,
                        qty=int(qty),
                        price=None,
                        ts_ns=int(submitted_at_ns),
                        decision_id=decision_id,
                        model_version="paper-replay",
                        journal_path=compliance_journal_path,
                    )

            seen.add(coid)
            n_recorded += 1

    pnl = book.realized_pnl_ytd()
    return {
        "rows_seen": n_seen,
        "rows_skipped_dedup_or_invalid": n_skipped,
        "rows_recorded": n_recorded,
        "fills_replayed_into_book": n_filled,
        "realized_pnl_short_term": str(pnl.short_term),
        "realized_pnl_long_term": str(pnl.long_term),
        "realized_pnl_total": str(pnl.total),
        "dry_run": str(dry_run),
        "completed_at_ns": now_ns(),
    }


def _status_to_event(status: str) -> str | None:
    mapping = {
        "SUBMITTED": "ORDER_SUBMITTED",
        "FILLED": "ORDER_FILLED",
        "CANCELLED": "ORDER_CANCELLED",
        "CANCELED": "ORDER_CANCELLED",
        "REJECTED": "ORDER_REJECTED",
    }
    return mapping.get(status.upper())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--journal-root",
        type=Path,
        default=_DEFAULT_PAPER_JOURNAL_ROOT,
        help="Paper journal root (default: data/journal/paper)",
    )
    parser.add_argument(
        "--compliance-journal",
        type=Path,
        default=None,
        help="Override compliance journal path (default: data/compliance/journal.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Replay through tax-lot book + print stats, but do not write compliance journal.",
    )
    args = parser.parse_args(argv)

    compliance_path = (
        args.compliance_journal
        if args.compliance_journal is not None
        else default_journal_path()
    )

    stats = replay(
        paper_journal_root=args.journal_root,
        compliance_journal_path=compliance_path,
        dry_run=args.dry_run,
    )
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
