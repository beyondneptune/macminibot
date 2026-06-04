"""compliance.monthly_report — generate the operator-facing monthly markdown.

Reads the compliance journal (``data/compliance/journal.parquet``),
filters events to the calendar month requested, replays them through a
fresh :class:`FifoTaxLotBook`, and writes a markdown summary to
``data/compliance/reports/YYYY-MM.md``.

Runnable as a script::

    python -m compliance.monthly_report          # previous calendar month
    python -m compliance.monthly_report 2026-05  # explicit month

The report is **regenerated from the journal** every time — the journal
is authoritative (CONTEXT.md convention).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa

from compliance.journal import (
    JOURNAL_SCHEMA,
    default_journal_path,
    read_journal,
)
from compliance.tax_lots import FifoTaxLotBook
from shared_core.types import NS_PER_SECOND, ns_to_datetime_utc

__all__ = [
    "DEFAULT_REPORTS_DIR",
    "generate_monthly_report",
    "main",
    "month_bounds_ns",
]


_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR: Path = _REPO_ROOT / "data" / "compliance" / "reports"


def month_bounds_ns(year: int, month: int) -> tuple[int, int]:
    """Return ``(start_ns_inclusive, end_ns_exclusive)`` for a calendar month UTC."""

    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return int(start.timestamp()) * NS_PER_SECOND, int(end.timestamp()) * NS_PER_SECOND


def _year_start_ns(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()) * NS_PER_SECOND


def _filter_table_by_ns(
    table: pa.Table, start_ns: int, end_ns: int
) -> pa.Table:
    """Filter the journal to events with ``start_ns <= ts_ns < end_ns``."""

    if table.num_rows == 0:
        return table
    ts = table.column("ts_ns")
    mask = pa.compute.and_(
        pa.compute.greater_equal(ts, pa.scalar(start_ns, type=pa.int64())),
        pa.compute.less(ts, pa.scalar(end_ns, type=pa.int64())),
    )
    return table.filter(mask)


def _format_decimal(value: Decimal, places: int = 2) -> str:
    quant = Decimal("1." + "0" * places) if places else Decimal("1")
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value).quantize(quant)}"


def generate_monthly_report(
    year: int,
    month: int,
    *,
    journal_path: Path | None = None,
    reports_dir: Path | None = None,
) -> Path:
    """Generate the YYYY-MM markdown report. Returns the written path."""

    journal_path = journal_path if journal_path is not None else default_journal_path()
    reports_dir = reports_dir if reports_dir is not None else DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    start_ns, end_ns = month_bounds_ns(year, month)

    # Build the YTD book by replaying everything up to and including the
    # target month (so realized_pnl_ytd correctly attributes prior-month
    # sales). The journal is the authoritative input.
    full = read_journal(journal_path)

    # Realized P&L since 1 Jan of the target year — book replayed in
    # ts_ns order so FIFO matching is deterministic.
    book = FifoTaxLotBook()
    if full.num_rows > 0:
        sorted_table = full.sort_by("ts_ns")
        ytd_start = _year_start_ns(year)
        for row in sorted_table.to_pylist():
            if row["ts_ns"] >= end_ns:
                break
            if row["event_type"] != "ORDER_FILLED":
                continue
            if row["price"] is None:
                continue
            side = "BUY" if int(row["qty"]) > 0 else "SELL"
            # qty in journal is positive; side is inferred from event payload.
            # The compliance journal stores absolute qty + decision_id; we
            # rely on the recorder to have written one event per BUY/SELL.
            # By convention, ORDER_FILLED rows imply the *filled* side —
            # see scripts/43_compliance_replay.py for the mapping.
            side = row.get("_side", side)  # type: ignore[assignment]
            # Fall back: use the absolute value with side parsing below.
            book.apply_fill(
                symbol=row["symbol"],
                side=_infer_side(row),
                qty=int(row["qty"]),
                price=float(row["price"]),
                ts_ns=int(row["ts_ns"]),
            )

        ytd_pnl_all = book.realized_pnl_ytd(year_start_ns=ytd_start)
    else:
        ytd_pnl_all = book.realized_pnl_ytd()

    month_table = _filter_table_by_ns(full, start_ns, end_ns)

    label = f"{year:04d}-{month:02d}"
    out_path = reports_dir / f"{label}.md"

    lines: list[str] = []
    lines.append(f"# Compliance Report — {label}")
    lines.append("")
    lines.append(
        f"Generated from journal at `{journal_path}`. "
        f"This file is regenerable; the journal is authoritative."
    )
    lines.append("")

    # ---- Trades section --------------------------------------------------
    lines.append("## Trades")
    lines.append("")
    if month_table.num_rows == 0:
        lines.append("_No journal events in this month._")
    else:
        lines.append(
            "| Timestamp (UTC) | Event | Symbol | Qty | Price | Decision ID | Model |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in month_table.sort_by("ts_ns").to_pylist():
            ts = ns_to_datetime_utc(int(row["ts_ns"])).strftime("%Y-%m-%d %H:%M:%S")
            price_s = (
                _format_decimal(Decimal(str(row["price"])))
                if row["price"] is not None
                else "—"
            )
            lines.append(
                f"| {ts} | {row['event_type']} | {row['symbol']} | "
                f"{row['qty']} | {price_s} | "
                f"`{row['decision_id'][:16]}…` | {row['model_version']} |"
            )
    lines.append("")

    # ---- Realized P&L ----------------------------------------------------
    lines.append("## Realized P&L (Year-to-Date)")
    lines.append("")
    lines.append("| Bucket | Amount |")
    lines.append("|---|---|")
    lines.append(f"| Short-term | {_format_decimal(ytd_pnl_all.short_term)} |")
    lines.append(f"| Long-term  | {_format_decimal(ytd_pnl_all.long_term)} |")
    lines.append(f"| **Total**  | **{_format_decimal(ytd_pnl_all.total)}** |")
    lines.append("")

    # Per-symbol breakdown (only symbols with any sales)
    symbols = sorted({s.symbol for s in book.sales()})
    if symbols:
        lines.append("### Per-symbol YTD breakdown")
        lines.append("")
        lines.append("| Symbol | Short-term | Long-term | Total |")
        lines.append("|---|---|---|---|")
        ytd_start = _year_start_ns(year)
        for sym in symbols:
            p = book.realized_pnl_ytd(symbol=sym, year_start_ns=ytd_start)
            lines.append(
                f"| {sym} | {_format_decimal(p.short_term)} | "
                f"{_format_decimal(p.long_term)} | "
                f"{_format_decimal(p.total)} |"
            )
        lines.append("")

    # ---- Open tax lots ---------------------------------------------------
    lines.append("## Open Tax Lots (as of end of month)")
    lines.append("")
    open_lots = book.open_lots()
    if not open_lots:
        lines.append("_No open positions._")
    else:
        lines.append("| Symbol | Qty | Cost Basis | Acquired (UTC) |")
        lines.append("|---|---|---|---|")
        for lot in open_lots:
            ts = ns_to_datetime_utc(lot.acquired_ts_ns).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(
                f"| {lot.symbol} | {lot.qty} | "
                f"{_format_decimal(lot.price, places=4)} | {ts} |"
            )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _infer_side(row: dict[str, object]) -> str:
    """Decode side from a journal row.

    The compliance journal does not store an explicit ``side`` column —
    the upstream recorder writes ``ORDER_FILLED`` once per fill. To keep
    FIFO determinism we encode the side into ``decision_id``'s suffix
    (``:BUY`` or ``:SELL``) so replay can recover it without a schema
    bump. Callers that want explicit side handling should write a row
    per side and use this helper. Defaults to BUY for robustness.
    """

    did = str(row.get("decision_id", ""))
    if did.endswith(":SELL"):
        return "SELL"
    if did.endswith(":BUY"):
        return "BUY"
    # Fallback: if the model_version or symbol carries a hint (legacy),
    # default to BUY — the replay script always tags side explicitly.
    return "BUY"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a compliance monthly report (markdown)."
    )
    parser.add_argument(
        "month",
        nargs="?",
        default=None,
        help="YYYY-MM (defaults to the previous calendar month UTC).",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=None,
        help="Override journal parquet path (debug/test only).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Override output dir (debug/test only).",
    )
    args = parser.parse_args(argv)

    if args.month is None:
        now = datetime.now(timezone.utc)
        if now.month == 1:
            year, month = now.year - 1, 12
        else:
            year, month = now.year, now.month - 1
    else:
        try:
            year_s, month_s = args.month.split("-")
            year, month = int(year_s), int(month_s)
        except (ValueError, AttributeError):
            print(f"error: invalid month {args.month!r}; expected YYYY-MM", file=sys.stderr)
            return 2

    path = generate_monthly_report(
        year, month, journal_path=args.journal, reports_dir=args.reports_dir
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
