"""Daily-bar backfill for the starter universe.

Reads (asset_class, name, head_trades_ts, head_midpoint_ts) from data/universe.db,
pulls all-history 1-day bars via reqHistoricalData, prefers TRADES where the account
has permissions and falls back to MIDPOINT otherwise. Writes to ArcticDB libraries
`<asset_class>.daily_bars` with symbol `<exchange>.<name>`.

Pacing-aware: reqHistoricalData with '1 day' barSize accepts up to 1 Y per request,
so all-history equity backfill is just ceil(years / 1) requests per contract. The 40
contracts × ~10 requests average ≈ 400 requests, well under 60/10min for a single
clientId if we space ~10s apart. We just go sequentially with a small sleep.

Read-only (readonly=True). Idempotent: rewrites the symbol each run.

Run from project root:

    .venv/bin/python scripts/03_backfill_daily_bars.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from ib_async import IB, Contract, util

sys.path.insert(0, str(Path(__file__).parent))
from _storage import ensure_library, library_name, open_store, symbol_name  # noqa: E402
from _universe import UNIVERSE  # noqa: E402

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4001
CLIENT_ID = 13
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.db"

# Spacing — 1 day bars are very cheap (1 Y per request), pacing budget is not the
# bottleneck. Sleep modest amount to be polite.
INTER_REQUEST_SLEEP = 0.5

util.logToConsole(level="WARNING")


def load_universe_rows() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Order by shortest expected history first (newest head_ts → fewest 1Y requests).
    # COALESCE prefers trades head; falls back to midpoint. NULLs (any missing head)
    # sort last via the IFNULL '0000' trick so they don't crash the run.
    rows = list(
        conn.execute(
            """
            SELECT * FROM universe
            ORDER BY
              COALESCE(head_trades_ts, head_midpoint_ts, '0000') DESC,
              asset_class, name
            """
        )
    )
    conn.close()
    return [dict(r) for r in rows]


def reconstruct_contract(row: dict) -> Contract:
    """Reconstruct a qualified Contract from the universe.db row."""
    return Contract(
        secType=row["sec_type"],
        conId=row["con_id"],
        symbol=row["local_symbol"] or row["name"],
        exchange=row["exchange"],
        primaryExchange=row["primary_exchange"] or "",
        currency=row["currency"],
        localSymbol=row["local_symbol"] or "",
    )


def pick_what_to_show(row: dict) -> str | None:
    """Prefer TRADES if account has permissions, else MIDPOINT.

    The head-timestamp pre-pass populated head_trades_ts only when the TRADES request
    succeeded; if it's NULL the account doesn't have TRADES permissions for that
    contract's exchange — fall back to MIDPOINT. FX always uses MIDPOINT (TRADES
    returns empty for FX even with permissions).
    """
    if row["asset_class"] == "fx":
        return "MIDPOINT" if row["head_midpoint_ts"] else None
    if row["head_trades_ts"]:
        return "TRADES"
    if row["head_midpoint_ts"]:
        return "MIDPOINT"
    return None


def earliest_available(row: dict, what: str) -> datetime | None:
    raw = row["head_trades_ts"] if what == "TRADES" else row["head_midpoint_ts"]
    if not raw:
        return None
    # Strings like "1980-12-12 14:30:00" from IBKR — assume UTC for simplicity since
    # we set formatDate=1 in the pre-pass (TWS-local timezone, but for daily bars the
    # date is what matters).
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        # IBKR sometimes returns with TZ suffix
        cleaned = raw.split(" US/")[0].split(" Europe/")[0].strip()
        return datetime.fromisoformat(cleaned)


def backfill_one(ib: IB, row: dict) -> int:
    """Returns number of bars written, or 0 on skip/failure."""
    what = pick_what_to_show(row)
    if what is None:
        print(f"  {row['name']:8s}  SKIP — no available whatToShow")
        return 0

    earliest = earliest_available(row, what)
    if earliest is None:
        print(f"  {row['name']:8s}  SKIP — no head timestamp")
        return 0

    contract = reconstruct_contract(row)

    # Walk back in 1-year chunks from now. Use empty endDateTime for "now" then walk.
    all_bars: list = []
    end_dt: str = ""  # empty = now
    bars_pulled_this_loop = -1
    safety_cap = 60  # 60 yrs max — way more than any IBKR depth
    iters = 0
    while bars_pulled_this_loop != 0 and iters < safety_cap:
        iters += 1
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr="1 Y",
                barSizeSetting="1 day",
                whatToShow=what,
                useRTH=False,
                formatDate=2,  # UTC datetime
                timeout=60,
            )
        except Exception as exc:
            print(f"  {row['name']:8s}  ERROR at end={end_dt!r}: {type(exc).__name__}: {exc}")
            break
        bars_pulled_this_loop = len(bars)
        if bars_pulled_this_loop == 0:
            break
        all_bars = list(bars) + all_bars  # bars are oldest-first within a request
        # Step end back to the earliest bar's date for the next request.
        # For 1-day bars, bar.date is datetime.date; for intraday it's datetime.
        oldest = bars[0].date
        if isinstance(oldest, datetime):
            oldest_naive = oldest.replace(tzinfo=None) if oldest.tzinfo else oldest
        else:  # datetime.date
            oldest_naive = datetime(oldest.year, oldest.month, oldest.day)
        end_dt = oldest_naive.strftime("%Y%m%d %H:%M:%S")
        # Stop if we've reached/passed earliest available
        if oldest_naive <= earliest:
            break
        time.sleep(INTER_REQUEST_SLEEP)

    if not all_bars:
        print(f"  {row['name']:8s}  NO BARS RETURNED ({what})")
        return 0

    # Dedupe (boundary bars may repeat) and write.
    # Normalize bar.date to naive datetime for a uniform time-series index.
    def _to_dt(d):
        if isinstance(d, datetime):
            return d.replace(tzinfo=None) if d.tzinfo else d
        return datetime(d.year, d.month, d.day)

    df = pl.DataFrame(
        {
            "date": [_to_dt(b.date) for b in all_bars],
            "open": [float(b.open) for b in all_bars],
            "high": [float(b.high) for b in all_bars],
            "low": [float(b.low) for b in all_bars],
            "close": [float(b.close) for b in all_bars],
            "volume": [float(b.volume) for b in all_bars],
            "bar_count": [b.barCount for b in all_bars],
            "wap": [float(b.average) for b in all_bars],
        }
    ).unique(subset=["date"]).sort("date")

    store = open_store()
    lib_name = library_name(row["asset_class"], "daily_bars")
    lib = ensure_library(store, lib_name)
    sym = symbol_name(row["primary_exchange"] or row["exchange"], row["name"])
    metadata = {
        "what_to_show": what,
        "con_id": row["con_id"],
        "currency": row["currency"],
        "sec_type": row["sec_type"],
        "earliest_in_data": str(df["date"].min()),
        "latest_in_data": str(df["date"].max()),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    lib.write(sym, df.to_pandas().set_index("date"), metadata=metadata)
    n = len(df)
    print(
        f"  {row['name']:8s}  {what:8s}  {n:6d} bars  "
        f"{df['date'].min()} → {df['date'].max()}  "
        f"-> {lib_name}/{sym}"
    )
    return n


def main() -> int:
    rows = load_universe_rows()
    if not rows:
        print("universe.db is empty — run scripts/02_head_timestamps.py first")
        return 1

    ib = IB()
    print(f"connecting to IBKR Gateway @ {GATEWAY_HOST}:{GATEWAY_PORT} clientId={CLIENT_ID}")
    t0 = time.perf_counter()
    try:
        ib.connect(GATEWAY_HOST, GATEWAY_PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    except Exception as exc:
        print(f"connect failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"connected in {(time.perf_counter() - t0) * 1000:.0f} ms (readonly)\n")

    print(f"backfilling daily bars for {len(rows)} contracts...")
    total_bars = 0
    ok = 0
    for row in rows:
        try:
            n = backfill_one(ib, row)
        except Exception as exc:
            print(f"  {row['name']:8s}  TOP-LEVEL ERROR: {type(exc).__name__}: {exc}")
            continue
        if n > 0:
            ok += 1
            total_bars += n
        time.sleep(INTER_REQUEST_SLEEP)

    ib.disconnect()
    print(f"\ndone. {ok}/{len(rows)} contracts written, {total_bars:,} total bars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
