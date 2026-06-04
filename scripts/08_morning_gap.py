"""Morning overnight-gap feature pull.

Runs at 09:35 ET each trading day (via launchd). For each contract in the active
ML-research universe, pulls one 5-minute bar covering 09:30–09:35 ET and computes
the overnight gap: log(open_t / close_{t-1}). Writes to ArcticDB library
`<asset_class>.gap_features` keyed by date.

This is the mitigation Lou/Polk/Skouras (2019) requires when training on T-1 dollar
bars: yesterday's features systematically misrepresent today's intraday regime for
exactly the names where overnight news mattered most. The gap is the regime input
the model conditions on, AND the position-sizer uses to down-size on |gap| > 1.5σ
of trailing 20-day gap distribution (strategy-engine I8).

Read-only on IBKR. Run from project root:

    PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/08_morning_gap.py \\
        > data/08_morning_gap.log 2>&1

Env vars:
    TARGETS=AAPL,MSFT,...     comma-separated names (default: top-5)
    CLIENT_ID=24              IBKR clientId
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ib_async import IB, Contract, util

sys.path.insert(0, str(Path(__file__).parent))
from _storage import ensure_library, library_name, open_store, symbol_name  # noqa: E402

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4001
CLIENT_ID = int(os.environ.get("CLIENT_ID", "24"))
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.db"

DEFAULT_TARGETS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "ES", "GLD", "EURUSD"]
TARGETS = [n.strip() for n in os.environ.get("TARGETS", ",".join(DEFAULT_TARGETS)).split(",") if n.strip()]

util.logToConsole(level="WARNING")


def load_rows(targets: list[str]) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(targets))
    rows = list(conn.execute(f"SELECT * FROM universe WHERE name IN ({placeholders})", targets))
    conn.close()
    return [dict(r) for r in rows]


def reconstruct_contract(row: dict) -> Contract:
    return Contract(
        secType=row["sec_type"],
        conId=row["con_id"],
        symbol=row["local_symbol"] or row["name"],
        exchange=row["exchange"],
        primaryExchange=row["primary_exchange"] or "",
        currency=row["currency"],
        localSymbol=row["local_symbol"] or "",
    )


def pull_gap_one(ib: IB, row: dict) -> dict | None:
    """Pull the 09:30-09:35 ET 5-min bar and compute overnight gap.

    For FX (24/5), the "gap" is the change from prior day close to today's first
    5-min bar — meaningful but smaller magnitude than equity gaps.
    """
    contract = reconstruct_contract(row)
    is_fx = row["asset_class"] == "fx"
    what = "MIDPOINT" if is_fx else "TRADES"

    # Get today's first 5-min bar (will be 09:30-09:35 ET for equities/futures)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="5 mins",
        whatToShow=what,
        useRTH=False,
        formatDate=2,
        timeout=30,
    )
    if not bars:
        print(f"  {row['name']:8s}  NO BARS", flush=True)
        return None

    # The most recent bar is "now" or the close of the most recent 5-min period.
    # For an open-bar feature we want the bar at 09:30:00 of the latest trading day.
    latest = bars[-1]
    # Look up prior close from daily bars in ArcticDB
    store = open_store()
    primary = row["primary_exchange"] or row["exchange"]
    sym = symbol_name(primary, row["name"])
    daily_lib_name = library_name(row["asset_class"], "daily_bars")
    if sym not in store[daily_lib_name].list_symbols():
        print(f"  {row['name']:8s}  no daily bars in ArcticDB", flush=True)
        return None
    daily_df = store[daily_lib_name].read(sym).data
    if daily_df.empty:
        return None
    prev_close = float(daily_df["close"].iloc[-1])
    open_5min = float(latest.open)
    gap = math.log(open_5min / prev_close) if prev_close > 0 else 0.0

    print(
        f"  {row['name']:8s}  prev_close={prev_close:.4f} "
        f"open={open_5min:.4f}  gap={gap*100:+.3f}%",
        flush=True,
    )
    return {
        "symbol": sym,
        "asset_class": row["asset_class"],
        "bar_time": latest.date,
        "prev_close": prev_close,
        "open_5min": open_5min,
        "high_5min": float(latest.high),
        "low_5min": float(latest.low),
        "close_5min": float(latest.close),
        "volume_5min": float(latest.volume),
        "gap_log_return": gap,
    }


def write_gap_features(store, rows_data: list[dict]) -> None:
    """Append today's gaps to each contract's gap_features symbol."""
    by_asset_class: dict[str, list[dict]] = {}
    for d in rows_data:
        by_asset_class.setdefault(d["asset_class"], []).append(d)
    for asset_class, entries in by_asset_class.items():
        lib_name = library_name(asset_class, "gap_features")
        lib = ensure_library(store, lib_name)
        for entry in entries:
            df = pd.DataFrame([entry])
            df.index = pd.to_datetime(df["bar_time"], utc=True).dt.tz_localize(None)
            df.index.name = "time"
            df = df.drop(columns=["bar_time", "symbol", "asset_class"])
            sym = entry["symbol"]
            if sym in lib.list_symbols():
                existing = lib.read(sym).data
                combined = pd.concat([existing, df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                lib.write(sym, combined, metadata={"updated_at": datetime.now(timezone.utc).isoformat()})
            else:
                lib.write(sym, df, metadata={"created_at": datetime.now(timezone.utc).isoformat()})


def main() -> int:
    rows = load_rows(TARGETS)
    if not rows:
        print(f"no targets in universe.db for: {TARGETS}", flush=True)
        return 1
    print(f"targets: {[r['name'] for r in rows]}  client_id={CLIENT_ID}", flush=True)

    ib = IB()
    print(f"connecting to IBKR Gateway @ {GATEWAY_HOST}:{GATEWAY_PORT} clientId={CLIENT_ID}", flush=True)
    try:
        ib.connect(GATEWAY_HOST, GATEWAY_PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    except Exception as exc:
        print(f"connect failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("connected (readonly)\n", flush=True)

    store = open_store()
    results = []
    for row in rows:
        try:
            d = pull_gap_one(ib, row)
            if d:
                results.append(d)
        except Exception as exc:
            print(f"  {row['name']:8s}  ERROR: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(0.5)

    if results:
        write_gap_features(store, results)
        print(f"\n  wrote {len(results)} gap-feature rows", flush=True)
    ib.disconnect()
    print("morning gap pull complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
