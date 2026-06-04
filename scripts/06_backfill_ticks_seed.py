"""Historical tick backfill — small "seed" pull for top-5 most-liquid contracts.

Pulls the last SEED_HOURS hours of historical ticks via reqHistoricalTicks for a
small subset of the universe and runs them through the dollar-bar accumulator,
writing emitted bars to ArcticDB library `<asset_class>.dollar_bars_seed` (a
separate library from the live stream's `dollar_bars` to avoid append-order
conflicts on overlapping timestamps).

Top-5 chosen for high-tick-rate diversity across asset classes:
  - AAPL (US equity, NASDAQ, TRADES)
  - MSFT (US equity, NASDAQ, TRADES)
  - ES   (E-mini S&P future, CME, TRADES)
  - GLD  (gold ETF, ARCA, TRADES — works despite earlier permission warning)
  - EURUSD (FX, IDEALPRO, MIDPOINT)

Pacing-aware: 60 req/10min per clientId. Uses clientId 21 (distinct from live stream
which uses 20) so the two run in parallel without competing for budget.

Raw ticks discarded after derivation — only dollar bars persist (per architecture).

Read-only. Run from project root:

    PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/06_backfill_ticks_seed.py \\
        > data/06_seed.log 2>&1 &
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from ib_async import IB, Contract, util

sys.path.insert(0, str(Path(__file__).parent))
from _bars import DollarBarAccumulator  # noqa: E402
from _storage import ensure_library, library_name, open_store, symbol_name  # noqa: E402
from _thresholds import threshold_for  # noqa: E402

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4001
CLIENT_ID = 21
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.db"

# How far back from "now" to pull. Override via env var SEED_HOURS for longer runs.
import os
SEED_HOURS = float(os.environ.get("SEED_HOURS", "1"))

# Contracts to seed
SEED_NAMES = ["AAPL", "MSFT", "ES", "GLD", "EURUSD"]

# Pacing — conservative 11s between requests to stay well below 60/10min
INTER_REQUEST_SLEEP = 11.0

util.logToConsole(level="WARNING")


def load_rows() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(SEED_NAMES))
    rows = list(
        conn.execute(
            f"SELECT * FROM universe WHERE name IN ({placeholders})",
            SEED_NAMES,
        )
    )
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


def write_bars(store, asset_class: str, sym: str, bars: list[dict]) -> None:
    if not bars:
        return
    lib_name = library_name(asset_class, "dollar_bars_seed")
    lib = ensure_library(store, lib_name)
    df = pd.DataFrame(bars)
    df = df.set_index(pd.to_datetime(df["end_time"], utc=True).dt.tz_localize(None))
    df.index.name = "time"
    df = df.drop(columns=["end_time"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if sym in lib.list_symbols():
        # Combine + dedupe + sort to handle multi-run resumes
        existing = lib.read(sym).data
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        lib.write(sym, combined, metadata={"updated_at": datetime.now(timezone.utc).isoformat()})
    else:
        lib.write(sym, df, metadata={"created_at": datetime.now(timezone.utc).isoformat()})


def backfill_one(ib: IB, store, row: dict) -> int:
    threshold, is_midpoint = threshold_for(row["name"], row["asset_class"])
    what = "MIDPOINT" if (is_midpoint or row["asset_class"] == "fx") else "TRADES"
    acc = DollarBarAccumulator(threshold=threshold, is_midpoint=(what == "MIDPOINT"))

    contract = reconstruct_contract(row)
    sym = symbol_name(row["primary_exchange"] or row["exchange"], row["name"])
    print(
        f"\n[{row['name']}] starting — what={what} threshold={threshold} "
        f"hours_back={SEED_HOURS} → library={row['asset_class']}.dollar_bars_seed/{sym}",
        flush=True,
    )

    end_time = datetime.now(timezone.utc)
    earliest_target = end_time - timedelta(hours=SEED_HOURS)
    pending_bars: list[dict] = []
    total_ticks = 0
    request_count = 0
    safety_cap = 10_000  # protect against runaway pacing
    iter_count = 0
    last_flush = time.time()

    while iter_count < safety_cap:
        iter_count += 1
        end_str = end_time.strftime("%Y%m%d-%H:%M:%S")
        try:
            ticks = ib.reqHistoricalTicks(
                contract,
                startDateTime="",
                endDateTime=end_str,
                numberOfTicks=1000,
                whatToShow=what,
                useRth=False,
                ignoreSize=False,
            )
        except Exception as exc:
            print(f"  [{row['name']}] ERROR at end={end_str}: {type(exc).__name__}: {exc}", flush=True)
            break
        request_count += 1
        if not ticks:
            print(f"  [{row['name']}] no more ticks at end={end_str} — done", flush=True)
            break
        # ticks are returned earliest-first; walk forward through them
        for tick in ticks:
            ts = tick.time
            if what == "TRADES":
                price = float(tick.price)
                size = float(tick.size)
                bar = acc.on_trade(ts, price, size)
            else:
                mid = float(getattr(tick, "midPoint", getattr(tick, "price", 0)))
                if mid <= 0:
                    continue
                bar = acc.on_midpoint(ts, mid)
            total_ticks += 1
            if bar:
                pending_bars.append(bar)

        # Step back: new end_time = earliest tick we just received
        earliest_tick_ts = ticks[0].time
        if isinstance(earliest_tick_ts, datetime):
            if earliest_tick_ts.tzinfo is None:
                earliest_tick_ts = earliest_tick_ts.replace(tzinfo=timezone.utc)
            end_time = earliest_tick_ts

        # Stop if we've gone past the target window
        if end_time <= earliest_target:
            print(f"  [{row['name']}] reached target window ({end_time}) — done", flush=True)
            break

        # Periodic flush
        if time.time() - last_flush > 60 or len(pending_bars) >= 100:
            write_bars(store, row["asset_class"], sym, pending_bars)
            print(
                f"  [{row['name']}] flush: ticks={total_ticks:,} req={request_count} "
                f"bars_emitted={len(pending_bars)} end={end_time}",
                flush=True,
            )
            pending_bars.clear()
            last_flush = time.time()

        time.sleep(INTER_REQUEST_SLEEP)

    # Final flush
    write_bars(store, row["asset_class"], sym, pending_bars)
    print(
        f"[{row['name']}] DONE — ticks={total_ticks:,} requests={request_count} "
        f"bars_total={len(pending_bars)}",
        flush=True,
    )
    return len(pending_bars)


def main() -> int:
    rows = load_rows()
    if not rows:
        print("no seed contracts found in universe.db", flush=True)
        return 1
    print(f"seed targets: {[r['name'] for r in rows]}  hours_back={SEED_HOURS}", flush=True)

    ib = IB()
    print(f"connecting to IBKR Gateway @ {GATEWAY_HOST}:{GATEWAY_PORT} clientId={CLIENT_ID}", flush=True)
    try:
        ib.connect(GATEWAY_HOST, GATEWAY_PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    except Exception as exc:
        print(f"connect failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("connected (readonly)\n", flush=True)

    store = open_store()
    for row in rows:
        try:
            backfill_one(ib, store, row)
        except Exception as exc:
            print(f"[{row['name']}] TOP-LEVEL ERROR: {type(exc).__name__}: {exc}", flush=True)

    ib.disconnect()
    print("\nseed backfill complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
