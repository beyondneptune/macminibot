"""Nightly historical tick pull → dollar bars (production).

The ratified path forward after research synthesis on 2026-05-17:
- Live tick subscriptions denied on account U18581020 (no live market-data subs)
- Historical tick pull via reqHistoricalTicks works — verified by seed run
- Build dollar bars from yesterday's closed session, write to ArcticDB
- T-1 stale; rebalance once per day at session open with overnight-gap conditioning

This script pulls one trading day's ticks for the configured target universe and
derives dollar bars using the López de Prado threshold = median(daily $ volume) / 50.
Writes to `<asset_class>.dollar_bars` (production library, not `_seed`).

Default target: AAPL (proof-of-pipeline). To run the full top-10 nightly, set
TARGETS env var (comma-separated names from data/universe.db).

Pacing: 11s between requests = ~5.5 req/min ≈ within 60/10min budget per clientId.

Read-only. Run from project root:

    PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/07_nightly_tick_pull.py \\
        > data/07_nightly_tick_pull.log 2>&1 &

Env vars:
    TARGETS=AAPL,MSFT          comma-separated contract names (default: AAPL)
    SESSION_DATE=2026-05-15    YYYY-MM-DD trading day to pull (default: most recent)
    HOURS_BACK=8               session length in hours to pull (default: 8 — US RTH+pre)
    CLIENT_ID=22               IBKR clientId (default: 22, distinct from live/seed)
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from ib_async import IB, Contract, util

sys.path.insert(0, str(Path(__file__).parent))
from _bars import DollarBarAccumulator  # noqa: E402
from _calibration import calibrate_threshold  # noqa: E402
from _storage import ensure_library, library_name, open_store, symbol_name  # noqa: E402

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4001
CLIENT_ID = int(os.environ.get("CLIENT_ID", "22"))
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.db"

TARGETS = [n.strip() for n in os.environ.get("TARGETS", "AAPL").split(",") if n.strip()]
HOURS_BACK = float(os.environ.get("HOURS_BACK", "8"))  # US RTH = 6.5h; pre/post adds buffer
SESSION_DATE_STR = os.environ.get("SESSION_DATE", "")  # blank → most recent

INTER_REQUEST_SLEEP = 11.0

util.logToConsole(level="WARNING")


def load_rows(targets: list[str]) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(targets))
    rows = list(
        conn.execute(
            f"SELECT * FROM universe WHERE name IN ({placeholders})",
            targets,
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


def resolve_session_end_utc() -> datetime:
    """Find the end-of-session UTC datetime to walk backwards from.

    If SESSION_DATE is set, use that date at 21:00 UTC (≈ 16:00 ET pre-DST cushion).
    Else, use 'now' rounded down to the nearest minute.
    """
    if SESSION_DATE_STR:
        d = datetime.strptime(SESSION_DATE_STR, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return d.replace(hour=21, minute=0, second=0)
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


def write_bars_append(store, asset_class: str, sym: str, bars: list[dict]) -> int:
    """Write bars to <asset_class>.dollar_bars/<sym>, appending if exists."""
    if not bars:
        return 0
    lib_name = library_name(asset_class, "dollar_bars")
    lib = ensure_library(store, lib_name)
    df = pd.DataFrame(bars)
    df.index = pd.to_datetime(df["end_time"], utc=True).dt.tz_localize(None)
    df.index.name = "time"
    df = df.drop(columns=["end_time"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if sym in lib.list_symbols():
        existing = lib.read(sym).data
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        lib.write(sym, combined, metadata={"updated_at": datetime.now(timezone.utc).isoformat()})
    else:
        lib.write(sym, df, metadata={"created_at": datetime.now(timezone.utc).isoformat()})
    return len(df)


def pull_one(ib: IB, store, row: dict) -> dict:
    is_fx = row["asset_class"] == "fx"
    # For midpoint contracts, calibration falls back to default tick-count threshold
    # since dollar-volume calibration doesn't apply to midpoint streams.
    is_midpoint = is_fx  # extend later for non-NASDAQ stocks
    primary = row["primary_exchange"] or row["exchange"]
    threshold, source = calibrate_threshold(row["name"], row["asset_class"], primary, is_midpoint)
    what = "MIDPOINT" if is_midpoint else "TRADES"
    acc = DollarBarAccumulator(threshold=threshold, is_midpoint=is_midpoint)

    contract = reconstruct_contract(row)
    sym = symbol_name(primary, row["name"])
    session_end = resolve_session_end_utc()
    earliest_target = session_end - timedelta(hours=HOURS_BACK)
    print(
        f"[{row['name']}] start session_end={session_end.isoformat()} "
        f"target_window={HOURS_BACK}h what={what} "
        f"threshold={threshold:,.0f} ({source})",
        flush=True,
    )

    end_time = session_end
    pending: list[dict] = []
    total_ticks = 0
    request_count = 0
    iter_count = 0
    safety = 5_000

    while iter_count < safety:
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
            print(f"[{row['name']}] error at end={end_str}: {type(exc).__name__}: {exc}", flush=True)
            break
        request_count += 1
        if not ticks:
            print(f"[{row['name']}] no more ticks at end={end_str} (request {request_count})", flush=True)
            break

        for tick in ticks:
            ts = tick.time
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if is_midpoint:
                mid = float(getattr(tick, "midPoint", getattr(tick, "price", 0)))
                if mid <= 0:
                    continue
                bar = acc.on_midpoint(ts, mid)
            else:
                bar = acc.on_trade(ts, float(tick.price), float(tick.size))
            total_ticks += 1
            if bar:
                pending.append(bar)

        # Step end back to earliest tick we just saw
        earliest_tick_ts = ticks[0].time
        if isinstance(earliest_tick_ts, datetime):
            if earliest_tick_ts.tzinfo is None:
                earliest_tick_ts = earliest_tick_ts.replace(tzinfo=timezone.utc)
            end_time = earliest_tick_ts

        if end_time <= earliest_target:
            print(
                f"[{row['name']}] reached target window ({end_time.isoformat()}) "
                f"after {request_count} reqs / {total_ticks:,} ticks / {len(pending)} bars",
                flush=True,
            )
            break

        # Progress log every 25 pages
        if request_count % 25 == 0:
            print(
                f"[{row['name']}] progress: req={request_count} ticks={total_ticks:,} "
                f"bars={len(pending)} end={end_time.isoformat()}",
                flush=True,
            )

        time.sleep(INTER_REQUEST_SLEEP)

    # Final flush
    written = write_bars_append(store, row["asset_class"], sym, pending)
    print(
        f"[{row['name']}] DONE: requests={request_count} ticks={total_ticks:,} "
        f"bars_emitted={len(pending)} bars_written={written} → "
        f"{row['asset_class']}.dollar_bars/{sym}",
        flush=True,
    )
    return {
        "name": row["name"],
        "requests": request_count,
        "ticks": total_ticks,
        "bars": len(pending),
    }


def main() -> int:
    rows = load_rows(TARGETS)
    if not rows:
        print(f"no targets found in universe.db for: {TARGETS}", flush=True)
        return 1
    print(f"targets: {[r['name'] for r in rows]}  hours_back={HOURS_BACK}  client_id={CLIENT_ID}", flush=True)

    ib = IB()
    print(f"connecting to IBKR Gateway @ {GATEWAY_HOST}:{GATEWAY_PORT} clientId={CLIENT_ID}", flush=True)
    try:
        ib.connect(GATEWAY_HOST, GATEWAY_PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    except Exception as exc:
        print(f"connect failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    print(f"connected (readonly)  server={ib.client.serverVersion()}\n", flush=True)

    store = open_store()
    results = []
    for row in rows:
        try:
            r = pull_one(ib, store, row)
            results.append(r)
        except Exception as exc:
            print(f"[{row['name']}] TOP-LEVEL ERROR: {type(exc).__name__}: {exc}", flush=True)

    ib.disconnect()
    print("\n=== summary ===", flush=True)
    for r in results:
        print(
            f"  {r['name']:8s}  reqs={r['requests']:4d}  ticks={r['ticks']:>10,}  bars={r['bars']:4d}",
            flush=True,
        )
    print("nightly tick pull complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
