"""Head-timestamp pre-pass for the starter universe.

For every contract in scripts/_universe.py, resolve it via reqContractDetails and
query reqHeadTimeStamp to learn exactly how far back IBKR will serve data. Persists
results to data/universe.db (SQLite) so backfill scripts can plan precisely without
re-probing IBKR (which costs against pacing).

Read-only — never places orders. Connect with readonly=True.

Run from project root:

    .venv/bin/python scripts/02_head_timestamps.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

from ib_async import IB, util

sys.path.insert(0, str(Path(__file__).parent))
from _universe import UNIVERSE, UniverseEntry  # noqa: E402

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4001
CLIENT_ID = 12
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.db"

# Modest spacing — reqHeadTimeStamp doesn't share the 60/10min historical budget per
# IBKR docs, but better safe than 162'd. ~80 calls @ 0.3s = ~24s total.
INTER_REQUEST_SLEEP = 0.3

util.logToConsole(level="WARNING")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS universe (
            name TEXT PRIMARY KEY,
            asset_class TEXT NOT NULL,
            sec_type TEXT NOT NULL,
            exchange TEXT,
            primary_exchange TEXT,
            currency TEXT,
            con_id INTEGER,
            local_symbol TEXT,
            long_name TEXT,
            time_zone_id TEXT,
            what_to_show_trades TEXT,
            head_trades_ts TEXT,
            head_midpoint_ts TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def probe(ib: IB, entry: UniverseEntry) -> dict | None:
    details_list = ib.reqContractDetails(entry.contract)
    if not details_list:
        print(f"  {entry.name:8s}  NO CONTRACT (entitlement or symbol error)")
        return None
    cd = details_list[0]
    qual = cd.contract  # fully qualified with conId

    time.sleep(INTER_REQUEST_SLEEP)
    head_trades = None
    head_midpoint = None
    if entry.what_to_show_trades == "TRADES":
        try:
            head_trades = ib.reqHeadTimeStamp(qual, whatToShow="TRADES", useRTH=False, formatDate=1)
        except Exception as exc:
            print(f"  {entry.name:8s}  TRADES head failed: {exc}")
    time.sleep(INTER_REQUEST_SLEEP)
    try:
        head_midpoint = ib.reqHeadTimeStamp(qual, whatToShow="MIDPOINT", useRTH=False, formatDate=1)
    except Exception as exc:
        print(f"  {entry.name:8s}  MIDPOINT head failed: {exc}")

    head_disp = head_trades or head_midpoint
    print(
        f"  {entry.name:8s}  conId={qual.conId:<12d} "
        f"primary={qual.primaryExchange or '-':10s} "
        f"head={head_disp}"
    )
    return {
        "sec_type": qual.secType,
        "exchange": qual.exchange,
        "primary_exchange": qual.primaryExchange or "",
        "currency": qual.currency,
        "con_id": qual.conId,
        "local_symbol": qual.localSymbol,
        "long_name": cd.longName,
        "time_zone_id": cd.timeZoneId,
        "head_trades_ts": str(head_trades) if head_trades else None,
        "head_midpoint_ts": str(head_midpoint) if head_midpoint else None,
    }


def main() -> int:
    ib = IB()
    print(f"connecting to IBKR Gateway @ {GATEWAY_HOST}:{GATEWAY_PORT} clientId={CLIENT_ID}")
    t0 = time.perf_counter()
    try:
        ib.connect(GATEWAY_HOST, GATEWAY_PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    except Exception as exc:
        print(f"connect failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"connected in {(time.perf_counter() - t0) * 1000:.0f} ms (readonly)")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    ensure_schema(conn)

    print(f"\nprobing {len(UNIVERSE)} contracts...")
    ok = 0
    fail = 0
    for entry in UNIVERSE:
        try:
            data = probe(ib, entry)
        except Exception as exc:
            print(f"  {entry.name:8s}  EXCEPTION: {type(exc).__name__}: {exc}")
            fail += 1
            continue
        if data is None:
            fail += 1
            continue
        ok += 1
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn.execute(
            """
            INSERT INTO universe (
                name, asset_class, sec_type, exchange, primary_exchange, currency,
                con_id, local_symbol, long_name, time_zone_id, what_to_show_trades,
                head_trades_ts, head_midpoint_ts, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                asset_class=excluded.asset_class,
                sec_type=excluded.sec_type,
                exchange=excluded.exchange,
                primary_exchange=excluded.primary_exchange,
                currency=excluded.currency,
                con_id=excluded.con_id,
                local_symbol=excluded.local_symbol,
                long_name=excluded.long_name,
                time_zone_id=excluded.time_zone_id,
                what_to_show_trades=excluded.what_to_show_trades,
                head_trades_ts=excluded.head_trades_ts,
                head_midpoint_ts=excluded.head_midpoint_ts,
                updated_at=excluded.updated_at
            """,
            (
                entry.name,
                entry.asset_class,
                data["sec_type"],
                data["exchange"],
                data["primary_exchange"],
                data["currency"],
                data["con_id"],
                data["local_symbol"],
                data["long_name"],
                data["time_zone_id"],
                entry.what_to_show_trades,
                data["head_trades_ts"],
                data["head_midpoint_ts"],
                now_iso,
            ),
        )
        conn.commit()

    ib.disconnect()
    conn.close()
    print(f"\ndone. ok={ok}  fail={fail}  db={DB_PATH}")
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
