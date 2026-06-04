"""09_alpaca_daily_backfill — pull missing daily bars via the Alpaca data API
and upsert into ArcticDB. Used when IBKR Gateway is offline so the per-asset
runner has fresh bars before its 14:35 BST fire.

Run:
    .venv/bin/python scripts/09_alpaca_daily_backfill.py            # last 5 days
    .venv/bin/python scripts/09_alpaca_daily_backfill.py --date 2026-05-18
    .venv/bin/python scripts/09_alpaca_daily_backfill.py --from 2026-05-15 --to 2026-05-19

Uses ArcticDB's `update()` which is row-level upsert: overlapping dates are
replaced, new dates appended. Safe to re-run.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

ASSET_MAP = {
    "AAPL":  ("equity.daily_bars", "NASDAQ.AAPL"),
    "MSFT":  ("equity.daily_bars", "NASDAQ.MSFT"),
    "GOOGL": ("equity.daily_bars", "NASDAQ.GOOGL"),
    "AMZN":  ("equity.daily_bars", "NASDAQ.AMZN"),
    "META":  ("equity.daily_bars", "NASDAQ.META"),
    "TSLA":  ("equity.daily_bars", "NASDAQ.TSLA"),
    "NVDA":  ("equity.daily_bars", "NASDAQ.NVDA"),
    "GLD":   ("etf.daily_bars",    "ARCA.GLD"),
    # v3 leveraged ETF candidates (added 2026-05-21)
    "SOXL":  ("etf.daily_bars",    "ARCA.SOXL"),
    "NVDL":  ("etf.daily_bars",    "ARCA.NVDL"),
    "TSLL":  ("etf.daily_bars",    "ARCA.TSLL"),
}


def parse_args() -> tuple[datetime, datetime]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, help="single ISO date YYYY-MM-DD")
    ap.add_argument("--from", dest="from_", type=str, help="ISO date start")
    ap.add_argument("--to", dest="to_", type=str, help="ISO date end (exclusive)")
    a = ap.parse_args()
    if a.date:
        d = datetime.strptime(a.date, "%Y-%m-%d")
        return d, d + timedelta(days=1)
    if a.from_:
        start = datetime.strptime(a.from_, "%Y-%m-%d")
        end = datetime.strptime(a.to_, "%Y-%m-%d") if a.to_ else start + timedelta(days=1)
        return start, end
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return end - timedelta(days=5), end + timedelta(days=1)


def main() -> int:
    start, end = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ["ALPACA_API_KEY"]
    secret = os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"]

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from scripts._storage import open_store

    client = StockHistoricalDataClient(api_key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=list(ASSET_MAP.keys()),
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    print(f"Pulling daily bars {start.date()} → {end.date()} for "
          f"{len(ASSET_MAP)} symbols...")
    bars = client.get_stock_bars(req)
    store = open_store()

    print("=" * 80)
    print(f"{'ticker':<7} {'lib':<20} {'arctic_sym':<16} {'new_bars':<10} {'last_date'}")
    print("-" * 80)
    total_new = 0
    for ticker, (lib_name, sym) in ASSET_MAP.items():
        rows = bars.data.get(ticker, [])
        if not rows:
            print(f"{ticker:<7} {lib_name:<20} {sym:<16} 0          (no data)")
            continue
        records = []
        for b in rows:
            # Alpaca timestamps are UTC tz-aware datetimes (stdlib); coerce to
            # pd.Timestamp and normalize to naive date level because the existing
            # ArcticDB schema uses a naive `date` index.
            ts = pd.Timestamp(b.timestamp).tz_convert("UTC").tz_localize(None).normalize()
            records.append({
                "date": ts,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
                "bar_count": int(b.trade_count or 0),
                "wap": float(b.vwap),
            })
        df = pd.DataFrame(records).set_index("date").sort_index()
        df.index.name = "date"

        lib = store.get_library(lib_name)
        try:
            existing = lib.read(sym).data
            had_dates = set(existing.index.normalize())
        except Exception:
            had_dates = set()

        new_dates = set(df.index) - had_dates
        # ArcticDB `update` does range-replace + extend — safe upsert.
        lib.update(sym, df, upsert=True)
        last = df.index[-1].strftime("%Y-%m-%d")
        print(f"{ticker:<7} {lib_name:<20} {sym:<16} "
              f"{len(new_dates):<10} {last}")
        total_new += len(new_dates)

    print("=" * 80)
    print(f"Total NEW bars added: {total_new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
