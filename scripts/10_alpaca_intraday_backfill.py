"""10_alpaca_intraday_backfill — pull 5-minute (or 1-minute) bars from Alpaca
for the per-asset bake-off universe and upsert into ArcticDB.

This is the substrate-change experiment recommended at the end of the
comprehensive daily-bar search (scripts 73–77). Daily bars cap empirically
around +28–35% annualized on this universe; intraday gives ~78× more
decision points per day per asset and is the next bake-off substrate.

Storage:
    equity.minute_bars  →  NASDAQ.AAPL, NASDAQ.MSFT, ... (7 names)
    etf.minute_bars     →  ARCA.GLD

Run:
    .venv/bin/python scripts/10_alpaca_intraday_backfill.py --tf 5min --years 2
    .venv/bin/python scripts/10_alpaca_intraday_backfill.py --tf 1min --years 1 \
        --tickers AAPL                                      # AAPL-only smoke
    .venv/bin/python scripts/10_alpaca_intraday_backfill.py --tf 5min --years 2 \
        --replace                                           # wipe and re-pull

Approach: per asset, pull in 30-day chunks (Alpaca's bar endpoint caps at
~10k bars per call; 30 days × ~78 5m bars = 2,340 rows, well under cap).
Resumable: re-running skips dates already in ArcticDB unless --replace.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

ASSET_MAP = {
    "AAPL":  ("equity.minute_bars", "NASDAQ.AAPL"),
    "MSFT":  ("equity.minute_bars", "NASDAQ.MSFT"),
    "GOOGL": ("equity.minute_bars", "NASDAQ.GOOGL"),
    "AMZN":  ("equity.minute_bars", "NASDAQ.AMZN"),
    "META":  ("equity.minute_bars", "NASDAQ.META"),
    "TSLA":  ("equity.minute_bars", "NASDAQ.TSLA"),
    "NVDA":  ("equity.minute_bars", "NASDAQ.NVDA"),
    "GLD":   ("etf.minute_bars",    "ARCA.GLD"),
    # v3 leveraged ETF candidates (added 2026-05-21)
    "SOXL":  ("etf.minute_bars",    "ARCA.SOXL"),
    "NVDL":  ("etf.minute_bars",    "ARCA.NVDL"),
    "TSLL":  ("etf.minute_bars",    "ARCA.TSLL"),
}


def get_timeframe(tf: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    if tf == "1min":
        return TimeFrame(1, TimeFrameUnit.Minute)
    if tf == "5min":
        return TimeFrame(5, TimeFrameUnit.Minute)
    if tf == "15min":
        return TimeFrame(15, TimeFrameUnit.Minute)
    raise ValueError(f"unsupported timeframe {tf!r}")


def ensure_library(store, lib_name: str) -> None:
    if lib_name not in set(store.list_libraries()):
        store.create_library(lib_name)


def existing_index(store, lib_name: str, sym: str) -> pd.DatetimeIndex:
    try:
        df = store.get_library(lib_name).read(sym).data
        return df.index
    except Exception:
        return pd.DatetimeIndex([])


def fetch_chunk(client, ticker: str, tf, start_utc: datetime,
                end_utc: datetime) -> pd.DataFrame:
    """One Alpaca call. Returns DataFrame with the ArcticDB-target schema."""
    from alpaca.data.requests import StockBarsRequest
    req = StockBarsRequest(
        symbol_or_symbols=ticker, timeframe=tf,
        start=start_utc, end=end_utc,
    )
    bars = client.get_stock_bars(req)
    rows = bars.data.get(ticker, [])
    if not rows:
        return pd.DataFrame()
    records = []
    for b in rows:
        ts = pd.Timestamp(b.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        records.append({
            "timestamp": ts,
            "open": float(b.open), "high": float(b.high), "low": float(b.low),
            "close": float(b.close), "volume": float(b.volume),
            "bar_count": int(b.trade_count or 0), "wap": float(b.vwap),
        })
    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["1min", "5min", "15min"], default="5min")
    ap.add_argument("--years", type=float, default=2.0,
                    help="how many years back to pull (default 2)")
    ap.add_argument("--tickers", type=str, default=None,
                    help="comma-separated subset of universe (default: all 8)")
    ap.add_argument("--replace", action="store_true",
                    help="wipe ArcticDB symbol before pulling")
    ap.add_argument("--chunk-days", type=int, default=30,
                    help="days per API call (default 30)")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ["ALPACA_API_KEY"]
    secret = os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"]

    from alpaca.data.historical import StockHistoricalDataClient
    from scripts._storage import open_store

    client = StockHistoricalDataClient(api_key, secret)
    store = open_store()
    tf = get_timeframe(args.tf)

    tickers = (args.tickers.split(",") if args.tickers
               else list(ASSET_MAP.keys()))
    tickers = [t.strip() for t in tickers if t.strip()]
    for t in tickers:
        if t not in ASSET_MAP:
            raise ValueError(f"{t} not in universe; choose from {list(ASSET_MAP)}")

    end_utc = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_utc = end_utc - timedelta(days=int(args.years * 365.25))

    print()
    print("=" * 92)
    print(f"  Intraday backfill  tf={args.tf}  span={start_utc.date()} → {end_utc.date()}  "
          f"chunk={args.chunk_days}d")
    print(f"  tickers: {tickers}")
    print("=" * 92)

    for ticker in tickers:
        lib_name, sym = ASSET_MAP[ticker]
        ensure_library(store, lib_name)
        lib = store.get_library(lib_name)
        if args.replace:
            try:
                lib.delete(sym)
            except Exception:
                pass

        existing_dates = set(d.date() for d in existing_index(store, lib_name, sym))
        cursor = start_utc
        total_new = 0
        chunks_pulled = 0
        chunks_skipped = 0

        while cursor < end_utc:
            chunk_end = min(cursor + timedelta(days=args.chunk_days), end_utc)
            # Skip the chunk if EVERY trading day inside it is already in
            # the existing index (resume path).
            chunk_days = pd.bdate_range(cursor.date(), chunk_end.date() - timedelta(days=1))
            if not args.replace and len(chunk_days) > 0 and \
                    all(d.date() in existing_dates for d in chunk_days):
                chunks_skipped += 1
                cursor = chunk_end
                continue
            try:
                df = fetch_chunk(client, ticker, tf, cursor, chunk_end)
            except Exception as exc:
                print(f"  {ticker}  {cursor.date()}→{chunk_end.date()}  ERR {type(exc).__name__}: {exc}")
                cursor = chunk_end
                continue
            if df.empty:
                chunks_pulled += 1
                cursor = chunk_end
                continue
            # Drop already-existing timestamps to keep upsert lean.
            if not args.replace:
                df = df[~df.index.isin(existing_index(store, lib_name, sym))]
            if not df.empty:
                # ArcticDB requires tz-naive index for some library configs;
                # we keep UTC tz-aware to preserve the wall-clock semantics.
                # (If the existing library demands naive, drop tz here.)
                try:
                    lib.update(sym, df, upsert=True)
                except Exception:
                    # Fallback: drop tz and retry.
                    df2 = df.copy()
                    df2.index = df2.index.tz_convert("UTC").tz_localize(None)
                    df2.index.name = "timestamp"
                    lib.update(sym, df2, upsert=True)
                total_new += len(df)
            chunks_pulled += 1
            cursor = chunk_end
            # Rate-limit-friendly pacing (Alpaca free tier = 200 req/min).
            time.sleep(0.05)

        # Final size summary.
        try:
            now = lib.read(sym).data
            n = len(now)
            first = now.index[0]; last = now.index[-1]
        except Exception:
            n, first, last = 0, "—", "—"
        print(f"  {ticker:<5}  pulled {chunks_pulled} chunks ({chunks_skipped} skipped), "
              f"added {total_new:>6} rows  →  total {n:>7} rows  "
              f"({first} → {last})")

    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
