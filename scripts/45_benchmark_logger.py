"""45_benchmark_logger — daily post-close logger of bot P&L vs adversarial
benchmarks (B&H equal-weight + uniform-stop overlays at 1σ/2σ/3σ).

Fires daily at 21:30 BST (16:30 ET) via launchd. Writes one row per
trading day to `data/benchmark_journal.parquet`.

Robustness fix (2026-05-27): previously frozen because ArcticDB
`equity.daily_bars` last bar was 2026-05-20 (nightly-tick-pull job has
not been writing fresh bars). Now: when ArcticDB is stale, gap-fill
the missing daily bars from Alpaca's StockBarsRequest before computing
benchmarks. Logs an explicit `STALE` warning so the freeze is visible.

Run manually:
    .venv/bin/python scripts/45_benchmark_logger.py
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from scripts._storage import open_store  # noqa: E402

NET_LIQ_NOMINAL = 100000.0
NOTIONAL = NET_LIQ_NOMINAL / 8.0
WARMUP = 252
DEPLOY_DATE = pd.Timestamp("2026-05-19")

UNIVERSE = [
    ("AAPL",  "equity.daily_bars", "NASDAQ.AAPL"),
    ("MSFT",  "equity.daily_bars", "NASDAQ.MSFT"),
    ("GOOGL", "equity.daily_bars", "NASDAQ.GOOGL"),
    ("AMZN",  "equity.daily_bars", "NASDAQ.AMZN"),
    ("META",  "equity.daily_bars", "NASDAQ.META"),
    ("TSLA",  "equity.daily_bars", "NASDAQ.TSLA"),
    ("NVDA",  "equity.daily_bars", "NASDAQ.NVDA"),
    ("GLD",   "etf.daily_bars",    "ARCA.GLD"),
]


def _last_trading_day(today: pd.Timestamp) -> pd.Timestamp:
    """Most recent weekday ≤ today (rough: ignores US holidays). The
    Alpaca pull will simply return what it has, so calendar correctness
    only matters for the staleness threshold."""
    d = today.normalize()
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= pd.Timedelta(days=1)
    return d


def _fetch_alpaca_daily_gap(ticker: str, start: pd.Timestamp,
                             end: pd.Timestamp) -> pd.DataFrame:
    """Pull daily bars from Alpaca for [start, end] inclusive.

    Returns DataFrame indexed by date with OHLCV columns. Empty frame
    on error / no bars.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not secret:
        return pd.DataFrame()
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        return pd.DataFrame()
    client = StockHistoricalDataClient(api_key, secret)
    # IEX is the free-tier feed; SIP requires a paid subscription. The runner
    # uses IEX quotes for live trading, so benchmarking on IEX is the
    # apples-to-apples comparison.
    req = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        feed=DataFeed.IEX,
    )
    try:
        bars = client.get_stock_bars(req).df
    except Exception as exc:
        print(f"WARN: alpaca gap-fetch {ticker} failed: {exc}", file=sys.stderr)
        return pd.DataFrame()
    if bars is None or bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        # alpaca-py uses lowercase "symbol" as the outer level
        try:
            bars = bars.xs(ticker, level="symbol")
        except KeyError:
            # fallback: index level 0
            bars = bars.xs(ticker, level=0)
    bars = bars[["open", "high", "low", "close", "volume"]].copy()
    # The Alpaca SDK returns tz-aware UTC; convert to naive UTC then normalize.
    idx = pd.to_datetime(bars.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    bars.index = idx.normalize()
    return bars


def load_bars(lib_name: str, arctic_sym: str, ticker: str) -> tuple[pd.DataFrame, bool]:
    """Load bars from ArcticDB; if stale (>3 calendar days behind today),
    gap-fill from Alpaca. Returns (df, was_gap_filled)."""
    store = open_store()
    df = store.get_library(lib_name).read(arctic_sym).data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[["open", "high", "low", "close", "volume"]]

    today = pd.Timestamp(datetime.now(timezone.utc).date())
    last_trading = _last_trading_day(today)
    last_in_db = df.index[-1].normalize() if len(df) else pd.Timestamp("1970-01-01")
    gap_days = (last_trading - last_in_db).days
    if gap_days <= 3:
        return df, False

    gap_start = last_in_db + pd.Timedelta(days=1)
    gap_end = last_trading + pd.Timedelta(days=1)  # inclusive of last trading day
    gap_df = _fetch_alpaca_daily_gap(ticker, gap_start, gap_end)
    if gap_df.empty:
        return df, False
    gap_df = gap_df[~gap_df.index.isin(df.index)]
    merged = pd.concat([df, gap_df]).sort_index()
    return merged, True


def compute_bh_daily_returns(asset_bars: pd.DataFrame, notional: float,
                              end_date: pd.Timestamp) -> dict:
    df = asset_bars[asset_bars.index <= end_date]
    if len(df) < 2:
        return {"today_pnl": 0.0, "cum_pnl_from_deploy": 0.0, "qty": 0}
    today_pnl = float(df["close"].iloc[-1] - df["close"].iloc[-2])
    deploy_slice = asset_bars[asset_bars.index >= DEPLOY_DATE]
    qty_cum = 0
    cum_pnl = 0.0
    if not deploy_slice.empty:
        anchor = float(deploy_slice["close"].iloc[0])
        qty_cum = int(notional // anchor) if anchor > 0 else 0
        if len(deploy_slice) >= 1:
            cum_pnl = float(deploy_slice["close"].iloc[-1] - anchor) * qty_cum
    yest_anchor = float(df["close"].iloc[-2])
    today_qty = int(notional // yest_anchor) if yest_anchor > 0 else 0
    return {"today_pnl": today_pnl * today_qty,
            "cum_pnl_from_deploy": cum_pnl, "qty": today_qty}


def compute_unif_stop_today_pnl(asset_bars: pd.DataFrame, feat: pd.DataFrame,
                                  notional: float, trail_sigma: float,
                                  end_date: pd.Timestamp) -> dict:
    df = asset_bars[asset_bars.index <= end_date]
    if len(df) < WARMUP + 2:
        return {"today_pnl": 0.0, "cum_pnl_from_deploy": 0.0}
    closes = df["close"].astype(float).values
    vol = feat["vol_21"].reindex(df.index).values
    deploy_idx = df.index.get_indexer([DEPLOY_DATE], method="bfill")[0]
    if deploy_idx < 0 or deploy_idx >= len(df):
        return {"today_pnl": 0.0, "cum_pnl_from_deploy": 0.0}

    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = None
    daily_pnl: list[float] = []

    for i in range(deploy_idx, len(df)):
        sigma = float(vol[i])
        if not math.isfinite(sigma) or sigma <= 0:
            daily_pnl.append(0.0)
            continue
        close = float(closes[i])
        if i > deploy_idx and state == "IN_LONG" and entry_qty > 0:
            day_pnl = (close - float(closes[i - 1])) * entry_qty
        else:
            day_pnl = 0.0
        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            stop = peak - entry_price * entry_sigma * trail_sigma
            if close <= stop:
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
        if state == "FLAT":
            qty = int(notional // close) if close > 0 else 0
            if qty > 0:
                state = "IN_LONG"
                entry_price = close
                entry_sigma = sigma
                entry_qty = qty
                peak = close
        daily_pnl.append(day_pnl)

    return {"today_pnl": daily_pnl[-1] if daily_pnl else 0.0,
            "cum_pnl_from_deploy": float(sum(daily_pnl))}


def get_alpaca_equity() -> float:
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.trading.client import TradingClient
    tc = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
        paper=True,
    )
    acct = tc.get_account()
    return float(acct.portfolio_value)


def main() -> int:
    state_dir = PROJECT_ROOT / "data"
    journal = state_dir / "benchmark_journal.parquet"
    today_utc = datetime.now(timezone.utc).date()
    today_ts = pd.Timestamp(today_utc)
    last_trading = _last_trading_day(today_ts)

    try:
        equity_today = get_alpaca_equity()
    except Exception as exc:
        print(f"WARN: alpaca equity fetch failed: {exc}", file=sys.stderr)
        equity_today = float("nan")

    bh_today_pnl = 0.0
    bh_cum_pnl = 0.0
    us = {"1.0": {"today": 0.0, "cum": 0.0},
          "2.0": {"today": 0.0, "cum": 0.0},
          "3.0": {"today": 0.0, "cum": 0.0}}

    gap_filled_any = False
    stale_assets: list[str] = []
    for asset, lib, sym in UNIVERSE:
        try:
            bars, was_filled = load_bars(lib, sym, asset)
        except Exception as exc:
            print(f"WARN: {asset} bars load failed: {exc}", file=sys.stderr)
            continue
        if was_filled:
            gap_filled_any = True
        # Check post-fill staleness
        last_bar = bars.index[-1].normalize() if len(bars) else pd.Timestamp("1970-01-01")
        if (last_trading - last_bar).days > 1:
            stale_assets.append(f"{asset}:{last_bar.date()}")
        feat = add_per_asset_features(bars)
        bh = compute_bh_daily_returns(bars, NOTIONAL, last_trading)
        bh_today_pnl += bh["today_pnl"]
        bh_cum_pnl += bh["cum_pnl_from_deploy"]
        for trail in (1.0, 2.0, 3.0):
            res = compute_unif_stop_today_pnl(bars, feat, NOTIONAL, trail, last_trading)
            us[f"{trail}"]["today"] += res["today_pnl"]
            us[f"{trail}"]["cum"] += res["cum_pnl_from_deploy"]

    row = {
        "date": today_ts,
        "bot_equity": equity_today,
        "bot_cum_pnl": equity_today - NET_LIQ_NOMINAL if not math.isnan(equity_today) else float("nan"),
        "bh_today_pnl": bh_today_pnl,
        "bh_cum_pnl": bh_cum_pnl,
        "unifstop_1sigma_today_pnl": us["1.0"]["today"],
        "unifstop_1sigma_cum_pnl": us["1.0"]["cum"],
        "unifstop_2sigma_today_pnl": us["2.0"]["today"],
        "unifstop_2sigma_cum_pnl": us["2.0"]["cum"],
        "unifstop_3sigma_today_pnl": us["3.0"]["today"],
        "unifstop_3sigma_cum_pnl": us["3.0"]["cum"],
        "data_gap_filled": gap_filled_any,
        "stale_assets": ",".join(stale_assets) if stale_assets else "",
    }
    df = pd.DataFrame([row])

    if journal.exists():
        existing = pd.read_parquet(journal)
        existing = existing[existing["date"] != today_ts]
        for c in df.columns:
            if c not in existing.columns:
                existing[c] = None
        df = pd.concat([existing, df], ignore_index=True).sort_values("date")
    df.to_parquet(journal, index=False)

    print(f"=== Benchmark journal {today_ts.date()} (last trading day {last_trading.date()}) ===")
    if gap_filled_any:
        print(f"  *** ArcticDB was stale — gap-filled from Alpaca")
    if stale_assets:
        print(f"  *** STILL STALE after fill: {stale_assets}")
    print(f"  bot equity:           ${equity_today:>12,.2f}")
    print(f"  bot cum P&L:          ${row['bot_cum_pnl']:>+12,.2f}")
    print(f"  B&H cum P&L:          ${row['bh_cum_pnl']:>+12,.2f}")
    print(f"  α vs B&H cum:         ${row['bot_cum_pnl'] - row['bh_cum_pnl']:>+12,.2f}")
    print(f"  UnifStop-1σ cum:      ${row['unifstop_1sigma_cum_pnl']:>+12,.2f}")
    print(f"  UnifStop-2σ cum:      ${row['unifstop_2sigma_cum_pnl']:>+12,.2f}")
    print(f"  UnifStop-3σ cum:      ${row['unifstop_3sigma_cum_pnl']:>+12,.2f}")
    best_us = max(row['unifstop_1sigma_cum_pnl'], row['unifstop_2sigma_cum_pnl'], row['unifstop_3sigma_cum_pnl'])
    print(f"  α vs best-UnifStop:   ${row['bot_cum_pnl'] - best_us:>+12,.2f}")
    print(f"  journal → {journal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
