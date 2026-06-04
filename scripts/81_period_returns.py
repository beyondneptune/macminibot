"""81_period_returns — daily / weekly / monthly return distribution for the
deployed hybrid bot over the OOS test window (2026-01-01 → 2026-05-19).

Replays every deployed config, aggregates per-bar MTM into per-day P&L
buckets, sums across the 8 assets to a portfolio-level daily series, then
rolls to weekly (5-trading-day) and monthly (21-trading-day) sums.

Reports:
  • daily P&L distribution (mean, std, min, max, percentiles, win-rate)
  • weekly P&L distribution
  • monthly P&L distribution
  • best/worst day, week, month
  • projection: at this rate, expected $/day, $/week, $/month, $/year

Run:
    .venv/bin/python scripts/81_period_returns.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml_research.asset_specialists import compute_signal_for  # noqa: E402
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from scripts._storage import open_store  # noqa: E402

NET_LIQ = 100000.0
NOTIONAL = NET_LIQ / 8.0
SPLIT = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-05-19")

WARMUP_DAILY = 252
WARMUP_INTRADAY = 500

DEPLOYED = [
    ("AAPL",  "bollinger",     3.0, 3.0, True,  "equity.minute_bars", "NASDAQ.AAPL"),
    ("MSFT",  "bollinger",     3.0, 3.0, True,  "equity.minute_bars", "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0, False, "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      0.5, 3.0, False, "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0, False, "etf.daily_bars",     "ARCA.GLD"),
]


def load_bars(lib, sym, intraday):
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    if intraday and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def precompute_signal(asset, bars, feat, mode, warmup):
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "baseline":
        for i in range(warmup, n):
            try:
                out[i] = int(round(float(compute_signal_for(asset, bars.iloc[: i + 1]))))
            except Exception:
                pass
        return out
    if mode == "buy_and_hold":
        out[warmup:] = 1
        return out
    if mode == "bollinger":
        close = bars["close"]
        ma = close.rolling(20).mean().shift(1)
        sd = close.rolling(20).std().shift(1)
        z = (close.shift(1) - ma) / sd
        out = np.where(z < -2, 1, np.where(z > 2, -1, 0)).astype(np.int8)
        out[:warmup] = 0
        return out
    raise ValueError(f"unknown mode {mode}")


def replay_daily_pnl(bars, feat, sig, start_idx, end_idx, trail, pt):
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    bucket = {}
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0
    vol = feat["vol_21"].values

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig[i])
        close = float(closes[i])
        dk = bars.index[i].normalize()

        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            step = (close - prev_close) * prev_qty * prev_side
            bucket[dk] = bucket.get(dk, 0.0) + step

        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop = peak - band
            reason = None
            if close <= stop:
                reason = "trail"
            elif pt is not None and close >= entry_price * (1 + entry_sigma * pt):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            band = entry_price * entry_sigma * trail
            stop = trough + band
            reason = None
            if close >= stop:
                reason = "trail"
            elif pt is not None and close <= entry_price * (1 - entry_sigma * pt):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        if state == "FLAT" and sig_dir != 0:
            qty = int(NOTIONAL // close) if close > 0 else 0
            if qty > 0:
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price = close
                entry_sigma = sigma
                entry_qty = qty
                if sig_dir > 0:
                    peak, trough = close, None
                else:
                    trough, peak = close, None

        prev_close = close
        prev_state = state
        prev_qty = entry_qty
        prev_side = 1 if state == "IN_LONG" else (-1 if state == "IN_SHORT" else 0)

    return bucket


def main() -> int:
    print()
    print("=" * 100)
    print(f"  Portfolio period returns on OOS window {SPLIT.date()} → {END.date()}")
    print(f"  Account = ${NET_LIQ:,.0f}  •  per-name notional = ${NOTIONAL:,.2f}")
    print("=" * 100)

    # Aggregate per-day P&L across all 8 assets.
    portfolio_pnl: dict[pd.Timestamp, float] = {}

    for asset, sigmode, trail, pt, intraday, lib, sym in DEPLOYED:
        warmup = WARMUP_INTRADAY if intraday else WARMUP_DAILY
        bars = load_bars(lib, sym, intraday)
        feat = add_per_asset_features(bars)
        sig = precompute_signal(asset, bars, feat, sigmode, warmup)
        split_idx = bars.index.get_indexer([SPLIT], method="bfill")[0]
        end_idx = bars.index.get_indexer([END], method="ffill")[0] + 1
        end_idx = min(end_idx, len(bars))
        bucket = replay_daily_pnl(bars, feat, sig, split_idx, end_idx, trail, pt)
        for k, v in bucket.items():
            portfolio_pnl[k] = portfolio_pnl.get(k, 0.0) + v

    daily = pd.Series(portfolio_pnl).sort_index()
    daily_pct = daily / NET_LIQ * 100.0

    def summarize(series: pd.Series, label: str, period_days: int):
        if len(series) == 0:
            return
        pct = series / NET_LIQ * 100.0
        print(f"\n  {label}  (n = {len(series)})")
        print(f"    mean         {pct.mean():>+7.3f}%   ${series.mean():>+9,.2f}")
        print(f"    median       {pct.median():>+7.3f}%   ${series.median():>+9,.2f}")
        print(f"    std dev      {pct.std():>7.3f}%   ${series.std():>9,.2f}")
        print(f"    best          {pct.max():>+7.3f}%   ${series.max():>+9,.2f}  "
              f"({series.idxmax().date()})")
        print(f"    worst         {pct.min():>+7.3f}%   ${series.min():>+9,.2f}  "
              f"({series.idxmin().date()})")
        print(f"    25th pctile  {pct.quantile(0.25):>+7.3f}%   ${series.quantile(0.25):>+9,.2f}")
        print(f"    75th pctile  {pct.quantile(0.75):>+7.3f}%   ${series.quantile(0.75):>+9,.2f}")
        pos = (series > 0).sum()
        print(f"    positive     {pos}/{len(series)}  ({pos/len(series)*100:.1f}%)")
        # Annualized projection (mean × periods/year)
        annual_periods = 252 / period_days
        proj = series.sum() / len(series) * annual_periods
        print(f"    → projected annualized: ${proj:>+9,.0f} "
              f"({proj/NET_LIQ*100:>+6.2f}% on ${NET_LIQ:,.0f})")

    summarize(daily, "DAILY", 1)

    # Weekly buckets (group by ISO week starting Monday).
    weekly = daily.groupby(pd.Grouper(freq="W-MON", label="left", closed="left")).sum()
    weekly = weekly[weekly != 0]  # drop empty weeks
    summarize(weekly, "WEEKLY (5 trading days)", 5)

    # Monthly buckets.
    monthly = daily.groupby(pd.Grouper(freq="MS")).sum()
    monthly = monthly[monthly != 0]
    summarize(monthly, "MONTHLY", 21)

    # Cumulative / final
    print()
    print("=" * 100)
    cum = daily.cumsum()
    peak_eq = cum.cummax()
    dd = (cum - peak_eq)
    print(f"  CUMULATIVE over OOS window:")
    print(f"    total P&L      ${cum.iloc[-1]:+,.2f}")
    print(f"    return %       {cum.iloc[-1]/NET_LIQ*100:+.2f}%")
    print(f"    max drawdown   ${dd.min():+,.2f}  "
          f"({dd.min()/NET_LIQ*100:+.2f}%) on {dd.idxmin().date()}")
    print(f"    days in DD     {(dd < -100).sum()} / {len(dd)}")
    print(f"    test window    {len(daily)} sessions ≈ "
          f"{len(daily)/252.0:.2f}y → annualized {((1 + cum.iloc[-1]/NET_LIQ) ** (252/len(daily)) - 1) * 100:+.1f}%")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
