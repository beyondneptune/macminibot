"""105_strategy_zoo_sweep — comprehensive sweep across the Strategy Library.

For every (asset, strategy_template, param_combo) cell, runs the Crucible's
replay engine on 7 days of minute_bars (with daily-bar fallback) and reports
survivors at multiple Sharpe thresholds.

Goal: see which signal classes have ANY edge, even if weak — useful before
deciding which classes to specialise for production deployment.

Output: prints ranked tables + writes
``data/strategy_zoo_results.parquet`` with the full grid.
"""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from backtesting.fees import equity_tiered  # noqa: E402
from backtesting.slippage import slippage_bps_with_floor  # noqa: E402
from factory.strategy_library import STRATEGY_ZOO, iter_param_combinations  # noqa: E402


UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "GLD",
            "SOXL", "NVDL"]
PER_NAME_NOTIONAL = 12_500.0
WARMUP = 60


def load_bars(symbol: str) -> tuple[pd.DataFrame, str]:
    """Try ArcticDB minute_bars first, then daily Alpaca fallback."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from _storage import open_store  # type: ignore
        store = open_store()
        lookup = {
            "AAPL": ("equity.minute_bars", "NASDAQ.AAPL"),
            "MSFT": ("equity.minute_bars", "NASDAQ.MSFT"),
            "GOOGL": ("equity.minute_bars", "NASDAQ.GOOGL"),
            "AMZN": ("equity.minute_bars", "NASDAQ.AMZN"),
            "META": ("equity.minute_bars", "NASDAQ.META"),
            "TSLA": ("equity.minute_bars", "NASDAQ.TSLA"),
            "NVDA": ("equity.minute_bars", "NASDAQ.NVDA"),
            "GLD":  ("etf.minute_bars", "ARCA.GLD"),
        }
        if symbol in lookup:
            lib, sym = lookup[symbol]
            try:
                df = store.get_library(lib).read(sym).data.copy()
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                df = df.sort_index()
                cutoff = df.index[-1] - pd.Timedelta(days=7)
                df = df[df.index >= cutoff][["open", "high", "low", "close", "volume"]]
                if len(df) > WARMUP:
                    return df, "minute_bars"
            except Exception:
                pass
    except Exception:
        pass

    # Alpaca daily fallback
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
    )
    end = datetime.utcnow()
    start = end - timedelta(days=180)
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Day,
                            start=start, end=end, feed=DataFeed.IEX)
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame(), "none"
    if hasattr(bars.index, "levels"):
        bars = bars.xs(symbol, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    return out.sort_index(), "daily_bars"


def fill_cost(qty: int, price: float, year: int,
                daily_vol_bps: float, adv: float) -> float:
    if qty <= 0 or price <= 0:
        return 0.0
    commission = equity_tiered(float(qty), float(price)).total
    notional = qty * price
    slip = slippage_bps_with_floor(
        volume_traded=float(qty), adv=float(max(adv, 1.0)),
        daily_vol_bps=float(max(daily_vol_bps, 1.0)),
        asset_class="equity", k=0.1,
    )
    return commission + notional * (slip.total_bps / 1e4)


def replay(bars: pd.DataFrame, sig: np.ndarray, notional: float) -> dict:
    closes = bars["close"].astype(float).values
    log_ret = np.log(pd.Series(closes) / pd.Series(closes).shift(1))
    rv = log_ret.rolling(21).std().fillna(0.02).values
    adv = bars["volume"].rolling(21).mean().bfill().fillna(1.0).values

    state = 0
    entry_price = 0.0
    qty = 0
    daily_pnl = []
    trades = 0

    for i in range(WARMUP, len(bars)):
        sig_dir = int(sig[i]) if i < len(sig) else 0
        close = float(closes[i])
        year = bars.index[i].year if hasattr(bars.index[i], "year") else 2026
        vbps = float(rv[i]) * math.sqrt(252) * 10_000
        a = max(float(adv[i]), 1.0)

        pnl = 0.0
        if i > WARMUP and state != 0 and qty > 0:
            pnl = (close - float(closes[i - 1])) * qty * state

        if sig_dir != state:
            # close
            if state != 0 and qty > 0:
                cost = fill_cost(qty, close, year, vbps, a)
                pnl -= cost
                trades += 1
            # open
            if sig_dir != 0:
                new_qty = int(notional // close) if close > 0 else 0
                if new_qty > 0:
                    cost = fill_cost(new_qty, close, year, vbps, a)
                    pnl -= cost
                    trades += 1
                    state = sig_dir
                    entry_price = close
                    qty = new_qty
                else:
                    state = 0
                    qty = 0
            else:
                state = 0
                qty = 0
        daily_pnl.append(pnl)

    # unrealized
    open_unrz = 0.0
    if state != 0 and qty > 0 and len(closes):
        open_unrz = (float(closes[-1]) - entry_price) * qty * state

    net = float(sum(daily_pnl)) + open_unrz
    arr = np.asarray(daily_pnl)
    sharpe = float("nan")
    if arr.std() > 0:
        sharpe = float(arr.mean() / arr.std() * math.sqrt(252))
    cum = np.cumsum(arr)
    if len(cum):
        peak = np.maximum.accumulate(cum)
        max_dd = float((cum - peak).min())
    else:
        max_dd = 0.0
    dd_pct = abs(max_dd) / notional
    return {"net": net, "sharpe": sharpe, "max_dd": max_dd,
            "dd_pct": dd_pct, "trades": trades}


def main() -> int:
    print("=" * 130)
    print("  STRATEGY ZOO SWEEP — 15 templates × multiple params × 10 assets")
    print(f"  Substrate: 7-day minute_bars (v2 assets) / 180-day daily (SOXL/NVDL)")
    print(f"  Costs: IBKR-tiered + Almgren-Chriss + spread floor")
    print("=" * 130)

    all_rows = []
    total_strategies = sum(len(iter_param_combinations(s.param_grid)) for s in STRATEGY_ZOO)
    print(f"\n  Total strategy variants: {total_strategies} × {len(UNIVERSE)} assets = "
          f"{total_strategies * len(UNIVERSE)} evaluations\n")

    for asset in UNIVERSE:
        bars, substrate = load_bars(asset)
        if bars.empty or len(bars) < WARMUP + 20:
            print(f"  {asset:<6}  skipped (insufficient bars)")
            continue
        print(f"  {asset:<6}  bars={len(bars):>5}  substrate={substrate}", end="  ")
        n_evals = 0
        for spec in STRATEGY_ZOO:
            for params in iter_param_combinations(spec.param_grid):
                try:
                    sig = spec.fn(bars, **params)
                    res = replay(bars, sig, PER_NAME_NOTIONAL)
                    all_rows.append({
                        "asset": asset, "substrate": substrate,
                        "strategy": spec.name, "params": str(params),
                        **res,
                    })
                    n_evals += 1
                except Exception as exc:
                    all_rows.append({
                        "asset": asset, "substrate": substrate,
                        "strategy": spec.name, "params": str(params),
                        "error": str(exc)[:80],
                    })
        print(f"evals={n_evals}")

    df = pd.DataFrame(all_rows)
    out_path = PROJECT_ROOT / "data" / "strategy_zoo_results.parquet"
    df.to_parquet(out_path, index=False)

    # Filter to successful evals only
    succ = df[df["sharpe"].notna()].copy() if "sharpe" in df.columns else pd.DataFrame()
    if succ.empty:
        print("\n  No successful evaluations.")
        return 1

    print("\n" + "=" * 130)
    print("  SURVIVOR COUNTS BY SHARPE THRESHOLD")
    print("=" * 130)
    for thr in (0.5, 1.0, 1.5, 2.0, 2.5):
        n = int((succ["sharpe"] >= thr).sum())
        n_dd_pass = int(((succ["sharpe"] >= thr) & (succ["dd_pct"] <= 0.10)).sum())
        print(f"  Sharpe ≥ {thr:.1f}: {n:>5} cells   "
              f"(of which |DD| ≤ 10%: {n_dd_pass:>4})")

    print("\n" + "=" * 130)
    print("  TOP-15 BY SHARPE (any DD)")
    print("=" * 130)
    top = succ.nlargest(15, "sharpe")
    print(f"  {'asset':<6} {'strategy':<22} {'sharpe':<8} {'net':<10} {'dd%':<7} "
          f"{'trades':<7} params")
    print("-" * 130)
    for _, r in top.iterrows():
        print(f"  {r['asset']:<6} {r['strategy']:<22} {r['sharpe']:>+6.2f}   "
              f"${r['net']:>+7,.0f}   {r['dd_pct']*100:>5.1f}%   "
              f"{int(r['trades']):>5}   {r['params']}")

    print("\n" + "=" * 130)
    print("  TOP-15 BY SHARPE WITH |DD| ≤ 10%")
    print("=" * 130)
    constrained = succ[succ["dd_pct"] <= 0.10].nlargest(15, "sharpe")
    if constrained.empty:
        print("  (none)")
    else:
        print(f"  {'asset':<6} {'strategy':<22} {'sharpe':<8} {'net':<10} {'dd%':<7} "
              f"{'trades':<7} params")
        print("-" * 130)
        for _, r in constrained.iterrows():
            print(f"  {r['asset']:<6} {r['strategy']:<22} {r['sharpe']:>+6.2f}   "
                  f"${r['net']:>+7,.0f}   {r['dd_pct']*100:>5.1f}%   "
                  f"{int(r['trades']):>5}   {r['params']}")

    print("\n" + "=" * 130)
    print("  PER-STRATEGY-CLASS MEDIAN SHARPE (rough comparability)")
    print("=" * 130)
    by_class = succ.groupby("strategy")["sharpe"].agg(["count", "median", "max"]).sort_values("median", ascending=False)
    print(by_class.to_string())

    # Cross-asset robustness check: which (strategy, params) win on multiple assets?
    print("\n" + "=" * 130)
    print("  CROSS-ASSET ROBUSTNESS — same (strategy, params) wins on N assets")
    print("=" * 130)
    succ["key"] = succ["strategy"] + " | " + succ["params"]
    winners = succ[(succ["sharpe"] >= 1.0) & (succ["dd_pct"] <= 0.10)]
    if len(winners):
        grouped = winners.groupby("key").agg(
            n_assets=("asset", "nunique"),
            assets=("asset", lambda s: ",".join(sorted(s.unique()))),
            median_sharpe=("sharpe", "median"),
            max_sharpe=("sharpe", "max"),
            median_dd_pct=("dd_pct", "median"),
        ).sort_values(["n_assets", "median_sharpe"], ascending=False)
        # Show only those winning on ≥2 assets
        multi = grouped[grouped["n_assets"] >= 2]
        if len(multi):
            print(f"  Found {len(multi)} (strategy, params) combos that pass Sharpe≥1 + DD≤10% on multiple assets:\n")
            for key, row in multi.head(20).iterrows():
                print(f"    n={int(row['n_assets'])}  median_Sh={row['median_sharpe']:+.2f}  "
                      f"median_DD={row['median_dd_pct']*100:.1f}%  "
                      f"assets=[{row['assets']}]")
                print(f"      → {key}")
        else:
            print("  No combo wins on 2+ assets (no robust cross-asset edge).")

        # By strategy CLASS — does a class consistently appear across assets?
        print("\n  Strategy classes with wins on multiple assets:")
        class_robust = winners.groupby("strategy").agg(
            n_assets=("asset", "nunique"),
            assets=("asset", lambda s: ",".join(sorted(s.unique()))),
            n_winners=("sharpe", "count"),
            best_sharpe=("sharpe", "max"),
        ).sort_values(["n_assets", "n_winners"], ascending=False)
        cr_multi = class_robust[class_robust["n_assets"] >= 2]
        for cls, row in cr_multi.iterrows():
            print(f"    {cls:<22} n_assets={int(row['n_assets'])}  "
                  f"n_winners={int(row['n_winners'])}  best_Sh={row['best_sharpe']:+.2f}  "
                  f"assets=[{row['assets']}]")
    else:
        print("  No cells pass Sharpe≥1 + DD≤10%.")
    print(f"\n  Full results → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
