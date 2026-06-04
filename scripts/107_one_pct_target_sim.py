"""107_one_pct_target_sim — parallel simulation targeting 1%/day on a
$100k Alpaca paper account using 8 leveraged 3×/2× ETFs.

Sizing: $50k notional per name × 8 names = $400k book = 4× stacked on
$100k seed. Within Alpaca intraday-margin cap (which is 4×). Each name
is a 3× ETF so effective underlying exposure ≈ 12×.

Sweep grid:
  8 ETFs × 6 signal families × 3 trail σ × 3 PT σ = 432 per-asset cells.
  Each (signal, trail, pt) config is then aggregated across all 8 assets
  to produce a portfolio-level daily-return series.

For each portfolio config we compute:
  - mean daily % return
  - daily-return std
  - annualised Sharpe
  - max drawdown (% of seed)
  - fraction of days ≥ +1.0%
  - distance from the +1.0%/day target (abs(mean - 1.0))

Then rank by:
  (a) closest to +1.0%/day (the user's target)
  (b) highest Sharpe (the empirical winner)
"""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED_USD = 100_000.0
PER_NAME_NOTIONAL = 50_000.0   # $50k × 8 = $400k = 4× intraday-margin stack
N_NAMES = 8
WARMUP = 60
TARGET_DAILY_PCT = 1.0   # the user's bar

# 8-asset leveraged sleeve
UNIVERSE = [
    "SOXL",  # 3× semis
    "NVDL",  # 2× NVDA
    "TQQQ",  # 3× QQQ
    "UPRO",  # 3× SPX
    "TNA",   # 3× Russell 2000
    "FAS",   # 3× financials
    "LABU",  # 3× biotech
    "TSLL",  # 2× TSLA
]

# Signal grid — keep tight, focus on classes that survived earlier sweeps
SIGNAL_PARAMS = [
    ("rsi2_mr_10_90", {"low": 10, "high": 90}),
    ("rsi14_mr",      {"low": 30, "high": 70}),
    ("macd_hist",     {}),
    ("zscore_ema",    {"window": 20, "entry_z": 2.0}),
    ("bollinger",     {"window": 20, "entry_z": 2.0}),
    ("buy_and_hold",  {}),
]

TRAIL_SIGMAS = [0.5, 1.0, 2.0]
PT_SIGMAS = [None, 3.0]


# ---------------------------------------------------------------------------
# Bars loader
# ---------------------------------------------------------------------------


def load_alpaca_daily(ticker: str, years: float = 3.0) -> pd.DataFrame:
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
    )
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(years * 365.25))
    req = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day,
                            start=start, end=end, feed=DataFeed.IEX)
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    out.index.name = "date"
    return out.sort_index()


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_signal(name: str, params: dict, bars: pd.DataFrame) -> np.ndarray:
    close = bars["close"].astype(float)
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if name == "buy_and_hold":
        out[WARMUP:] = 1
        return out
    if name == "rsi2_mr_10_90":
        r = _rsi(close, 2).shift(1).values
        out = np.where(r < params["low"], 1,
                        np.where(r > params["high"], -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if name == "rsi14_mr":
        r = _rsi(close, 14).shift(1).values
        out = np.where(r < params["low"], 1,
                        np.where(r > params["high"], -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if name == "macd_hist":
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=9, adjust=False).mean()
        hist = (macd - sig).shift(1).values
        out = np.where(hist > 0, 1, np.where(hist < 0, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if name == "zscore_ema":
        w = params["window"]
        ema = close.ewm(span=w, adjust=False).mean().shift(1)
        sd = close.rolling(w).std().shift(1)
        z = (close.shift(1) - ema) / sd.replace(0, np.nan)
        z = z.values
        out = np.where(z < -params["entry_z"], 1,
                        np.where(z > params["entry_z"], -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if name == "bollinger":
        w = params["window"]
        sma = close.rolling(w).mean().shift(1)
        sd = close.rolling(w).std().shift(1)
        z = (close.shift(1) - sma) / sd.replace(0, np.nan)
        z = z.values
        out = np.where(z < -params["entry_z"], 1,
                        np.where(z > params["entry_z"], -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    raise ValueError(f"unknown signal {name}")


# ---------------------------------------------------------------------------
# Cost-aware replay producing a daily-P&L series (in USD)
# ---------------------------------------------------------------------------


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


def replay_daily(bars: pd.DataFrame, sig: np.ndarray, trail_sigma: float,
                  pt_sigma: float | None, notional: float) -> pd.Series:
    """Run the standard trail-stop replay and return a per-bar P&L series
    (indexed by date)."""
    closes = bars["close"].astype(float).values
    log_ret = np.log(pd.Series(closes) / pd.Series(closes).shift(1))
    vol_21 = log_ret.rolling(21).std().fillna(0.02).values
    adv = bars["volume"].rolling(21).mean().bfill().fillna(1.0).values

    state = 0
    entry_price = entry_sigma = 0.0
    qty = 0
    peak = trough = None
    daily_pnl = np.zeros(len(bars), dtype=np.float64)

    for i in range(WARMUP, len(bars)):
        sigma = float(vol_21[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig[i]) if i < len(sig) else 0
        close = float(closes[i])
        year = bars.index[i].year if hasattr(bars.index[i], "year") else 2026
        vbps = sigma * math.sqrt(252) * 10_000.0
        a = max(float(adv[i]), 1.0)

        pnl = 0.0
        # MTM on prior position
        if state != 0 and qty > 0 and i > WARMUP:
            pnl = (close - float(closes[i - 1])) * qty * state

        # Exit logic
        exit_now = False
        if state == 1:
            if peak is None or close > peak:
                peak = close
            stop = peak - entry_price * entry_sigma * trail_sigma
            if close <= stop:
                exit_now = True
            elif pt_sigma is not None and close >= entry_price * (1 + entry_sigma * pt_sigma):
                exit_now = True
            elif sig_dir < 0:
                exit_now = True
        elif state == -1:
            if trough is None or close < trough:
                trough = close
            stop = trough + entry_price * entry_sigma * trail_sigma
            if close >= stop:
                exit_now = True
            elif pt_sigma is not None and close <= entry_price * (1 - entry_sigma * pt_sigma):
                exit_now = True
            elif sig_dir > 0:
                exit_now = True

        if exit_now and qty > 0:
            cost = fill_cost(qty, close, year, vbps, a)
            pnl -= cost
            state = 0
            qty = 0
            peak = trough = None

        # Entry logic
        if state == 0 and sig_dir != 0:
            new_qty = int(notional // close) if close > 0 else 0
            if new_qty > 0:
                cost = fill_cost(new_qty, close, year, vbps, a)
                pnl -= cost
                state = sig_dir
                entry_price = close
                entry_sigma = sigma
                qty = new_qty
                if sig_dir > 0:
                    peak, trough = close, None
                else:
                    trough, peak = close, None

        daily_pnl[i] = pnl

    return pd.Series(daily_pnl, index=bars.index)


# ---------------------------------------------------------------------------
# Cell-level evaluator (the parallel worker)
# ---------------------------------------------------------------------------


def evaluate_cell(args: tuple) -> dict:
    asset, bars, sig_name, sig_params, trail, pt = args
    try:
        sig = build_signal(sig_name, sig_params, bars)
        pnl = replay_daily(bars, sig, trail, pt, PER_NAME_NOTIONAL)
        return {
            "asset": asset,
            "signal": sig_name,
            "trail": trail,
            "pt": pt if pt is not None else "None",
            "pnl_series": pnl.to_dict(),   # picklable
        }
    except Exception as exc:
        return {
            "asset": asset, "signal": sig_name, "trail": trail,
            "pt": pt if pt is not None else "None",
            "error": str(exc)[:120],
        }


def _config_key(c: dict) -> str:
    return f"{c['signal']}|tr={c['trail']}|pt={c['pt']}"


def main() -> int:
    print("=" * 130)
    print("  1%/DAY TARGET SIMULATION — 8 leveraged ETFs × 6 signals × 3 trail × 3 PT")
    print(f"  Notional per name: ${PER_NAME_NOTIONAL:,.0f}  ×  {N_NAMES} names  "
          f"= ${PER_NAME_NOTIONAL*N_NAMES:,.0f} book on ${SEED_USD:,.0f} seed (4× stacked)")
    print(f"  Costs: IBKR-tiered + Almgren-Chriss + spread floor")
    print(f"  Target: +{TARGET_DAILY_PCT:.2f}%/day mean")
    print("=" * 130)

    # 1. Load bars
    print("\n[1/3] Loading bars for 8 leveraged ETFs from Alpaca...")
    bars_dict: dict[str, pd.DataFrame] = {}
    for asset in UNIVERSE:
        try:
            df = load_alpaca_daily(asset, years=3.0)
            if df.empty or len(df) < WARMUP + 50:
                print(f"  {asset:<6} skipped (insufficient bars: {len(df)})")
                continue
            bars_dict[asset] = df
            print(f"  {asset:<6} bars={len(df):>5}  range={df.index[0].date()} → {df.index[-1].date()}")
        except Exception as exc:
            print(f"  {asset:<6} ERROR {exc}")
    if not bars_dict:
        print("FATAL: no bars loaded")
        return 1

    # 2. Build cell list
    cells = [
        (asset, bars, sname, sparams, trail, pt)
        for asset, bars in bars_dict.items()
        for sname, sparams in SIGNAL_PARAMS
        for trail in TRAIL_SIGMAS
        for pt in PT_SIGMAS
    ]
    print(f"\n[2/3] Evaluating {len(cells)} cells in parallel "
           f"(8 ETFs × {len(SIGNAL_PARAMS)} signals × {len(TRAIL_SIGMAS)} trails × {len(PT_SIGMAS)} PTs)...")

    n_workers = min(8, os.cpu_count() or 4)
    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(evaluate_cell, c) for c in cells]
        for fut in as_completed(futures):
            results.append(fut.result())
            completed += 1
            if completed % 50 == 0:
                print(f"  progress: {completed}/{len(cells)}")
    print(f"  done — {len(results)} cells")

    # 3. Aggregate per portfolio config
    print("\n[3/3] Aggregating across assets per (signal, trail, pt) config...")
    by_config: dict[str, list[pd.Series]] = {}
    for r in results:
        if "error" in r:
            continue
        key = _config_key(r)
        s = pd.Series(r["pnl_series"])
        s.index = pd.to_datetime(s.index)
        by_config.setdefault(key, []).append(s)

    portfolio_rows = []
    for key, series_list in by_config.items():
        if len(series_list) < 4:    # need at least 4 of 8 assets
            continue
        df = pd.concat(series_list, axis=1).fillna(0)
        n_assets_used = df.shape[1]
        portfolio_pnl = df.sum(axis=1)
        # Daily % return = $ P&L / SEED (we report as percent of seed, NOT of book)
        daily_pct = portfolio_pnl / SEED_USD * 100
        # Trim warmup zeros
        nonzero = daily_pct[daily_pct.abs() > 1e-9]
        if len(nonzero) < 30:
            continue
        mean_pct = float(daily_pct.mean())
        std_pct = float(daily_pct.std())
        sharpe = (mean_pct / std_pct * math.sqrt(252)) if std_pct > 0 else float("nan")
        cum = portfolio_pnl.cumsum()
        peak = cum.cummax()
        max_dd_usd = float((cum - peak).min())
        max_dd_pct = max_dd_usd / SEED_USD * 100
        days_geq_1 = float((daily_pct >= TARGET_DAILY_PCT).mean() * 100)
        target_dist = abs(mean_pct - TARGET_DAILY_PCT)
        total_usd = float(portfolio_pnl.sum())
        annual_geom = ((1 + mean_pct / 100) ** 252 - 1) * 100 if math.isfinite(mean_pct) else float("nan")
        portfolio_rows.append({
            "config": key,
            "n_assets": n_assets_used,
            "mean_daily_pct": mean_pct,
            "std_daily_pct": std_pct,
            "sharpe": sharpe,
            "max_dd_pct": max_dd_pct,
            "days_geq_1pct": days_geq_1,
            "total_usd": total_usd,
            "ann_geom_pct": annual_geom,
            "target_distance": target_dist,
        })

    if not portfolio_rows:
        print("\nNo valid configs after aggregation.")
        return 1

    pdf = pd.DataFrame(portfolio_rows)
    out_path = PROJECT_ROOT / "data" / "one_pct_target_sim.parquet"
    pdf.to_parquet(out_path, index=False)
    print(f"  full results → {out_path}")

    # ---- Reports ----
    fmt_cols = ["config", "n_assets", "mean_daily_pct", "std_daily_pct",
                "sharpe", "max_dd_pct", "days_geq_1pct",
                "ann_geom_pct", "total_usd"]

    def fmt_table(df: pd.DataFrame, title: str, n: int = 15) -> None:
        print("\n" + "=" * 130)
        print(f"  {title}")
        print("=" * 130)
        if df.empty:
            print("  (empty)")
            return
        head = df.head(n).copy()
        head["mean_daily_pct"] = head["mean_daily_pct"].map(lambda x: f"{x:+.3f}%")
        head["std_daily_pct"] = head["std_daily_pct"].map(lambda x: f"{x:.3f}%")
        head["sharpe"] = head["sharpe"].map(lambda x: f"{x:+.2f}")
        head["max_dd_pct"] = head["max_dd_pct"].map(lambda x: f"{x:+.1f}%")
        head["days_geq_1pct"] = head["days_geq_1pct"].map(lambda x: f"{x:.1f}%")
        head["ann_geom_pct"] = head["ann_geom_pct"].map(lambda x: f"{x:+,.0f}%")
        head["total_usd"] = head["total_usd"].map(lambda x: f"${x:+,.0f}")
        print(head[fmt_cols].to_string(index=False))

    # Ranking 1: closest to +1.00%/day
    nearest = pdf.sort_values("target_distance").reset_index(drop=True)
    fmt_table(nearest, "TOP 15 — CLOSEST to +1.00%/day target", 15)

    # Ranking 2: highest mean daily return (independent of distance)
    by_mean = pdf.sort_values("mean_daily_pct", ascending=False).reset_index(drop=True)
    fmt_table(by_mean, "TOP 15 — HIGHEST mean daily % (regardless of DD)", 15)

    # Ranking 3: highest Sharpe (the empirical winner)
    by_sharpe = pdf[pdf["sharpe"].notna()].sort_values("sharpe", ascending=False).reset_index(drop=True)
    fmt_table(by_sharpe, "TOP 10 — HIGHEST Sharpe", 10)

    # Ranking 4: best risk-adjusted hit on the target — mean ≥ 0.5% AND |DD| ≤ 40%
    realistic = pdf[(pdf["mean_daily_pct"] >= 0.5) & (pdf["max_dd_pct"] >= -40)]
    realistic = realistic.sort_values("mean_daily_pct", ascending=False).reset_index(drop=True)
    fmt_table(realistic, "TOP 10 — REALISTIC (mean ≥ 0.5%/day AND DD ≤ 40% of seed)", 10)

    # Headline summary
    print("\n" + "=" * 130)
    print("  HEADLINE")
    print("=" * 130)
    best_mean = by_mean.iloc[0]
    closest = nearest.iloc[0]
    print(f"  Best mean daily return found: {best_mean['mean_daily_pct']:+.3f}%/day")
    print(f"    config: {best_mean['config']}")
    print(f"    Sharpe: {best_mean['sharpe']:+.2f}  max DD: {best_mean['max_dd_pct']:+.1f}% of seed")
    print(f"  Closest to +1.00%/day target: {closest['mean_daily_pct']:+.3f}%/day")
    print(f"    config: {closest['config']}")
    print(f"    Sharpe: {closest['sharpe']:+.2f}  max DD: {closest['max_dd_pct']:+.1f}% of seed")
    print()
    n_hit = int((pdf["mean_daily_pct"] >= TARGET_DAILY_PCT).sum())
    print(f"  Configs hitting target (mean ≥ +{TARGET_DAILY_PCT:.2f}%/day): {n_hit}/{len(pdf)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
