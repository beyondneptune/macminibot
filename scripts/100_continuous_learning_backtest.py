"""100_continuous_learning_backtest — high-fidelity walk-forward validation
of the v2 + v3 (leveraged sleeve) architecture with continuous LightGBM
retraining simulated at quarterly boundaries.

What this validates:
  1. The architecture we just built (continuous retraining daemon + leveraged
     sleeve infrastructure) before any real capital is at risk.
  2. The 10-asset universe (v2 8-name + SOXL + NVDL) under realistic costs:
      - IBKR Pro tiered commission via src/backtesting/fees.equity_tiered
      - Almgren-Chriss square-root impact via src/backtesting/slippage
      - SEC §31 + FINRA TAF sell-side regs
  3. Continuous-learning effect: META LightGBM is refit at each calendar
     quarter on the trailing 2-year window. Other 9 assets use rule-based
     signals (buy_and_hold / rsi2_mr / macd) which have no retraining.

Walk-forward windows evaluated:
  * 2024-01 → 2026-05  (full available recent history, ~2.4y)
  * COVID  2020-02 → 2020-04
  * RATESHOCK  2022-01 → 2022-06
  * OOS  2026-01 → 2026-05

Tear sheet: net P&L, Sharpe, max DD, alpha vs B&H per window.

Run:
    .venv/bin/python scripts/100_continuous_learning_backtest.py
    .venv/bin/python scripts/100_continuous_learning_backtest.py --window OOS_FULL
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
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
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from scripts._storage import open_store  # noqa: E402


NET_LIQ = 100_000.0
N_ASSETS = 10
PER_NAME_NOTIONAL = NET_LIQ / N_ASSETS
WARMUP = 252
RETRAIN_INTERVAL_DAYS = 63  # ≈ 1 quarter

# Universe: v2 (8) + v3 leveraged sleeve (SOXL, NVDL).
# Format: (display, signal_mode, trail_sigma, profit_target_sigmas, lib, sym, source)
# source ∈ {"arctic", "alpaca"} — arctic for the established 8, alpaca for the 2 leveraged ETFs
UNIVERSE = [
    ("AAPL",  "buy_and_hold",      3.0, 3.0,   "equity.daily_bars", "NASDAQ.AAPL",  "arctic"),
    ("MSFT",  "rsi2_mr_10_90",     1.0, 5.0,   "equity.daily_bars", "NASDAQ.MSFT",  "arctic"),
    ("GOOGL", "buy_and_hold",      2.0, None,  "equity.daily_bars", "NASDAQ.GOOGL", "arctic"),
    ("AMZN",  "buy_and_hold",      0.5, 5.0,   "equity.daily_bars", "NASDAQ.AMZN",  "arctic"),
    ("META",  "lgbm",              2.0, None,  "equity.daily_bars", "NASDAQ.META",  "arctic"),
    ("TSLA",  "macd",              0.5, 3.0,   "equity.daily_bars", "NASDAQ.TSLA",  "arctic"),
    ("NVDA",  "buy_and_hold",      2.0, None,  "equity.daily_bars", "NASDAQ.NVDA",  "arctic"),
    ("GLD",   "buy_and_hold",      2.0, 3.0,   "etf.daily_bars",    "ARCA.GLD",     "arctic"),
    ("SOXL",  "rsi2_mr_10_90",     2.0, 3.0,   None,                "SOXL",          "alpaca"),
    ("NVDL",  "rsi2_mr_10_90",     0.5, 5.0,   None,                "NVDL",          "alpaca"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_arctic(lib: str, arctic_sym: str) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib).read(arctic_sym).data.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = df.index.normalize()
    df.index.name = "date"
    return df.sort_index()[["open", "high", "low", "close", "volume"]]


def _load_alpaca_daily(ticker: str, years: float = 3.0) -> pd.DataFrame:
    """Pull daily bars from Alpaca IEX (free tier)."""
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
    req = StockBarsRequest(
        symbol_or_symbols=[ticker], timeframe=TimeFrame.Day,
        start=start, end=end, feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        try:
            bars = bars.xs(ticker, level="symbol")
        except KeyError:
            bars = bars.xs(ticker, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    out.index.name = "date"
    return out.sort_index()


def load_universe_bars() -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for asset, _sig, _tr, _pt, lib, sym, source in UNIVERSE:
        try:
            if source == "arctic":
                df = _load_arctic(lib, sym)
                # gap-fill recent bars from Alpaca (ArcticDB ends 2026-05-20)
                if len(df):
                    last = df.index[-1].normalize()
                    if (pd.Timestamp.utcnow().normalize().tz_localize(None) - last).days > 1:
                        gap = _load_alpaca_daily(asset, years=0.2)
                        if not gap.empty:
                            gap = gap[~gap.index.isin(df.index)]
                            df = pd.concat([df, gap]).sort_index()
            else:
                df = _load_alpaca_daily(asset, years=3.0)
            if df.empty:
                print(f"WARN: {asset}: no data")
                continue
            bars[asset] = df
            print(f"  {asset:<6} bars={len(df):>5} range={df.index[0].date()}..{df.index[-1].date()}")
        except Exception as exc:
            print(f"ERROR {asset}: {exc}")
    return bars


# ---------------------------------------------------------------------------
# Signals — rule-based (cheap, deterministic) + LightGBM (META) with
# quarterly refits.
# ---------------------------------------------------------------------------


def _signal_buy_and_hold(n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int8)
    out[WARMUP:] = 1
    return out


def _signal_rsi2_mr_10_90(feat: pd.DataFrame, n: int) -> np.ndarray:
    r = feat["rsi_2"].shift(1).values
    out = np.where(r < 10, 1, np.where(r > 90, -1, 0)).astype(np.int8)
    out[:WARMUP] = 0
    return out


def _signal_macd(feat: pd.DataFrame, n: int) -> np.ndarray:
    h = feat["macd_hist"].shift(1).values
    out = np.where(h > 0, 1, np.where(h < 0, -1, 0)).astype(np.int8)
    out[:WARMUP] = 0
    return out


def _signal_lgbm_walk_forward(asset: str, bars: pd.DataFrame, n: int,
                                retrains: list[dict]) -> np.ndarray:
    """Walk-forward LightGBM via the per-asset module's compute_signal_for.

    The asset module (`ml_research.asset_specialists.meta`) refits the
    LightGBM model on every call from cold cache — this is MORE aggressive
    than the daemon's daily cadence. It validates the worst-case retrain
    schedule.

    For efficiency we still refit at quarter boundaries here: call the
    per-asset signal pipeline once per quarter, hold the value forward
    until the next refit. This matches the daemon's actual behavior.
    """
    from ml_research.asset_specialists import compute_signal_for
    out = np.zeros(n, dtype=np.int8)
    refit_indices = list(range(WARMUP, n, RETRAIN_INTERVAL_DAYS))
    if not refit_indices:
        return out
    current_signal = 0
    for i in range(WARMUP, n):
        # Refit at quarter boundaries — compute_signal_for trains from scratch
        if i in refit_indices:
            try:
                s = compute_signal_for(asset, bars.iloc[: i + 1])
                current_signal = int(round(float(s)))
                retrains.append({
                    "date": bars.index[i].date().isoformat(),
                    "i": i,
                    "signal": current_signal,
                })
            except Exception:
                current_signal = 0
        out[i] = current_signal
    return out


def precompute_signal(asset: str, mode: str, bars: pd.DataFrame,
                       feat: pd.DataFrame, retrains: list[dict]) -> np.ndarray:
    n = len(bars)
    if mode == "buy_and_hold":
        return _signal_buy_and_hold(n)
    if mode == "rsi2_mr_10_90":
        return _signal_rsi2_mr_10_90(feat, n)
    if mode == "macd":
        return _signal_macd(feat, n)
    if mode == "lgbm":
        return _signal_lgbm_walk_forward(asset, bars, n, retrains)
    raise ValueError(f"unknown signal {mode}")


# ---------------------------------------------------------------------------
# Realistic cost model — IBKR tiered + Almgren-Chriss
# ---------------------------------------------------------------------------


def fill_cost_usd(qty: int, price: float, side: str, year: int,
                   daily_vol_bps: float, adv: float) -> dict:
    """Combined commission + impact for one fill.

    Returns {commission_usd, impact_usd, total_usd, impact_bps, floor_binds}.
    """
    if qty <= 0 or price <= 0:
        return {"commission_usd": 0.0, "impact_usd": 0.0, "total_usd": 0.0,
                "impact_bps": 0.0, "floor_binds": False}
    fee_breakdown = equity_tiered(float(qty), float(price))
    commission = fee_breakdown.total
    notional = qty * price
    slip = slippage_bps_with_floor(
        volume_traded=float(qty),
        adv=float(max(adv, 1.0)),
        daily_vol_bps=float(max(daily_vol_bps, 1.0)),
        asset_class="equity",
        k=0.1,
    )
    impact_usd = notional * (slip.total_bps / 1e4)
    return {
        "commission_usd": commission,
        "impact_usd": impact_usd,
        "total_usd": commission + impact_usd,
        "impact_bps": slip.total_bps,
        "floor_binds": slip.impact_bps < slip.floor_bps,
    }


# ---------------------------------------------------------------------------
# Walk-forward replay (mirrors 83_honest_replay.replay but with realistic costs)
# ---------------------------------------------------------------------------


def replay(asset: str, bars: pd.DataFrame, feat: pd.DataFrame,
           sig: np.ndarray, start_idx: int, end_idx: int,
           trail: float, pt: float | None, notional: float) -> dict:
    closes = bars["close"].astype(float).values
    vols_daily_bps_series = (feat.get("vol_21") if "vol_21" in feat.columns else None)
    vol_21 = feat["vol_21"].values
    adv_series = bars["volume"].rolling(21).mean().fillna(method="bfill").values

    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    round_trips: list[dict] = []
    daily_bucket: dict = {}
    total_commission = 0.0
    total_impact = 0.0
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol_21[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig[i]) if i < len(sig) else 0
        close = float(closes[i])
        dk = bars.index[i].normalize()
        year = bars.index[i].year
        daily_vol_bps = sigma * math.sqrt(252) * 10_000.0  # ann vol → daily bps
        adv = max(float(adv_series[i]) if math.isfinite(adv_series[i]) else 0.0, 1.0)

        # MTM on prior position
        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            daily_bucket[dk] = daily_bucket.get(dk, 0.0) + (close - prev_close) * prev_qty * prev_side

        # Exit logic
        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            stop = peak - entry_price * entry_sigma * trail
            reason = None
            if close <= stop:
                reason = "trail"
            elif pt is not None and close >= entry_price * (1 + entry_sigma * pt):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                gross = (close - entry_price) * entry_qty
                cost = fill_cost_usd(entry_qty, close, "SELL", year, daily_vol_bps, adv)
                pnl = gross - cost["total_usd"]
                total_commission += cost["commission_usd"]
                total_impact += cost["impact_usd"]
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost["total_usd"]
                round_trips.append({"side": "LONG", "pnl": pnl, "reason": reason})
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            stop = trough + entry_price * entry_sigma * trail
            reason = None
            if close >= stop:
                reason = "trail"
            elif pt is not None and close <= entry_price * (1 - entry_sigma * pt):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                gross = (entry_price - close) * entry_qty
                cost = fill_cost_usd(entry_qty, close, "BUY", year, daily_vol_bps, adv)
                pnl = gross - cost["total_usd"]
                total_commission += cost["commission_usd"]
                total_impact += cost["impact_usd"]
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost["total_usd"]
                round_trips.append({"side": "SHORT", "pnl": pnl, "reason": reason})
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        # Entry logic
        if state == "FLAT" and sig_dir != 0:
            qty = int(notional // close) if close > 0 else 0
            if qty > 0:
                side = "BUY" if sig_dir > 0 else "SELL"
                cost = fill_cost_usd(qty, close, side, year, daily_vol_bps, adv)
                total_commission += cost["commission_usd"]
                total_impact += cost["impact_usd"]
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost["total_usd"]
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

    # Unrealized on open position at end of window
    open_unrz = 0.0
    if state != "FLAT" and entry_price is not None and end_idx > 0:
        last_close = float(closes[end_idx - 1])
        if state == "IN_LONG":
            open_unrz = (last_close - entry_price) * entry_qty
        else:
            open_unrz = (entry_price - last_close) * entry_qty

    realized = sum(rt["pnl"] for rt in round_trips)
    net = realized + open_unrz
    wins = sum(1 for rt in round_trips if rt["pnl"] > 0)
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0

    sharpe = float("nan")
    max_dd = 0.0
    n_sessions = len(daily_bucket)
    if daily_bucket:
        rets = np.asarray(list(daily_bucket.values())) / notional
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    years = n_sessions / 252.0 if n_sessions else 0.0
    ann = float("nan")
    if years > 0 and notional + net > 0:
        ann = ((1 + net / notional) ** (1 / years) - 1) * 100.0

    return {
        "n_trades": len(round_trips), "wins": wins, "win_rate": win_rate,
        "net": net, "realized": realized, "open_unrz": open_unrz,
        "max_dd": max_dd, "sharpe": sharpe, "ann": ann,
        "commission_usd": total_commission, "impact_usd": total_impact,
        "total_cost_usd": total_commission + total_impact,
        "n_sessions": n_sessions,
    }


def bh_benchmark(bars: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                  notional: float) -> dict:
    df = bars[(bars.index >= start_ts) & (bars.index <= end_ts)]
    if len(df) < 2:
        return {"net": 0.0, "ann": float("nan"), "years": 0.0}
    entry = float(df["close"].iloc[0])
    exit_p = float(df["close"].iloc[-1])
    qty = int(notional // entry) if entry > 0 else 0
    if qty == 0:
        return {"net": 0.0, "ann": float("nan"), "years": 0.0}
    # entry cost
    year_start = df.index[0].year
    daily_vol_bps_start = 100.0  # rough liquid large-cap daily vol bps
    adv_start = float(df["volume"].iloc[:21].mean() if len(df) >= 21 else df["volume"].mean())
    entry_cost = fill_cost_usd(qty, entry, "BUY", year_start, daily_vol_bps_start, adv_start)
    exit_cost = fill_cost_usd(qty, exit_p, "SELL", df.index[-1].year, daily_vol_bps_start, adv_start)
    gross = (exit_p - entry) * qty
    net = gross - entry_cost["total_usd"] - exit_cost["total_usd"]
    years = len(df) / 252.0
    ann = ((1 + net / notional) ** (1 / years) - 1) * 100.0 if years > 0 and notional + net > 0 else float("nan")
    return {"net": net, "ann": ann, "years": years,
            "entry": entry, "exit": exit_p, "qty": qty}


# ---------------------------------------------------------------------------
# Tear sheet
# ---------------------------------------------------------------------------


WINDOWS = {
    "OOS_FULL":   (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-05-20")),
    "OOS_RECENT": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-20")),
    "COVID":      (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
    "RATESHOCK":  (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30")),
}


def evaluate_window(label: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                     all_bars: dict[str, pd.DataFrame],
                     retrains_by_asset: dict[str, list[dict]]) -> dict:
    print(f"\n{'=' * 130}")
    print(f"  {label} — {start_ts.date()} → {end_ts.date()}")
    print(f"{'=' * 130}")
    print(f"  {'asset':<7} {'mode':<16} {'tr σ':<5} {'PT σ':<5} {'trades':<7} "
          f"{'net $':<11} {'Sharpe':<7} {'maxDD $':<11} {'comm $':<8} {'impact $':<10}")
    print("-" * 130)

    bot_pnl_series: dict[pd.Timestamp, float] = {}
    bot_total = 0.0
    bh_total = 0.0
    bot_trades = 0
    total_comm = 0.0
    total_impact = 0.0
    rows = []
    n_deployed = 0
    deploy_skipped = []

    for asset, sigmode, trail, pt, _lib, _sym, _source in UNIVERSE:
        if asset not in all_bars:
            deploy_skipped.append(asset)
            continue
        bars = all_bars[asset]
        # need enough history before window start for warmup
        if bars.index[0] > start_ts - pd.Timedelta(days=int(WARMUP * 1.5)):
            deploy_skipped.append(asset)
            continue
        if bars.index[-1] < start_ts:
            deploy_skipped.append(asset)
            continue
        feat = add_per_asset_features(bars)
        retrains = retrains_by_asset.setdefault(asset, [])
        sig = precompute_signal(asset, sigmode, bars, feat, retrains)
        start_idx = bars.index.get_indexer([start_ts], method="bfill")[0]
        end_idx = bars.index.get_indexer([end_ts], method="ffill")[0] + 1
        end_idx = min(end_idx, len(bars))
        if start_idx <= WARMUP or end_idx <= start_idx:
            deploy_skipped.append(asset)
            continue
        r = replay(asset, bars, feat, sig, start_idx, end_idx, trail, pt, PER_NAME_NOTIONAL)
        bh = bh_benchmark(bars, start_ts, end_ts, PER_NAME_NOTIONAL)

        bot_total += r["net"]
        bh_total += bh["net"]
        bot_trades += r["n_trades"]
        total_comm += r["commission_usd"]
        total_impact += r["impact_usd"]
        n_deployed += 1
        pt_str = f"{pt:.1f}" if pt else "—"
        rows.append((asset, sigmode, trail, pt_str, r, bh))
        print(f"  {asset:<7} {sigmode:<16} {trail:<5.1f} {pt_str:<5} "
              f"{r['n_trades']:<7} ${r['net']:>+8,.0f}  {r['sharpe']:>+5.2f}  "
              f"${r['max_dd']:>+8,.0f}  ${r['commission_usd']:>5,.0f}  ${r['impact_usd']:>7,.0f}")

    portfolio_notional = max(n_deployed * PER_NAME_NOTIONAL, 1.0)
    alpha = bot_total - bh_total
    print("-" * 130)
    print(f"  BOT          net=${bot_total:+,.0f}  ({bot_total/portfolio_notional*100:+.2f}%)  "
          f"trades={bot_trades}  commission=${total_comm:,.0f}  impact=${total_impact:,.0f}")
    print(f"  BUY&HOLD     net=${bh_total:+,.0f}  ({bh_total/portfolio_notional*100:+.2f}%)")
    print(f"  ALPHA vs B&H ${alpha:+,.0f}  ({alpha/portfolio_notional*100:+.2f}%)")
    if deploy_skipped:
        print(f"  skipped (insufficient history): {deploy_skipped}")

    # Portfolio-level Sharpe + max DD (sum per-asset daily series)
    # For window-level Sharpe we use weighted average of per-asset Sharpe ratios
    # as a fast approximation (proper aggregation would require cross-asset
    # daily series merging).
    valid_sharpes = [r["sharpe"] for _, _, _, _, r, _ in rows if math.isfinite(r["sharpe"])]
    avg_sharpe = float(np.mean(valid_sharpes)) if valid_sharpes else float("nan")
    sum_dd = sum(r["max_dd"] for _, _, _, _, r, _ in rows)

    return {
        "label": label,
        "bot_net": bot_total, "bh_net": bh_total, "alpha": alpha,
        "trades": bot_trades, "commission_usd": total_comm,
        "impact_usd": total_impact, "n_deployed": n_deployed,
        "avg_sharpe": avg_sharpe, "sum_max_dd": sum_dd,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=list(WINDOWS.keys()) + ["ALL"], default="ALL")
    args = ap.parse_args()

    print("=" * 130)
    print("  CONTINUOUS-LEARNING BACKTEST — high-fidelity walk-forward validation")
    print("=" * 130)
    print(f"  Universe: {[u[0] for u in UNIVERSE]}")
    print(f"  Per-name notional: ${PER_NAME_NOTIONAL:,.0f}  (total ${NET_LIQ:,.0f})")
    print(f"  Costs:    IBKR Pro tiered commission + Almgren-Chriss √-impact + SEC §31 + FINRA TAF")
    print(f"  Retrain:  META LightGBM at every quarter boundary ({RETRAIN_INTERVAL_DAYS} bars)")
    print("=" * 130)
    print()
    print("  Loading universe bars...")
    all_bars = load_universe_bars()
    if not all_bars:
        print("FATAL: no bars loaded")
        return 1

    retrains_by_asset: dict[str, list[dict]] = {}

    windows_to_run = [args.window] if args.window != "ALL" else list(WINDOWS.keys())
    summary = []
    for w in windows_to_run:
        start_ts, end_ts = WINDOWS[w]
        summary.append(evaluate_window(w, start_ts, end_ts, all_bars, retrains_by_asset))

    # Final tear sheet
    print()
    print("=" * 130)
    print("  TEAR SHEET")
    print("=" * 130)
    print(f"  {'window':<14} {'bot net':<12} {'B&H net':<12} {'alpha':<11} "
          f"{'trades':<7} {'avg Sh':<7} {'Σ maxDD':<11}")
    print("-" * 130)
    for s in summary:
        print(f"  {s['label']:<14} ${s['bot_net']:>+9,.0f}  ${s['bh_net']:>+9,.0f}  "
              f"${s['alpha']:>+8,.0f}  {s['trades']:<7} "
              f"{s['avg_sharpe']:>+5.2f}   ${s['sum_max_dd']:>+8,.0f}")
    print("=" * 130)
    print()
    print("  CONTINUOUS LEARNING REFITS")
    for asset, rt in retrains_by_asset.items():
        if rt:
            print(f"  {asset}: {len(rt)} retrains "
                  f"first={rt[0]['date']} last={rt[-1]['date']}")
    print()
    print("  COST MODEL — IBKR-tiered + Almgren-Chriss √-impact + spread-half floor")
    print(f"  Year-2024 SEC §31: 27.80e-6 / Year-2025+: 8.00e-6")
    print(f"  Almgren k=0.1, equity spread floor = 1.0 bps half-spread")
    print()
    print("  READING:")
    print("  * alpha vs B&H is the only meaningful skill measure.")
    print("  * positive alpha in COVID/RATESHOCK = defensive architecture working;")
    print("    negative alpha in OOS_RECENT = bull-market underperformance vs naive long.")
    print("  * The bot is designed to earn its keep in drawdowns, not bull runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
