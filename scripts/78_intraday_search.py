"""78_intraday_search — the comprehensive bake-off on 5-minute bars.

Mirrors scripts/77_comprehensive_search.py exactly (same 6 signals × 4 trails
× 3 exits = 72 cells per asset) but consumes the 5-min substrate loaded by
scripts/10_alpaca_intraday_backfill.py.

Honest comparison: does 78× more decision points per asset per day actually
deliver more OOS P&L, or just more whipsaw?

Bar timeframe: 5 minutes. ~133 bars/day (regular + extended hours).
Train/test split: 2026-01-01 — gives ~4.5 months of OOS test (Jan-May 2026).
Warmup: 500 5-min bars (~6 trading days) for indicator stability.

Run:
    .venv/bin/python scripts/78_intraday_search.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from scripts._storage import open_store  # noqa: E402

WARMUP = 500   # ~6 trading days of 5-min bars

INTRADAY_ASSETS = {
    "AAPL":  ("equity.minute_bars", "NASDAQ.AAPL"),
    "MSFT":  ("equity.minute_bars", "NASDAQ.MSFT"),
    "GOOGL": ("equity.minute_bars", "NASDAQ.GOOGL"),
    "AMZN":  ("equity.minute_bars", "NASDAQ.AMZN"),
    "META":  ("equity.minute_bars", "NASDAQ.META"),
    "TSLA":  ("equity.minute_bars", "NASDAQ.TSLA"),
    "NVDA":  ("equity.minute_bars", "NASDAQ.NVDA"),
    "GLD":   ("etf.minute_bars",    "ARCA.GLD"),
}

SIGNAL_TYPES = ["buy_and_hold", "macd", "rsi2", "bollinger", "donchian", "mom21"]
TRAIL_SIGMAS = [0.5, 1.0, 2.0, 3.0]
PROFIT_TARGETS = [None, 3.0, 5.0]


def load_intraday_bars(lib_name: str, sym: str) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib_name).read(sym).data.copy()
    # Drop tz to align with daily-bar feature pipeline expectations.
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def precompute_signal(bars: pd.DataFrame, feat: pd.DataFrame,
                      mode: str) -> np.ndarray:
    """Same logic as the daily comprehensive search — re-applied on 5-min bars.

    All windows are interpreted in BARS, not days. So vol_21 on 5-min bars =
    vol over last 21 5-min bars (~1.75 hours). This is the naive translation;
    if the bake-off proves intraday useful, the next iteration recalibrates
    window sizes to session-equivalent scales.
    """
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "buy_and_hold":
        out[WARMUP:] = 1
        return out
    if mode == "macd":
        h = feat["macd_hist"].shift(1)
        out = np.where(h > 0, 1, np.where(h < 0, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "rsi2":
        r = feat["rsi_2"].shift(1)
        out = np.where(r < 10, 1, np.where(r > 90, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "bollinger":
        close = bars["close"]
        ma20 = close.rolling(20).mean().shift(1)
        sd20 = close.rolling(20).std().shift(1)
        prev_close = close.shift(1)
        z = (prev_close - ma20) / sd20
        out = np.where(z < -2, 1, np.where(z > 2, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "donchian":
        high20 = bars["high"].rolling(20).max().shift(1)
        low20 = bars["low"].rolling(20).min().shift(1)
        prev_close = bars["close"].shift(1)
        out = np.where(prev_close >= high20, 1,
                       np.where(prev_close <= low20, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "mom21":
        m = feat["ret_21"].shift(1)
        # Intraday momentum threshold — much smaller than daily because
        # 21-5min-bar return is tiny vs 21-day return.
        out = np.where(m > 0.003, 1, np.where(m < -0.003, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    raise ValueError(f"unknown signal mode {mode!r}")


def replay(bars: pd.DataFrame, feat: pd.DataFrame, sig_array: np.ndarray,
           start_idx: int, end_idx: int, notional: float,
           trail: float, profit_target_sigmas: float | None) -> dict:
    if start_idx < WARMUP:
        start_idx = WARMUP
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    round_trips: list[dict] = []
    # Daily MTM on intraday is too granular; aggregate by session date.
    daily_pnl_by_date: dict[pd.Timestamp, float] = {}
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0

    vol_series = feat["vol_21"].values

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol_series[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig_array[i])
        close = float(closes[i])
        date_key = bars.index[i].normalize()

        # Intra-bar MTM contribution to the session bucket.
        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            pnl_step = (close - prev_close) * prev_qty * prev_side
            daily_pnl_by_date[date_key] = daily_pnl_by_date.get(date_key, 0.0) + pnl_step

        # Exit checks first.
        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop = peak - band
            reason = None
            if close <= stop:
                reason = "trail"
            elif (profit_target_sigmas is not None and
                  close >= entry_price * (1 + entry_sigma * profit_target_sigmas)):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                round_trips.append({"side": "LONG",
                                    "pnl": (close - entry_price) * entry_qty})
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
            elif (profit_target_sigmas is not None and
                  close <= entry_price * (1 - entry_sigma * profit_target_sigmas)):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                round_trips.append({"side": "SHORT",
                                    "pnl": (entry_price - close) * entry_qty})
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        # New entry.
        if state == "FLAT" and sig_dir != 0:
            qty = int(notional // close) if close > 0 else 0
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

    open_unrz = 0.0
    if state != "FLAT" and entry_price is not None:
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
    if daily_pnl_by_date and notional > 0:
        # Aggregate intra-bar MTM into per-session daily returns.
        daily_arr = np.asarray(list(daily_pnl_by_date.values())) / notional
        if daily_arr.std() > 0 and len(daily_arr) > 1:
            sharpe = float(daily_arr.mean() / daily_arr.std() * math.sqrt(252))
        cum = np.cumsum(list(daily_pnl_by_date.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())

    sessions = len(daily_pnl_by_date)
    years = sessions / 252.0 if sessions else 0.0
    ann_ret = float("nan")
    if years > 0 and notional + net > 0:
        ann_ret = ((1 + net / notional) ** (1 / years) - 1) * 100.0

    return {
        "n_trades": len(round_trips), "wins": wins,
        "n_long": sum(1 for r in round_trips if r["side"] == "LONG"),
        "n_short": sum(1 for r in round_trips if r["side"] == "SHORT"),
        "win_rate": win_rate, "realized": realized, "open_unrz": open_unrz,
        "net": net, "max_dd": max_dd, "sharpe": sharpe,
        "ann_return_pct": ann_ret, "sessions": sessions,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2026-01-01")
    ap.add_argument("--net-liq", type=float, default=100000.0)
    args = ap.parse_args()

    split_ts = pd.Timestamp(args.split_date)
    notional = args.net_liq / 7.0

    print()
    print("=" * 130)
    print(f"  INTRADAY (5-min) comprehensive search  •  "
          f"{len(INTRADAY_ASSETS)} assets × {len(SIGNAL_TYPES)} signals × "
          f"{len(TRAIL_SIGMAS)} trails × {len(PROFIT_TARGETS)} exits = "
          f"{len(INTRADAY_ASSETS)*len(SIGNAL_TYPES)*len(TRAIL_SIGMAS)*len(PROFIT_TARGETS)} cells")
    print(f"  Split = {split_ts.date()}  •  per-name notional = ${notional:,.0f}")
    print("=" * 130)

    all_results = []
    for asset, (lib_name, sym) in INTRADAY_ASSETS.items():
        try:
            bars = load_intraday_bars(lib_name, sym)
        except Exception as exc:
            print(f"  {asset}: load failed → {type(exc).__name__}: {exc}")
            continue
        feat = add_per_asset_features(bars)
        split_idx = bars.index.get_indexer([split_ts], method="bfill")[0]
        if split_idx <= WARMUP or split_idx >= len(bars):
            print(f"  {asset}: split out of range")
            continue

        sig_cache = {m: precompute_signal(bars, feat, m) for m in SIGNAL_TYPES}

        cells = []
        for sigm in SIGNAL_TYPES:
            for tr in TRAIL_SIGMAS:
                for pt in PROFIT_TARGETS:
                    train_s = replay(bars, feat, sig_cache[sigm], WARMUP, split_idx,
                                     notional, tr, pt)
                    test_s = replay(bars, feat, sig_cache[sigm], split_idx, len(bars),
                                    notional, tr, pt)
                    cells.append({"signal": sigm, "trail": tr, "pt": pt,
                                  "train": train_s, "test": test_s})

        viable = [c for c in cells
                  if c["train"]["net"] > 0 and c["train"]["sharpe"] > 0]
        best = (max(viable, key=lambda c: c["train"]["net"] / (abs(c["train"]["max_dd"]) + 1.0))
                if viable else None)
        ceiling = max(cells, key=lambda c: c["test"]["net"])

        all_results.append({"asset": asset, "best_train": best,
                            "ceiling_test": ceiling, "n_cells": len(cells)})

        if best:
            pt_str = f"{best['pt']}σ" if best["pt"] else "—"
            print(f"  {asset:<6} train-best: signal={best['signal']:<13} "
                  f"trail={best['trail']}σ pt={pt_str:<4} | "
                  f"TRAIN trades={best['train']['n_trades']:>4} "
                  f"net=${best['train']['net']:>8,.0f} Sh={best['train']['sharpe']:+.2f} | "
                  f"TEST trades={best['test']['n_trades']:>4} "
                  f"net=${best['test']['net']:>8,.0f} Sh={best['test']['sharpe']:+.2f}")
        else:
            print(f"  {asset:<6} NO VIABLE config")

    deployable = [r for r in all_results if r["best_train"] is not None]
    portfolio_test_net = sum(r["best_train"]["test"]["net"] for r in deployable)
    portfolio_test_trades = sum(r["best_train"]["test"]["n_trades"] for r in deployable)
    portfolio_train_net = sum(r["best_train"]["train"]["net"] for r in deployable)
    portfolio_notional = len(deployable) * notional
    ceiling_net = sum(r["ceiling_test"]["test"]["net"] for r in all_results)
    ceiling_notional = len(all_results) * notional

    test_years = (all_results[0]["best_train"]["test"]["sessions"] / 252.0
                  if deployable else 0.0)
    test_ann = (((1 + portfolio_test_net / portfolio_notional) ** (1 / test_years) - 1) * 100.0
                if portfolio_notional > 0 and test_years > 0
                and portfolio_notional + portfolio_test_net > 0 else float("nan"))

    print()
    print("=" * 130)
    print("  INTRADAY portfolio (equal-weight across deployable assets)")
    print("=" * 130)
    print(f"  deployable: {len(deployable)}/{len(all_results)}  "
          f"({sorted(r['asset'] for r in deployable)})")
    print()
    print(f"  TRAIN net (picked configs):  ${portfolio_train_net:>+12,.0f}")
    print(f"  TEST  net (same configs):    ${portfolio_test_net:>+12,.0f}  "
          f"({portfolio_test_net/portfolio_notional*100:+.2f}% on "
          f"${portfolio_notional:,.0f} over {test_years:.2f}y → "
          f"{test_ann:+.1f}% annualized)")
    print(f"  TEST  trades total:          {portfolio_test_trades:>5}  "
          f"(avg {portfolio_test_trades/max(1,len(deployable)):.0f}/asset)")
    print(f"  TEST  ceiling (look-ahead):  ${ceiling_net:>+12,.0f}  "
          f"({ceiling_net/ceiling_notional*100:+.2f}% on ${ceiling_notional:,.0f})")
    print()
    print(f"  Reference — DAILY baseline 7-asset bot: ~+$79,634 (+80%) over 2.4y "
          f"≈ +28% annualized")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
