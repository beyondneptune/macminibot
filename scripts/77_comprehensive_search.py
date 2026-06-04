"""77_comprehensive_search — exhaustive empirical sweep across:
   • 17 candidate assets (7 equities + 10 ETFs already in ArcticDB)
   • 6 candidate signal types (buy_and_hold, MACD, RSI-2, Bollinger, Donchian,
     momentum-21)
   • 4 trail-σ multipliers (0.5×, 1.0×, 2.0×, 3.0×)
   • 3 exit policies (trailing-only, trail+PT-3σ, trail+PT-5σ)

Per asset: 6×4×3 = 72 candidates. Each candidate replayed on TRAIN window,
picked by risk-adjusted train metric (net / |maxDD|+1), then evaluated on
TEST. Final portfolio = equal-weight across the assets where the picked
config produces positive train net AND positive train sharpe (the
deployment gate).

Run:
    .venv/bin/python scripts/77_comprehensive_search.py
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

WARMUP = 252

CANDIDATE_ASSETS = {
    # ticker → (arctic_lib, arctic_symbol)
    "AAPL":  ("equity.daily_bars", "NASDAQ.AAPL"),
    "MSFT":  ("equity.daily_bars", "NASDAQ.MSFT"),
    "GOOGL": ("equity.daily_bars", "NASDAQ.GOOGL"),
    "AMZN":  ("equity.daily_bars", "NASDAQ.AMZN"),
    "META":  ("equity.daily_bars", "NASDAQ.META"),
    "TSLA":  ("equity.daily_bars", "NASDAQ.TSLA"),
    "NVDA":  ("equity.daily_bars", "NASDAQ.NVDA"),
    "BRK_B": ("equity.daily_bars", "NYSE.BRK B"),
    "JPM":   ("equity.daily_bars", "NYSE.JPM"),
    "V":     ("equity.daily_bars", "NYSE.V"),
    "SPY":   ("etf.daily_bars",    "ARCA.SPY"),
    "QQQ":   ("etf.daily_bars",    "NASDAQ.QQQ"),
    "IWM":   ("etf.daily_bars",    "ARCA.IWM"),
    "XLE":   ("etf.daily_bars",    "ARCA.XLE"),
    "GLD":   ("etf.daily_bars",    "ARCA.GLD"),
    "SLV":   ("etf.daily_bars",    "ARCA.SLV"),
    "GDX":   ("etf.daily_bars",    "ARCA.GDX"),
}

SIGNAL_TYPES = ["buy_and_hold", "macd", "rsi2", "bollinger", "donchian", "mom21"]
TRAIL_SIGMAS = [0.5, 1.0, 2.0, 3.0]
PROFIT_TARGETS = [None, 3.0, 5.0]   # in entry-σ units


def load_bars(lib_name: str, sym: str) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib_name).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def precompute_signal(bars: pd.DataFrame, feat: pd.DataFrame,
                      mode: str) -> np.ndarray:
    """Return sig_dir per bar as int8 array. ALL signals are CAUSAL — they use
    .shift(1) so signal at bar i depends only on bars[<i]."""
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
        # 20d Bollinger band breakout (mean reversion variant)
        close = bars["close"]
        ma20 = close.rolling(20).mean().shift(1)
        sd20 = close.rolling(20).std().shift(1)
        prev_close = close.shift(1)
        z = (prev_close - ma20) / sd20
        out = np.where(z < -2, 1, np.where(z > 2, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "donchian":
        # 20d Donchian channel breakout (trend-following)
        high20 = bars["high"].rolling(20).max().shift(1)
        low20 = bars["low"].rolling(20).min().shift(1)
        prev_close = bars["close"].shift(1)
        out = np.where(prev_close >= high20, 1,
                       np.where(prev_close <= low20, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "mom21":
        # 21-day momentum (lagged by 1)
        m = feat["ret_21"].shift(1)
        # Threshold so it doesn't always emit — fire only on strong moves
        out = np.where(m > 0.05, 1, np.where(m < -0.05, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    raise ValueError(f"unknown signal mode {mode!r}")


def replay(bars: pd.DataFrame, feat: pd.DataFrame, sig_array: np.ndarray,
           start_idx: int, end_idx: int, notional: float,
           trail: float, profit_target_sigmas: float | None) -> dict:
    if start_idx < WARMUP:
        start_idx = WARMUP
    closes = bars["close"].astype(float).values
    idxs = bars.index
    state = "FLAT"
    entry_price = entry_sigma = entry_ts = None
    entry_qty = 0
    peak = trough = None
    round_trips: list[dict] = []
    daily_pnl: list[float] = []
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(feat["vol_21"].iloc[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig_array[i])
        close = float(closes[i])

        # Daily MTM on the prior position before any action this bar.
        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            daily_pnl.append((close - prev_close) * prev_qty * prev_side)
        else:
            daily_pnl.append(0.0)

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
                entry_ts, entry_qty = None, 0
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
                entry_ts, entry_qty = None, 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

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
    losses = sum(1 for rt in round_trips if rt["pnl"] <= 0)
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0

    sharpe = float("nan")
    max_dd = 0.0
    if daily_pnl and notional > 0:
        rets = np.asarray(daily_pnl) / notional
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(daily_pnl)
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())

    n_bars = end_idx - start_idx
    years = n_bars / 252.0 if n_bars else 0.0
    ann_ret = float("nan")
    if years > 0 and notional + net > 0:
        ann_ret = ((1 + net / notional) ** (1 / years) - 1) * 100.0

    return {
        "n_trades": len(round_trips), "wins": wins, "losses": losses,
        "n_long": sum(1 for r in round_trips if r["side"] == "LONG"),
        "n_short": sum(1 for r in round_trips if r["side"] == "SHORT"),
        "win_rate": win_rate, "realized": realized, "open_unrz": open_unrz,
        "net": net, "max_dd": max_dd, "sharpe": sharpe,
        "ann_return_pct": ann_ret,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2024-01-01")
    ap.add_argument("--net-liq", type=float, default=100000.0)
    args = ap.parse_args()

    split_ts = pd.Timestamp(args.split_date)

    print()
    print("=" * 130)
    print(f"  Comprehensive search  •  {len(CANDIDATE_ASSETS)} assets × "
          f"{len(SIGNAL_TYPES)} signals × {len(TRAIL_SIGMAS)} trails × "
          f"{len(PROFIT_TARGETS)} exits = "
          f"{len(CANDIDATE_ASSETS)*len(SIGNAL_TYPES)*len(TRAIL_SIGMAS)*len(PROFIT_TARGETS)} cells")
    print(f"  Split = {split_ts.date()}  •  Test: bars after split → last bar")
    print("=" * 130)

    all_results = []

    for asset, (lib_name, sym) in CANDIDATE_ASSETS.items():
        try:
            bars = load_bars(lib_name, sym)
        except Exception as exc:
            print(f"{asset}: load failed → {type(exc).__name__}")
            continue
        feat = add_per_asset_features(bars)
        split_idx = bars.index.get_indexer([split_ts], method="bfill")[0]
        if split_idx <= WARMUP or split_idx >= len(bars):
            print(f"{asset}: split out of range — skip")
            continue

        # We split the per-asset $14,285 notional across the candidate pool
        # for sizing-comparability across assets. The "deployable" pool size
        # is determined after train-best picks; we use the original
        # notional/7 placeholder so per-asset values are comparable.
        per_asset_notional = args.net_liq / 7.0

        # Precompute signals once per (asset, signal).
        sig_cache = {m: precompute_signal(bars, feat, m) for m in SIGNAL_TYPES}

        # Replay every cell on TRAIN.
        cells = []
        for sigm in SIGNAL_TYPES:
            for tr in TRAIL_SIGMAS:
                for pt in PROFIT_TARGETS:
                    train_s = replay(bars, feat, sig_cache[sigm], WARMUP, split_idx,
                                     per_asset_notional, tr, pt)
                    test_s = replay(bars, feat, sig_cache[sigm], split_idx, len(bars),
                                    per_asset_notional, tr, pt)
                    cells.append({"signal": sigm, "trail": tr, "pt": pt,
                                  "train": train_s, "test": test_s})

        # Pick train-best (risk-adjusted): net / (|maxDD|+1), gated by net>0 + sharpe>0.
        viable = [c for c in cells
                  if c["train"]["net"] > 0 and c["train"]["sharpe"] > 0]
        if not viable:
            best = None
        else:
            best = max(viable,
                       key=lambda c: c["train"]["net"] / (abs(c["train"]["max_dd"]) + 1.0))
        # For reporting, also identify the test ceiling (look-ahead).
        ceiling = max(cells, key=lambda c: c["test"]["net"])

        all_results.append({
            "asset": asset, "best_train": best, "ceiling_test": ceiling,
            "n_cells": len(cells), "any_viable": bool(viable),
        })

        if best:
            pt_str = f"{best['pt']}σ" if best["pt"] else "—"
            print(f"  {asset:<6} train-best: signal={best['signal']:<13} "
                  f"trail={best['trail']}σ pt={pt_str:<4} | "
                  f"TRAIN net=${best['train']['net']:>9,.0f} "
                  f"Sh={best['train']['sharpe']:+.2f} | "
                  f"TEST net=${best['test']['net']:>9,.0f} "
                  f"Sh={best['test']['sharpe']:+.2f} "
                  f"L/S={best['test']['n_long']}/{best['test']['n_short']}")
        else:
            print(f"  {asset:<6} no viable config (every train cell either net≤0 or sharpe≤0)")

    # Portfolio summary.
    print()
    print("=" * 130)
    print("  Portfolio if we deploy ALL assets with viable train picks (equal-weight)")
    print("=" * 130)
    deployable = [r for r in all_results if r["best_train"] is not None]
    rejected = [r for r in all_results if r["best_train"] is None]
    portfolio_test_net = sum(r["best_train"]["test"]["net"] for r in deployable)
    portfolio_train_net = sum(r["best_train"]["train"]["net"] for r in deployable)
    portfolio_notional = len(deployable) * (args.net_liq / 7.0)
    ceiling_total = sum(r["ceiling_test"]["test"]["net"] for r in all_results)
    ceiling_notional = len(all_results) * (args.net_liq / 7.0)

    print(f"  deployable assets:    {len(deployable)} / {len(all_results)}  "
          f"({sorted(r['asset'] for r in deployable)})")
    print(f"  rejected (no viable): {len(rejected)}  ({sorted(r['asset'] for r in rejected)})")
    print()
    print(f"  TRAIN net (train-picked configs):  ${portfolio_train_net:>+12,.0f}")
    print(f"  TEST  net (same configs):          ${portfolio_test_net:>+12,.0f}  "
          f"({portfolio_test_net/portfolio_notional*100:+.2f}% on "
          f"${portfolio_notional:,.0f} deployed)")
    print(f"  TEST  ceiling (look-ahead):        ${ceiling_total:>+12,.0f}  "
          f"({ceiling_total/ceiling_notional*100:+.2f}% on "
          f"${ceiling_notional:,.0f} crystal-ball)")
    print()
    print(f"  Reference: current 7-asset deployment baseline test net ≈ +$79,634 (+79.6% on $100k)")
    print("=" * 130)

    # Per-asset detail table — show the picked config + how it compares to a
    # naive long-only-trail-1σ default to highlight what the search learned.
    print()
    print("  Per-asset detail (TRAIN-best vs TEST-ceiling)")
    print("-" * 130)
    print(f"  {'asset':<6} {'picked sig':<14} {'tr':<5} {'pt':<5} "
          f"{'TEST $':<11} {'TEST Sh':<8} {'L/S':<8} "
          f"|  {'ceil sig':<14} {'tr':<5} {'pt':<5} {'ceil $':<11}")
    print("-" * 130)
    for r in all_results:
        a = r["asset"]
        if r["best_train"]:
            b = r["best_train"]
            pt_b = f"{b['pt']}σ" if b["pt"] else "—"
            ls = f"{b['test']['n_long']}/{b['test']['n_short']}"
            cells_left = (f"{b['signal']:<14} {b['trail']}σ{'':<2} {pt_b:<5} "
                          f"${b['test']['net']:>+8,.0f}  {b['test']['sharpe']:>+6.2f}  "
                          f"{ls:<8}")
        else:
            cells_left = "NO VIABLE"
        c = r["ceiling_test"]
        pt_c = f"{c['pt']}σ" if c["pt"] else "—"
        cells_right = (f"{c['signal']:<14} {c['trail']}σ{'':<2} {pt_c:<5} "
                       f"${c['test']['net']:>+8,.0f}")
        print(f"  {a:<6} {cells_left}  |  {cells_right}")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
