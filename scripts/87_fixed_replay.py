"""87_fixed_replay — empirical test of "what if Fix A + Fix B had been live?"

Fix A: Widen AMZN/TSLA trail σ from 0.5 → 2.0 (per agent 2 + agent 3 findings).
       Applied to both daily and intraday exit logic.
Fix B: Profit-lock — don't trail-stop until position is +1.0σ in profit.
       Eliminates entry-day whipsaw.

Compares against current v2 baseline on the same 3 windows + same costs.

Run:
    .venv/bin/python scripts/87_fixed_replay.py
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
WARMUP = 252
COST_BP = 3.0

# Two config variants — same signals/PT, only trail differs for AMZN/TSLA
CURRENT_V2 = [
    ("AAPL",  "buy_and_hold",  3.0, 3.0,  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT",  "baseline",      1.0, 5.0,  "equity.daily_bars",  "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0,  "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      0.5, 3.0,  "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  "etf.daily_bars",     "ARCA.GLD"),
]

# Fix A: AMZN/TSLA widened to 2.0σ
FIXED_V3 = [
    ("AAPL",  "buy_and_hold",  3.0, 3.0,  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT",  "baseline",      1.0, 5.0,  "equity.daily_bars",  "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  2.0, 5.0,  "equity.daily_bars",  "NASDAQ.AMZN"),  # 0.5→2.0
    ("META",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      2.0, 3.0,  "equity.daily_bars",  "NASDAQ.TSLA"),  # 0.5→2.0
    ("NVDA",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  "etf.daily_bars",     "ARCA.GLD"),
]

WINDOWS = [
    ("OOS",      pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-20")),
    ("COVID",    pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
    ("RATESHOCK", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30")),
]


def load_bars(lib, sym):
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()[["open", "high", "low", "close", "volume"]]


def precompute_signal(asset, bars, feat, mode):
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "baseline":
        for i in range(WARMUP, n):
            try:
                out[i] = int(round(float(compute_signal_for(asset, bars.iloc[: i + 1]))))
            except Exception:
                pass
        return out
    if mode == "buy_and_hold":
        out[WARMUP:] = 1
        return out
    raise ValueError(mode)


def replay(bars, feat, sig, start_idx, end_idx, trail, pt, cost_bp,
           profit_lock_enabled=False, profit_lock_sigmas=1.0):
    """Replay with optional profit-lock (don't trail until +1σ in profit)."""
    cost_frac = cost_bp / 1e4
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    entry_side = 0
    peak = trough = None
    rt = []
    daily_bucket = {}
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
            daily_bucket[dk] = daily_bucket.get(dk, 0.0) + step

        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop = peak - band
            reason = None
            # Fix B: profit-lock — only trail when position has reached +profit_lock_sigmas
            profit_now_sigmas = (close - entry_price) / (entry_price * entry_sigma)
            trail_active = (not profit_lock_enabled) or (profit_now_sigmas >= profit_lock_sigmas)
            if trail_active and close <= stop:
                reason = "trail"
            elif pt is not None and close >= entry_price * (1 + entry_sigma * pt):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                gross = (close - entry_price) * entry_qty
                cost = entry_qty * close * cost_frac
                rt.append({"pnl": gross - cost, "reason": reason})
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty, entry_side = 0, 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            band = entry_price * entry_sigma * trail
            stop = trough + band
            profit_now_sigmas = (entry_price - close) / (entry_price * entry_sigma)
            trail_active = (not profit_lock_enabled) or (profit_now_sigmas >= profit_lock_sigmas)
            reason = None
            if trail_active and close >= stop:
                reason = "trail"
            elif pt is not None and close <= entry_price * (1 - entry_sigma * pt):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                gross = (entry_price - close) * entry_qty
                cost = entry_qty * close * cost_frac
                rt.append({"pnl": gross - cost, "reason": reason})
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty, entry_side = 0, 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        if state == "FLAT" and sig_dir != 0:
            qty = int(NOTIONAL // close) if close > 0 else 0
            if qty > 0:
                cost = qty * close * cost_frac
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price, entry_sigma = close, sigma
                entry_qty = qty
                entry_side = 1 if sig_dir > 0 else -1
                if sig_dir > 0:
                    peak, trough = close, None
                else:
                    trough, peak = close, None

        prev_close = close
        prev_state = state
        prev_qty = entry_qty
        prev_side = entry_side

    open_unrz = 0.0
    if state != "FLAT" and entry_price is not None:
        last_close = float(closes[end_idx - 1])
        open_unrz = (last_close - entry_price) * entry_qty * entry_side

    realized = sum(r["pnl"] for r in rt)
    net = realized + open_unrz
    wins = sum(1 for r in rt if r["pnl"] > 0)
    win_rate = (wins / len(rt) * 100.0) if rt else 0.0
    sharpe = float("nan")
    max_dd = 0.0
    if daily_bucket:
        rets = np.asarray(list(daily_bucket.values())) / NOTIONAL
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    n_sessions = len(daily_bucket)
    years = n_sessions / 252.0 if n_sessions else 0.0
    ann = float("nan")
    if years > 0 and NOTIONAL + net > 0:
        ann = ((1 + net / NOTIONAL) ** (1 / years) - 1) * 100.0
    return {"n_trades": len(rt), "wins": wins, "win_rate": win_rate,
            "net": net, "sharpe": sharpe, "max_dd": max_dd, "ann": ann}


def run_portfolio(configs, profit_lock_enabled):
    """Returns dict {window_label: portfolio_net}."""
    results = {}
    for label, start, end in WINDOWS:
        portfolio_net = 0.0
        portfolio_trades = 0
        portfolio_wins = 0
        for asset, mode, trail, pt, lib, sym in configs:
            bars = load_bars(lib, sym)
            feat = add_per_asset_features(bars)
            sig = precompute_signal(asset, bars, feat, mode)
            s = bars.index.get_indexer([start], method="bfill")[0]
            e = bars.index.get_indexer([end], method="ffill")[0] + 1
            if s <= WARMUP or e <= s:
                continue
            r = replay(bars, feat, sig, s, e, trail, pt, COST_BP,
                       profit_lock_enabled=profit_lock_enabled)
            portfolio_net += r["net"]
            portfolio_trades += r["n_trades"]
            portfolio_wins += r["wins"]
        wr = (portfolio_wins / portfolio_trades * 100.0) if portfolio_trades else 0.0
        ret = portfolio_net / NET_LIQ * 100.0
        # Annualize the OOS window (which is ~4.5 mo).
        years = (end - start).days / 365.25
        ann = float("nan")
        if years > 0 and NET_LIQ + portfolio_net > 0:
            ann = ((1 + portfolio_net / NET_LIQ) ** (1 / years) - 1) * 100.0
        results[label] = {"net": portfolio_net, "trades": portfolio_trades,
                          "wr": wr, "ret": ret, "ann": ann}
    return results


def main():
    print()
    print("=" * 130)
    print("  Fixed-replay test — Fix A (AMZN/TSLA 0.5σ → 2σ) + Fix B (profit-lock at +1σ)")
    print("=" * 130)

    print("\n  Running CURRENT v2 (control)...")
    v2 = run_portfolio(CURRENT_V2, profit_lock_enabled=False)
    print("  Running FIX A only (widen AMZN/TSLA)...")
    fa = run_portfolio(FIXED_V3, profit_lock_enabled=False)
    print("  Running FIX A + FIX B (widen + profit-lock)...")
    fab = run_portfolio(FIXED_V3, profit_lock_enabled=True)
    print("  Running FIX B only (profit-lock with original trails)...")
    fb = run_portfolio(CURRENT_V2, profit_lock_enabled=True)

    print()
    print("=" * 130)
    print(f"  {'config':<25} {'OOS net':<12} {'OOS ann':<9} | "
          f"{'COVID net':<12} {'COVID ann':<10} | {'RATESHOCK net':<14} {'RS ann':<8} | "
          f"{'Δ vs v2 (3-window)':<18}")
    print("-" * 130)

    def line(label, r):
        v2_total = sum(v2[w]["net"] for w in v2)
        r_total = sum(r[w]["net"] for w in r)
        delta = r_total - v2_total
        print(f"  {label:<25} ${r['OOS']['net']:>+8,.0f}  {r['OOS']['ann']:>+5.1f}%  | "
              f"${r['COVID']['net']:>+8,.0f}  {r['COVID']['ann']:>+5.1f}%  | "
              f"${r['RATESHOCK']['net']:>+10,.0f}  {r['RATESHOCK']['ann']:>+5.1f}%  | "
              f"${delta:>+10,.0f}")

    line("v2 baseline", v2)
    line("Fix A only", fa)
    line("Fix B only", fb)
    line("Fix A + Fix B", fab)

    print()
    print("  Per-day OOS averages (assumes 95 trading days in window):")
    days = 95
    for label, r in [("v2 baseline", v2), ("Fix A only", fa),
                     ("Fix B only", fb), ("Fix A + Fix B", fab)]:
        per_day = r["OOS"]["net"] / days
        per_day_pct = per_day / NET_LIQ * 100.0
        print(f"    {label:<20} ${per_day:>+7,.0f}/day  ({per_day_pct:+.3f}%)")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
