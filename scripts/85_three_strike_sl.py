"""85_three_strike_sl — empirical test of the user's "3-strike SL" rule.

Idea: instead of immediately exiting when stop-loss is hit, give the
position N chances (default 3 bars) to reverse back above the SL price.
Only force-exit when SL is hit on N consecutive bars or position close is
at/below SL after N bars total below it.

Hypothesis: many SL hits are whipsaw — price dips through the level then
recovers within 1-2 bars. A patience filter avoids those false exits but
caps the max loss higher than the nominal SL.

Same framework as 84_fixed_tp_sl: same signals, same windows, same costs.

Run:
    .venv/bin/python scripts/85_three_strike_sl.py
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

DEPLOYED = [
    ("AAPL",  "buy_and_hold",  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT",  "baseline",      "equity.daily_bars",  "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "baseline",      "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  "etf.daily_bars",     "ARCA.GLD"),
]

# (TP %, SL %, max_strikes_before_forced_exit)
CONFIGS = [
    (0.010, 0.0050, 1),  # baseline: 1.0% TP / 0.5% SL, instant exit (control)
    (0.010, 0.0050, 2),  # 2 strikes
    (0.010, 0.0050, 3),  # 3 strikes
    (0.010, 0.0050, 5),  # 5 strikes
    (0.020, 0.0100, 1),  # 2.0% TP / 1.0% SL, instant
    (0.020, 0.0100, 3),  # 3 strikes
]

WINDOWS = [
    ("OOS",      pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-19")),
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
    raise ValueError(f"unknown {mode}")


def replay_three_strike(bars, feat, sig, start_idx, end_idx,
                         tp_pct, sl_pct, max_strikes, cost_bp):
    """Replay with N-strike SL. Each bar where close is at/below SL counts
    as a strike. When strikes >= max_strikes, force exit. If price reverses
    back above SL, strike count resets to 0."""
    cost_frac = cost_bp / 1e4
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = None
    entry_qty = 0
    entry_side = 0
    strikes = 0
    round_trips = []

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sig_dir = int(sig[i])
        close = float(closes[i])

        if state == "IN_LONG" and entry_price is not None:
            tp_px = entry_price * (1 + tp_pct)
            sl_px = entry_price * (1 - sl_pct)
            if close <= sl_px:
                strikes += 1
                if strikes >= max_strikes:
                    pnl = (close - entry_price) * entry_qty
                    cost = entry_qty * close * cost_frac
                    pnl -= cost
                    round_trips.append({"pnl": pnl, "reason": "SL_strike", "side": "LONG"})
                    state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
                    strikes = 0
                    continue
                # else: hold through this bar; max loss could deepen
            else:
                strikes = 0  # reset if recovered above SL
                if close >= tp_px:
                    pnl = (close - entry_price) * entry_qty
                    cost = entry_qty * close * cost_frac
                    pnl -= cost
                    round_trips.append({"pnl": pnl, "reason": "TP", "side": "LONG"})
                    state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
                    continue
                if sig_dir < 0:
                    pnl = (close - entry_price) * entry_qty
                    cost = entry_qty * close * cost_frac
                    pnl -= cost
                    round_trips.append({"pnl": pnl, "reason": "flip", "side": "LONG"})
                    state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
                    continue

        elif state == "IN_SHORT" and entry_price is not None:
            tp_px = entry_price * (1 - tp_pct)
            sl_px = entry_price * (1 + sl_pct)
            if close >= sl_px:
                strikes += 1
                if strikes >= max_strikes:
                    pnl = (entry_price - close) * entry_qty
                    cost = entry_qty * close * cost_frac
                    pnl -= cost
                    round_trips.append({"pnl": pnl, "reason": "SL_strike", "side": "SHORT"})
                    state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
                    strikes = 0
                    continue
            else:
                strikes = 0
                if close <= tp_px:
                    pnl = (entry_price - close) * entry_qty
                    cost = entry_qty * close * cost_frac
                    pnl -= cost
                    round_trips.append({"pnl": pnl, "reason": "TP", "side": "SHORT"})
                    state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
                    continue
                if sig_dir > 0:
                    pnl = (entry_price - close) * entry_qty
                    cost = entry_qty * close * cost_frac
                    pnl -= cost
                    round_trips.append({"pnl": pnl, "reason": "flip", "side": "SHORT"})
                    state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
                    continue

        if state == "FLAT" and sig_dir != 0:
            qty = int(NOTIONAL // close) if close > 0 else 0
            if qty > 0:
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price = close
                entry_qty = qty
                entry_side = 1 if sig_dir > 0 else -1
                strikes = 0

    open_unrz = 0.0
    if state != "FLAT" and entry_price is not None:
        last_close = float(closes[end_idx - 1])
        open_unrz = (last_close - entry_price) * entry_qty * entry_side

    realized = sum(rt["pnl"] for rt in round_trips)
    net = realized + open_unrz
    wins = sum(1 for rt in round_trips if rt["pnl"] > 0)
    tps = sum(1 for rt in round_trips if rt["reason"] == "TP")
    sls = sum(1 for rt in round_trips if rt["reason"] == "SL_strike")
    flips = sum(1 for rt in round_trips if rt["reason"] == "flip")
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0
    worst = min((rt["pnl"] for rt in round_trips), default=0.0)
    return {"n_trades": len(round_trips), "tps": tps, "sls": sls, "flips": flips,
            "wins": wins, "win_rate": win_rate, "net": net, "worst_trade": worst}


def main():
    print()
    print("=" * 130)
    print(f"  3-strike SL test  •  cost={COST_BP}bp/side  •  per-name ${NOTIONAL:,.0f}")
    print("  (max_strikes=N: how many consecutive bars at-or-below SL before forced exit)")
    print("=" * 130)

    results = {}
    for label, start, end in WINDOWS:
        results[label] = {}
        for tp_pct, sl_pct, strikes in CONFIGS:
            portfolio_net = 0.0
            portfolio_trades = 0
            portfolio_tps = 0
            portfolio_sls = 0
            portfolio_wins = 0
            worst = 0.0
            for asset, mode, lib, sym in DEPLOYED:
                bars = load_bars(lib, sym)
                feat = add_per_asset_features(bars)
                sig = precompute_signal(asset, bars, feat, mode)
                s = bars.index.get_indexer([start], method="bfill")[0]
                e = bars.index.get_indexer([end], method="ffill")[0] + 1
                if s <= WARMUP or e <= s:
                    continue
                r = replay_three_strike(bars, feat, sig, s, e,
                                         tp_pct, sl_pct, strikes, COST_BP)
                portfolio_net += r["net"]
                portfolio_trades += r["n_trades"]
                portfolio_tps += r["tps"]
                portfolio_sls += r["sls"]
                portfolio_wins += r["wins"]
                worst = min(worst, r["worst_trade"])
            results[label][(tp_pct, sl_pct, strikes)] = {
                "net": portfolio_net, "trades": portfolio_trades,
                "tps": portfolio_tps, "sls": portfolio_sls,
                "wins": portfolio_wins, "worst": worst,
            }

    for label, start, end in WINDOWS:
        print(f"\n{label} window")
        print(f"  {'TP':<6} {'SL':<6} {'strikes':<8} {'trades':<7} "
              f"{'TPs':<6} {'SLs':<6} {'win%':<7} {'NET $':<13} {'worst trade $'}")
        print("-" * 100)
        for tp_pct, sl_pct, strikes in CONFIGS:
            r = results[label][(tp_pct, sl_pct, strikes)]
            wr = (r["wins"] / r["trades"] * 100.0) if r["trades"] else 0.0
            mark = " ← N=1 baseline" if strikes == 1 else ""
            print(f"  {tp_pct*100:.2f}%  {sl_pct*100:.2f}%  {strikes:<8} {r['trades']:<7} "
                  f"{r['tps']:<6} {r['sls']:<6} {wr:>5.1f}%  "
                  f"${r['net']:>+10,.0f}  ${r['worst']:>+10,.0f}{mark}")
    print()
    print("  N=1 = instant SL exit (control). Higher N = more patience, deeper potential losses.")
    print("=" * 130)


if __name__ == "__main__":
    sys.exit(main())
