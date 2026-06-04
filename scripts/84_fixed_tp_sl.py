"""84_fixed_tp_sl — empirical test of the user's "tight TP / tighter SL"
strategy variant. Replaces the trailing-stop exit policy with a fixed
percentage take-profit AND a fixed percentage stop-loss.

The hypothesis: "If we take small quick profits (+X%) and cut losses faster
(−X/2%) we win in volume even if each trade is small."

Tests this on the SAME 3 windows as script 83 (OOS, COVID, RATESHOCK),
with cost-included replay, against the current deployed v2 (trailing-stop
based) and the agent-required UnifStop benchmarks.

TP/SL ratios tested:
  TP=0.5% / SL=0.25%  (2:1 reward:risk)
  TP=1.0% / SL=0.5%
  TP=2.0% / SL=1.0%
  TP=0.5% / SL=0.5%   (1:1 — control)
  TP=1.0% / SL=1.0%   (1:1 — control)

Same per-asset signals as v2 deployment for the entries.

Run:
    .venv/bin/python scripts/84_fixed_tp_sl.py
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

# Same per-asset signal as deployed (script 30 universe).
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

TP_SL_GRID = [
    (0.005, 0.0025),  # 0.5% / 0.25% (2:1)
    (0.010, 0.0050),  # 1.0% / 0.5% (2:1)
    (0.020, 0.0100),  # 2.0% / 1.0% (2:1)
    (0.005, 0.0050),  # 0.5% / 0.5% (1:1 control)
    (0.010, 0.0100),  # 1.0% / 1.0% (1:1 control)
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


def replay_fixed_tpsl(bars, feat, sig, start_idx, end_idx,
                      tp_pct, sl_pct, cost_bp):
    """Daily-bar replay with fixed-percentage TP/SL exits.
    Important caveat: daily-bar HIGH/LOW range can hit both TP and SL on
    same bar. We resolve this by assuming worst-case (SL hit first) when
    both are in range — matches the bake-off's 'sl_first' priority.
    """
    cost_frac = cost_bp / 1e4
    closes = bars["close"].astype(float).values
    highs = bars["high"].astype(float).values
    lows = bars["low"].astype(float).values
    state = "FLAT"
    entry_price = None
    entry_qty = 0
    entry_side = 0  # +1 long, -1 short
    round_trips = []
    daily_pnl = {}

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sig_dir = int(sig[i])
        close = float(closes[i])
        high = float(highs[i])
        low = float(lows[i])
        dk = bars.index[i].normalize()

        # Track daily MTM on prior position
        if state != "FLAT" and entry_qty > 0:
            prev_close = float(closes[i - 1]) if i > 0 else close
            mtm = (close - prev_close) * entry_qty * entry_side
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + mtm

        # Check TP/SL exits on the intra-bar HIGH/LOW range.
        if state == "IN_LONG" and entry_price is not None:
            tp_px = entry_price * (1 + tp_pct)
            sl_px = entry_price * (1 - sl_pct)
            # sl_first priority — if both in range, SL fires
            if low <= sl_px:
                pnl = (sl_px - entry_price) * entry_qty
                exit_cost = entry_qty * sl_px * cost_frac
                pnl -= exit_cost
                daily_pnl[dk] = daily_pnl.get(dk, 0.0) - exit_cost
                round_trips.append({"pnl": pnl, "reason": "SL", "side": "LONG"})
                state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
            elif high >= tp_px:
                pnl = (tp_px - entry_price) * entry_qty
                exit_cost = entry_qty * tp_px * cost_frac
                pnl -= exit_cost
                daily_pnl[dk] = daily_pnl.get(dk, 0.0) - exit_cost
                round_trips.append({"pnl": pnl, "reason": "TP", "side": "LONG"})
                state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
            elif sig_dir < 0:
                # Signal flip exit
                pnl = (close - entry_price) * entry_qty
                exit_cost = entry_qty * close * cost_frac
                pnl -= exit_cost
                daily_pnl[dk] = daily_pnl.get(dk, 0.0) - exit_cost
                round_trips.append({"pnl": pnl, "reason": "flip", "side": "LONG"})
                state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0

        elif state == "IN_SHORT" and entry_price is not None:
            tp_px = entry_price * (1 - tp_pct)
            sl_px = entry_price * (1 + sl_pct)
            if high >= sl_px:
                pnl = (entry_price - sl_px) * entry_qty
                exit_cost = entry_qty * sl_px * cost_frac
                pnl -= exit_cost
                daily_pnl[dk] = daily_pnl.get(dk, 0.0) - exit_cost
                round_trips.append({"pnl": pnl, "reason": "SL", "side": "SHORT"})
                state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
            elif low <= tp_px:
                pnl = (entry_price - tp_px) * entry_qty
                exit_cost = entry_qty * tp_px * cost_frac
                pnl -= exit_cost
                daily_pnl[dk] = daily_pnl.get(dk, 0.0) - exit_cost
                round_trips.append({"pnl": pnl, "reason": "TP", "side": "SHORT"})
                state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0
            elif sig_dir > 0:
                pnl = (entry_price - close) * entry_qty
                exit_cost = entry_qty * close * cost_frac
                pnl -= exit_cost
                daily_pnl[dk] = daily_pnl.get(dk, 0.0) - exit_cost
                round_trips.append({"pnl": pnl, "reason": "flip", "side": "SHORT"})
                state, entry_price, entry_qty, entry_side = "FLAT", None, 0, 0

        # New entry
        if state == "FLAT" and sig_dir != 0:
            qty = int(NOTIONAL // close) if close > 0 else 0
            if qty > 0:
                entry_cost = qty * close * cost_frac
                daily_pnl[dk] = daily_pnl.get(dk, 0.0) - entry_cost
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price = close
                entry_qty = qty
                entry_side = 1 if sig_dir > 0 else -1

    # End-of-window open position
    open_unrz = 0.0
    if state != "FLAT" and entry_price is not None:
        last_close = float(closes[end_idx - 1])
        open_unrz = (last_close - entry_price) * entry_qty * entry_side

    realized = sum(rt["pnl"] for rt in round_trips)
    net = realized + open_unrz
    wins = sum(1 for rt in round_trips if rt["pnl"] > 0)
    tps = sum(1 for rt in round_trips if rt["reason"] == "TP")
    sls = sum(1 for rt in round_trips if rt["reason"] == "SL")
    flips = sum(1 for rt in round_trips if rt["reason"] == "flip")
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0
    return {
        "n_trades": len(round_trips),
        "tps": tps, "sls": sls, "flips": flips,
        "wins": wins, "win_rate": win_rate,
        "realized": realized, "open_unrz": open_unrz, "net": net,
    }


def main():
    print()
    print("=" * 130)
    print(f"  Fixed TP/SL test  •  cost={COST_BP}bp/side  •  per-name ${NOTIONAL:,.0f}")
    print("=" * 130)

    # For each window × TP/SL grid: compute portfolio total
    results = {}
    for label, start, end in WINDOWS:
        results[label] = {}
        for tp_pct, sl_pct in TP_SL_GRID:
            portfolio_net = 0.0
            portfolio_trades = 0
            portfolio_tps = 0
            portfolio_sls = 0
            portfolio_flips = 0
            portfolio_wins = 0
            for asset, mode, lib, sym in DEPLOYED:
                bars = load_bars(lib, sym)
                feat = add_per_asset_features(bars)
                sig = precompute_signal(asset, bars, feat, mode)
                s = bars.index.get_indexer([start], method="bfill")[0]
                e = bars.index.get_indexer([end], method="ffill")[0] + 1
                if s <= WARMUP or e <= s:
                    continue
                r = replay_fixed_tpsl(bars, feat, sig, s, e,
                                       tp_pct, sl_pct, COST_BP)
                portfolio_net += r["net"]
                portfolio_trades += r["n_trades"]
                portfolio_tps += r["tps"]
                portfolio_sls += r["sls"]
                portfolio_flips += r["flips"]
                portfolio_wins += r["wins"]
            results[label][(tp_pct, sl_pct)] = {
                "net": portfolio_net,
                "trades": portfolio_trades,
                "tps": portfolio_tps, "sls": portfolio_sls,
                "flips": portfolio_flips, "wins": portfolio_wins,
            }

    # Print the matrix.
    for label, start, end in WINDOWS:
        print(f"\n{label} window {start.date()} → {end.date()}")
        print(f"  {'TP':<6} {'SL':<6} {'ratio':<6} {'trades':<7} "
              f"{'TP exits':<9} {'SL exits':<9} {'flips':<6} "
              f"{'win%':<7} {'NET $':<13} {'return %'}")
        print("-" * 100)
        for tp_pct, sl_pct in TP_SL_GRID:
            r = results[label][(tp_pct, sl_pct)]
            n = r["trades"]
            wr = (r["wins"] / n * 100.0) if n else 0.0
            ratio = tp_pct / sl_pct if sl_pct > 0 else 0
            pn = 8 * NOTIONAL
            ret = r["net"] / pn * 100.0
            print(f"  {tp_pct*100:.2f}%  {sl_pct*100:.2f}%  {ratio:.1f}:1   "
                  f"{n:<7} {r['tps']:<9} {r['sls']:<9} {r['flips']:<6} "
                  f"{wr:>5.1f}%  ${r['net']:>+10,.0f}  {ret:>+6.2f}%")

    print()
    print("=" * 130)
    print("  Reference: current v2 deployment (trailing-stop) on same windows:")
    print("    OOS       +$19,144 (+19.14%)")
    print("    COVID     +$11,202 (+11.20%)")
    print("    RATESHOCK   -$7,081 (-7.08%)")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
