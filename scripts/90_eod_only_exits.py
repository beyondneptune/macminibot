"""90_eod_only_exits — empirical test: does removing intraday exit checks help?

Hypothesis under test
=====================
The live intraday-exit-monitor (36_intraday_exit_monitor.py) polls mid-quote
prices every 5 min during the US session and exits when (peak − band) is
touched. Five of five historically observed intraday exits were premature
(per ops notes). Question: does disabling intra-session stops and relying
only on the daily close (or pure signal-flip) produce better net P&L?

Replay variants tested
======================
A. v2_baseline       — current 87_fixed_replay logic. Stop check uses the
                       daily CLOSE price (no intra-bar touch). This is the
                       deployed daily strategy.
B. intra_bar         — adds intra-bar HIGH/LOW touch detection: a long stop
                       fires the moment the day's LOW reaches the stop, a
                       short stop fires when the day's HIGH reaches it. PT
                       likewise resolves intra-bar. This APPROXIMATES the
                       live intraday-exit-monitor's worst-case behaviour
                       (the monitor sees mid-quotes ≈ continuous price).
C. eod_only          — pure close-based exit. Functionally identical to A
                       for the daily replay, but flagged so the comparison
                       A − B isolates the cost of intra-bar checks.
D. signal_flip_only  — NO trailing stop, NO profit target. Only exit when
                       the signal flips. Removes all stop-loss / take-profit
                       logic. The most extreme "let winners run / let losers
                       ride" variant.

Windows: OOS, COVID, RATESHOCK   Costs: COST_BP=3 per side (6bp round-trip).

Note about the test framing
===========================
The user originally framed this as "87 = close-only vs 83 = HIGH/LOW".
Reading the code: both 87 and 83 currently use the CLOSE price for stop
checks. So the meaningful empirical comparison is intra-bar-touch
(approximating the live intraday monitor) vs close-only vs no-stops.

Run:
    .venv/bin/python scripts/90_eod_only_exits.py
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

# Deployed v2 configs (same as 87_fixed_replay CURRENT_V2)
DEPLOYED_CONFIGS = [
    ("AAPL",  "buy_and_hold",  3.0, 3.0,  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT",  "baseline",      1.0, 5.0,  "equity.daily_bars",  "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0,  "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      0.5, 3.0,  "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  "etf.daily_bars",     "ARCA.GLD"),
]

WINDOWS = [
    ("OOS",       pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-20")),
    ("COVID",     pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
    ("RATESHOCK", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30")),
]

VARIANTS = ["v2_baseline", "intra_bar", "eod_only", "signal_flip_only"]


def load_bars(lib, sym):
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()[["open", "high", "low", "close", "volume"]]


def precompute_signal(asset, bars, mode):
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


def replay(bars, feat, sig, start_idx, end_idx, trail, pt, cost_bp, variant):
    """Replay a single asset.

    variant ∈ { "v2_baseline", "intra_bar", "eod_only", "signal_flip_only" }
    """
    cost_frac = cost_bp / 1e4
    closes = bars["close"].astype(float).values
    highs = bars["high"].astype(float).values
    lows = bars["low"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    entry_side = 0
    peak = trough = None
    round_trips = []
    daily_bucket = {}
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0
    vol = feat["vol_21"].values

    use_intra = (variant == "intra_bar")
    use_stops = (variant != "signal_flip_only")

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig[i])
        close = float(closes[i])
        bar_high = float(highs[i])
        bar_low = float(lows[i])
        dk = bars.index[i].normalize()

        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            step = (close - prev_close) * prev_qty * prev_side
            daily_bucket[dk] = daily_bucket.get(dk, 0.0) + step

        if state == "IN_LONG":
            # Update peak. For intra_bar variant, the high COULD have lifted peak.
            ref_peak_candidate = bar_high if use_intra else close
            if peak is None or ref_peak_candidate > peak:
                peak = ref_peak_candidate
            band = entry_price * entry_sigma * trail
            stop = peak - band
            pt_price = (entry_price * (1 + entry_sigma * pt)) if pt is not None else None
            reason = None
            exit_at = close  # default exit price = close
            if use_stops:
                if use_intra:
                    # Stop fires the moment LOW touches stop. PT fires when HIGH reaches PT.
                    if bar_low <= stop:
                        reason = "trail"
                        exit_at = stop  # worst-case fill at stop level
                    elif pt_price is not None and bar_high >= pt_price:
                        reason = "pt"
                        exit_at = pt_price
                else:
                    # close-only
                    if close <= stop:
                        reason = "trail"
                    elif pt_price is not None and close >= pt_price:
                        reason = "pt"
            if reason is None and sig_dir < 0:
                reason = "flip"
                exit_at = close
            if reason:
                gross = (exit_at - entry_price) * entry_qty
                cost = entry_qty * exit_at * cost_frac
                round_trips.append({"pnl": gross - cost, "reason": reason})
                # daily bucket bookkeeping — adjust today's step to exit_at instead of close
                if exit_at != close and prev_close is not None:
                    # remove the close-based step and replace with exit_at-based
                    step_close = (close - prev_close) * prev_qty * prev_side
                    step_exit = (exit_at - prev_close) * prev_qty * prev_side
                    daily_bucket[dk] = daily_bucket.get(dk, 0.0) - step_close + step_exit
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty, entry_side = 0, 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        elif state == "IN_SHORT":
            ref_trough_candidate = bar_low if use_intra else close
            if trough is None or ref_trough_candidate < trough:
                trough = ref_trough_candidate
            band = entry_price * entry_sigma * trail
            stop = trough + band
            pt_price = (entry_price * (1 - entry_sigma * pt)) if pt is not None else None
            reason = None
            exit_at = close
            if use_stops:
                if use_intra:
                    if bar_high >= stop:
                        reason = "trail"
                        exit_at = stop
                    elif pt_price is not None and bar_low <= pt_price:
                        reason = "pt"
                        exit_at = pt_price
                else:
                    if close >= stop:
                        reason = "trail"
                    elif pt_price is not None and close <= pt_price:
                        reason = "pt"
            if reason is None and sig_dir > 0:
                reason = "flip"
                exit_at = close
            if reason:
                gross = (entry_price - exit_at) * entry_qty
                cost = entry_qty * exit_at * cost_frac
                round_trips.append({"pnl": gross - cost, "reason": reason})
                if exit_at != close and prev_close is not None:
                    step_close = (close - prev_close) * prev_qty * prev_side
                    step_exit = (exit_at - prev_close) * prev_qty * prev_side
                    daily_bucket[dk] = daily_bucket.get(dk, 0.0) - step_close + step_exit
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

    realized = sum(r["pnl"] for r in round_trips)
    net = realized + open_unrz
    wins = sum(1 for r in round_trips if r["pnl"] > 0)
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0
    max_dd = 0.0
    if daily_bucket:
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    return {
        "n_trades": len(round_trips), "wins": wins, "win_rate": win_rate,
        "net": net, "max_dd": max_dd,
    }


def run_portfolio(variant):
    """Returns dict {window_label: {net, max_dd, trades, wr}}."""
    results = {}
    for label, start, end in WINDOWS:
        portfolio_net = 0.0
        portfolio_trades = 0
        portfolio_wins = 0
        portfolio_max_dd = 0.0
        for asset, mode, trail, pt, lib, sym in DEPLOYED_CONFIGS:
            bars = load_bars(lib, sym)
            feat = add_per_asset_features(bars)
            sig = precompute_signal(asset, bars, mode)
            s = bars.index.get_indexer([start], method="bfill")[0]
            e = bars.index.get_indexer([end], method="ffill")[0] + 1
            if s <= WARMUP or e <= s:
                continue
            r = replay(bars, feat, sig, s, e, trail, pt, COST_BP, variant)
            portfolio_net += r["net"]
            portfolio_trades += r["n_trades"]
            portfolio_wins += r["wins"]
            # naive sum of per-asset DDs as a portfolio worst-case approximation
            portfolio_max_dd += r["max_dd"]
        wr = (portfolio_wins / portfolio_trades * 100.0) if portfolio_trades else 0.0
        ret_pct = portfolio_net / NET_LIQ * 100.0
        results[label] = {
            "net": portfolio_net, "trades": portfolio_trades, "wr": wr,
            "ret": ret_pct, "max_dd": portfolio_max_dd,
        }
    return results


def main():
    print()
    print("=" * 130)
    print(f"  90_eod_only_exits  •  cost={COST_BP}bp/side  •  variants={VARIANTS}")
    print("=" * 130)

    all_results = {}
    for v in VARIANTS:
        print(f"  Running {v}...")
        all_results[v] = run_portfolio(v)

    print()
    print("=" * 130)
    print(f"  {'variant':<20} | "
          f"{'OOS net':>10} {'OOS DD':>9} {'OOS trd':>7} | "
          f"{'COVID net':>11} {'COVID DD':>9} {'COVID trd':>10} | "
          f"{'RS net':>10} {'RS DD':>9} {'RS trd':>7}")
    print("-" * 130)
    for v in VARIANTS:
        r = all_results[v]
        print(
            f"  {v:<20} | "
            f"${r['OOS']['net']:>+8,.0f} ${r['OOS']['max_dd']:>+7,.0f} "
            f"{r['OOS']['trades']:>7} | "
            f"${r['COVID']['net']:>+9,.0f} ${r['COVID']['max_dd']:>+7,.0f} "
            f"{r['COVID']['trades']:>10} | "
            f"${r['RATESHOCK']['net']:>+8,.0f} ${r['RATESHOCK']['max_dd']:>+7,.0f} "
            f"{r['RATESHOCK']['trades']:>7}"
        )
    print("=" * 130)

    print()
    print("  Deltas vs v2_baseline (positive = variant beats baseline):")
    print("-" * 90)
    base = all_results["v2_baseline"]
    for v in VARIANTS:
        if v == "v2_baseline":
            continue
        r = all_results[v]
        print(
            f"  {v:<20} | "
            f"OOS Δ=${r['OOS']['net'] - base['OOS']['net']:>+8,.0f}  "
            f"COVID Δ=${r['COVID']['net'] - base['COVID']['net']:>+9,.0f}  "
            f"RS Δ=${r['RATESHOCK']['net'] - base['RATESHOCK']['net']:>+8,.0f}  "
            f"TOTAL Δ=${sum(r[w]['net'] - base[w]['net'] for w in r):>+9,.0f}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
