"""86_multi_position — pyramiding test: allow each asset to hold up to
N independent tranches simultaneously, each with its own trailing stop.

Hypothesis (user's): "If each asset can have 3 positions instead of 1, we
get 3× the upside on persistent signals."

Tested empirically against the current 1-tranche v2 deployment.

Sizing: per-TRANCHE notional = $100k / (8 × max_tranches). So 3-tranche
mode uses ~$4,166 per tranche (= 3 × $4,166 ≈ $12,500 per asset).

Tested configs:
  max_tranches = 1  (control = current v2)
  max_tranches = 2
  max_tranches = 3
  max_tranches = 5

Same per-asset signals + trail σ + PT as deployed v2.

Run:
    .venv/bin/python scripts/86_multi_position.py
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
PORTFOLIO_SIZE = 8
WARMUP = 252
COST_BP = 3.0

# Same per-asset signals + trail/PT as v2 deployment.
DEPLOYED = [
    ("AAPL",  "buy_and_hold",  3.0, 3.0,  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT",  "baseline",      1.0, 5.0,  "equity.daily_bars",  "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0,  "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      0.5, 3.0,  "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  "etf.daily_bars",     "ARCA.GLD"),
]

TRANCHE_CONFIGS = [1, 2, 3, 5]

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


def replay_multi_tranche(bars, feat, sig, start_idx, end_idx,
                          trail_sigma, pt_sigmas, max_tranches, notional_per_tranche):
    """Replay where each asset can hold up to max_tranches simultaneous
    positions, each with its own trailing stop / PT / sigma."""
    cost_frac = COST_BP / 1e4
    closes = bars["close"].astype(float).values
    vol = feat["vol_21"].values
    tranches: list[dict] = []  # active positions
    round_trips = []
    cost_total = 0.0
    daily_pnl = {}

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig[i])
        close = float(closes[i])
        dk = bars.index[i].normalize()

        # MTM on all active tranches from prior bar's close to today.
        prev_close = float(closes[i - 1]) if i > 0 else close
        for t in tranches:
            mtm = (close - prev_close) * t["qty"] * t["side"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + mtm

        # Check exits for each active tranche (iterate copy so we can remove).
        surviving = []
        for t in tranches:
            if t["side"] == 1:  # LONG
                if t["peak"] is None or close > t["peak"]:
                    t["peak"] = close
                band = t["entry_price"] * t["entry_sigma"] * trail_sigma
                stop = t["peak"] - band
                reason = None
                if close <= stop:
                    reason = "trail"
                elif pt_sigmas is not None and close >= t["entry_price"] * (1 + t["entry_sigma"] * pt_sigmas):
                    reason = "pt"
                elif sig_dir < 0:
                    reason = "flip"
                if reason:
                    gross = (close - t["entry_price"]) * t["qty"]
                    cost = t["qty"] * close * cost_frac
                    cost_total += cost
                    daily_pnl[dk] = daily_pnl.get(dk, 0.0) - cost
                    round_trips.append({"pnl": gross - cost, "reason": reason, "side": "LONG"})
                    continue
            else:  # SHORT
                if t["trough"] is None or close < t["trough"]:
                    t["trough"] = close
                band = t["entry_price"] * t["entry_sigma"] * trail_sigma
                stop = t["trough"] + band
                reason = None
                if close >= stop:
                    reason = "trail"
                elif pt_sigmas is not None and close <= t["entry_price"] * (1 - t["entry_sigma"] * pt_sigmas):
                    reason = "pt"
                elif sig_dir > 0:
                    reason = "flip"
                if reason:
                    gross = (t["entry_price"] - close) * t["qty"]
                    cost = t["qty"] * close * cost_frac
                    cost_total += cost
                    daily_pnl[dk] = daily_pnl.get(dk, 0.0) - cost
                    round_trips.append({"pnl": gross - cost, "reason": reason, "side": "SHORT"})
                    continue
            surviving.append(t)
        tranches = surviving

        # New entry if signal fires AND we're below max_tranches AND no tranche in opposite direction.
        if sig_dir != 0 and len(tranches) < max_tranches:
            # Check direction-consistency: don't open opposing tranches simultaneously
            if tranches:
                existing_side = tranches[0]["side"]
                requested_side = 1 if sig_dir > 0 else -1
                if existing_side != requested_side:
                    pass  # signal flipped — let existing tranches exit via flip first
                else:
                    qty = int(notional_per_tranche // close) if close > 0 else 0
                    if qty > 0:
                        cost = qty * close * cost_frac
                        cost_total += cost
                        daily_pnl[dk] = daily_pnl.get(dk, 0.0) - cost
                        tranches.append({
                            "side": requested_side, "qty": qty,
                            "entry_price": close, "entry_sigma": sigma,
                            "peak": close if requested_side == 1 else None,
                            "trough": close if requested_side == -1 else None,
                        })
            else:
                # No existing tranches — fresh entry
                requested_side = 1 if sig_dir > 0 else -1
                qty = int(notional_per_tranche // close) if close > 0 else 0
                if qty > 0:
                    cost = qty * close * cost_frac
                    cost_total += cost
                    daily_pnl[dk] = daily_pnl.get(dk, 0.0) - cost
                    tranches.append({
                        "side": requested_side, "qty": qty,
                        "entry_price": close, "entry_sigma": sigma,
                        "peak": close if requested_side == 1 else None,
                        "trough": close if requested_side == -1 else None,
                    })

    # End-of-window: mark all open tranches to last close.
    open_unrz = 0.0
    if tranches:
        last_close = float(closes[end_idx - 1])
        for t in tranches:
            if t["side"] == 1:
                open_unrz += (last_close - t["entry_price"]) * t["qty"]
            else:
                open_unrz += (t["entry_price"] - last_close) * t["qty"]

    realized = sum(rt["pnl"] for rt in round_trips)
    net = realized + open_unrz
    wins = sum(1 for rt in round_trips if rt["pnl"] > 0)
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0
    return {
        "n_trades": len(round_trips),
        "wins": wins, "win_rate": win_rate,
        "net": net, "cost": cost_total,
        "max_tranches_seen": max((t["side"] for t in tranches), default=0),
    }


def main():
    print()
    print("=" * 130)
    print(f"  Multi-position (pyramiding) test  •  cost={COST_BP}bp/side")
    print("=" * 130)

    for label, start, end in WINDOWS:
        print(f"\n{label} window {start.date()} → {end.date()}")
        print(f"  {'max':<5} {'per-tranche':<13} {'trades':<7} {'win%':<7} "
              f"{'cost $':<10} {'NET $':<13} {'return %'}")
        print("-" * 100)
        for max_tranches in TRANCHE_CONFIGS:
            # Per-tranche notional: $100k / (8 assets × max_tranches)
            notional_per_tranche = NET_LIQ / (PORTFOLIO_SIZE * max_tranches)
            portfolio_net = 0.0
            portfolio_trades = 0
            portfolio_wins = 0
            portfolio_cost = 0.0
            for asset, mode, trail, pt, lib, sym in DEPLOYED:
                bars = load_bars(lib, sym)
                feat = add_per_asset_features(bars)
                sig = precompute_signal(asset, bars, feat, mode)
                s = bars.index.get_indexer([start], method="bfill")[0]
                e = bars.index.get_indexer([end], method="ffill")[0] + 1
                if s <= WARMUP or e <= s:
                    continue
                r = replay_multi_tranche(bars, feat, sig, s, e,
                                          trail, pt, max_tranches, notional_per_tranche)
                portfolio_net += r["net"]
                portfolio_trades += r["n_trades"]
                portfolio_wins += r["wins"]
                portfolio_cost += r["cost"]
            wr = (portfolio_wins / portfolio_trades * 100.0) if portfolio_trades else 0.0
            ret = portfolio_net / NET_LIQ * 100.0
            mark = " ← current v2" if max_tranches == 1 else ""
            print(f"  {max_tranches:<5} ${notional_per_tranche:>10,.0f}    {portfolio_trades:<7} "
                  f"{wr:>5.1f}%  ${portfolio_cost:>7,.0f}  "
                  f"${portfolio_net:>+10,.0f}  {ret:>+6.2f}%{mark}")

    print()
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
