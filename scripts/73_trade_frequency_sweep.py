"""73_trade_frequency_sweep — does tightening the trail-sigma multiplier
actually give us more trades AND more P&L? Or just more whipsaw?

Sweeps trail_multiplier ∈ {0.1, 0.25, 0.5, 1.0, 2.0} over the OOS test
window (default 2024-01-01 → last bar). For each setting, replays each
deployable asset and reports:
  - trades / year
  - win rate
  - net P&L (realized + open MTM)
  - annualized return
  - Sharpe (daily-returns, ann.)
  - max drawdown

1.0× is the bake-off winner trail. Smaller = tighter = more trades.

Run:
    .venv/bin/python scripts/73_trade_frequency_sweep.py
    .venv/bin/python scripts/73_trade_frequency_sweep.py --split-date 2024-01-01
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

from ml_research.asset_specialists import (  # noqa: E402
    DEPLOYABLE_ASSETS, compute_signal_for, winner_for,
)
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from strategy_engine.per_asset_runner import default_bar_loader  # noqa: E402

WARMUP = 252


def _latest_sigma(feat: pd.DataFrame) -> float | None:
    if "vol_21" not in feat.columns or feat.empty:
        return None
    v = feat["vol_21"].iloc[-1]
    return float(v) if math.isfinite(v) and v > 0 else None


def replay_window(asset: str, bars: pd.DataFrame, start_idx: int,
                  end_idx: int, notional: float, trail: float) -> dict:
    """Same state machine as 72_train_test_split, just parameterized on trail."""
    if start_idx < WARMUP:
        start_idx = WARMUP
    state = "FLAT"
    entry_price = entry_sigma = entry_ts = None
    entry_qty = 0
    peak = None
    round_trips: list[dict] = []
    daily_pnl: list[tuple[pd.Timestamp, float]] = []
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0

    for i in range(start_idx, end_idx):
        bars_so_far = bars.iloc[: i + 1]
        feat = add_per_asset_features(bars_so_far)
        sigma = _latest_sigma(feat)
        if sigma is None:
            continue
        try:
            sig = compute_signal_for(asset, bars_so_far)
        except Exception:
            continue
        sig_dir = int(round(float(sig)))
        close = float(bars["close"].iloc[i])
        ts = bars.index[i]

        if prev_close is not None and prev_state == "IN_LONG" and prev_qty > 0:
            daily_pnl.append((ts, (close - prev_close) * prev_qty))
        else:
            daily_pnl.append((ts, 0.0))

        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop_price = peak - band
            exit_reason = None
            if close <= stop_price:
                exit_reason = "trailing_stop"
            elif sig_dir < 0:
                exit_reason = "signal_flip"
            if exit_reason is not None:
                pnl = (close - entry_price) * entry_qty
                round_trips.append({
                    "entry_ts": entry_ts, "exit_ts": ts, "pnl": pnl,
                    "hold_days": (ts - entry_ts).days,
                    "reason": exit_reason,
                })
                state = "FLAT"
                entry_price = entry_sigma = peak = entry_ts = None
                entry_qty = 0
                prev_state = "FLAT"
                prev_qty = 0
                prev_close = close
                continue

        if state == "FLAT" and sig_dir > 0:
            qty = int(notional // close) if close > 0 else 0
            if qty > 0:
                state = "IN_LONG"
                entry_price = close
                entry_sigma = sigma
                entry_ts = ts
                entry_qty = qty
                peak = close

        prev_close = close
        prev_state = state
        prev_qty = entry_qty

    open_unrz = 0.0
    if state == "IN_LONG" and entry_price is not None:
        last_close = float(bars["close"].iloc[end_idx - 1])
        open_unrz = (last_close - entry_price) * entry_qty

    return {
        "round_trips": round_trips,
        "open_unrealized": open_unrz,
        "n_bars": end_idx - start_idx,
        "daily_pnl": daily_pnl,
    }


def stats(window: dict, notional: float) -> dict:
    rts = window["round_trips"]
    realized = sum(rt["pnl"] for rt in rts)
    net = realized + window["open_unrealized"]
    wins = [rt for rt in rts if rt["pnl"] > 0]
    losses = [rt for rt in rts if rt["pnl"] <= 0]
    win_rate = (len(wins) / len(rts) * 100.0) if rts else 0.0
    avg_hold = (sum(rt["hold_days"] for rt in rts) / len(rts)) if rts else 0.0

    daily = window["daily_pnl"]
    sharpe = float("nan")
    max_dd = 0.0
    days_above_1pct = 0
    days_below_neg1pct = 0
    worst_day = 0.0
    if daily and notional > 0:
        rets = np.asarray([pnl / notional for _, pnl in daily], dtype=float)
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum([pnl for _, pnl in daily])
        peak = np.maximum.accumulate(cum)
        max_dd = float((cum - peak).min())
        days_above_1pct = int((rets >= 0.01).sum())
        days_below_neg1pct = int((rets <= -0.01).sum())
        worst_day = float(rets.min() * notional)

    n_bars = window["n_bars"]
    years = n_bars / 252.0 if n_bars else 0.0
    trades_per_year = len(rts) / years if years > 0 else 0.0
    ann_ret_pct = float("nan")
    if years > 0 and notional + net > 0:
        ann_ret_pct = ((1 + net / notional) ** (1 / years) - 1) * 100.0

    return {
        "n_trades": len(rts), "trades_per_year": trades_per_year,
        "win_rate": win_rate, "avg_hold_days": avg_hold,
        "realized": realized, "open_unrealized": window["open_unrealized"],
        "net": net, "return_pct": (net / notional * 100.0) if notional else 0.0,
        "ann_return_pct": ann_ret_pct,
        "sharpe": sharpe, "max_dd": max_dd, "max_dd_pct": (max_dd / notional * 100.0),
        "days_above_1pct": days_above_1pct,
        "days_below_neg1pct": days_below_neg1pct,
        "worst_day_dollar": worst_day,
        "years": years,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2024-01-01")
    ap.add_argument("--net-liq", type=float, default=100000.0)
    ap.add_argument("--multipliers", type=str,
                    default="0.1,0.25,0.5,1.0,2.0")
    args = ap.parse_args()

    notional = args.net_liq / 7.0
    multipliers = [float(x) for x in args.multipliers.split(",")]
    split_ts = pd.Timestamp(args.split_date)

    print()
    print("=" * 120)
    print(f"  Trade-frequency sweep on OOS test window  "
          f"(split={split_ts.date()}, net_liq=${args.net_liq:,.0f}, "
          f"per-name=${notional:,.2f})")
    print("=" * 120)

    # Pre-load bar histories.
    histories = {}
    base_trails = {}
    split_idxs = {}
    for asset in DEPLOYABLE_ASSETS:
        winner = winner_for(asset)
        if winner is None:
            continue
        bars = default_bar_loader(asset)
        idx = bars.index.get_indexer([split_ts], method="bfill")[0]
        if idx <= WARMUP or idx >= len(bars):
            continue
        histories[asset] = bars
        base_trails[asset] = float(winner["trail_sigma"])
        split_idxs[asset] = idx

    # Sweep.
    portfolio_rows = []
    for mult in multipliers:
        per_asset = []
        for asset, bars in histories.items():
            trail = base_trails[asset] * mult
            test_w = replay_window(asset, bars, split_idxs[asset],
                                   len(bars), notional, trail)
            s = stats(test_w, notional)
            s["asset"] = asset
            s["trail"] = trail
            per_asset.append(s)

        net_total = sum(s["net"] for s in per_asset)
        trades_total = sum(s["n_trades"] for s in per_asset)
        wins_total = sum(s["n_trades"] * (s["win_rate"] / 100.0) for s in per_asset)
        wr = (wins_total / trades_total * 100.0) if trades_total else 0.0
        max_dd_portfolio = sum(s["max_dd"] for s in per_asset)  # rough upper bound
        n_deployed = len(per_asset)
        port_notional = notional * n_deployed
        years = per_asset[0]["years"] if per_asset else 0.0
        ann_ret = (((1 + net_total / port_notional) ** (1 / years) - 1) * 100.0
                   if years > 0 and port_notional + net_total > 0 else float("nan"))

        days_pos = sum(s["days_above_1pct"] for s in per_asset)
        days_neg = sum(s["days_below_neg1pct"] for s in per_asset)

        portfolio_rows.append({
            "mult": mult,
            "trades": trades_total,
            "trades_per_year_per_asset": trades_total / (n_deployed * years) if years else 0,
            "win_rate": wr,
            "net": net_total,
            "return_pct": net_total / port_notional * 100.0,
            "ann_return_pct": ann_ret,
            "max_dd": max_dd_portfolio,
            "max_dd_pct": max_dd_portfolio / port_notional * 100.0,
            "days_above_1pct": days_pos,
            "days_below_neg1pct": days_neg,
            "per_asset": per_asset,
        })

    # Print portfolio summary.
    print()
    print(f"{'mult':<6} {'trades':<8} {'tr/y/asset':<11} {'win%':<7} "
          f"{'net $':<14} {'return%':<9} {'ann%':<9} {'maxDD $':<14} "
          f"{'maxDD%':<8} {'>=+1% days':<11} {'<=-1% days':<11}")
    print("-" * 120)
    for row in portfolio_rows:
        mark = " ←" if row["mult"] == 1.0 else ""
        print(f"{row['mult']:<6.2f} {row['trades']:<8} {row['trades_per_year_per_asset']:>5.1f}      "
              f"{row['win_rate']:>5.1f}%  ${row['net']:>11,.0f}  "
              f"{row['return_pct']:>+7.1f}%  {row['ann_return_pct']:>+6.1f}%  "
              f"${row['max_dd']:>11,.0f}  {row['max_dd_pct']:>+6.1f}%  "
              f"{row['days_above_1pct']:>5}      {row['days_below_neg1pct']:>5}{mark}")

    # Per-asset detail at each multiplier.
    print()
    print("=" * 120)
    print("  Per-asset detail (OOS test only)")
    print("=" * 120)
    for row in portfolio_rows:
        print(f"\n  trail × {row['mult']}  →  portfolio net ${row['net']:+,.0f}  "
              f"({row['ann_return_pct']:+.1f}% ann.)  Sharpe-DD ratio = "
              f"{abs(row['net']/row['max_dd']) if row['max_dd'] != 0 else float('inf'):.2f}")
        print(f"  {'asset':<6} {'trail':<7} {'trades':<7} {'win%':<7} "
              f"{'net $':<13} {'ann %':<8} {'Sharpe':<8} {'maxDD $':<12}")
        for s in row["per_asset"]:
            print(f"  {s['asset']:<6} {s['trail']:<7.3f} {s['n_trades']:<7} "
                  f"{s['win_rate']:>5.1f}%  ${s['net']:>9,.0f}  "
                  f"{s['ann_return_pct']:>+6.1f}%  {s['sharpe']:>+6.2f}  "
                  f"${s['max_dd']:>9,.0f}")

    # Verdict.
    best = max(portfolio_rows, key=lambda r: (r["net"]))
    most_active = max(portfolio_rows, key=lambda r: r["trades"])
    best_riskadj = max(portfolio_rows,
                       key=lambda r: (r["net"] / abs(r["max_dd"])) if r["max_dd"] != 0 else 0)
    print()
    print("=" * 120)
    print(f"  Best NET P&L:        trail × {best['mult']}  →  "
          f"${best['net']:+,.0f}  ({best['ann_return_pct']:+.1f}% ann.)")
    print(f"  Most trades:         trail × {most_active['mult']}  →  "
          f"{most_active['trades']} trades  "
          f"({most_active['trades_per_year_per_asset']:.1f}/yr/asset)  "
          f"NET ${most_active['net']:+,.0f}")
    print(f"  Best risk-adj (NET/MaxDD):  trail × {best_riskadj['mult']}  →  "
          f"ratio {abs(best_riskadj['net']/best_riskadj['max_dd']):.2f}  "
          f"({best_riskadj['ann_return_pct']:+.1f}% ann.)")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
