"""72_train_test_split — replay the per-asset trailing-stop state machine
separately on a TRAIN window and a held-out TEST window and report P&L,
Sharpe, win rate, and max drawdown on each.

This is the honest OOS check the user should look at before flipping on
paper trading: the bake-off promotion gate ran walk-forward folds, but
its DSR/Holm/CI numbers are sensitive to the cutoff. A blunt chronological
split shows whether the deployed configs hold up on the most recent slice
that's genuinely after the bake-off date.

Default split: 2024-01-01 (configurable via --split-date).
  TRAIN:  asset's full history up to (and excluding) split date
  TEST:   split date through last bar
Each half is replayed independently from a FLAT state with feature warmup
using prior bars (so the test half is not biased by train-window state).

Run:
    .venv/bin/python scripts/72_train_test_split.py
    .venv/bin/python scripts/72_train_test_split.py --split-date 2023-06-01
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
    """Replay [start_idx, end_idx) bars starting FLAT. Bars before start_idx
    are still available for indicator computation (no peek-ahead — only
    bars_so_far up through current i are used). Returns realized round
    trips + open-position MTM at end."""
    if start_idx < WARMUP:
        start_idx = WARMUP
    if end_idx <= start_idx:
        return {"trades": [], "round_trips": [], "open_unrealized": 0.0,
                "open_qty": 0, "n_bars": 0, "daily_returns": []}

    state = "FLAT"
    entry_price = entry_sigma = entry_ts = None
    entry_qty = 0
    peak = None
    trades: list[dict] = []
    round_trips: list[dict] = []
    # Daily returns of the strategy in account-equity terms (notional-anchored)
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

        # Daily MTM on yesterday's position (before any action this bar).
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
                pnl_pct = (close / entry_price - 1.0) * 100.0
                trades.append({"ts": ts, "side": "SELL", "price": close,
                               "reason": exit_reason, "qty": entry_qty,
                               "pnl": pnl, "pnl_pct": pnl_pct})
                round_trips.append({
                    "entry_ts": entry_ts, "entry_price": entry_price,
                    "exit_ts": ts, "exit_price": close, "qty": entry_qty,
                    "reason": exit_reason, "pnl": pnl, "pnl_pct": pnl_pct,
                    "hold_days": (ts - entry_ts).days,
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
                trades.append({"ts": ts, "side": "BUY", "price": close,
                               "reason": "entry", "qty": qty})
                state = "IN_LONG"
                entry_price = close
                entry_sigma = sigma
                entry_ts = ts
                entry_qty = qty
                peak = close

        prev_close = close
        prev_state = state
        prev_qty = entry_qty

    open_unrealized = 0.0
    open_qty = 0
    if state == "IN_LONG" and entry_price is not None:
        last_close = float(bars["close"].iloc[end_idx - 1])
        open_unrealized = (last_close - entry_price) * entry_qty
        open_qty = entry_qty

    return {
        "trades": trades, "round_trips": round_trips,
        "open_unrealized": open_unrealized, "open_qty": open_qty,
        "n_bars": end_idx - start_idx,
        "daily_pnl": daily_pnl,
    }


def stats(window_result: dict, notional: float) -> dict:
    rts = window_result["round_trips"]
    realized = sum(rt["pnl"] for rt in rts)
    open_unrz = window_result["open_unrealized"]
    net = realized + open_unrz
    wins = [rt for rt in rts if rt["pnl"] > 0]
    losses = [rt for rt in rts if rt["pnl"] <= 0]
    win_rate = (len(wins) / len(rts) * 100.0) if rts else 0.0
    avg_win = (sum(rt["pnl"] for rt in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(rt["pnl"] for rt in losses) / len(losses)) if losses else 0.0
    avg_hold = (sum(rt["hold_days"] for rt in rts) / len(rts)) if rts else 0.0

    # Daily-return Sharpe on strategy equity curve.
    daily = window_result["daily_pnl"]
    if daily and notional > 0:
        rets = [pnl / notional for _, pnl in daily]
        rets = np.asarray(rets, dtype=float)
        if rets.std() > 0 and len(rets) > 1:
            sharpe_daily = rets.mean() / rets.std()
            sharpe_annual = sharpe_daily * math.sqrt(252)
        else:
            sharpe_annual = float("nan")
        # Max DD on cumulative P&L.
        cum = np.cumsum([pnl for _, pnl in daily])
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = float(dd.min()) if len(dd) else 0.0
    else:
        sharpe_annual = float("nan")
        max_dd = 0.0

    n_bars = window_result["n_bars"]
    years = n_bars / 252.0 if n_bars else 0.0
    return {
        "n_trades": len(rts),
        "n_bars": n_bars,
        "years": years,
        "win_rate": win_rate,
        "wins": len(wins), "losses": len(losses),
        "realized": realized,
        "open_unrealized": open_unrz,
        "net": net,
        "return_pct": (net / notional * 100.0) if notional else 0.0,
        "annualized_return_pct": (
            ((1 + net / notional) ** (1 / years) - 1) * 100.0
            if years > 0 and notional + net > 0 else float("nan")
        ),
        "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_hold_days": avg_hold,
        "sharpe_annual": sharpe_annual,
        "max_dd": max_dd,
    }


def fmt_pnl(x: float) -> str:
    return f"${x:+,.2f}" if x else "$0.00"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2024-01-01")
    ap.add_argument("--net-liq", type=float, default=100000.0)
    args = ap.parse_args()

    notional = args.net_liq / 7.0
    split_ts = pd.Timestamp(args.split_date)

    print()
    print("=" * 116)
    print(f"  TRAIN / TEST split — cutoff = {split_ts.date()}   "
          f"(net_liq=${args.net_liq:,.0f}, per-name=${notional:,.2f})")
    print("=" * 116)

    train_totals = {"realized": 0.0, "open_unrealized": 0.0,
                     "n_trades": 0, "wins": 0, "losses": 0}
    test_totals = {"realized": 0.0, "open_unrealized": 0.0,
                    "n_trades": 0, "wins": 0, "losses": 0}

    for asset in DEPLOYABLE_ASSETS:
        winner = winner_for(asset)
        if winner is None:
            print(f"{asset}: no winner — skip")
            continue
        trail = float(winner["trail_sigma"])
        bars = default_bar_loader(asset)
        split_idx = bars.index.get_indexer([split_ts], method="bfill")[0]
        if split_idx <= WARMUP or split_idx >= len(bars):
            print(f"{asset}: split date out of range — skip")
            continue

        train_w = replay_window(asset, bars, WARMUP, split_idx, notional, trail)
        test_w = replay_window(asset, bars, split_idx, len(bars), notional, trail)
        train_s = stats(train_w, notional)
        test_s = stats(test_w, notional)

        train_totals["realized"] += train_s["realized"]
        train_totals["open_unrealized"] += train_s["open_unrealized"]
        train_totals["n_trades"] += train_s["n_trades"]
        train_totals["wins"] += train_s["wins"]
        train_totals["losses"] += train_s["losses"]
        test_totals["realized"] += test_s["realized"]
        test_totals["open_unrealized"] += test_s["open_unrealized"]
        test_totals["n_trades"] += test_s["n_trades"]
        test_totals["wins"] += test_s["wins"]
        test_totals["losses"] += test_s["losses"]

        train_date_start = bars.index[WARMUP].date()
        train_date_end = bars.index[split_idx - 1].date()
        test_date_start = bars.index[split_idx].date()
        test_date_end = bars.index[-1].date()

        print(f"\n{asset}  •  {winner['model']}  trail={trail}σ")
        print(f"  TRAIN  ({train_date_start} → {train_date_end},  "
              f"{train_s['years']:.1f}y, {train_s['n_bars']} bars):")
        print(f"    trades={train_s['n_trades']:<4}  win_rate={train_s['win_rate']:>5.1f}%  "
              f"realized={fmt_pnl(train_s['realized']):<15}  "
              f"net={fmt_pnl(train_s['net']):<15}  "
              f"ann_ret={train_s['annualized_return_pct']:>6.2f}%  "
              f"sharpe={train_s['sharpe_annual']:>5.2f}  "
              f"max_DD={fmt_pnl(train_s['max_dd'])}")
        print(f"  TEST   ({test_date_start} → {test_date_end},   "
              f"{test_s['years']:.1f}y, {test_s['n_bars']} bars):")
        print(f"    trades={test_s['n_trades']:<4}  win_rate={test_s['win_rate']:>5.1f}%  "
              f"realized={fmt_pnl(test_s['realized']):<15}  "
              f"net={fmt_pnl(test_s['net']):<15}  "
              f"ann_ret={test_s['annualized_return_pct']:>6.2f}%  "
              f"sharpe={test_s['sharpe_annual']:>5.2f}  "
              f"max_DD={fmt_pnl(test_s['max_dd'])}")

        # Honest verdict per asset.
        train_pos = train_s["net"] > 0
        test_pos = test_s["net"] > 0
        verdict = (
            "✓ holds OOS"        if train_pos and test_pos
            else "✗ FAILS OOS"     if train_pos and not test_pos
            else "? both negative" if not train_pos and not test_pos
            else "? test-only win (suspicious)"
        )
        print(f"    verdict: {verdict}")

    train_net = train_totals["realized"] + train_totals["open_unrealized"]
    test_net = test_totals["realized"] + test_totals["open_unrealized"]
    portfolio_notional = 7 * notional
    train_winrate = (train_totals["wins"] / (train_totals["wins"] + train_totals["losses"]) * 100.0
                     if (train_totals["wins"] + train_totals["losses"]) else 0.0)
    test_winrate = (test_totals["wins"] / (test_totals["wins"] + test_totals["losses"]) * 100.0
                    if (test_totals["wins"] + test_totals["losses"]) else 0.0)
    print()
    print("=" * 116)
    print("  PORTFOLIO (7 assets, equal-weight)")
    print(f"  TRAIN  trades={train_totals['n_trades']:<5}  win_rate={train_winrate:>5.1f}%  "
          f"realized={fmt_pnl(train_totals['realized']):<15}  net={fmt_pnl(train_net):<15}  "
          f"return={train_net/portfolio_notional*100:+6.2f}%")
    print(f"  TEST   trades={test_totals['n_trades']:<5}  win_rate={test_winrate:>5.1f}%  "
          f"realized={fmt_pnl(test_totals['realized']):<15}  net={fmt_pnl(test_net):<15}  "
          f"return={test_net/portfolio_notional*100:+6.2f}%")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    sys.exit(main())
