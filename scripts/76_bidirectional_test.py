"""76_bidirectional_test — empirical answer: does going SHORT (not just LONG)
add P&L when applied to all 7 assets?

The current deployment uses asset-specific signals; 4 of 7 are buy_and_hold
which always emits +1 (long-only, never short). 2 are bidirectional
(MSFT rsi2 and TSLA macd) and 1 is lgbm (META). The user asked: what if we
go aggressive with bidirectional signals on every asset so the bot actually
takes short positions too?

This script replays the state machine with two alternative signal modes per
asset over the same OOS test window and reports the comparison.

  baseline  = current deployed config (winner signal per asset)
  macd      = bidirectional MACD signal for ALL assets (-1, 0, +1)
  rsi2      = bidirectional RSI(2) mean-reversion for ALL assets

For each variant we record: trades, fraction that were SHORTS, win rate, net
P&L, max DD, Sharpe.

Run:
    .venv/bin/python scripts/76_bidirectional_test.py
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
from ml_research.asset_specialists._signals import (  # noqa: E402
    add_per_asset_features, signal_macd, signal_rsi2_mr_10_90,
)
from strategy_engine.per_asset_runner import default_bar_loader  # noqa: E402

WARMUP = 252


def _latest_sigma(row: pd.Series) -> float | None:
    v = row.get("vol_21", float("nan"))
    return float(v) if math.isfinite(v) and v > 0 else None


def precompute_signal(asset: str, bars: pd.DataFrame, mode: str) -> np.ndarray:
    """Return sig_dir per bar as a numpy int8 array."""
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "baseline":
        for i in range(WARMUP, n):
            try:
                s = compute_signal_for(asset, bars.iloc[: i + 1])
                out[i] = int(round(float(s)))
            except Exception:
                pass
        return out
    if mode == "macd":
        feat = add_per_asset_features(bars)
        h = feat["macd_hist"].shift(1)
        sig = pd.Series(0, index=feat.index, dtype="int8")
        sig = sig.mask(h > 0, 1).mask(h < 0, -1)
        out = sig.fillna(0).astype(np.int8).values
        out[:WARMUP] = 0
        return out
    if mode == "rsi2":
        feat = add_per_asset_features(bars)
        rsi_prev = feat["rsi_2"].shift(1)
        sig = pd.Series(0, index=feat.index, dtype="int8")
        sig = sig.mask(rsi_prev < 10, 1).mask(rsi_prev > 90, -1)
        out = sig.fillna(0).astype(np.int8).values
        out[:WARMUP] = 0
        return out
    raise ValueError(f"unknown signal mode {mode!r}")


def replay(asset: str, bars: pd.DataFrame, feat: pd.DataFrame,
           sig_array: np.ndarray, start_idx: int, end_idx: int,
           notional: float, trail: float) -> dict:
    if start_idx < WARMUP:
        start_idx = WARMUP
    closes = bars["close"].astype(float).values
    idxs = bars.index
    state = "FLAT"
    entry_price = entry_sigma = entry_ts = None
    entry_qty = 0
    peak = trough = None
    round_trips: list[dict] = []
    daily_pnl: list[tuple[pd.Timestamp, float]] = []
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0  # +1 long, -1 short, 0 flat

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        row = feat.iloc[i]
        sigma = _latest_sigma(row)
        if sigma is None:
            continue
        sig_dir = int(sig_array[i])
        close = float(closes[i])
        ts = idxs[i]

        # Daily MTM (mark prior position to today's close).
        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            pnl_today = (close - prev_close) * prev_qty * prev_side
            daily_pnl.append((ts, pnl_today))
        else:
            daily_pnl.append((ts, 0.0))

        # Maintain extremes + check exits.
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
                    "side": "LONG", "entry_ts": entry_ts, "exit_ts": ts,
                    "entry_price": entry_price, "exit_price": close,
                    "qty": entry_qty, "pnl": pnl, "reason": exit_reason,
                })
                state, entry_price, entry_sigma, peak, entry_ts, entry_qty = \
                    "FLAT", None, None, None, None, 0
                prev_state = "FLAT"; prev_qty = 0; prev_side = 0
                prev_close = close
                continue

        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            band = entry_price * entry_sigma * trail
            stop_price = trough + band
            exit_reason = None
            if close >= stop_price:
                exit_reason = "trailing_stop"
            elif sig_dir > 0:
                exit_reason = "signal_flip"
            if exit_reason is not None:
                pnl = (entry_price - close) * entry_qty
                round_trips.append({
                    "side": "SHORT", "entry_ts": entry_ts, "exit_ts": ts,
                    "entry_price": entry_price, "exit_price": close,
                    "qty": entry_qty, "pnl": pnl, "reason": exit_reason,
                })
                state, entry_price, entry_sigma, trough, entry_ts, entry_qty = \
                    "FLAT", None, None, None, None, 0
                prev_state = "FLAT"; prev_qty = 0; prev_side = 0
                prev_close = close
                continue

        # New entry.
        if state == "FLAT" and sig_dir != 0:
            qty = int(notional // close) if close > 0 else 0
            if qty > 0:
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price = close
                entry_sigma = sigma
                entry_ts = ts
                entry_qty = qty
                if sig_dir > 0:
                    peak = close; trough = None
                else:
                    trough = close; peak = None

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
    longs = [rt for rt in rts if rt["side"] == "LONG"]
    shorts = [rt for rt in rts if rt["side"] == "SHORT"]
    wins = [rt for rt in rts if rt["pnl"] > 0]
    win_rate = (len(wins) / len(rts) * 100.0) if rts else 0.0
    daily = window["daily_pnl"]
    sharpe = float("nan")
    max_dd = 0.0
    if daily and notional > 0:
        rets = np.asarray([pnl / notional for _, pnl in daily])
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum([pnl for _, pnl in daily])
        peak = np.maximum.accumulate(cum)
        max_dd = float((cum - peak).min())
    n_bars = window["n_bars"]
    years = n_bars / 252.0 if n_bars else 0.0
    ann_ret = float("nan")
    if years > 0 and notional + net > 0:
        ann_ret = ((1 + net / notional) ** (1 / years) - 1) * 100.0
    long_pnl = sum(rt["pnl"] for rt in longs)
    short_pnl = sum(rt["pnl"] for rt in shorts)
    return {
        "n_trades": len(rts), "n_long": len(longs), "n_short": len(shorts),
        "long_pnl": long_pnl, "short_pnl": short_pnl,
        "win_rate": win_rate, "net": net, "max_dd": max_dd,
        "sharpe": sharpe, "ann_return_pct": ann_ret,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2024-01-01")
    ap.add_argument("--net-liq", type=float, default=100000.0)
    args = ap.parse_args()

    notional = args.net_liq / 7.0
    split_ts = pd.Timestamp(args.split_date)
    modes = ["baseline", "macd", "rsi2"]

    print()
    print("=" * 122)
    print(f"  Bidirectional signal test on OOS  (split={split_ts.date()}, "
          f"per-name=${notional:,.2f})")
    print("=" * 122)

    totals = {m: {"net": 0.0, "trades": 0, "longs": 0, "shorts": 0,
                  "long_pnl": 0.0, "short_pnl": 0.0} for m in modes}

    for asset in DEPLOYABLE_ASSETS:
        winner = winner_for(asset)
        if winner is None:
            continue
        trail = float(winner["trail_sigma"])
        bars = default_bar_loader(asset)
        feat = add_per_asset_features(bars)
        split_idx = bars.index.get_indexer([split_ts], method="bfill")[0]
        if split_idx <= WARMUP or split_idx >= len(bars):
            continue

        print(f"\n  {asset}  (winner={winner['model']}, trail={trail}σ)")
        print(f"  {'mode':<10} {'trades':<7} {'L/S':<10} {'win%':<7} "
              f"{'long $':<11} {'short $':<11} {'NET $':<13} {'ann%':<7} {'Sharpe'}")
        for m in modes:
            sig_arr = precompute_signal(asset, bars, m)
            w = replay(asset, bars, feat, sig_arr, split_idx, len(bars),
                       notional, trail)
            s = stats(w, notional)
            mark = " ←current" if m == "baseline" else ""
            print(f"  {m:<10} {s['n_trades']:<7} "
                  f"{s['n_long']}/{s['n_short']:<7} {s['win_rate']:>5.1f}%  "
                  f"${s['long_pnl']:>+8,.0f}  ${s['short_pnl']:>+8,.0f}  "
                  f"${s['net']:>+10,.0f}  {s['ann_return_pct']:>+5.1f}%  "
                  f"{s['sharpe']:>+5.2f}{mark}")
            totals[m]["net"] += s["net"]
            totals[m]["trades"] += s["n_trades"]
            totals[m]["longs"] += s["n_long"]
            totals[m]["shorts"] += s["n_short"]
            totals[m]["long_pnl"] += s["long_pnl"]
            totals[m]["short_pnl"] += s["short_pnl"]

    print()
    print("=" * 122)
    print("  PORTFOLIO totals across 7 assets (OOS test)")
    print("=" * 122)
    print(f"  {'mode':<10} {'trades':<7} {'L/S':<13} {'long $':<14} "
          f"{'short $':<14} {'NET $':<14} {'vs baseline'}")
    base = totals["baseline"]["net"]
    for m in modes:
        t = totals[m]
        d = t["net"] - base
        mark = " ←current" if m == "baseline" else (f"  Δ=${d:+,.0f}")
        print(f"  {m:<10} {t['trades']:<7} {t['longs']}/{t['shorts']:<11} "
              f"${t['long_pnl']:>+11,.0f}  ${t['short_pnl']:>+11,.0f}  "
              f"${t['net']:>+11,.0f}{mark}")
    print("=" * 122)
    return 0


if __name__ == "__main__":
    sys.exit(main())
