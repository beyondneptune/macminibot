"""71_single_day_report — what the per-asset state machine did on a single bar.

Daily-bar granularity ⇒ one decision per asset per day. This walks the
replay forward through the full history, then prints what fired on the
*last* bar (default), or on any specific date you ask for via --date.

Run:
    .venv/bin/python scripts/71_single_day_report.py            # last bar
    .venv/bin/python scripts/71_single_day_report.py --date 2026-05-15
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from ml_research.asset_specialists import (  # noqa: E402
    DEPLOYABLE_ASSETS, compute_signal_for, winner_for,
)
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from strategy_engine.per_asset_runner import default_bar_loader  # noqa: E402


def _latest_sigma(feat: pd.DataFrame) -> float | None:
    if "vol_21" not in feat.columns or feat.empty:
        return None
    v = feat["vol_21"].iloc[-1]
    return float(v) if math.isfinite(v) and v > 0 else None


def walk_until(asset: str, target_date: pd.Timestamp | None,
               notional: float) -> dict:
    """Run the state machine bar-by-bar; report what happened ON target_date
    (or last bar if None). Returns dict with action, mark-to-market P&L,
    and the running state going into / out of that bar."""
    winner = winner_for(asset)
    if winner is None:
        return {"asset": asset, "action": "no_winner"}
    trail = float(winner["trail_sigma"])
    bars = default_bar_loader(asset)
    if len(bars) < 30:
        return {"asset": asset, "action": "insufficient_bars"}

    if target_date is None:
        target_idx = len(bars) - 1
    else:
        idx = bars.index.get_indexer([target_date], method="pad")[0]
        if idx < 0:
            return {"asset": asset, "action": f"date {target_date.date()} not found"}
        target_idx = idx

    if target_idx < 252:
        return {"asset": asset, "action": "target before warmup"}

    state = "FLAT"
    entry_price = entry_sigma = entry_ts = None
    entry_qty = 0
    peak = None
    prev_state = "FLAT"
    prev_entry_price = None
    prev_entry_qty = 0
    prev_peak = None

    action = None  # what fired on target bar
    action_pnl = 0.0
    action_pnl_pct = 0.0

    for i in range(252, target_idx + 1):
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
        is_target = (i == target_idx)

        if is_target:
            prev_state = state
            prev_entry_price = entry_price
            prev_entry_qty = entry_qty
            prev_peak = peak

        fired = None
        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop_price = peak - band
            if close <= stop_price:
                fired = ("SELL", "trailing_stop", close, entry_qty)
            elif sig_dir < 0:
                fired = ("SELL", "signal_flip", close, entry_qty)
            if fired is not None:
                pnl = (close - entry_price) * entry_qty
                pnl_pct = (close / entry_price - 1.0) * 100.0
                if is_target:
                    action = fired
                    action_pnl = pnl
                    action_pnl_pct = pnl_pct
                state = "FLAT"
                entry_price = entry_sigma = peak = entry_ts = None
                entry_qty = 0
                continue
        if state == "FLAT" and sig_dir > 0:
            qty = int(notional // close) if close > 0 else 0
            if qty > 0:
                fired = ("BUY", "entry", close, qty)
                if is_target:
                    action = fired
                state = "IN_LONG"
                entry_price = close
                entry_sigma = sigma
                entry_ts = ts
                entry_qty = qty
                peak = close

    target_close = float(bars["close"].iloc[target_idx])
    target_date_str = bars.index[target_idx].strftime("%Y-%m-%d")
    prev_close = float(bars["close"].iloc[target_idx - 1])

    # If no entry/exit fired today but we were holding, compute daily MTM.
    mtm_pnl = 0.0
    mtm_pct = 0.0
    if action is None and prev_state == "IN_LONG" and prev_entry_qty > 0:
        mtm_pnl = (target_close - prev_close) * prev_entry_qty
        mtm_pct = (target_close / prev_close - 1.0) * 100.0

    return {
        "asset": asset, "winner": winner, "trail_sigma": trail,
        "date": target_date_str, "close": target_close, "prev_close": prev_close,
        "day_return_pct": (target_close / prev_close - 1.0) * 100.0,
        "state_in": prev_state, "state_out": state,
        "qty_in": prev_entry_qty, "qty_out": entry_qty,
        "entry_price_in": prev_entry_price,
        "entry_price_out": entry_price,
        "action": action,            # tuple (side, reason, price, qty) or None
        "action_pnl": action_pnl,
        "action_pnl_pct": action_pnl_pct,
        "mtm_pnl": mtm_pnl,
        "mtm_pct": mtm_pct,
    }


def _color(v: float) -> str:
    if v > 0: return "\033[32m"   # green
    if v < 0: return "\033[31m"   # red
    return ""


RESET = "\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None,
                    help="ISO date (YYYY-MM-DD). Default: last bar in data.")
    ap.add_argument("--net-liq", type=float, default=100000.0)
    args = ap.parse_args()

    target = pd.Timestamp(args.date) if args.date else None
    notional = args.net_liq / 7.0

    rows = []
    for asset in DEPLOYABLE_ASSETS:
        try:
            rows.append(walk_until(asset, target, notional))
        except Exception as exc:
            rows.append({"asset": asset, "action": f"error: {exc}"})

    # Find the date we ended up using (assets share the same trading calendar
    # for the deployable universe, so any row's date is fine).
    used_date = next((r.get("date") for r in rows if "date" in r), str(target))

    print()
    print("=" * 98)
    print(f"  Single-day report — {used_date}   (net_liq=${args.net_liq:,.0f}, "
          f"per-name notional=${notional:,.2f})")
    print("=" * 98)
    print(f"{'asset':<6} {'close':<9} {'day Δ':<9} {'state in':<10} "
          f"{'action':<23} {'qty':<5} {'realized':<13} {'MTM':<13}")
    print("-" * 98)

    tot_realized = 0.0
    tot_mtm = 0.0
    for r in rows:
        a = r.get("asset", "?")
        if "winner" not in r:
            print(f"{a:<6} (skipped: {r.get('action', 'unknown')})")
            continue
        ret = r["day_return_pct"]
        ret_col = _color(ret)
        action_txt = "HOLD"
        action_pnl_txt = "—"
        action_qty_txt = "—"
        realized_color = ""
        if r["action"] is not None:
            side, reason, price, qty = r["action"]
            action_qty_txt = str(qty)
            if side == "BUY":
                action_txt = f"BUY (entry) @ ${price:.2f}"
            else:
                action_txt = f"SELL ({reason}) @ ${price:.2f}"
                realized_color = _color(r["action_pnl"])
                action_pnl_txt = (f"{realized_color}${r['action_pnl']:+,.2f} "
                                  f"({r['action_pnl_pct']:+.2f}%){RESET}")
                tot_realized += r["action_pnl"]
        elif r["state_in"] == "IN_LONG":
            action_txt = f"HOLD (in_long since entry @ ${r['entry_price_in']:.2f})"
            action_qty_txt = str(r["qty_in"])
        # MTM only meaningful when holding through the day (no entry/exit)
        mtm_txt = "—"
        if r["mtm_pnl"]:
            mtm_col = _color(r["mtm_pnl"])
            mtm_txt = f"{mtm_col}${r['mtm_pnl']:+,.2f} ({r['mtm_pct']:+.2f}%){RESET}"
            tot_mtm += r["mtm_pnl"]
        print(f"{a:<6} ${r['close']:<8.2f} {ret_col}{ret:>+6.2f}%{RESET}   "
              f"{r['state_in']:<10} {action_txt:<23} {action_qty_txt:<5} "
              f"{action_pnl_txt:<22} {mtm_txt}")

    print("-" * 98)
    net_total = tot_realized + tot_mtm
    nc = _color(net_total)
    print(f"  Day total: realized ${tot_realized:+,.2f}  +  "
          f"MTM (open positions) ${tot_mtm:+,.2f}  =  "
          f"{nc}${net_total:+,.2f}{RESET}")
    print("=" * 98)
    return 0


if __name__ == "__main__":
    sys.exit(main())
