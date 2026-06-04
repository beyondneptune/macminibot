"""82_monday_simulation — bar-by-bar replay of the LIVE deployed configs
on Monday 2026-05-18's market data.

For each of the 8 deployed configs:
  • Loads full ArcticDB history (so all features warm up against years of data)
  • Starts FLAT at the start of the target day (clean book)
  • Walks every bar within Monday only, logging every entry / exit / hold
  • Aggregates realized P&L + open-position MTM at Monday's close

Intraday (AAPL, MSFT): ~78 regular-session 5-min bars + ~55 extended-hours
Daily (everyone else): 1 bar (Monday's close)

Run:
    .venv/bin/python scripts/82_monday_simulation.py
    .venv/bin/python scripts/82_monday_simulation.py --date 2026-05-15
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

from ml_research.asset_specialists import compute_signal_for  # noqa: E402
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from scripts._storage import open_store  # noqa: E402

NET_LIQ = 100000.0
NOTIONAL = NET_LIQ / 8.0
WARMUP_DAILY = 252
WARMUP_INTRADAY = 500

DEPLOYED = [
    ("AAPL",  "bollinger",     3.0, 3.0,  True,  "equity.minute_bars", "NASDAQ.AAPL"),
    ("MSFT",  "bollinger",     3.0, 3.0,  True,  "equity.minute_bars", "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0,  False, "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      0.5, 3.0,  False, "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  False, "etf.daily_bars",     "ARCA.GLD"),
]


def load_bars(lib, sym, intraday):
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    if intraday and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def precompute_signal(asset, bars, feat, mode, warmup):
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "baseline":
        for i in range(warmup, n):
            try:
                out[i] = int(round(float(compute_signal_for(asset, bars.iloc[: i + 1]))))
            except Exception:
                pass
        return out
    if mode == "buy_and_hold":
        out[warmup:] = 1
        return out
    if mode == "bollinger":
        close = bars["close"]
        ma = close.rolling(20).mean().shift(1)
        sd = close.rolling(20).std().shift(1)
        z = (close.shift(1) - ma) / sd
        out = np.where(z < -2, 1, np.where(z > 2, -1, 0)).astype(np.int8)
        out[:warmup] = 0
        return out
    raise ValueError(f"unknown mode {mode}")


def simulate_one_day(asset, sigmode, trail, pt, intraday, lib, sym,
                     target_day: pd.Timestamp) -> dict:
    warmup = WARMUP_INTRADAY if intraday else WARMUP_DAILY
    bars = load_bars(lib, sym, intraday)
    feat = add_per_asset_features(bars)
    sig = precompute_signal(asset, bars, feat, sigmode, warmup)
    closes = bars["close"].astype(float).values
    vol = feat["vol_21"].values

    # Slice indices for the target day only.
    day_start = bars.index.normalize() == target_day
    day_idxs = np.where(day_start)[0]
    if len(day_idxs) == 0:
        return {"asset": asset, "events": [], "net": 0.0,
                "open_unrz": 0.0, "n_bars": 0, "last_close": None}

    state = "FLAT"
    entry_price = entry_sigma = entry_ts = None
    entry_qty = 0
    peak = trough = None
    events: list[dict] = []
    realized = 0.0

    for i in day_idxs:
        sigma = float(vol[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig[i])
        close = float(closes[i])
        ts = bars.index[i]

        # Exit checks.
        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop = peak - band
            reason = None
            if close <= stop:
                reason = "trailing_stop"
            elif pt is not None and close >= entry_price * (1 + entry_sigma * pt):
                reason = "profit_target"
            elif sig_dir < 0:
                reason = "signal_flip"
            if reason:
                pnl = (close - entry_price) * entry_qty
                pnl_pct = (close / entry_price - 1.0) * 100.0
                realized += pnl
                events.append({
                    "ts": ts, "action": "SELL", "reason": reason,
                    "price": close, "qty": entry_qty,
                    "pnl": pnl, "pnl_pct": pnl_pct,
                    "entry_price": entry_price, "signal": sig_dir,
                })
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
                continue
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            band = entry_price * entry_sigma * trail
            stop = trough + band
            reason = None
            if close >= stop:
                reason = "trailing_stop"
            elif pt is not None and close <= entry_price * (1 - entry_sigma * pt):
                reason = "profit_target"
            elif sig_dir > 0:
                reason = "signal_flip"
            if reason:
                pnl = (entry_price - close) * entry_qty
                pnl_pct = (entry_price / close - 1.0) * 100.0
                realized += pnl
                events.append({
                    "ts": ts, "action": "BUY-to-cover", "reason": reason,
                    "price": close, "qty": entry_qty,
                    "pnl": pnl, "pnl_pct": pnl_pct,
                    "entry_price": entry_price, "signal": sig_dir,
                })
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                continue

        # New entry.
        if state == "FLAT" and sig_dir != 0:
            qty = int(NOTIONAL // close) if close > 0 else 0
            if qty > 0:
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price = close
                entry_sigma = sigma
                entry_ts = ts
                entry_qty = qty
                if sig_dir > 0:
                    peak, trough = close, None
                else:
                    trough, peak = close, None
                events.append({
                    "ts": ts, "action": "BUY" if sig_dir > 0 else "SHORT",
                    "reason": "entry", "price": close, "qty": qty,
                    "pnl": 0.0, "pnl_pct": 0.0, "entry_price": close,
                    "signal": sig_dir,
                })

    # Open-position MTM at Monday close.
    last_close = float(closes[day_idxs[-1]])
    open_unrz = 0.0
    if state == "IN_LONG" and entry_price is not None:
        open_unrz = (last_close - entry_price) * entry_qty
    elif state == "IN_SHORT" and entry_price is not None:
        open_unrz = (entry_price - last_close) * entry_qty

    return {
        "asset": asset, "events": events, "realized": realized,
        "open_unrz": open_unrz, "net": realized + open_unrz,
        "n_bars": len(day_idxs), "last_close": last_close,
        "ending_state": state, "open_entry": entry_price, "open_qty": entry_qty,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default="2026-05-18")
    args = ap.parse_args()

    target_day = pd.Timestamp(args.date).normalize()
    print()
    print("=" * 110)
    print(f"  Monday-style simulation  •  target day = {target_day.date()}  "
          f"•  account ${NET_LIQ:,.0f}  •  per-name ${NOTIONAL:,.0f}")
    print(f"  Starting state: FLAT on every asset (clean book at session open)")
    print("=" * 110)

    portfolio_realized = 0.0
    portfolio_open = 0.0
    portfolio_trades = 0
    results = []

    for asset, sigmode, trail, pt, intraday, lib, sym in DEPLOYED:
        r = simulate_one_day(asset, sigmode, trail, pt, intraday, lib, sym, target_day)
        results.append((r, intraday, sigmode, trail, pt))
        portfolio_realized += r["realized"]
        portfolio_open += r["open_unrz"]
        portfolio_trades += len([e for e in r["events"] if e["reason"] == "entry"])

        print(f"\n  {asset}  ({'intraday 5-min' if intraday else 'daily'}, "
              f"{sigmode} trail={trail}σ pt={f'{pt}σ' if pt else '—'})  "
              f"— {r['n_bars']} bars on target day")
        if not r["events"]:
            print(f"    (no signal — HOLD all day; last close ${r['last_close']:.2f})")
        else:
            for ev in r["events"]:
                ts_s = ev["ts"].strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d")
                if ev["reason"] == "entry":
                    print(f"    {ts_s}  {ev['action']:<13}  "
                          f"qty={ev['qty']:<4}  @ ${ev['price']:.2f}  "
                          f"signal={ev['signal']:+d}")
                else:
                    sign = "+" if ev["pnl"] >= 0 else ""
                    print(f"    {ts_s}  {ev['action']:<13}  "
                          f"qty={ev['qty']:<4}  @ ${ev['price']:.2f}  "
                          f"reason={ev['reason']}  "
                          f"P&L={sign}${ev['pnl']:,.2f} ({ev['pnl_pct']:+.2f}%)")
        if r["ending_state"] != "FLAT" and r["open_entry"] is not None:
            print(f"    >>> end-of-day OPEN: {r['ending_state']}  "
                  f"entry=${r['open_entry']:.2f}  qty={r['open_qty']}  "
                  f"MTM=${r['open_unrz']:+,.2f}")
        net_str = f"${r['net']:+,.2f}"
        print(f"    SUBTOTAL: realized ${r['realized']:+,.2f} "
              f"+ open MTM ${r['open_unrz']:+,.2f} = {net_str}")

    print()
    print("=" * 110)
    portfolio_net = portfolio_realized + portfolio_open
    print(f"  MONDAY PORTFOLIO (8 assets, clean-book simulation)")
    print(f"    realized P&L:       ${portfolio_realized:+,.2f}")
    print(f"    open positions MTM: ${portfolio_open:+,.2f}")
    print(f"    NET P&L:            ${portfolio_net:+,.2f}  "
          f"({portfolio_net / NET_LIQ * 100:+.3f}%)")
    print(f"    entries fired:      {portfolio_trades}")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
