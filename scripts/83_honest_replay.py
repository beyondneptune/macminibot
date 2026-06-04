"""83_honest_replay — replay deployed configs WITH realistic transaction
costs, alongside a buy-and-hold benchmark and historical crisis windows.

What changed vs 79/80/82:
   1. Every entry AND every exit pays a round-trip cost = `--cost-bp` basis
      points of notional. Default 3bp (1.5bp slippage + 1bp spread per side).
   2. A buy-and-hold benchmark is computed for every window: enter at window
      open, hold to window close, equal-weight across the 8 names.
   3. Three windows are evaluated:
        OOS         2026-01-01 → 2026-05-19  (current OOS window)
        COVID       2020-02-01 → 2020-04-30  (Q1 2020 crash)
        RATESHOCK   2022-01-01 → 2022-06-30  (rate-shock drawdown)
      For COVID and RATESHOCK only the 6 DAILY assets are evaluated — we have
      no 5-min bars before 2024-05 for AAPL/MSFT.

Run:
    .venv/bin/python scripts/83_honest_replay.py
    .venv/bin/python scripts/83_honest_replay.py --cost-bp 5
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
    # all-daily v2 deployment (post-agent-review, intraday sleeve paused).
    ("AAPL",  "buy_and_hold",  3.0, 3.0,  False, "equity.daily_bars",  "NASDAQ.AAPL",  "equity.daily_bars", "NASDAQ.AAPL"),
    ("MSFT",  "baseline",      1.0, 5.0,  False, "equity.daily_bars",  "NASDAQ.MSFT",  "equity.daily_bars", "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.GOOGL", "equity.daily_bars", "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0,  False, "equity.daily_bars",  "NASDAQ.AMZN",  "equity.daily_bars", "NASDAQ.AMZN"),
    ("META",  "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.META",  "equity.daily_bars", "NASDAQ.META"),
    ("TSLA",  "baseline",      0.5, 3.0,  False, "equity.daily_bars",  "NASDAQ.TSLA",  "equity.daily_bars", "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      2.0, None, False, "equity.daily_bars",  "NASDAQ.NVDA",  "equity.daily_bars", "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  False, "etf.daily_bars",     "ARCA.GLD",     "etf.daily_bars",    "ARCA.GLD"),
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


def replay(bars, feat, sig, start_idx, end_idx, trail, pt, cost_bp):
    """Replay with cost: each entry AND each exit pays cost_bp basis points
    of the trade notional. cost_bp=3 means 0.03% per side = 0.06% round-trip."""
    cost_frac = cost_bp / 1e4
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    round_trips = []
    daily_bucket = {}
    total_cost = 0.0
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
            if close <= stop:
                reason = "trail"
            elif pt is not None and close >= entry_price * (1 + entry_sigma * pt):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                gross = (close - entry_price) * entry_qty
                exit_cost = entry_qty * close * cost_frac
                pnl = gross - exit_cost
                total_cost += exit_cost
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - exit_cost
                round_trips.append({"side": "LONG", "pnl": pnl, "cost": exit_cost})
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            band = entry_price * entry_sigma * trail
            stop = trough + band
            reason = None
            if close >= stop:
                reason = "trail"
            elif pt is not None and close <= entry_price * (1 - entry_sigma * pt):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                gross = (entry_price - close) * entry_qty
                exit_cost = entry_qty * close * cost_frac
                pnl = gross - exit_cost
                total_cost += exit_cost
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - exit_cost
                round_trips.append({"side": "SHORT", "pnl": pnl, "cost": exit_cost})
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        if state == "FLAT" and sig_dir != 0:
            qty = int(NOTIONAL // close) if close > 0 else 0
            if qty > 0:
                entry_cost = qty * close * cost_frac
                total_cost += entry_cost
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - entry_cost
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price = close
                entry_sigma = sigma
                entry_qty = qty
                if sig_dir > 0:
                    peak, trough = close, None
                else:
                    trough, peak = close, None

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

    realized = sum(rt["pnl"] for rt in round_trips)
    net = realized + open_unrz
    wins = sum(1 for rt in round_trips if rt["pnl"] > 0)
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0
    sharpe = float("nan")
    max_dd = 0.0
    n_sessions = len(daily_bucket)
    if daily_bucket:
        rets = np.asarray(list(daily_bucket.values())) / NOTIONAL
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    years = n_sessions / 252.0 if n_sessions else 0.0
    ann = float("nan")
    if years > 0 and NOTIONAL + net > 0:
        ann = ((1 + net / NOTIONAL) ** (1 / years) - 1) * 100.0
    return {
        "n_trades": len(round_trips), "wins": wins, "win_rate": win_rate,
        "net": net, "max_dd": max_dd, "sharpe": sharpe, "ann": ann,
        "total_cost": total_cost, "years": years,
    }


def uniform_stop_benchmark(daily_bars: pd.DataFrame, feat: pd.DataFrame,
                            start_idx: int, end_idx: int, notional: float,
                            cost_bp: float, trail_sigma: float = 2.0) -> dict:
    """Long-only generic trailing-stop overlay — the falsification agent's
    null hypothesis. Buy on day 1, exit when close drops below peak − 2σ.
    Re-enter on the next bar. No per-asset signal selection. If the deployed
    bot can't beat THIS, the "skill" is just the stop-loss overlay."""
    cost_frac = cost_bp / 1e4
    closes = daily_bars["close"].astype(float).values
    vol = feat["vol_21"].values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = None
    round_trips = []
    total_cost = 0.0
    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        close = float(closes[i])
        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            stop = peak - entry_price * entry_sigma * trail_sigma
            if close <= stop:
                gross = (close - entry_price) * entry_qty
                cost = entry_qty * close * cost_frac
                round_trips.append({"pnl": gross - cost})
                total_cost += cost
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
                continue
        if state == "FLAT":  # uniform overlay always re-enters next bar
            qty = int(notional // close) if close > 0 else 0
            if qty > 0:
                total_cost += qty * close * cost_frac
                state = "IN_LONG"
                entry_price = close
                entry_sigma = sigma
                entry_qty = qty
                peak = close
    open_unrz = 0.0
    if state == "IN_LONG" and entry_price is not None:
        last_close = float(closes[end_idx - 1])
        open_unrz = (last_close - entry_price) * entry_qty
    net = sum(rt["pnl"] for rt in round_trips) + open_unrz
    return {"net": net, "trades": len(round_trips), "cost": total_cost}


def bh_benchmark(daily_bars: pd.DataFrame, start_ts: pd.Timestamp,
                 end_ts: pd.Timestamp, notional: float,
                 cost_bp: float) -> dict:
    """Buy at start, hold to end. Entry + exit both pay cost_bp.
    Returns net P&L for that one asset's portion."""
    df = daily_bars[(daily_bars.index >= start_ts) & (daily_bars.index <= end_ts)]
    if len(df) < 2:
        return {"net": 0.0, "ann": float("nan"), "years": 0.0,
                "entry_price": None, "exit_price": None}
    cost_frac = cost_bp / 1e4
    entry = float(df["close"].iloc[0])
    exit_p = float(df["close"].iloc[-1])
    qty = int(notional // entry) if entry > 0 else 0
    if qty == 0:
        return {"net": 0.0, "ann": float("nan"), "years": 0.0,
                "entry_price": entry, "exit_price": exit_p}
    entry_cost = qty * entry * cost_frac
    exit_cost = qty * exit_p * cost_frac
    gross = (exit_p - entry) * qty
    net = gross - entry_cost - exit_cost
    years = len(df) / 252.0
    ann = float("nan")
    if years > 0 and notional + net > 0:
        ann = ((1 + net / notional) ** (1 / years) - 1) * 100.0
    return {"net": net, "ann": ann, "years": years,
            "entry_price": entry, "exit_price": exit_p, "qty": qty}


WINDOWS = [
    ("OOS",      pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-19"), True),
    ("COVID",    pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"), False),
    ("RATESHOCK", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30"), False),
]


def evaluate_window(label, start_ts, end_ts, allow_intraday, cost_bp):
    print(f"\n{'=' * 138}")
    print(f"  {label} window  {start_ts.date()} → {end_ts.date()}  cost={cost_bp}bp per side")
    print(f"{'=' * 138}")
    print(f"  {'asset':<6} {'config':<28} "
          f"{'trades':<7} {'NET $':<11} {'B&H $':<11} {'UnifStop $':<12} "
          f"{'α vs B&H':<11} {'α vs UnifStop':<14}")
    print("-" * 138)

    bot_total = 0.0
    bh_total = 0.0
    us_total = 0.0
    bot_trades = 0
    cost_total = 0.0
    n_deployed = 0

    for asset, sigmode, trail, pt, intraday, ilib, isym, dlib, dsym in DEPLOYED:
        use_intraday = intraday and allow_intraday
        warmup = WARMUP_INTRADAY if use_intraday else WARMUP_DAILY
        lib = ilib if use_intraday else dlib
        sym = isym if use_intraday else dsym
        try:
            bars = load_bars(lib, sym, use_intraday)
        except Exception as exc:
            print(f"  {asset}: load error {type(exc).__name__}")
            continue
        feat = add_per_asset_features(bars)
        sig = precompute_signal(asset, bars, feat,
                                 sigmode if use_intraday or not intraday else "baseline",
                                 warmup)
        start_idx = bars.index.get_indexer([start_ts], method="bfill")[0]
        end_idx = bars.index.get_indexer([end_ts], method="ffill")[0] + 1
        end_idx = min(end_idx, len(bars))
        if start_idx <= warmup or end_idx <= start_idx:
            print(f"  {asset}: out of range")
            continue

        # If we're in crisis window but the deployed strategy is intraday,
        # we use the DAILY equivalent of that asset's baseline (since the
        # intraday bake-off never trained on crisis data).
        result_sigmode = sigmode if use_intraday or not intraday else "baseline"
        result = replay(bars, feat, sig, start_idx, end_idx, trail, pt, cost_bp)

        # B&H benchmark uses daily bars regardless.
        daily_bars = load_bars(dlib, dsym, False)
        bh = bh_benchmark(daily_bars, start_ts, end_ts, NOTIONAL, cost_bp)

        # Uniform-stop overlay benchmark on the same DAILY bars.
        daily_feat = add_per_asset_features(daily_bars)
        d_start = daily_bars.index.get_indexer([start_ts], method="bfill")[0]
        d_end = daily_bars.index.get_indexer([end_ts], method="ffill")[0] + 1
        d_end = min(d_end, len(daily_bars))
        if d_start <= WARMUP_DAILY:
            d_start = WARMUP_DAILY
        us = uniform_stop_benchmark(daily_bars, daily_feat, d_start, d_end,
                                     NOTIONAL, cost_bp, trail_sigma=2.0)

        bot_total += result["net"]
        bh_total += bh["net"]
        us_total += us["net"]
        bot_trades += result["n_trades"]
        cost_total += result["total_cost"]
        n_deployed += 1

        alpha_bh = result["net"] - bh["net"]
        alpha_us = result["net"] - us["net"]
        cfg = f"{result_sigmode} tr={trail}σ pt={f'{pt}σ' if pt else '—'}"
        print(f"  {asset:<6} {cfg:<28} "
              f"{result['n_trades']:<7} ${result['net']:>+8,.0f}  "
              f"${bh['net']:>+8,.0f}  ${us['net']:>+8,.0f}    "
              f"${alpha_bh:>+8,.0f}  ${alpha_us:>+8,.0f}")

    portfolio_notional = n_deployed * NOTIONAL
    alpha_bh_total = bot_total - bh_total
    alpha_us_total = bot_total - us_total
    print("-" * 138)
    print(f"  BOT          net=${bot_total:+,.0f}  ({bot_total/portfolio_notional*100:+.2f}%)  "
          f"trades={bot_trades}  costs=${cost_total:,.0f}")
    print(f"  BUY&HOLD     net=${bh_total:+,.0f}  ({bh_total/portfolio_notional*100:+.2f}%)")
    print(f"  UNIF-STOP-2σ net=${us_total:+,.0f}  ({us_total/portfolio_notional*100:+.2f}%)  "
          f"← falsification null: long basket + generic 2σ trailing stop")
    print(f"  ALPHA vs B&H        ${alpha_bh_total:+,.0f}  ({alpha_bh_total/portfolio_notional*100:+.2f}%)")
    print(f"  ALPHA vs UnifStop   ${alpha_us_total:+,.0f}  ({alpha_us_total/portfolio_notional*100:+.2f}%)"
          f"  ← THIS is the agent-required skill measure")
    return {"bot_net": bot_total, "bh_net": bh_total, "us_net": us_total,
            "alpha_bh": alpha_bh_total, "alpha_us": alpha_us_total,
            "trades": bot_trades, "costs": cost_total,
            "n_deployed": n_deployed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bp", type=float, default=3.0,
                    help="cost per side in bps (default 3 = 1.5bp spread + 1.5bp slippage)")
    args = ap.parse_args()

    print()
    print(f"  Honest replay  •  net_liq=${NET_LIQ:,.0f}  per-name=${NOTIONAL:,.0f}  "
          f"cost={args.cost_bp}bp/side ({args.cost_bp*2}bp round-trip)")

    results = {}
    for label, start, end, allow_intraday in WINDOWS:
        results[label] = evaluate_window(label, start, end, allow_intraday,
                                          args.cost_bp)

    # Final summary.
    print()
    print("=" * 130)
    print(f"  HONEST SUMMARY across {len(WINDOWS)} windows (cost={args.cost_bp}bp/side)")
    print("=" * 130)
    print(f"  {'window':<12} {'bot net':<12} {'B&H net':<12} {'UnifStop net':<14} "
          f"{'α vs B&H':<12} {'α vs UnifStop':<14}")
    print("-" * 138)
    for label, _, _, _ in WINDOWS:
        r = results[label]
        pn = r["n_deployed"] * NOTIONAL
        print(f"  {label:<12} ${r['bot_net']:>+9,.0f}  ${r['bh_net']:>+9,.0f}  "
              f"${r['us_net']:>+9,.0f}     "
              f"${r['alpha_bh']:>+9,.0f}  ${r['alpha_us']:>+9,.0f}")
    print("=" * 138)
    print()
    print("  READING:")
    print("  • α vs B&H = bot − naive buy-and-hold. Tests 'long bias' explanation.")
    print("  • α vs UnifStop = bot − (long + uniform 2σ trailing stop). Tests")
    print("    the falsification agent's null: if the bot can't beat a generic")
    print("    stop-loss overlay, the per-asset 'skill' is just the stops.")
    print("  • Intraday Bollinger PAUSED post-agent-review. AAPL/MSFT now run")
    print("    their daily-bar comprehensive-search-best configs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
