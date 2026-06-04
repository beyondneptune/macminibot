"""88_profit_lock_sweep — sensitivity sweep of Fix B profit-lock threshold.

Reuses replay() from 87_fixed_replay.py — only sweeps `profit_lock_sigmas`
across {0.5, 1.0, 1.5, 2.0, 3.0} plus a 0.0 (== disabled / baseline) anchor.

Reports per-threshold portfolio:
  - total net P&L on OOS window (2026-01-01 → 2026-05-20)
  - number of round-trip trades + exit-reason mix (trail / pt / flip)
  - win rate
  - max DD (sum-of-asset, conservative aggregation)
  - annualized return
  - avg loss-on-trail (gives us the "small-loss trail" check at high N)

Run:
    .venv/bin/python scripts/88_profit_lock_sweep.py
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

# Deployed v2 config (matches 87_fixed_replay.CURRENT_V2)
CURRENT_V2 = [
    ("AAPL",  "buy_and_hold",  3.0, 3.0,  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT",  "baseline",      1.0, 5.0,  "equity.daily_bars",  "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0,  "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      0.5, 3.0,  "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  "etf.daily_bars",     "ARCA.GLD"),
]

OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-05-20")

THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


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


def replay_detailed(bars, feat, sig, start_idx, end_idx, trail, pt, cost_bp,
                    profit_lock_enabled=False, profit_lock_sigmas=1.0):
    """Replay that ALSO returns per-trade exit reasons + daily P&L stream.

    Mirrors logic of scripts/87_fixed_replay.py::replay verbatim — only the
    returned dict is richer (rt list + daily_bucket) so the caller can compute
    portfolio-level Sharpe / DD / exit-mix.
    """
    cost_frac = cost_bp / 1e4
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    entry_side = 0
    peak = trough = None
    rt = []
    daily_bucket = {}
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
            profit_now_sigmas = (close - entry_price) / (entry_price * entry_sigma)
            trail_active = (not profit_lock_enabled) or (profit_now_sigmas >= profit_lock_sigmas)
            if trail_active and close <= stop:
                reason = "trail"
            elif pt is not None and close >= entry_price * (1 + entry_sigma * pt):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                gross = (close - entry_price) * entry_qty
                cost = entry_qty * close * cost_frac
                rt.append({"pnl": gross - cost, "reason": reason})
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty, entry_side = 0, 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            band = entry_price * entry_sigma * trail
            stop = trough + band
            profit_now_sigmas = (entry_price - close) / (entry_price * entry_sigma)
            trail_active = (not profit_lock_enabled) or (profit_now_sigmas >= profit_lock_sigmas)
            reason = None
            if trail_active and close >= stop:
                reason = "trail"
            elif pt is not None and close <= entry_price * (1 - entry_sigma * pt):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                gross = (entry_price - close) * entry_qty
                cost = entry_qty * close * cost_frac
                rt.append({"pnl": gross - cost, "reason": reason})
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

    realized = sum(r["pnl"] for r in rt)
    net = realized + open_unrz
    return {"rt": rt, "daily_bucket": daily_bucket, "net": net,
            "realized": realized, "open_unrz": open_unrz}


def aggregate_portfolio(per_asset_results, years):
    """Aggregate per-asset replay_detailed outputs into a portfolio summary."""
    all_trades = []
    combined_daily = {}
    portfolio_net = 0.0
    for r in per_asset_results:
        all_trades.extend(r["rt"])
        portfolio_net += r["net"]
        for dk, v in r["daily_bucket"].items():
            combined_daily[dk] = combined_daily.get(dk, 0.0) + v

    n_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    win_rate = (wins / n_trades * 100.0) if n_trades else 0.0
    exits = {"trail": 0, "pt": 0, "flip": 0}
    trail_pnls = []
    for t in all_trades:
        exits[t["reason"]] = exits.get(t["reason"], 0) + 1
        if t["reason"] == "trail":
            trail_pnls.append(t["pnl"])

    # Sharpe + DD from combined daily P&L stream
    sharpe = float("nan")
    max_dd = 0.0
    if combined_daily:
        sorted_keys = sorted(combined_daily.keys())
        daily_pnl = np.asarray([combined_daily[k] for k in sorted_keys])
        rets = daily_pnl / NET_LIQ
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(daily_pnl)
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())

    ann = float("nan")
    if years > 0 and NET_LIQ + portfolio_net > 0:
        ann = ((1 + portfolio_net / NET_LIQ) ** (1 / years) - 1) * 100.0

    avg_trail_loss = float("nan")
    losing_trails = [p for p in trail_pnls if p < 0]
    if losing_trails:
        avg_trail_loss = float(np.mean(losing_trails))
    avg_trail_pnl = float(np.mean(trail_pnls)) if trail_pnls else float("nan")

    return {
        "net": portfolio_net,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "exits": exits,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann": ann,
        "avg_trail_pnl": avg_trail_pnl,
        "avg_trail_loss": avg_trail_loss,
        "n_losing_trails": len(losing_trails),
    }


def main():
    print()
    print("=" * 110)
    print("  Profit-lock sensitivity sweep — OOS window 2026-01-01 → 2026-05-20")
    print(f"  Config: deployed v2 (8 assets), NET_LIQ=${NET_LIQ:,.0f}, COST_BP={COST_BP}")
    print("=" * 110)

    # Precompute bars/feat/sig ONCE per asset — independent of profit_lock
    print("\n  Precomputing bars + signals (one pass per asset)...")
    precomputed = []
    for asset, mode, trail, pt, lib, sym in CURRENT_V2:
        bars = load_bars(lib, sym)
        feat = add_per_asset_features(bars)
        sig = precompute_signal(asset, bars, mode)
        s = bars.index.get_indexer([OOS_START], method="bfill")[0]
        e = bars.index.get_indexer([OOS_END], method="ffill")[0] + 1
        precomputed.append({
            "asset": asset, "bars": bars, "feat": feat, "sig": sig,
            "s": s, "e": e, "trail": trail, "pt": pt,
        })
        print(f"    {asset:<6} bars={len(bars):>5} s={s} e={e} window_days={e-s}")

    years = (OOS_END - OOS_START).days / 365.25

    summaries = []
    for lock in THRESHOLDS:
        enabled = lock > 0.0
        per_asset = []
        for pc in precomputed:
            if pc["s"] <= WARMUP or pc["e"] <= pc["s"]:
                continue
            r = replay_detailed(pc["bars"], pc["feat"], pc["sig"],
                                pc["s"], pc["e"], pc["trail"], pc["pt"], COST_BP,
                                profit_lock_enabled=enabled,
                                profit_lock_sigmas=lock)
            per_asset.append(r)
        summ = aggregate_portfolio(per_asset, years)
        summ["lock"] = lock
        summaries.append(summ)

    print()
    print("=" * 110)
    print("  RESULTS")
    print("=" * 110)
    print(f"  {'lock(σ)':<9} {'trades':<7} {'win%':<7} {'net $':<11} {'ann %':<8} "
          f"{'maxDD $':<10} {'Sharpe':<7} | {'trail':<6} {'pt':<5} {'flip':<5} "
          f"{'avg trail $':<12} {'avg trail-loss $':<16}")
    print("-" * 110)
    for s in summaries:
        baseline_tag = "  ← baseline" if s["lock"] == 0.0 else ""
        print(f"  {s['lock']:<9.1f} {s['n_trades']:<7d} {s['win_rate']:<6.1f}% "
              f"${s['net']:>+8,.0f}  {s['ann']:>+5.1f}%  "
              f"${s['max_dd']:>+7,.0f}  {s['sharpe']:>+5.2f}  | "
              f"{s['exits']['trail']:<6d} {s['exits']['pt']:<5d} {s['exits']['flip']:<5d} "
              f"${s['avg_trail_pnl']:>+8,.0f}    "
              f"${s['avg_trail_loss']:>+8,.0f} (n={s['n_losing_trails']}){baseline_tag}")

    # Δ vs baseline
    base = next(s for s in summaries if s["lock"] == 0.0)
    print()
    print("  Δ vs baseline (lock=0.0):")
    print(f"  {'lock(σ)':<9} {'Δ net $':<11} {'Δ trades':<9} {'Δ trail-exits':<14} {'Δ ann %':<8}")
    print("-" * 60)
    for s in summaries:
        if s["lock"] == 0.0:
            continue
        d_net = s["net"] - base["net"]
        d_tr = s["n_trades"] - base["n_trades"]
        d_te = s["exits"]["trail"] - base["exits"]["trail"]
        d_ann = s["ann"] - base["ann"]
        print(f"  {s['lock']:<9.1f} ${d_net:>+8,.0f}  {d_tr:>+9d} {d_te:>+14d} {d_ann:>+6.2f}%")

    # Sweet spots
    best_net = max(summaries, key=lambda x: x["net"])
    valid_sharpe = [s for s in summaries if math.isfinite(s["sharpe"])]
    best_sharpe = max(valid_sharpe, key=lambda x: x["sharpe"]) if valid_sharpe else None
    print()
    print("  Sweet spot:")
    print(f"    Max NET: lock={best_net['lock']}σ  net=${best_net['net']:+,.0f}  ann={best_net['ann']:+.2f}%")
    if best_sharpe:
        print(f"    Max Sharpe: lock={best_sharpe['lock']}σ  Sharpe={best_sharpe['sharpe']:+.2f}  net=${best_sharpe['net']:+,.0f}")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
