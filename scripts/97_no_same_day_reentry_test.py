"""97_no_same_day_reentry_test — empirical test of "no same-day re-entry" rule.

Hypothesis: After exiting an asset on day D, blocking re-entry on day D
prevents intra-day whipsaw losses (e.g. yesterday's AMZN -$187 round-trip:
in 14:35, out 15:05) without giving up next-day opportunity.

Mirrors 87_fixed_replay.py but tracks per-asset `last_exit_date` in state
and blocks any entry whose bar-date == last_exit_date.

Tested on OOS / COVID / RATESHOCK at 3 bp. Uses the CURRENT_V2 configs
deployed today.

Run:
    .venv/bin/python scripts/97_no_same_day_reentry_test.py
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

# Mirrors 87_fixed_replay.py — the currently deployed v2 sleeve.
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

WINDOWS = [
    ("OOS",       pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-20")),
    ("COVID",     pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
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
    raise ValueError(mode)


def replay(bars, feat, sig, start_idx, end_idx, trail, pt, cost_bp,
           no_same_day_reentry=False):
    """Replay with optional no-same-day-reentry rule.

    Returns dict with summary stats AND a `blocked_entries` list of
    (date, asset_close) for entries that the rule suppressed, so we can
    measure the rule's side-effect (positions it would have prevented
    that would have been winners).
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

    # No-same-day-reentry state
    last_exit_date = None  # pd.Timestamp (normalized) | None
    blocked_entries = []   # list of (date, close, sig_dir) — entries that the rule suppressed

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
                cost = entry_qty * close * cost_frac
                rt.append({"pnl": gross - cost, "reason": reason, "exit_date": dk})
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                last_exit_date = dk  # arm same-day-reentry block
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
            reason = None
            if close >= stop:
                reason = "trail"
            elif pt is not None and close <= entry_price * (1 - entry_sigma * pt):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                gross = (entry_price - close) * entry_qty
                cost = entry_qty * close * cost_frac
                rt.append({"pnl": gross - cost, "reason": reason, "exit_date": dk})
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                last_exit_date = dk
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty, entry_side = 0, 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        if state == "FLAT" and sig_dir != 0:
            # No-same-day-reentry check
            if no_same_day_reentry and last_exit_date is not None and dk == last_exit_date:
                blocked_entries.append((dk, close, sig_dir, i))
            else:
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
    wins = sum(1 for r in rt if r["pnl"] > 0)
    win_rate = (wins / len(rt) * 100.0) if rt else 0.0
    sharpe = float("nan")
    max_dd = 0.0
    if daily_bucket:
        rets = np.asarray(list(daily_bucket.values())) / NOTIONAL
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    n_sessions = len(daily_bucket)
    years = n_sessions / 252.0 if n_sessions else 0.0
    ann = float("nan")
    if years > 0 and NOTIONAL + net > 0:
        ann = ((1 + net / NOTIONAL) ** (1 / years) - 1) * 100.0
    return {"n_trades": len(rt), "wins": wins, "win_rate": win_rate,
            "net": net, "sharpe": sharpe, "max_dd": max_dd, "ann": ann,
            "blocked_entries": blocked_entries, "rt": rt}


def run_portfolio(configs, no_same_day_reentry):
    """Returns dict {window_label: {portfolio stats, per-asset breakdown}}."""
    results = {}
    for label, start, end in WINDOWS:
        portfolio_net = 0.0
        portfolio_trades = 0
        portfolio_wins = 0
        portfolio_blocked = 0
        per_asset = {}
        for asset, mode, trail, pt, lib, sym in configs:
            bars = load_bars(lib, sym)
            feat = add_per_asset_features(bars)
            sig = precompute_signal(asset, bars, feat, mode)
            s = bars.index.get_indexer([start], method="bfill")[0]
            e = bars.index.get_indexer([end], method="ffill")[0] + 1
            if s <= WARMUP or e <= s:
                continue
            r = replay(bars, feat, sig, s, e, trail, pt, COST_BP,
                       no_same_day_reentry=no_same_day_reentry)
            portfolio_net += r["net"]
            portfolio_trades += r["n_trades"]
            portfolio_wins += r["wins"]
            portfolio_blocked += len(r["blocked_entries"])
            per_asset[asset] = r
        wr = (portfolio_wins / portfolio_trades * 100.0) if portfolio_trades else 0.0
        ret = portfolio_net / NET_LIQ * 100.0
        years = (end - start).days / 365.25
        ann = float("nan")
        if years > 0 and NET_LIQ + portfolio_net > 0:
            ann = ((1 + portfolio_net / NET_LIQ) ** (1 / years) - 1) * 100.0
        results[label] = {"net": portfolio_net, "trades": portfolio_trades, "wins": portfolio_wins,
                          "wr": wr, "ret": ret, "ann": ann, "blocked": portfolio_blocked,
                          "per_asset": per_asset}
    return results


def hypothetical_blocked_pnl(asset, configs, baseline_results):
    """For each entry the rule blocked, simulate what the position WOULD have
    earned by replaying from that entry under the same exit rules. Returns
    {window: list of hypothetical realised P&L for blocked entries}.
    """
    # Find config for asset
    cfg = next((c for c in configs if c[0] == asset), None)
    if cfg is None:
        return {}
    _, mode, trail, pt, lib, sym = cfg
    bars = load_bars(lib, sym)
    feat = add_per_asset_features(bars)
    sig = precompute_signal(asset, bars, feat, mode)
    cost_frac = COST_BP / 1e4
    closes = bars["close"].astype(float).values
    vol = feat["vol_21"].values

    results = {}
    for window, start, end in WINDOWS:
        s = bars.index.get_indexer([start], method="bfill")[0]
        e = bars.index.get_indexer([end], method="ffill")[0] + 1
        baseline_blocked = baseline_results[window]["per_asset"].get(asset, {}).get("blocked_entries", [])
        hypos = []
        for dk, close_entry, sig_dir, entry_idx in baseline_blocked:
            # Simulate a position starting at entry_idx with the blocked signal
            # using same trail/pt rules, no further re-entry logic
            sigma_e = float(vol[entry_idx])
            if not math.isfinite(sigma_e) or sigma_e <= 0:
                continue
            qty = int(NOTIONAL // close_entry) if close_entry > 0 else 0
            if qty <= 0:
                continue
            side = 1 if sig_dir > 0 else -1
            entry_cost = qty * close_entry * cost_frac
            peak = close_entry if side > 0 else None
            trough = close_entry if side < 0 else None
            pnl = -entry_cost
            exited = False
            for j in range(entry_idx + 1, e):
                c = float(closes[j])
                s_dir = int(sig[j])
                if side > 0:
                    if peak is None or c > peak:
                        peak = c
                    stop = peak - close_entry * sigma_e * trail
                    if c <= stop or s_dir < 0 or (pt is not None and c >= close_entry * (1 + sigma_e * pt)):
                        gross = (c - close_entry) * qty
                        exit_cost = qty * c * cost_frac
                        pnl += gross - exit_cost
                        exited = True
                        break
                else:
                    if trough is None or c < trough:
                        trough = c
                    stop = trough + close_entry * sigma_e * trail
                    if c >= stop or s_dir > 0 or (pt is not None and c <= close_entry * (1 - sigma_e * pt)):
                        gross = (close_entry - c) * qty
                        exit_cost = qty * c * cost_frac
                        pnl += gross - exit_cost
                        exited = True
                        break
            if not exited:
                # Mark-to-end-of-window
                last_c = float(closes[e - 1])
                gross = (last_c - close_entry) * qty * side
                pnl += gross
            hypos.append(pnl)
        results[window] = hypos
    return results


def main():
    print()
    print("=" * 130)
    print("  No-same-day-reentry test (v2 deployed sleeve, 3bp cost, OOS/COVID/RATESHOCK)")
    print("=" * 130)

    print("\n  Running CURRENT v2 (baseline, no rule)...")
    base = run_portfolio(CURRENT_V2, no_same_day_reentry=False)
    print("  Running CURRENT v2 + no-same-day-reentry...")
    rule = run_portfolio(CURRENT_V2, no_same_day_reentry=True)

    print()
    print("=" * 130)
    print(f"  {'window':<12} | {'baseline net':>14} {'#tr':>5} {'WR':>6} | "
          f"{'with-rule net':>14} {'#tr':>5} {'WR':>6} {'blocked':>8} | "
          f"{'Δ net':>11}  {'Δ #tr':>7}")
    print("-" * 130)
    for window in ["OOS", "COVID", "RATESHOCK"]:
        b, r = base[window], rule[window]
        d_net = r["net"] - b["net"]
        d_tr  = r["trades"] - b["trades"]
        print(f"  {window:<12} | ${b['net']:>+12,.0f} {b['trades']:>5} {b['wr']:>5.1f}% | "
              f"${r['net']:>+12,.0f} {r['trades']:>5} {r['wr']:>5.1f}% {r['blocked']:>8} | "
              f"${d_net:>+9,.0f}  {d_tr:>+7}")

    b_total = sum(base[w]["net"] for w in base)
    r_total = sum(rule[w]["net"] for w in rule)
    b_tr    = sum(base[w]["trades"] for w in base)
    r_tr    = sum(rule[w]["trades"] for w in rule)
    print("-" * 130)
    print(f"  {'TOTAL':<12} | ${b_total:>+12,.0f} {b_tr:>5}        | "
          f"${r_total:>+12,.0f} {r_tr:>5}                 | "
          f"${r_total - b_total:>+9,.0f}  {r_tr - b_tr:>+7}")
    print("=" * 130)

    # Per-asset breakdown
    print()
    print("  Per-asset Δ (net = with-rule minus baseline):")
    print(f"  {'asset':<6} | {'window':<11} {'base':>11} {'rule':>11} {'Δ':>11} "
          f"{'base #tr':>9} {'rule #tr':>9} {'blocked':>8}")
    print("-" * 95)
    for asset, *_ in CURRENT_V2:
        for window in ["OOS", "COVID", "RATESHOCK"]:
            ba = base[window]["per_asset"].get(asset, None)
            ra = rule[window]["per_asset"].get(asset, None)
            if ba is None or ra is None:
                continue
            print(f"  {asset:<6} | {window:<11} ${ba['net']:>+9,.0f} ${ra['net']:>+9,.0f} "
                  f"${ra['net'] - ba['net']:>+9,.0f} {ba['n_trades']:>9} {ra['n_trades']:>9} "
                  f"{len(ra['blocked_entries']):>8}")
        print()

    # Side effect: how would blocked entries have done?
    print()
    print("  SIDE EFFECT — blocked entries simulated forward under same trail/pt rules:")
    print(f"  {'asset':<6} | {'window':<11} {'#blocked':>9} {'avg P&L':>10} {'tot P&L':>11} "
          f"{'wins':>5} {'losses':>7}")
    print("-" * 80)
    asset_window_blocked_pnl = {}
    for asset, *_ in CURRENT_V2:
        hypos = hypothetical_blocked_pnl(asset, CURRENT_V2, rule)
        asset_window_blocked_pnl[asset] = hypos
        for window in ["OOS", "COVID", "RATESHOCK"]:
            hs = hypos.get(window, [])
            if not hs:
                continue
            avg = sum(hs) / len(hs)
            wins = sum(1 for h in hs if h > 0)
            losses = len(hs) - wins
            print(f"  {asset:<6} | {window:<11} {len(hs):>9} ${avg:>+8,.0f} ${sum(hs):>+9,.0f} "
                  f"{wins:>5} {losses:>7}")

    print()
    print("=" * 130)
    print("  Recommendation:")
    delta_total = r_total - b_total
    if delta_total > 0 and r_tr < b_tr:
        rec = "DEPLOY — rule improves P&L while reducing trades"
    elif delta_total > 0:
        rec = "DEPLOY — rule improves P&L"
    elif delta_total >= 0 and r_tr < b_tr:
        rec = "DEPLOY — same P&L, lower trade count = lower variance/cost"
    elif delta_total < 0 and abs(delta_total) < 500:
        rec = "MARGINAL — tiny degradation; depends on out-of-sample stability"
    else:
        rec = "DO NOT DEPLOY — material P&L degradation"
    print(f"    Δ total P&L: ${delta_total:+,.0f}   Δ trades: {r_tr - b_tr:+}")
    print(f"    → {rec}")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
