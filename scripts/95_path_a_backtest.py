"""95_path_a_backtest — empirical test of Path A: drop AMZN/GOOGL/TSLA, add SOXL/NVDL/TSLL.

Path A universe (8 assets, $12.5k each = $100k):
    Kept from v2: AAPL, MSFT, META, NVDA, GLD
    Added:        SOXL, NVDL, TSLL

Each asset uses its bake-off / search winner config:
    AAPL  buy_and_hold     trail 3.0σ  pt 3.0σ
    MSFT  rsi2_mr_10_90    trail 1.0σ  pt 5.0σ
    META  lgbm             trail 2.0σ  pt None
    NVDA  buy_and_hold     trail 2.0σ  pt None
    GLD   buy_and_hold     trail 2.0σ  pt 3.0σ
    SOXL  rsi2_mr_10_90    trail 2.0σ  pt 3.0σ   (agent 3, soxl.py WINNER)
    NVDL  rsi2_mr_10_90    trail 0.5σ  pt 5.0σ   (agent 3, nvdl.py WINNER)
    TSLL  rsi2_mr_10_90    trail 1.0σ  pt 3.0σ   (agent 3 bake-off; no module WINNER yet)

Windows: OOS (2026-01-01 → 2026-05-20), COVID, RATESHOCK at 3bp.

CAVEAT: SOXL/NVDL/TSLL daily bars start 2024-01-02 — they cover OOS only,
not COVID/RATESHOCK. Path A is therefore costed on OOS in full; for COVID
and RATESHOCK the leveraged sleeve contributes $0 (no data, no positions).
This is the honest replay; treat it as a lower bound on crisis exposure.

Run:
    .venv/bin/python scripts/95_path_a_backtest.py
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
from ml_research.asset_specialists._signals import (  # noqa: E402
    add_per_asset_features,
    signal_rsi2_mr_10_90,
)
from scripts._storage import open_store  # noqa: E402

NET_LIQ = 100000.0
NOTIONAL = NET_LIQ / 8.0  # $12,500
WARMUP = 252
COST_BP = 3.0

# v2 baseline — same as scripts/87_fixed_replay CURRENT_V2.
V2_BASELINE = [
    ("AAPL",  "specialist",    3.0, 3.0,  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT",  "specialist",    1.0, 5.0,  "equity.daily_bars",  "NASDAQ.MSFT"),
    ("GOOGL", "specialist",    2.0, None, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0,  "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "specialist",    2.0, None, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "specialist",    0.5, 3.0,  "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "specialist",    2.0, None, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  "etf.daily_bars",     "ARCA.GLD"),
]

# Path A — drop AMZN/GOOGL/TSLA, add SOXL/NVDL/TSLL.
PATH_A = [
    ("AAPL", "specialist",   3.0, 3.0,  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT", "specialist",   1.0, 5.0,  "equity.daily_bars",  "NASDAQ.MSFT"),
    ("META", "specialist",   2.0, None, "equity.daily_bars",  "NASDAQ.META"),
    ("NVDA", "specialist",   2.0, None, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",  "buy_and_hold", 2.0, 3.0,  "etf.daily_bars",     "ARCA.GLD"),
    ("SOXL", "rsi2_mr",      2.0, 3.0,  "etf.daily_bars",     "ARCA.SOXL"),
    ("NVDL", "rsi2_mr",      0.5, 5.0,  "etf.daily_bars",     "ARCA.NVDL"),
    ("TSLL", "rsi2_mr",      1.0, 3.0,  "etf.daily_bars",     "ARCA.TSLL"),
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
    """Precompute the entry signal array for `bars`.

    Modes:
      'specialist'  — dispatch to ml_research.asset_specialists.compute_signal_for
      'buy_and_hold' — always +1 after warmup
      'rsi2_mr'     — RSI(2) 10/90 mean-reversion (used for SOXL/NVDL/TSLL)
    """
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "specialist":
        for i in range(WARMUP, n):
            try:
                out[i] = int(round(float(compute_signal_for(asset, bars.iloc[: i + 1]))))
            except Exception:
                pass
        return out
    if mode == "buy_and_hold":
        out[WARMUP:] = 1
        return out
    if mode == "rsi2_mr":
        # Vectorised RSI(2) 10/90 — same as signal_rsi2_mr_10_90 but for the
        # whole series at once (faster than per-bar evaluation).
        rsi_prev = feat["rsi_2"].shift(1)
        sig = pd.Series(0, index=feat.index, dtype="int8")
        sig = sig.mask(rsi_prev < 10.0, 1).mask(rsi_prev > 90.0, -1)
        arr = sig.fillna(0).astype("int8").values
        # Suppress signals during warmup so feature stability holds.
        arr[:WARMUP] = 0
        return arr
    raise ValueError(mode)


def replay(bars, feat, sig, start_idx, end_idx, trail, pt, cost_bp):
    """Replay long+short bracket strategy on `bars` with trailing stop + optional PT.

    Direct copy of scripts/87_fixed_replay.py::replay with profit_lock disabled.
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
            if close <= stop:
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
    wins = sum(1 for r in rt if r["pnl"] > 0)
    win_rate = (wins / len(rt) * 100.0) if rt else 0.0
    return {
        "n_trades": len(rt),
        "wins": wins,
        "win_rate": win_rate,
        "net": net,
        "realized": realized,
        "open_unrz": open_unrz,
    }


def run_portfolio(configs, label="config"):
    """Returns dict of dicts:
        {window_label: {asset: {net, n_trades, ...}, '_total': portfolio_net}}
    """
    out = {}
    for win_label, start, end in WINDOWS:
        per_asset = {}
        total_net = 0.0
        for asset, mode, trail, pt, lib, sym in configs:
            try:
                bars = load_bars(lib, sym)
            except Exception as e:
                per_asset[asset] = {"net": 0.0, "n_trades": 0,
                                    "note": f"no data: {e}"}
                continue
            feat = add_per_asset_features(bars)

            # Window slicing — fail closed if data doesn't cover.
            if bars.index.max() < start or bars.index.min() > end:
                per_asset[asset] = {"net": 0.0, "n_trades": 0,
                                    "note": f"data {bars.index.min().date()}–{bars.index.max().date()} outside window"}
                continue

            # Need WARMUP + window. If first bar > start - WARMUP, then warmup
            # is incomplete for the window start; we still let it run but the
            # signal stays 0 until index ≥ WARMUP, so trades begin late.
            sig = precompute_signal(asset, bars, feat, mode)
            s = bars.index.get_indexer([start], method="bfill")[0]
            e = bars.index.get_indexer([end], method="ffill")[0] + 1
            if s < 0 or e <= s:
                per_asset[asset] = {"net": 0.0, "n_trades": 0,
                                    "note": "window outside bars"}
                continue
            if s <= WARMUP:
                # Bump start past warmup so signal is defined.
                s = WARMUP
                if s >= e:
                    per_asset[asset] = {"net": 0.0, "n_trades": 0,
                                        "note": f"warmup ({WARMUP} bars) overlaps window — skipping"}
                    continue
            r = replay(bars, feat, sig, s, e, trail, pt, COST_BP)
            per_asset[asset] = r
            total_net += r["net"]
        per_asset["_total"] = total_net
        out[win_label] = per_asset
    return out


def fmt_dollar(x):
    return f"${x:>+9,.0f}"


def main():
    print()
    print("=" * 130)
    print("  PATH A BACKTEST — drop AMZN/GOOGL/TSLA, add SOXL/NVDL/TSLL")
    print("=" * 130)
    print(f"  Net liq: ${NET_LIQ:,.0f}   notional/name: ${NOTIONAL:,.0f}   cost: {COST_BP:.1f}bp")
    print(f"  WARMUP: {WARMUP} bars")

    # Sanity check the leveraged universe exists.
    store = open_store()
    etf_syms = store.get_library("etf.daily_bars").list_symbols()
    for sym in ("ARCA.SOXL", "ARCA.NVDL", "ARCA.TSLL"):
        if sym not in etf_syms:
            print(f"  FATAL: {sym} not in etf.daily_bars")
            return 1
    print("  Verified: ARCA.SOXL, ARCA.NVDL, ARCA.TSLL all present in etf.daily_bars")

    print("\n  Running v2 baseline...")
    v2 = run_portfolio(V2_BASELINE, label="v2")
    print("  Running Path A (5 v2-keeps + SOXL/NVDL/TSLL)...")
    pa = run_portfolio(PATH_A, label="path_a")

    # Per-asset breakdown per window.
    print()
    print("=" * 130)
    print("  PER-ASSET NET P&L PER WINDOW")
    print("=" * 130)
    for win in ("OOS", "COVID", "RATESHOCK"):
        print(f"\n  --- {win} ---")
        v2_assets = sorted([a for a in v2[win] if a != "_total"])
        pa_assets = sorted([a for a in pa[win] if a != "_total"])
        union = sorted(set(v2_assets) | set(pa_assets))
        print(f"    {'asset':<8} {'v2_net':<12} {'pa_net':<12} {'trades_v2':<10} {'trades_pa':<10}")
        for asset in union:
            v2_r = v2[win].get(asset, {})
            pa_r = pa[win].get(asset, {})
            v2_net = v2_r.get("net", 0.0)
            pa_net = pa_r.get("net", 0.0)
            v2_tr = v2_r.get("n_trades", 0)
            pa_tr = pa_r.get("n_trades", 0)
            note = ""
            if "note" in pa_r:
                note = f"  ({pa_r['note']})"
            elif "note" in v2_r:
                note = f"  ({v2_r['note']})"
            v2_str = fmt_dollar(v2_net) if asset in v2_assets else "    -    "
            pa_str = fmt_dollar(pa_net) if asset in pa_assets else "    -    "
            print(f"    {asset:<8} {v2_str:<12} {pa_str:<12} {v2_tr:<10} {pa_tr:<10}{note}")

    # Portfolio summary.
    print()
    print("=" * 130)
    print("  PORTFOLIO TOTAL vs V2 BASELINE")
    print("=" * 130)
    print(f"    {'window':<14} {'v2_net':<14} {'path_a_net':<14} {'delta':<14} {'beat?':<8}")
    print("    " + "-" * 70)
    pa_better_windows = []
    for win in ("OOS", "COVID", "RATESHOCK"):
        v2_total = v2[win]["_total"]
        pa_total = pa[win]["_total"]
        delta = pa_total - v2_total
        beat = "YES" if delta > 0 else "no"
        if delta > 0:
            pa_better_windows.append(win)
        print(f"    {win:<14} {fmt_dollar(v2_total):<14} {fmt_dollar(pa_total):<14} "
              f"{fmt_dollar(delta):<14} {beat:<8}")

    # Sum across the 3 windows.
    v2_grand = sum(v2[w]["_total"] for w in v2)
    pa_grand = sum(pa[w]["_total"] for w in pa)
    delta_grand = pa_grand - v2_grand
    print(f"    {'3-window sum':<14} {fmt_dollar(v2_grand):<14} {fmt_dollar(pa_grand):<14} "
          f"{fmt_dollar(delta_grand):<14} {'YES' if delta_grand > 0 else 'no':<8}")

    # Crisis behavior — which sleeve drove RATESHOCK?
    print()
    print("=" * 130)
    print("  CRISIS BEHAVIOR — RATESHOCK leveraged-sleeve attribution")
    print("=" * 130)
    for asset in ("SOXL", "NVDL", "TSLL"):
        r = pa["RATESHOCK"].get(asset, {})
        if "note" in r:
            print(f"    {asset}: NO DATA — {r['note']}")
        else:
            print(f"    {asset}: net {fmt_dollar(r.get('net', 0))}, trades {r.get('n_trades', 0)}")
    rs_pa = pa["RATESHOCK"]["_total"]
    rs_v2 = v2["RATESHOCK"]["_total"]
    print(f"    RATESHOCK delta vs v2: {fmt_dollar(rs_pa - rs_v2)}")

    # Recommendation.
    print()
    print("=" * 130)
    print("  RECOMMENDATION")
    print("=" * 130)
    if delta_grand > 0 and "RATESHOCK" not in pa_better_windows:
        print("    Path A beats v2 on 3-window sum but not on RATESHOCK — proceed with caution.")
    elif delta_grand > 0 and "RATESHOCK" in pa_better_windows:
        print("    Path A beats v2 on 3-window sum AND survives RATESHOCK — recommend deploy.")
    elif delta_grand <= 0:
        print("    Path A does NOT beat v2 on 3-window sum — DO NOT deploy.")
    print(f"    Path A better on: {pa_better_windows or 'no windows'}")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
