"""80_deployed_train_test — TRAIN + TEST report for every deployed config
in the live hybrid bot.

For each of the 8 deployed configs (2 intraday + 6 daily), replay the same
state machine on:
  TRAIN: warmup → 2026-01-01
  TEST : 2026-01-01 → 2026-05-19  (~4.5 months OOS)

Same per-name notional ($12,500) and same split as scripts 77+78+79. Shows
both halves so the operator can see (a) what the bake-off was promoting on
and (b) what the OOS evidence is for the deployed picks.

Run:
    .venv/bin/python scripts/80_deployed_train_test.py
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
SPLIT = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-05-19")

WARMUP_DAILY = 252
WARMUP_INTRADAY = 500

# Reflects the active hybrid deployment (memory: project_hybrid_deployment.md).
# Each entry: (signal_mode, trail_sigma, pt_sigmas, substrate, lib, sym, runner)
DEPLOYED_CONFIGS = [
    ("AAPL",  "bollinger",     3.0, 3.0, "intraday", "equity.minute_bars", "NASDAQ.AAPL", "intraday"),
    ("MSFT",  "bollinger",     3.0, 3.0, "intraday", "equity.minute_bars", "NASDAQ.MSFT", "intraday"),
    ("GOOGL", "baseline",      2.0, None, "daily",    "equity.daily_bars",  "NASDAQ.GOOGL","daily"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0, "daily",    "equity.daily_bars",  "NASDAQ.AMZN", "daily"),
    ("META",  "baseline",      2.0, None, "daily",    "equity.daily_bars",  "NASDAQ.META", "daily"),
    ("TSLA",  "baseline",      0.5, 3.0, "daily",    "equity.daily_bars",  "NASDAQ.TSLA", "daily"),
    ("NVDA",  "baseline",      2.0, None, "daily",    "equity.daily_bars",  "NASDAQ.NVDA", "daily"),
    ("GLD",   "buy_and_hold",  2.0, 3.0, "daily",    "etf.daily_bars",     "ARCA.GLD",    "daily"),
]


def load_bars(lib_name: str, sym: str, intraday: bool) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib_name).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    if intraday and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def precompute_signal(asset: str, bars: pd.DataFrame, feat: pd.DataFrame,
                      mode: str, warmup: int) -> np.ndarray:
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "baseline":
        for i in range(warmup, n):
            try:
                s = compute_signal_for(asset, bars.iloc[: i + 1])
                out[i] = int(round(float(s)))
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
    raise ValueError(f"unknown signal mode {mode}")


def replay(bars: pd.DataFrame, feat: pd.DataFrame, sig_array: np.ndarray,
           start_idx: int, end_idx: int, trail: float,
           pt_sigmas: float | None) -> dict:
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    round_trips: list[dict] = []
    daily_bucket: dict[pd.Timestamp, float] = {}
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
        sig_dir = int(sig_array[i])
        close = float(closes[i])
        date_key = bars.index[i].normalize()

        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            step = (close - prev_close) * prev_qty * prev_side
            daily_bucket[date_key] = daily_bucket.get(date_key, 0.0) + step

        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop = peak - band
            reason = None
            if close <= stop:
                reason = "trail"
            elif (pt_sigmas is not None and
                  close >= entry_price * (1 + entry_sigma * pt_sigmas)):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                round_trips.append({"side": "LONG",
                                    "pnl": (close - entry_price) * entry_qty})
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
            elif (pt_sigmas is not None and
                  close <= entry_price * (1 - entry_sigma * pt_sigmas)):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                round_trips.append({"side": "SHORT",
                                    "pnl": (entry_price - close) * entry_qty})
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        if state == "FLAT" and sig_dir != 0:
            qty = int(NOTIONAL // close) if close > 0 else 0
            if qty > 0:
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

    return {
        "n_trades": len(round_trips),
        "n_long": sum(1 for r in round_trips if r["side"] == "LONG"),
        "n_short": sum(1 for r in round_trips if r["side"] == "SHORT"),
        "win_rate": win_rate, "net": net, "max_dd": max_dd, "sharpe": sharpe,
        "ann": ann, "years": years,
    }


def main() -> int:
    print()
    print("=" * 138)
    print(f"  Deployed-config train/test report  •  split={SPLIT.date()}  "
          f"end={END.date()}  •  per-name notional ${NOTIONAL:,.0f}")
    print("=" * 138)
    print(f"  {'asset':<6} {'sub':<9} {'signal':<13} {'trail':<6} {'PT':<5} "
          f"|  {'TRAIN trades':<13} {'TR net':<13} {'TR ann':<8} {'TR Sh':<7} "
          f"|  {'TEST trades':<12} {'TE net':<11} {'TE ann':<8} {'TE Sh'}")
    print("-" * 138)

    totals = {"train_net": 0.0, "test_net": 0.0, "train_tr": 0, "test_tr": 0,
              "train_years_sample": 0.0, "test_years_sample": 0.0}

    for asset, sigmode, trail, pt, substrate, lib, sym, _runner in DEPLOYED_CONFIGS:
        intraday = (substrate == "intraday")
        warmup = WARMUP_INTRADAY if intraday else WARMUP_DAILY
        bars = load_bars(lib, sym, intraday)
        feat = add_per_asset_features(bars)
        sig = precompute_signal(asset, bars, feat, sigmode, warmup)

        split_idx = bars.index.get_indexer([SPLIT], method="bfill")[0]
        end_idx = bars.index.get_indexer([END], method="ffill")[0] + 1
        end_idx = min(end_idx, len(bars))
        train_start = warmup
        if split_idx <= warmup or end_idx <= split_idx:
            print(f"  {asset:<6} window out of range")
            continue

        tr = replay(bars, feat, sig, train_start, split_idx, trail, pt)
        te = replay(bars, feat, sig, split_idx, end_idx, trail, pt)

        pt_str = f"{pt}σ" if pt is not None else "—"
        print(f"  {asset:<6} {substrate:<9} {sigmode:<13} {trail}σ{'':<2} {pt_str:<5}"
              f"|  {tr['n_trades']:<13} ${tr['net']:>+9,.0f}  "
              f"{tr['ann']:>+5.1f}%  {tr['sharpe']:>+5.2f}  "
              f"|  {te['n_trades']:<12} ${te['net']:>+9,.0f}  "
              f"{te['ann']:>+5.1f}%  {te['sharpe']:>+5.2f}")

        totals["train_net"] += tr["net"]
        totals["test_net"] += te["net"]
        totals["train_tr"] += tr["n_trades"]
        totals["test_tr"] += te["n_trades"]
        totals["train_years_sample"] = max(totals["train_years_sample"], tr["years"])
        totals["test_years_sample"] = max(totals["test_years_sample"], te["years"])

    print("-" * 138)
    pn = 8 * NOTIONAL
    tr_years = totals["train_years_sample"]  # longest train window across assets
    te_years = totals["test_years_sample"]
    tr_ret = totals["train_net"] / pn * 100.0
    te_ret = totals["test_net"] / pn * 100.0
    tr_ann = (((1 + totals["train_net"] / pn) ** (1 / tr_years) - 1) * 100.0
              if tr_years > 0 and pn + totals["train_net"] > 0 else float("nan"))
    te_ann = (((1 + totals["test_net"] / pn) ** (1 / te_years) - 1) * 100.0
              if te_years > 0 and pn + totals["test_net"] > 0 else float("nan"))
    print(f"  PORTFOLIO (8 assets, ${pn:,.0f} deployed)")
    print(f"  TRAIN: {totals['train_tr']:<6} trades  net=${totals['train_net']:>+11,.0f}  "
          f"return {tr_ret:>+6.2f}%  (~{tr_ann:+.1f}% ann. on longest-train asset)")
    print(f"  TEST : {totals['test_tr']:<6} trades  net=${totals['test_net']:>+11,.0f}  "
          f"return {te_ret:>+6.2f}%  ({te_ann:+.1f}% ann.)")
    print("=" * 138)
    print()
    print("  NOTES:")
    print("  • TRAIN windows differ per asset:")
    print("      - 6 daily assets: full ArcticDB history up to 2026-01-01")
    print("        (varies 5y for META up to ~44y for AAPL)")
    print("      - 2 intraday assets: 5-min data only goes back to 2024-05")
    print("        (~1.6y train per asset)")
    print("  • TEST window is shared: 2026-01-01 → 2026-05-19 (~4.5 months OOS)")
    print("  • Train→test slippage = ratio of train Sharpe to test Sharpe;")
    print("    values < 1.5x mean the config generalizes; > 3x means overfit risk.")
    print()
    print("  Reference — same-window comparison from script 79:")
    print("    All daily baseline (old 7-asset deployment): +50.7% ann. on TEST")
    print("    All daily comprehensive:                     +12.2% ann. on TEST")
    print("    All intraday:                                 +9.4% ann. on TEST")
    print("    Hybrid (DEPLOYED):                           +66.8% ann. on TEST")
    print("=" * 138)
    return 0


if __name__ == "__main__":
    sys.exit(main())
