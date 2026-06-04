"""94_leveraged_intraday_bakeoff — intraday (5-min) comprehensive search
on the leveraged ETF candidates (SOXL, NVDL, TSLL).

Mirrors script 78_intraday_search but on the v3 leveraged sleeve assets.
Tests whether the higher intraday vol of leveraged ETFs lets signals
extract more edge per day.

Run:
    .venv/bin/python scripts/94_leveraged_intraday_bakeoff.py
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

from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from scripts._storage import open_store  # noqa: E402

WARMUP = 500
NOTIONAL = 20000.0  # sleeve per-name = $20k
COST_BP = 3.0

UNIVERSE = {
    "SOXL": ("etf.minute_bars", "ARCA.SOXL"),
    "NVDL": ("etf.minute_bars", "ARCA.NVDL"),
    "TSLL": ("etf.minute_bars", "ARCA.TSLL"),
}
SIGNALS = ["buy_and_hold", "macd", "rsi2", "bollinger", "donchian", "mom21"]
TRAILS = [0.5, 1.0, 2.0, 3.0]
PTS = [None, 3.0, 5.0]

SPLIT = pd.Timestamp("2026-01-01")


def load_bars(lib: str, sym: str) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def precompute(bars: pd.DataFrame, feat: pd.DataFrame, mode: str) -> np.ndarray:
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "buy_and_hold":
        out[WARMUP:] = 1
        return out
    if mode == "macd":
        h = feat["macd_hist"].shift(1)
        out = np.where(h > 0, 1, np.where(h < 0, -1, 0)).astype(np.int8)
    elif mode == "rsi2":
        r = feat["rsi_2"].shift(1)
        out = np.where(r < 10, 1, np.where(r > 90, -1, 0)).astype(np.int8)
    elif mode == "bollinger":
        close = bars["close"]
        ma = close.rolling(20).mean().shift(1)
        sd = close.rolling(20).std().shift(1)
        z = (close.shift(1) - ma) / sd
        out = np.where(z < -2, 1, np.where(z > 2, -1, 0)).astype(np.int8)
    elif mode == "donchian":
        high20 = bars["high"].rolling(20).max().shift(1)
        low20 = bars["low"].rolling(20).min().shift(1)
        prev_close = bars["close"].shift(1)
        out = np.where(prev_close >= high20, 1,
                       np.where(prev_close <= low20, -1, 0)).astype(np.int8)
    elif mode == "mom21":
        m = feat["ret_21"].shift(1)
        out = np.where(m > 0.003, 1, np.where(m < -0.003, -1, 0)).astype(np.int8)
    else:
        raise ValueError(mode)
    out[:WARMUP] = 0
    return out


def replay(bars, feat, sig, start_idx, end_idx, trail, pt, cost_bp):
    cost_frac = cost_bp / 1e4
    closes = bars["close"].astype(float).values
    vol = feat["vol_21"].values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    entry_side = 0
    peak = trough = None
    rt = []
    daily = {}
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0
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
            daily[dk] = daily.get(dk, 0.0) + (close - prev_close) * prev_qty * prev_side
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
                rt.append({"pnl": gross - cost})
                daily[dk] = daily.get(dk, 0.0) - cost
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
                rt.append({"pnl": gross - cost})
                daily[dk] = daily.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty, entry_side = 0, 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        if state == "FLAT" and sig_dir != 0:
            qty = int(NOTIONAL // close) if close > 0 else 0
            if qty > 0:
                cost = qty * close * cost_frac
                daily[dk] = daily.get(dk, 0.0) - cost
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price, entry_sigma = close, sigma
                entry_qty, entry_side = qty, 1 if sig_dir > 0 else -1
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
    win_rate = wins / len(rt) * 100.0 if rt else 0.0
    sharpe = float("nan")
    max_dd = 0.0
    if daily:
        rets = np.asarray(list(daily.values())) / NOTIONAL
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(list(daily.values()))
        peak_e = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_e).min())
    sessions = len(daily)
    years = sessions / 252.0 if sessions else 0
    ann = float("nan")
    if years > 0 and NOTIONAL + net > 0:
        ann = ((1 + net / NOTIONAL) ** (1 / years) - 1) * 100.0
    return {"n_trades": len(rt), "win_rate": win_rate, "net": net,
            "sharpe": sharpe, "max_dd": max_dd, "ann": ann, "sessions": sessions}


def main():
    print()
    print("=" * 130)
    print(f"  Leveraged ETF intraday bake-off  •  6 signals × 4 trails × 3 PTs = 72 cells per asset")
    print(f"  Split = {SPLIT.date()}  •  Per-name notional = ${NOTIONAL:,.0f}  •  Cost {COST_BP}bp/side")
    print("=" * 130)
    portfolio_test_net = 0.0
    portfolio_train_net = 0.0
    for asset, (lib, sym) in UNIVERSE.items():
        bars = load_bars(lib, sym)
        feat = add_per_asset_features(bars)
        split_idx = bars.index.get_indexer([SPLIT], method="bfill")[0]
        if split_idx <= WARMUP:
            print(f"  {asset}: split before warmup — skip")
            continue
        sig_cache = {m: precompute(bars, feat, m) for m in SIGNALS}
        cells = []
        for s in SIGNALS:
            for t in TRAILS:
                for pt in PTS:
                    train = replay(bars, feat, sig_cache[s], WARMUP, split_idx, t, pt, COST_BP)
                    test = replay(bars, feat, sig_cache[s], split_idx, len(bars), t, pt, COST_BP)
                    cells.append({"signal": s, "trail": t, "pt": pt,
                                  "train": train, "test": test})
        viable = [c for c in cells if c["train"]["net"] > 0 and c["train"]["sharpe"] > 0]
        best = (max(viable, key=lambda c: c["train"]["net"] / (abs(c["train"]["max_dd"]) + 1.0))
                if viable else None)
        ceiling = max(cells, key=lambda c: c["test"]["net"])
        if best:
            pt_str = f"{best['pt']}σ" if best["pt"] else "—"
            print(f"  {asset:<5} train-best: {best['signal']:<13} trail={best['trail']}σ pt={pt_str:<5} | "
                  f"TRAIN net=${best['train']['net']:>9,.0f} Sh={best['train']['sharpe']:+.2f} | "
                  f"TEST trades={best['test']['n_trades']:<5} net=${best['test']['net']:>9,.0f} "
                  f"Sh={best['test']['sharpe']:+.2f} ann={best['test']['ann']:+.1f}%")
            portfolio_test_net += best["test"]["net"]
            portfolio_train_net += best["train"]["net"]
            pt_c = f"{ceiling['pt']}σ" if ceiling["pt"] else "—"
            print(f"        ceiling (look-ahead): {ceiling['signal']:<13} trail={ceiling['trail']}σ pt={pt_c:<5} | "
                  f"TEST net=${ceiling['test']['net']:>9,.0f}")
        else:
            print(f"  {asset}: no viable config")
    print()
    print("=" * 130)
    print(f"  PORTFOLIO (3 leveraged ETFs, $60k deployed)")
    print(f"  TRAIN: ${portfolio_train_net:+,.0f}")
    print(f"  TEST : ${portfolio_test_net:+,.0f}  ({portfolio_test_net/60000*100:+.2f}% on $60k)")
    if portfolio_test_net > 0:
        # The test window is ~4.5 months
        test_days = 95
        per_day = portfolio_test_net / test_days
        per_day_pct = per_day / 60000 * 100
        print(f"  Per-day avg: ${per_day:+.0f}/day on $60k sleeve = {per_day_pct:+.3f}%/day")
        print(f"  As % of $100k account: {per_day / 100000 * 100:+.3f}%/day extra")
    print("=" * 130)


if __name__ == "__main__":
    sys.exit(main())
