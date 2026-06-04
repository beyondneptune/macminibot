"""92_kelly_sizing — empirical test of Kelly-fraction position sizing vs
equal-weight ($12,500 per name).

Hypothesis: per-asset Sharpes vary materially. Sizing positions by
Kelly fraction (proportional to per-asset Sharpe) should improve
portfolio Sharpe and net P&L vs the current uniform notional.

Kelly weights derived from script 80_deployed_train_test.py per-asset OOS
Sharpes (provided by the operator):

    Asset    OOS Sharpe    Kelly weight    $ notional (of $100k)
    AAPL     1.66          10.0%           $10,043
    MSFT     2.29          13.9%           $13,860
    GOOGL    3.60          21.8%           $21,786
    AMZN     1.34           8.1%            $8,107
    META     3.70          22.4%           $22,389
    TSLA     1.31           7.9%            $7,925
    NVDA     1.59           9.6%            $9,619
    GLD      1.04           6.3%            $6,292

(Sum of Sharpes = 16.53. weight_i = Sharpe_i / sum × 0.95 + floor 5%.)

Equal-weight comparison: all 8 assets at $12,500.

The signals, trails, and PTs are IDENTICAL to current v2 baseline (the
same configs replayed in scripts 87_fixed_replay.py). The ONLY difference
is the per-asset notional used for position sizing.

Run:
    .venv/bin/python scripts/92_kelly_sizing.py
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

NET_LIQ = 100_000.0
WARMUP = 252
COST_BP = 3.0

# Current v2 config — same signals / trails / PTs as production daily-v2.
# Source: scripts/87_fixed_replay.py CURRENT_V2.
DEPLOYED = [
    ("AAPL",  "buy_and_hold",  3.0, 3.0,  "equity.daily_bars",  "NASDAQ.AAPL"),
    ("MSFT",  "baseline",      1.0, 5.0,  "equity.daily_bars",  "NASDAQ.MSFT"),
    ("GOOGL", "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold",  0.5, 5.0,  "equity.daily_bars",  "NASDAQ.AMZN"),
    ("META",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.META"),
    ("TSLA",  "baseline",      0.5, 3.0,  "equity.daily_bars",  "NASDAQ.TSLA"),
    ("NVDA",  "baseline",      2.0, None, "equity.daily_bars",  "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold",  2.0, 3.0,  "etf.daily_bars",     "ARCA.GLD"),
]

# Per-asset OOS Sharpes from script 80_deployed_train_test.py
OOS_SHARPE = {
    "AAPL":  1.66,
    "MSFT":  2.29,
    "GOOGL": 3.60,
    "AMZN":  1.34,
    "META":  3.70,
    "TSLA":  1.31,
    "NVDA":  1.59,
    "GLD":   1.04,
}

# Equal-weight (current production v2): NET_LIQ / 8 = $12,500 per name.
EQUAL_NOTIONAL = {a: NET_LIQ / 8.0 for a in OOS_SHARPE}

# Kelly-weighted notionals (provided by operator; 0.25 Kelly fraction baked in
# via the weighting formula; floor 5% per name; total = $100k).
KELLY_NOTIONAL = {
    "AAPL":  10_043.0,
    "MSFT":  13_860.0,
    "GOOGL": 21_786.0,
    "AMZN":   8_107.0,
    "META":  22_389.0,
    "TSLA":   7_925.0,
    "NVDA":   9_619.0,
    "GLD":    6_292.0,
}

WINDOWS = [
    ("OOS",       pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-20")),
    ("COVID",     pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
    ("RATESHOCK", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30")),
]


def load_bars(lib: str, sym: str) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()[["open", "high", "low", "close", "volume"]]


def precompute_signal(asset: str, bars: pd.DataFrame, mode: str) -> np.ndarray:
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


def replay(bars: pd.DataFrame, feat: pd.DataFrame, sig: np.ndarray,
           start_idx: int, end_idx: int, trail: float, pt: float | None,
           cost_bp: float, notional: float) -> dict:
    """Replay one asset on one window using a per-asset notional.

    Returns dict with realized + open net P&L, per-trade list, and
    daily P&L bucket so the caller can aggregate at portfolio level."""
    cost_frac = cost_bp / 1e4
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    entry_side = 0
    peak = trough = None
    rt = []
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
            qty = int(notional // close) if close > 0 else 0
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
    return {
        "n_trades": len(rt),
        "wins": sum(1 for r in rt if r["pnl"] > 0),
        "realized": realized,
        "open_unrz": open_unrz,
        "net": net,
        "daily_bucket": daily_bucket,
    }


def run_portfolio(notional_map: dict[str, float], label: str) -> dict:
    """Replay all 8 assets on all 3 windows with the given per-asset notionals.

    Returns: dict {window_label: {per_asset: {...}, totals: {...}}}.
    """
    out: dict[str, dict] = {}
    print(f"  [{label}] replaying {len(DEPLOYED)} assets × {len(WINDOWS)} windows ...")
    for wlabel, start, end in WINDOWS:
        per_asset: dict[str, dict] = {}
        portfolio_daily: dict[pd.Timestamp, float] = {}
        for asset, mode, trail, pt, lib, sym in DEPLOYED:
            notional = notional_map[asset]
            bars = load_bars(lib, sym)
            feat = add_per_asset_features(bars)
            sig = precompute_signal(asset, bars, mode)
            s = bars.index.get_indexer([start], method="bfill")[0]
            e = bars.index.get_indexer([end], method="ffill")[0] + 1
            if s <= WARMUP or e <= s:
                per_asset[asset] = {"n_trades": 0, "wins": 0, "net": 0.0,
                                    "notional": notional, "sharpe": float("nan"),
                                    "max_dd": 0.0}
                continue
            r = replay(bars, feat, sig, s, e, trail, pt, COST_BP, notional)

            # Aggregate this asset's daily P&L into the portfolio bucket
            for k, v in r["daily_bucket"].items():
                portfolio_daily[k] = portfolio_daily.get(k, 0.0) + v

            # Per-asset Sharpe + DD (relative to per-asset notional)
            sharpe = float("nan")
            max_dd = 0.0
            if r["daily_bucket"]:
                rets = np.asarray(list(r["daily_bucket"].values())) / notional
                if rets.std() > 0 and len(rets) > 1:
                    sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
                cum = np.cumsum(list(r["daily_bucket"].values()))
                pk = np.maximum.accumulate(cum)
                max_dd = float((cum - pk).min())
            per_asset[asset] = {
                "n_trades": r["n_trades"],
                "wins": r["wins"],
                "net": r["net"],
                "notional": notional,
                "sharpe": sharpe,
                "max_dd": max_dd,
            }

        # Portfolio totals
        tot_net = sum(pa["net"] for pa in per_asset.values())
        tot_trades = sum(pa["n_trades"] for pa in per_asset.values())
        tot_wins = sum(pa["wins"] for pa in per_asset.values())

        # Portfolio Sharpe & DD relative to NET_LIQ (total capital deployed)
        sharpe = float("nan")
        max_dd = 0.0
        if portfolio_daily:
            rets = np.asarray(list(portfolio_daily.values())) / NET_LIQ
            if rets.std() > 0 and len(rets) > 1:
                sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
            cum = np.cumsum(list(portfolio_daily.values()))
            pk = np.maximum.accumulate(cum)
            max_dd = float((cum - pk).min())

        wr = (tot_wins / tot_trades * 100.0) if tot_trades else 0.0
        ret_pct = tot_net / NET_LIQ * 100.0
        years = (end - start).days / 365.25
        ann = float("nan")
        if years > 0 and NET_LIQ + tot_net > 0:
            ann = ((1 + tot_net / NET_LIQ) ** (1 / years) - 1) * 100.0

        out[wlabel] = {
            "per_asset": per_asset,
            "totals": {
                "net": tot_net,
                "trades": tot_trades,
                "wr": wr,
                "ret_pct": ret_pct,
                "ann": ann,
                "sharpe": sharpe,
                "max_dd": max_dd,
            },
        }
    return out


def main() -> int:
    print()
    print("=" * 130)
    print("  Kelly-fraction sizing vs equal-weight (current v2) — same signals/trails/PTs")
    print("=" * 130)
    print(f"\n  Configs: {len(DEPLOYED)} assets, {len(WINDOWS)} windows")
    print(f"  NET_LIQ = ${NET_LIQ:,.0f}   COST_BP = {COST_BP}")

    print("\n  Notional table:")
    print(f"    {'asset':<7} {'Sharpe':>7}  {'equal $':>10}  {'Kelly $':>10}  {'Δ':>9}")
    for asset, mode, *_ in DEPLOYED:
        sh = OOS_SHARPE[asset]
        eq = EQUAL_NOTIONAL[asset]
        kl = KELLY_NOTIONAL[asset]
        print(f"    {asset:<7} {sh:>7.2f}  ${eq:>9,.0f}  ${kl:>9,.0f}  ${kl-eq:>+8,.0f}")
    print(f"    {'TOTAL':<7} {sum(OOS_SHARPE.values()):>7.2f}  "
          f"${sum(EQUAL_NOTIONAL.values()):>9,.0f}  "
          f"${sum(KELLY_NOTIONAL.values()):>9,.0f}")

    eq_res = run_portfolio(EQUAL_NOTIONAL, "EQUAL")
    kl_res = run_portfolio(KELLY_NOTIONAL, "KELLY")

    # Per-asset, per-window net table
    print()
    print("=" * 130)
    print("  Per-asset net P&L by window (Kelly | equal)")
    print("=" * 130)
    for wlabel, _, _ in WINDOWS:
        print(f"\n  Window: {wlabel}")
        print(f"    {'asset':<7}  {'Kelly $':>10}  {'eq $':>10}  "
              f"{'Kelly net':>12}  {'eq net':>12}  {'Δ Kelly−eq':>12}  "
              f"{'K trd':>5}  {'eq trd':>6}  {'K Sh':>6}")
        for asset, *_ in DEPLOYED:
            kp = kl_res[wlabel]["per_asset"][asset]
            ep = eq_res[wlabel]["per_asset"][asset]
            delta = kp["net"] - ep["net"]
            print(f"    {asset:<7}  ${kp['notional']:>9,.0f}  ${ep['notional']:>9,.0f}  "
                  f"${kp['net']:>+10,.0f}  ${ep['net']:>+10,.0f}  ${delta:>+10,.0f}  "
                  f"{kp['n_trades']:>5}  {ep['n_trades']:>6}  {kp['sharpe']:>6.2f}")

    # Portfolio summary
    print()
    print("=" * 130)
    print(f"  PORTFOLIO TOTALS — Kelly vs equal-weight ({len(DEPLOYED)} assets × $100k total capital)")
    print("=" * 130)
    print(f"    {'window':<11}  {'Kelly net':>12}  {'eq net':>12}  {'Δ':>10}  "
          f"{'K Sharpe':>9}  {'eq Sharpe':>10}  {'K DD':>10}  {'eq DD':>10}  "
          f"{'K trd':>6}  {'eq trd':>6}")
    for wlabel, _, _ in WINDOWS:
        k = kl_res[wlabel]["totals"]
        e = eq_res[wlabel]["totals"]
        delta = k["net"] - e["net"]
        print(f"    {wlabel:<11}  ${k['net']:>+10,.0f}  ${e['net']:>+10,.0f}  ${delta:>+8,.0f}  "
              f"{k['sharpe']:>9.2f}  {e['sharpe']:>10.2f}  "
              f"${k['max_dd']:>+8,.0f}  ${e['max_dd']:>+8,.0f}  "
              f"{k['trades']:>6}  {e['trades']:>6}")

    # 3-window sum
    k_sum = sum(kl_res[w]["totals"]["net"] for w, _, _ in WINDOWS)
    e_sum = sum(eq_res[w]["totals"]["net"] for w, _, _ in WINDOWS)
    print(f"\n    {'3-WINDOW SUM':<11}  ${k_sum:>+10,.0f}  ${e_sum:>+10,.0f}  ${k_sum-e_sum:>+8,.0f}")

    # Cross-check vs operator's quoted baseline
    print()
    print("=" * 130)
    print("  Sanity check vs operator-quoted equal-weight baseline")
    print("=" * 130)
    expected = {"OOS": 19144.0, "COVID": 11202.0, "RATESHOCK": -7081.0}
    print(f"    {'window':<11}  {'this run (eq)':>14}  {'operator quote':>15}  {'diff':>10}")
    for wlabel in ("OOS", "COVID", "RATESHOCK"):
        got = eq_res[wlabel]["totals"]["net"]
        exp = expected[wlabel]
        print(f"    {wlabel:<11}  ${got:>+12,.0f}  ${exp:>+13,.0f}  ${got-exp:>+8,.0f}")

    print()
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
