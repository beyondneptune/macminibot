"""96_path_b_backtest — Path B: run v2 ($100k, 8 assets × $12.5k) AND
v3 leveraged sleeve ($60k, 3 ETFs × $20k) IN PARALLEL on the same $100k
account → 1.6x gross exposure (forced margin).

For each test window:
  * Run v2 portfolio (existing CURRENT_V2 configs from script 87).
  * Run v3 sleeve (SOXL/NVDL/TSLL with rsi2_mr per agent-3 bake-off).
  * Combine into a single daily P&L stream (additive on same $100k account).
  * Compute combined max DD (worst simultaneous-drawdown across both books).

Key question: at 1.6x gross, what -% on the $100k account triggers margin
issues? Reg-T initial margin 50%; maintenance margin typically 25-30%.
At 1.6x gross, equity = $100k, gross = $160k → margin used = $60k loan.
A -25% drawdown drops equity to $75k → maintenance ratio = 75/160 = 47%
(still well above 25-30% maint). But the absolute risk is large: -25% on
$100k = lose a quarter of capital.

Windows tested:
  * OOS:       2026-01-01 → 2026-05-20 (v3 sleeve has full data)
  * COVID:     2020-02-01 → 2020-04-30 (v3 = N/A, ETFs didn't exist yet)
  * RATESHOCK: 2022-01-01 → 2022-06-30 (v3 = N/A, ETFs didn't exist yet)

Note SOXL/NVDL/TSLL ArcticDB coverage starts 2024-01-02 (598 bars at the
time of writing). So the COVID/RATESHOCK rows show v3=N/A and combined =
v2-only. The OOS window is where Path B leverage actually matters.

Run:
    .venv/bin/python scripts/96_path_b_backtest.py
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

ACCOUNT_EQUITY = 100_000.0
V2_NOTIONAL = ACCOUNT_EQUITY / 8.0          # $12,500 per v2 name
V3_NOTIONAL = 20_000.0                       # $20k per leveraged ETF
WARMUP = 252
COST_BP = 3.0

# v2 baseline (mirrors script 87 CURRENT_V2 exactly)
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

# v3 leveraged sleeve — rsi2_mr_10_90 (per agent 3 bake-off)
# Format: (asset, mode, trail_sigma, profit_target, library, symbol)
V3_SLEEVE = [
    ("SOXL", "rsi2_mr", 2.0, 3.0, "etf.daily_bars", "ARCA.SOXL"),
    ("NVDL", "rsi2_mr", 0.5, 5.0, "etf.daily_bars", "ARCA.NVDL"),
    ("TSLL", "rsi2_mr", 1.0, 3.0, "etf.daily_bars", "ARCA.TSLL"),
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
    """Latest causal signal vector for `bars`.

    Modes:
      * baseline      — dispatch to asset specialist via compute_signal_for.
      * buy_and_hold  — always +1 after warmup.
      * rsi2_mr       — RSI(2) at T-1: <10 long, >90 short. Used by v3 sleeve.
    """
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
    if mode == "rsi2_mr":
        r = feat["rsi_2"].shift(1)
        arr = np.where(r < 10, 1, np.where(r > 90, -1, 0)).astype(np.int8)
        arr[:WARMUP] = 0
        return arr
    raise ValueError(mode)


def replay_with_daily(bars, feat, sig, start_idx, end_idx, trail, pt,
                      notional, cost_bp):
    """Same as script 87 replay but returns the full per-day P&L dict.

    Returns:
      dict with keys: n_trades, wins, win_rate, net, sharpe, max_dd, ann,
      daily_pnl (dict: pd.Timestamp -> dollar P&L on that calendar day).
    """
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

        # MTM of prior position
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
    wins = sum(1 for r in rt if r["pnl"] > 0)
    win_rate = (wins / len(rt) * 100.0) if rt else 0.0
    sharpe = float("nan")
    max_dd = 0.0
    if daily_bucket:
        vals = list(daily_bucket.values())
        rets = np.asarray(vals) / notional
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(vals)
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    n_sessions = len(daily_bucket)
    years = n_sessions / 252.0 if n_sessions else 0.0
    ann = float("nan")
    if years > 0 and notional + net > 0:
        ann = ((1 + net / notional) ** (1 / years) - 1) * 100.0
    return {
        "n_trades": len(rt), "wins": wins, "win_rate": win_rate,
        "net": net, "sharpe": sharpe, "max_dd": max_dd, "ann": ann,
        "daily_pnl": daily_bucket,
    }


def run_book(configs, notional, label, window_start, window_end):
    """Run a portfolio book over a single window.

    Returns dict with portfolio_net, max_dd_self, daily_series (Series),
    n_trades, wins, missing_assets (list).
    """
    portfolio_daily: dict[pd.Timestamp, float] = {}
    portfolio_net = 0.0
    portfolio_trades = 0
    portfolio_wins = 0
    missing_assets: list[str] = []
    per_asset_results: list[tuple[str, dict]] = []

    for asset, mode, trail, pt, lib, sym in configs:
        bars = load_bars(lib, sym)
        # Sleeve coverage check: window must fall within the symbol's bars.
        if bars.index.max() < window_start or bars.index.min() > window_end:
            missing_assets.append(asset)
            continue
        # We also need WARMUP bars before window_start.
        first_eligible = bars.index[WARMUP] if len(bars) > WARMUP else None
        if first_eligible is None or first_eligible > window_end:
            missing_assets.append(asset)
            continue

        feat = add_per_asset_features(bars)
        sig = precompute_signal(asset, bars, feat, mode)
        s = bars.index.get_indexer([window_start], method="bfill")[0]
        e = bars.index.get_indexer([window_end], method="ffill")[0] + 1
        if s < 0 or e <= s or s <= WARMUP:
            # If the window start is before warmup completes, push start forward.
            if e > WARMUP and s <= WARMUP:
                s = WARMUP + 1
            else:
                missing_assets.append(asset)
                continue

        r = replay_with_daily(bars, feat, sig, s, e, trail, pt,
                              notional, COST_BP)
        portfolio_net += r["net"]
        portfolio_trades += r["n_trades"]
        portfolio_wins += r["wins"]
        per_asset_results.append((asset, r))
        for dk, v in r["daily_pnl"].items():
            portfolio_daily[dk] = portfolio_daily.get(dk, 0.0) + v

    # Self max-DD on the *book's own* daily series
    max_dd_self = 0.0
    if portfolio_daily:
        sorted_days = sorted(portfolio_daily.keys())
        vals = [portfolio_daily[d] for d in sorted_days]
        cum = np.cumsum(vals)
        peak_eq = np.maximum.accumulate(cum)
        max_dd_self = float((cum - peak_eq).min())

    return {
        "label": label,
        "net": portfolio_net,
        "trades": portfolio_trades,
        "wins": portfolio_wins,
        "max_dd_self": max_dd_self,
        "daily_series": pd.Series(portfolio_daily).sort_index() if portfolio_daily else pd.Series(dtype=float),
        "missing_assets": missing_assets,
        "per_asset": per_asset_results,
    }


def combine_books(v2_book, v3_book):
    """Combine two daily P&L series on the same $100k account.

    Returns dict with combined net, combined daily series, combined max DD,
    and the worst single-day combined drawdown.
    """
    v2_d = v2_book["daily_series"]
    v3_d = v3_book["daily_series"]
    # Union of dates; treat missing days as 0 P&L.
    all_dates = sorted(set(v2_d.index) | set(v3_d.index))
    combined = pd.Series(0.0, index=all_dates)
    if not v2_d.empty:
        combined = combined.add(v2_d.reindex(all_dates, fill_value=0.0), fill_value=0.0)
    if not v3_d.empty:
        combined = combined.add(v3_d.reindex(all_dates, fill_value=0.0), fill_value=0.0)
    combined_net = float(combined.sum()) if len(combined) else 0.0

    if len(combined):
        cum = combined.cumsum().values
        peak_eq = np.maximum.accumulate(cum)
        combined_max_dd = float((cum - peak_eq).min())
        worst_day = float(combined.min())
        # Find days where both books were down simultaneously
        v2_aligned = v2_d.reindex(all_dates, fill_value=0.0)
        v3_aligned = v3_d.reindex(all_dates, fill_value=0.0)
        both_down_mask = (v2_aligned < 0) & (v3_aligned < 0)
        n_both_down = int(both_down_mask.sum())
        # Worst single day where both are down
        if n_both_down > 0:
            both_down_combined = combined[both_down_mask]
            worst_both_down = float(both_down_combined.min())
            worst_both_down_date = both_down_combined.idxmin()
        else:
            worst_both_down = float("nan")
            worst_both_down_date = None
    else:
        combined_max_dd = 0.0
        worst_day = 0.0
        n_both_down = 0
        worst_both_down = float("nan")
        worst_both_down_date = None

    return {
        "net": combined_net,
        "max_dd": combined_max_dd,
        "worst_day": worst_day,
        "n_both_down": n_both_down,
        "worst_both_down": worst_both_down,
        "worst_both_down_date": worst_both_down_date,
        "daily_series": combined,
    }


def fmt_money(x, width=10):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return f"{'N/A':>{width}}"
    return f"${x:>+{width-1},.0f}"


def fmt_pct(x, width=7):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return f"{'N/A':>{width}}"
    return f"{x:>+{width-1}.2f}%"


def main():
    print()
    print("=" * 130)
    print("  Path B backtest — v2 ($100k, 8 names × $12.5k) + v3 sleeve ($60k, 3 ETFs × $20k)")
    print("  Total deployed: $160k on $100k account → 1.6× gross exposure")
    print("=" * 130)

    window_results = []
    for label, start, end in WINDOWS:
        print(f"\n  Running window {label}  ({start.date()} → {end.date()})...")

        print(f"    v2 book ($100k, 8 names)...")
        v2_book = run_book(CURRENT_V2, V2_NOTIONAL, "v2", start, end)
        if v2_book["missing_assets"]:
            print(f"      v2 missing: {v2_book['missing_assets']}")

        print(f"    v3 sleeve ($60k, 3 leveraged ETFs)...")
        v3_book = run_book(V3_SLEEVE, V3_NOTIONAL, "v3", start, end)
        if v3_book["missing_assets"]:
            print(f"      v3 missing: {v3_book['missing_assets']} (no bars in window)")

        combined = combine_books(v2_book, v3_book)
        window_results.append({
            "label": label,
            "start": start,
            "end": end,
            "v2": v2_book,
            "v3": v3_book,
            "combined": combined,
        })

    # ---- Print summary ----
    print()
    print("=" * 130)
    print("  Per-window summary")
    print("=" * 130)
    print(f"  {'window':<11} {'v2 net':>11} {'v2 maxDD':>11} {'v3 net':>11} "
          f"{'v3 maxDD':>11} {'combined':>11} {'combo maxDD':>13} "
          f"{'combo DD %':>11} {'both-down days':>15}")
    print("-" * 130)
    for w in window_results:
        v2 = w["v2"]
        v3 = w["v3"]
        c = w["combined"]
        v3_has_data = bool(v3["daily_series"].size)
        v3_net_str = fmt_money(v3["net"]) if v3_has_data else f"{'N/A':>10}"
        v3_dd_str = fmt_money(v3["max_dd_self"]) if v3_has_data else f"{'N/A':>10}"
        combo_dd_pct = c["max_dd"] / ACCOUNT_EQUITY * 100.0
        both_str = f"{c['n_both_down']}" if v3_has_data else "N/A"
        print(f"  {w['label']:<11} {fmt_money(v2['net'])} {fmt_money(v2['max_dd_self'])} "
              f"{v3_net_str} {v3_dd_str} "
              f"{fmt_money(c['net'])} {fmt_money(c['max_dd'])} "
              f"{combo_dd_pct:>+10.2f}% {both_str:>15}")

    print()
    print("=" * 130)
    print("  Worst-case daily detail (combined book, days where BOTH books were down)")
    print("=" * 130)
    for w in window_results:
        v3_has_data = bool(w["v3"]["daily_series"].size)
        if not v3_has_data:
            print(f"  {w['label']:<11}  v3 has no data in this window — combined = v2-only")
            continue
        c = w["combined"]
        print(f"  {w['label']:<11}  "
              f"worst single combined day: {fmt_money(c['worst_day'])}  |  "
              f"# both-down days: {c['n_both_down']}  |  "
              f"worst both-down day: {fmt_money(c['worst_both_down'])} "
              f"({c['worst_both_down_date'].date() if c['worst_both_down_date'] is not None else 'n/a'})")

    print()
    print("=" * 130)
    print("  Margin call analysis (Reg-T, 1.6× gross, $100k equity, $60k loan)")
    print("=" * 130)
    print(f"  Account equity:   $100,000")
    print(f"  Gross exposure:   $160,000 (1.6× leverage)")
    print(f"  Maintenance min:  25% of gross = $40,000 equity required")
    print(f"  Buffer to margin call (loss before $100k → $40k): -$60,000  (-60% of account)")
    print()
    # Per-window: is combined max DD inside this buffer?
    print("  Per-window margin headroom:")
    for w in window_results:
        c = w["combined"]
        # Combined max DD is negative
        room = 60000.0 - abs(c["max_dd"])
        flag = "OK" if room > 20000 else ("TIGHT" if room > 0 else "MARGIN CALL")
        dd_pct = c["max_dd"] / ACCOUNT_EQUITY * 100.0
        print(f"    {w['label']:<11}  combined max DD = {fmt_money(c['max_dd'])} ({dd_pct:+.2f}%)  "
              f"→ headroom {fmt_money(60000.0 + c['max_dd'])}  [{flag}]")

    print()
    print("=" * 130)
    print("  Summary table — net P&L and DD by book")
    print("=" * 130)
    print(f"  {'metric':<28} {'OOS':>14} {'COVID':>14} {'RATESHOCK':>14}")
    print("-" * 130)
    rows = [
        ("v2 net",            lambda w: w["v2"]["net"]),
        ("v2 max DD ($)",     lambda w: w["v2"]["max_dd_self"]),
        ("v2 max DD (%)",     lambda w: w["v2"]["max_dd_self"] / ACCOUNT_EQUITY * 100.0),
        ("v3 net",            lambda w: w["v3"]["net"] if w["v3"]["daily_series"].size else float("nan")),
        ("v3 max DD ($)",     lambda w: w["v3"]["max_dd_self"] if w["v3"]["daily_series"].size else float("nan")),
        ("v3 max DD (% of $60k)", lambda w: (w["v3"]["max_dd_self"] / 60000.0 * 100.0) if w["v3"]["daily_series"].size else float("nan")),
        ("combined net",      lambda w: w["combined"]["net"]),
        ("combined max DD",   lambda w: w["combined"]["max_dd"]),
        ("combined max DD (% acct)", lambda w: w["combined"]["max_dd"] / ACCOUNT_EQUITY * 100.0),
        ("# both-down days",  lambda w: float(w["combined"]["n_both_down"]) if w["v3"]["daily_series"].size else float("nan")),
    ]
    for name, fn in rows:
        vals = [fn(w) for w in window_results]
        if "%" in name:
            cells = "  ".join(fmt_pct(v, width=12) for v in vals)
        elif "days" in name:
            cells = "  ".join((f"{int(v):>12}" if math.isfinite(v) else f"{'N/A':>12}") for v in vals)
        else:
            cells = "  ".join(fmt_money(v, width=12) for v in vals)
        print(f"  {name:<28}  {cells}")

    print()
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
