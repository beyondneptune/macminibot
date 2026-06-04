"""89_tod_filter_test — Empirical test: does blocking intraday exit checks
during the morning vol-cluster window improve net P&L?

Agent 3 found 60% of premature exits clustered in 09:30-10:30 ET (vol = 3.6x
daily mean). Profit-lock / widen-trails underperform on OOS. Time-of-day
filter is the last untested idea.

This replay uses:
  - DAILY bars to compute the entry signal (warmup 252) — exactly as v2
    deployment does, mirroring 87_fixed_replay's signal logic.
  - 5-MIN intraday bars (equity.minute_bars / etf.minute_bars) to walk the
    intraday session and check trail-stop / profit-target / signal-flip
    exits at 5-min granularity (analogous to live 36_intraday_exit_monitor).
  - HIGH/LOW within each 5-min bar to check exit hits (more honest than
    close-only — matches the task spec).
  - Optional time-of-day filter: skip exit checks if minute-of-ET-day falls
    in the blocked window.

Variants tested (OOS only, intraday data exists 2024-05-20 onwards):
  * NoFilter        (control)           — 09:30-16:00 ET, exit always live
  * Block 09:30-10:00 ET                — 30-min skip
  * Block 09:30-10:30 ET                — Agent 3's headline window
  * Block 09:30-11:00 ET                — 90-min skip

Same per-asset configs as v2 deployment:
    AAPL  buy_and_hold   trail=3.0  pt=3.0
    MSFT  rsi2_mr_10_90  trail=1.0  pt=5.0
    GOOGL buy_and_hold   trail=2.0  pt=None
    AMZN  buy_and_hold   trail=0.5  pt=5.0
    META  lgbm           trail=2.0  pt=None
    TSLA  macd_signal    trail=0.5  pt=3.0
    NVDA  buy_and_hold   trail=2.0  pt=None
    GLD   buy_and_hold   trail=2.0  pt=3.0

Run:
    .venv/bin/python scripts/89_tod_filter_test.py
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

# ---------------------------------------------------------------------------
# Sizing / cost constants — match 87_fixed_replay.py and v2 deployment.
# ---------------------------------------------------------------------------
NET_LIQ = 100000.0
NOTIONAL = NET_LIQ / 8.0
WARMUP = 252                 # days of daily warmup before signals fire
COST_BP = 3.0                # 3 bp round trip per side

# OOS window (intraday data exists 2024-05-20 onwards; head-room for warmup)
OOS_START = pd.Timestamp("2024-05-20")
OOS_END = pd.Timestamp("2026-05-19")

# Regular trading hours window in minutes-from-midnight ET
RTH_OPEN_MIN = 9 * 60 + 30       # 570
RTH_CLOSE_MIN = 16 * 60          # 960

# Per-asset config (asset, signal_mode, trail_sigma, pt_sigma, daily_lib,
# daily_sym, intraday_lib, intraday_sym)
CONFIGS = [
    ("AAPL",  "buy_and_hold", 3.0, 3.0,
     "equity.daily_bars", "NASDAQ.AAPL", "equity.minute_bars", "NASDAQ.AAPL"),
    ("MSFT",  "baseline",     1.0, 5.0,
     "equity.daily_bars", "NASDAQ.MSFT", "equity.minute_bars", "NASDAQ.MSFT"),
    ("GOOGL", "baseline",     2.0, None,
     "equity.daily_bars", "NASDAQ.GOOGL", "equity.minute_bars", "NASDAQ.GOOGL"),
    ("AMZN",  "buy_and_hold", 0.5, 5.0,
     "equity.daily_bars", "NASDAQ.AMZN", "equity.minute_bars", "NASDAQ.AMZN"),
    ("META",  "baseline",     2.0, None,
     "equity.daily_bars", "NASDAQ.META", "equity.minute_bars", "NASDAQ.META"),
    ("TSLA",  "baseline",     0.5, 3.0,
     "equity.daily_bars", "NASDAQ.TSLA", "equity.minute_bars", "NASDAQ.TSLA"),
    ("NVDA",  "baseline",     2.0, None,
     "equity.daily_bars", "NASDAQ.NVDA", "equity.minute_bars", "NASDAQ.NVDA"),
    ("GLD",   "buy_and_hold", 2.0, 3.0,
     "etf.daily_bars", "ARCA.GLD", "etf.minute_bars", "ARCA.GLD"),
]

# Time-of-day filter variants: (label, start_min_ET, end_min_ET).
# `None` means no filter (control).
VARIANTS = [
    ("NoFilter (control)", None, None),
    ("Block 09:30-10:00",  9 * 60 + 30, 10 * 60 + 0),
    ("Block 09:30-10:30",  9 * 60 + 30, 10 * 60 + 30),
    ("Block 09:30-11:00",  9 * 60 + 30, 11 * 60 + 0),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_daily(lib: str, sym: str) -> pd.DataFrame:
    """Daily OHLCV (index already datetime)."""
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()[["open", "high", "low", "close", "volume"]]


def load_intraday_rth(lib: str, sym: str) -> pd.DataFrame:
    """5-min bars, restricted to US regular trading hours (09:30-16:00 ET).

    Returns a frame with a `minute_of_et_day` column for fast TOD filtering.
    Index is the bar's *start* timestamp in UTC. We attach ET conversion as
    an extra column rather than re-tz-ing inside the inner loop.
    """
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[["open", "high", "low", "close", "volume"]]
    # Convert to ET to filter RTH and compute minute-of-day
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    et = df.index.tz_convert("America/New_York")
    minute_of_day = et.hour * 60 + et.minute
    weekday = et.weekday
    mask = (
        (minute_of_day >= RTH_OPEN_MIN)
        & (minute_of_day < RTH_CLOSE_MIN)
        & (weekday < 5)
    )
    df = df.loc[mask].copy()
    df["minute_of_et_day"] = minute_of_day[mask]
    df["et_date"] = et[mask].normalize().date  # python date for grouping
    return df


def precompute_signal(asset: str, bars: pd.DataFrame, mode: str) -> np.ndarray:
    """Returns int8 array of {-1,0,+1} per-row signal (same length as `bars`).

    Causally consistent: position uses info up to row i inclusive (matches
    87_fixed_replay convention).
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
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# Replay core
# ---------------------------------------------------------------------------
def replay_intraday(
    daily: pd.DataFrame,
    daily_feat: pd.DataFrame,
    sig: np.ndarray,
    intraday: pd.DataFrame,
    trail: float,
    pt: float | None,
    cost_bp: float,
    block_start_min: int | None,
    block_end_min: int | None,
) -> dict:
    """Walk 5-min bars; check trail-stop / PT / signal-flip exits with optional TOD filter.

    Entry: at the OPEN of the first 5-min bar of the trading day if (a) we are
    FLAT and (b) the daily signal for that day is non-zero. The daily signal
    for trading day D is `sig[idx_in_daily(D)]` — same convention as 87.

    Exit (checked on every 5-min bar):
      - If `tod_filter_active(bar)`: skip exit checks entirely (peak/trough
        still updates from the bar's high/low so the trail floor stays honest
        once we resume).
      - Else: trail-stop hits if the bar's intraday LOW (long) / HIGH (short)
        breaches the running stop. PT hits if HIGH (long) / LOW (short) hits
        target. Signal-flip uses the *current day's* daily sig — if it flips
        sign we exit at the current bar's close.

    End-of-day: if still in a position at 16:00 ET, we hold to the next
    trading day (positions can span days; matches v2 logic).

    Returns dict of summary stats + per-exit timestamps.
    """
    cost_frac = cost_bp / 1e4

    # Daily lookup: et_date -> daily_index
    daily_dates = daily.index.normalize()
    daily_dates_py = pd.DatetimeIndex(daily_dates).date
    daily_date_to_idx = {d: i for i, d in enumerate(daily_dates_py)}

    vol = daily_feat["vol_21"].values

    # Intraday columns as numpy for speed
    op = intraday["open"].astype(float).values
    hi = intraday["high"].astype(float).values
    lo = intraday["low"].astype(float).values
    cl = intraday["close"].astype(float).values
    mod = intraday["minute_of_et_day"].astype(int).values  # ET minute-of-day
    edate = intraday["et_date"].values  # python date array

    # State
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    entry_side = 0
    peak = trough = None
    rt = []  # round-trip log
    daily_bucket: dict = {}   # ET-date -> per-day pnl (for sharpe/dd)
    prev_close = None
    prev_qty = 0
    prev_side = 0
    prev_state = "FLAT"
    open_today_bar_idx = -1   # first intraday-bar index of current ET day
    cur_day_sig = 0
    cur_day_sigma = None

    n = len(intraday)
    if n == 0:
        return _empty_result()

    cur_date = None
    cur_day_dindex = None

    for i in range(n):
        bar_close = float(cl[i])
        bar_open = float(op[i])
        bar_high = float(hi[i])
        bar_low = float(lo[i])
        minute = int(mod[i])
        bar_date = edate[i]

        # On a new ET trading day, look up that day's signal/sigma from daily.
        # Causal alignment: signal computed at CLOSE of day D-1 applied at
        # OPEN of day D. So index = daily_idx(D) - 1 (or D if D-1 is not a
        # trading day — but every daily-bar row is a trading day, so D's
        # date should map to a daily idx and we step back one).
        if bar_date != cur_date:
            cur_date = bar_date
            d_idx = daily_date_to_idx.get(cur_date)
            if d_idx is None or d_idx == 0:
                # Day not in daily bars (rare) — fall back to previous valid idx
                cur_day_sig = 0
                cur_day_sigma = None
            else:
                src_idx = d_idx - 1  # signal/sigma from PRIOR day's close
                cur_day_sig = int(sig[src_idx])
                v = float(vol[src_idx])
                cur_day_sigma = v if (math.isfinite(v) and v > 0) else None
            cur_day_dindex = d_idx
            open_today_bar_idx = i

        # P&L accumulation: mark-to-market the previous bar's close→this bar's close
        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            step = (bar_close - prev_close) * prev_qty * prev_side
            daily_bucket[bar_date] = daily_bucket.get(bar_date, 0.0) + step

        in_tod_block = (
            block_start_min is not None
            and block_end_min is not None
            and block_start_min <= minute < block_end_min
        )

        # ---- Exit checks on the CURRENT bar ----
        if state == "IN_LONG":
            # Always update running peak from intraday high (otherwise the
            # trail floor is artificially lower once we resume after a block).
            if peak is None or bar_high > peak:
                peak = bar_high

            band = entry_price * entry_sigma * trail
            stop_price = peak - band
            reason = None
            exit_price = None

            if not in_tod_block:
                # Trail-stop: bar low breaches stop
                if bar_low <= stop_price:
                    reason = "trail"
                    # Exit at stop_price if the bar gapped through it; else at the low
                    exit_price = stop_price if bar_open > stop_price else bar_open
                # PT: bar high breaches target
                elif pt is not None:
                    pt_price = entry_price * (1 + entry_sigma * pt)
                    if bar_high >= pt_price:
                        reason = "pt"
                        exit_price = pt_price if bar_open < pt_price else bar_open
                # Signal flip vs *today's* daily sig
                elif cur_day_sig < 0:
                    reason = "flip"
                    exit_price = bar_close

            if reason:
                gross = (exit_price - entry_price) * entry_qty
                cost = entry_qty * exit_price * cost_frac
                rt.append({
                    "pnl": gross - cost,
                    "reason": reason,
                    "ts": intraday.index[i],
                    "minute_et": minute,
                })
                daily_bucket[bar_date] = daily_bucket.get(bar_date, 0.0) - cost
                # Realized intraday step: from prev_close → exit_price (not bar_close)
                # We already booked prev_close → bar_close above; correct by
                # adding (exit_price - bar_close) * qty * side.
                if prev_close is not None and prev_state != "FLAT":
                    correction = (exit_price - bar_close) * entry_qty * entry_side
                    daily_bucket[bar_date] += correction
                state = "FLAT"
                entry_price = entry_sigma = None
                peak = trough = None
                entry_qty = 0
                entry_side = 0
                prev_close = bar_close
                prev_state = "FLAT"
                prev_qty = 0
                prev_side = 0
                continue

        elif state == "IN_SHORT":
            if trough is None or bar_low < trough:
                trough = bar_low

            band = entry_price * entry_sigma * trail
            stop_price = trough + band
            reason = None
            exit_price = None

            if not in_tod_block:
                if bar_high >= stop_price:
                    reason = "trail"
                    exit_price = stop_price if bar_open < stop_price else bar_open
                elif pt is not None:
                    pt_price = entry_price * (1 - entry_sigma * pt)
                    if bar_low <= pt_price:
                        reason = "pt"
                        exit_price = pt_price if bar_open > pt_price else bar_open
                elif cur_day_sig > 0:
                    reason = "flip"
                    exit_price = bar_close

            if reason:
                gross = (entry_price - exit_price) * entry_qty
                cost = entry_qty * exit_price * cost_frac
                rt.append({
                    "pnl": gross - cost,
                    "reason": reason,
                    "ts": intraday.index[i],
                    "minute_et": minute,
                })
                daily_bucket[bar_date] = daily_bucket.get(bar_date, 0.0) - cost
                if prev_close is not None and prev_state != "FLAT":
                    correction = (exit_price - bar_close) * entry_qty * entry_side
                    daily_bucket[bar_date] += correction
                state = "FLAT"
                entry_price = entry_sigma = None
                peak = trough = None
                entry_qty = 0
                entry_side = 0
                prev_close = bar_close
                prev_state = "FLAT"
                prev_qty = 0
                prev_side = 0
                continue

        # ---- Entry: at the first RTH bar of the day, use its OPEN as entry ----
        if state == "FLAT" and i == open_today_bar_idx and cur_day_sig != 0 and cur_day_sigma is not None:
            entry_at = bar_open
            qty = int(NOTIONAL // entry_at) if entry_at > 0 else 0
            if qty > 0:
                cost = qty * entry_at * cost_frac
                daily_bucket[bar_date] = daily_bucket.get(bar_date, 0.0) - cost
                state = "IN_LONG" if cur_day_sig > 0 else "IN_SHORT"
                entry_price = entry_at
                entry_sigma = cur_day_sigma
                entry_qty = qty
                entry_side = 1 if cur_day_sig > 0 else -1
                # Initialize peak/trough at the bar's high/low (we entered at open)
                if cur_day_sig > 0:
                    peak = bar_high
                    trough = None
                else:
                    trough = bar_low
                    peak = None
                # Book entry-bar move: from entry_at → bar_close
                step = (bar_close - entry_at) * qty * entry_side
                daily_bucket[bar_date] = daily_bucket.get(bar_date, 0.0) + step

        prev_close = bar_close
        prev_state = state
        prev_qty = entry_qty
        prev_side = entry_side

    # End: mark-to-market open position at last close
    open_unrz = 0.0
    if state != "FLAT" and entry_price is not None:
        last_close = float(cl[n - 1])
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

    return {
        "n_trades": len(rt),
        "wins": wins,
        "win_rate": win_rate,
        "net": net,
        "realized": realized,
        "open_unrealized": open_unrz,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann": ann,
        "trades": rt,
    }


def _empty_result() -> dict:
    return {
        "n_trades": 0, "wins": 0, "win_rate": 0.0, "net": 0.0,
        "realized": 0.0, "open_unrealized": 0.0,
        "sharpe": float("nan"), "max_dd": 0.0, "ann": float("nan"),
        "trades": [],
    }


# ---------------------------------------------------------------------------
# Portfolio runner — runs every variant against every asset, returns dict-of-dicts.
# ---------------------------------------------------------------------------
def run_all_variants() -> dict:
    """Returns {variant_label: {asset: per-asset-result-dict}}."""
    # Pre-load every asset (slow, do once)
    print("\n  Loading bars and pre-computing signals...")
    asset_data = {}
    for (asset, mode, trail, pt, dlib, dsym, ilib, isym) in CONFIGS:
        print(f"    {asset}...", flush=True)
        daily = load_daily(dlib, dsym)
        feat = add_per_asset_features(daily)
        sig = precompute_signal(asset, daily, mode)
        intra = load_intraday_rth(ilib, isym)
        # Restrict intraday to OOS window
        intra = intra.loc[(intra.index >= OOS_START.tz_localize("UTC"))
                          & (intra.index <= OOS_END.tz_localize("UTC"))]
        # Restrict daily to OOS too (signals still pre-computed on full history
        # so warmup is honored)
        asset_data[asset] = (daily, feat, sig, intra, trail, pt)

    results: dict = {}
    for (label, b_start, b_end) in VARIANTS:
        print(f"\n  Variant: {label}", flush=True)
        results[label] = {}
        for (asset, mode, trail, pt, *_rest) in CONFIGS:
            daily, feat, sig, intra, trail_cfg, pt_cfg = asset_data[asset]
            r = replay_intraday(
                daily, feat, sig, intra,
                trail_cfg, pt_cfg, COST_BP,
                b_start, b_end,
            )
            results[label][asset] = r
            print(
                f"    {asset:>5}: trades={r['n_trades']:>3}  "
                f"win%={r['win_rate']:>5.1f}  "
                f"net=${r['net']:>+10,.0f}  "
                f"sharpe={r['sharpe']:>+5.2f}",
                flush=True,
            )
    return results


def summarize(results: dict) -> None:
    print()
    print("=" * 120)
    print("  TIME-OF-DAY FILTER BAKE-OFF (OOS intraday: 2024-05-20 → 2026-05-19)")
    print("=" * 120)

    # Header table: variant × portfolio total
    print()
    print(f"  {'Variant':<22} {'Trades':>8}  {'WinRate':>8}  {'Net PnL':>14}  "
          f"{'Ret %':>8}  {'Ann %':>8}  {'Sharpe':>7}  {'MaxDD':>11}  {'vs Ctrl':>10}")
    print("  " + "-" * 116)

    ctrl_net = None
    summary_rows = []
    for (label, b_start, b_end) in VARIANTS:
        per = results[label]
        portfolio_net = sum(r["net"] for r in per.values())
        portfolio_trades = sum(r["n_trades"] for r in per.values())
        portfolio_wins = sum(r["wins"] for r in per.values())
        wr = (portfolio_wins / portfolio_trades * 100.0) if portfolio_trades else 0.0
        ret = portfolio_net / NET_LIQ * 100.0
        years = (OOS_END - OOS_START).days / 365.25
        ann = float("nan")
        if years > 0 and NET_LIQ + portfolio_net > 0:
            ann = ((1 + portfolio_net / NET_LIQ) ** (1 / years) - 1) * 100.0

        # Portfolio-level sharpe (sum per-asset per-day buckets is unfair;
        # use mean of per-asset sharpes weighted by trade count instead, fallback to mean)
        sharpes = [r["sharpe"] for r in per.values()
                   if r["n_trades"] > 0 and math.isfinite(r["sharpe"])]
        sharpe_avg = float(np.mean(sharpes)) if sharpes else float("nan")
        max_dds = [r["max_dd"] for r in per.values()]
        max_dd_sum = float(np.sum(max_dds))

        if ctrl_net is None:
            ctrl_net = portfolio_net
            delta_str = "  (ctrl)"
        else:
            delta = portfolio_net - ctrl_net
            sign = "+" if delta >= 0 else ""
            delta_str = f"${sign}{delta:,.0f}"

        print(f"  {label:<22} {portfolio_trades:>8d}  {wr:>7.1f}%  "
              f"${portfolio_net:>+12,.0f}  {ret:>+6.2f}%  {ann:>+6.2f}%  "
              f"{sharpe_avg:>+6.2f}  ${max_dd_sum:>+8,.0f}  {delta_str:>10s}")
        summary_rows.append({
            "label": label, "net": portfolio_net, "trades": portfolio_trades,
            "wr": wr, "ret": ret, "ann": ann, "sharpe": sharpe_avg,
        })

    print()
    print("=" * 120)

    # Find winning variant
    ctrl = summary_rows[0]
    winners = sorted(summary_rows[1:], key=lambda r: r["net"], reverse=True)
    best = winners[0]
    delta = best["net"] - ctrl["net"]

    if delta > 0:
        print(f"\n  BEST VARIANT: {best['label']}  (+${delta:,.0f} vs control = +{delta/NET_LIQ*100:.2f}% net liq)")
    else:
        print(f"\n  NO VARIANT BEATS CONTROL — best is {best['label']} at {delta/NET_LIQ*100:+.2f}% vs ctrl")

    # Per-asset breakdown for the winning variant
    print()
    print(f"  PER-ASSET BREAKDOWN — {best['label']}")
    print(f"  {'Asset':<6} {'Trades':>7}  {'Win%':>6}  "
          f"{'Net':>12}  {'vs Ctrl':>12}  {'Sharpe':>7}")
    print("  " + "-" * 60)
    for asset in [c[0] for c in CONFIGS]:
        r_best = results[best["label"]][asset]
        r_ctrl = results[ctrl["label"]][asset]
        d = r_best["net"] - r_ctrl["net"]
        sign = "+" if d >= 0 else ""
        print(f"  {asset:<6} {r_best['n_trades']:>7d}  {r_best['win_rate']:>5.1f}%  "
              f"${r_best['net']:>+10,.0f}  ${sign}{d:>+10,.0f}  "
              f"{r_best['sharpe']:>+6.2f}")

    # Exit-count diagnostic: did the filter reduce exits?
    print()
    print("  EXIT-COUNT DIAGNOSTIC (should drop with more aggressive filtering):")
    for label, *_ in VARIANTS:
        total = sum(r["n_trades"] for r in results[label].values())
        print(f"    {label:<22} {total:>6d} trades")

    # Time-of-day distribution of exits in the control
    print()
    print("  EXIT TIME-OF-DAY DISTRIBUTION (control = NoFilter):")
    minute_buckets = {}
    for r in results["NoFilter (control)"].values():
        for t in r["trades"]:
            m = t["minute_et"]
            bucket = m - (m % 30)  # 30-min buckets
            minute_buckets[bucket] = minute_buckets.get(bucket, 0) + 1
    total_exits = sum(minute_buckets.values())
    if total_exits > 0:
        for bucket in sorted(minute_buckets.keys()):
            hh, mm = divmod(bucket, 60)
            count = minute_buckets[bucket]
            pct = count / total_exits * 100
            bar = "#" * int(pct / 2)
            print(f"    {hh:02d}:{mm:02d} ET  {count:>4d}  {pct:>5.1f}%  {bar}")


def main() -> int:
    print()
    print("=" * 120)
    print("  89_tod_filter_test — testing whether blocking morning exits beats baseline")
    print(f"  Window: {OOS_START.date()} → {OOS_END.date()}  ({(OOS_END - OOS_START).days} days)")
    print(f"  Variants: {[v[0] for v in VARIANTS]}")
    print("=" * 120)

    results = run_all_variants()
    summarize(results)

    print("\n" + "=" * 120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
