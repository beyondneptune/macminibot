"""98_asset_specific_tod_test — Asset-specific time-of-day filter bake-off.

Background:
  - Universal 30-min block (09:30-10:00 ET) is the current production filter
    (established by 89_tod_filter_test).
  - Earlier diagnostics found ~60% of premature exits cluster 09:30-10:30 ET,
    mostly on AMZN/TSLA (tight 0.5sigma trails).
  - Hypothesis: AMZN/TSLA need a LONGER block; AAPL/GLD (wider trails) may not
    need a block at all.

This replay mirrors 89_tod_filter_test exactly (daily signal + 5-min intraday
replay with HIGH/LOW honesty), but the time-of-day filter window is *per-asset*
not portfolio-wide. Each variant supplies a function asset -> (start_min, end_min)
or None to skip the block for that asset.

Variants:
  A  NoFilter                        — control, no filter for any asset.
  B  Universal 30-min                — block 09:30-10:00 ET for ALL assets (production).
  C  AMZN/TSLA 90-min, others 30-min — AMZN/TSLA block 09:30-11:00 ET; others 09:30-10:00.
  D  AMZN/TSLA 60-min, others NONE   — AMZN/TSLA block 09:30-10:30 ET; everyone else OFF.
  E  All-Off                         — same as A (sanity check / redundancy).

Run:
    .venv/bin/python scripts/98_asset_specific_tod_test.py
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
# Sizing / cost constants — match 89_tod_filter_test.py and v2 deployment.
# ---------------------------------------------------------------------------
NET_LIQ = 100000.0
NOTIONAL = NET_LIQ / 8.0
WARMUP = 252
COST_BP = 3.0

OOS_START = pd.Timestamp("2024-05-20")
OOS_END = pd.Timestamp("2026-05-19")

RTH_OPEN_MIN = 9 * 60 + 30       # 570
RTH_CLOSE_MIN = 16 * 60          # 960

# Per-asset config (matches v2 deployment / 89_tod_filter_test)
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

# Minute-of-day shortcuts for readability.
M_0930 = 9 * 60 + 30
M_1000 = 10 * 60
M_1030 = 10 * 60 + 30
M_1100 = 11 * 60


def _block_universal_30(_asset: str):
    return (M_0930, M_1000)


def _block_split_amzn_tsla_90(asset: str):
    if asset in ("AMZN", "TSLA"):
        return (M_0930, M_1100)   # 90-min block
    return (M_0930, M_1000)       # 30-min block for others


def _block_amzn_tsla_only_60(asset: str):
    if asset in ("AMZN", "TSLA"):
        return (M_0930, M_1030)   # 60-min block
    return None                   # no block


def _block_none(_asset: str):
    return None


# (label, asset -> (start_min, end_min) | None)
VARIANTS = [
    ("A  NoFilter",                     _block_none),
    ("B  Universal 30-min",             _block_universal_30),
    ("C  AMZN/TSLA 90 + others 30",     _block_split_amzn_tsla_90),
    ("D  AMZN/TSLA 60 only",            _block_amzn_tsla_only_60),
    ("E  All-Off",                      _block_none),
]


# ---------------------------------------------------------------------------
# Data loading (identical to 89_tod_filter_test)
# ---------------------------------------------------------------------------
def load_daily(lib: str, sym: str) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()[["open", "high", "low", "close", "volume"]]


def load_intraday_rth(lib: str, sym: str) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[["open", "high", "low", "close", "volume"]]
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
    df["et_date"] = et[mask].normalize().date
    return df


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


# ---------------------------------------------------------------------------
# Replay core (identical to 89_tod_filter_test, but block window is passed in)
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
    cost_frac = cost_bp / 1e4

    daily_dates = daily.index.normalize()
    daily_dates_py = pd.DatetimeIndex(daily_dates).date
    daily_date_to_idx = {d: i for i, d in enumerate(daily_dates_py)}

    vol = daily_feat["vol_21"].values

    op = intraday["open"].astype(float).values
    hi = intraday["high"].astype(float).values
    lo = intraday["low"].astype(float).values
    cl = intraday["close"].astype(float).values
    mod = intraday["minute_of_et_day"].astype(int).values
    edate = intraday["et_date"].values

    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    entry_side = 0
    peak = trough = None
    rt = []
    daily_bucket: dict = {}
    prev_close = None
    prev_qty = 0
    prev_side = 0
    prev_state = "FLAT"
    open_today_bar_idx = -1
    cur_day_sig = 0
    cur_day_sigma = None

    n = len(intraday)
    if n == 0:
        return _empty_result()

    cur_date = None

    for i in range(n):
        bar_close = float(cl[i])
        bar_open = float(op[i])
        bar_high = float(hi[i])
        bar_low = float(lo[i])
        minute = int(mod[i])
        bar_date = edate[i]

        if bar_date != cur_date:
            cur_date = bar_date
            d_idx = daily_date_to_idx.get(cur_date)
            if d_idx is None or d_idx == 0:
                cur_day_sig = 0
                cur_day_sigma = None
            else:
                src_idx = d_idx - 1
                cur_day_sig = int(sig[src_idx])
                v = float(vol[src_idx])
                cur_day_sigma = v if (math.isfinite(v) and v > 0) else None
            open_today_bar_idx = i

        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            step = (bar_close - prev_close) * prev_qty * prev_side
            daily_bucket[bar_date] = daily_bucket.get(bar_date, 0.0) + step

        in_tod_block = (
            block_start_min is not None
            and block_end_min is not None
            and block_start_min <= minute < block_end_min
        )

        if state == "IN_LONG":
            if peak is None or bar_high > peak:
                peak = bar_high

            band = entry_price * entry_sigma * trail
            stop_price = peak - band
            reason = None
            exit_price = None

            if not in_tod_block:
                if bar_low <= stop_price:
                    reason = "trail"
                    exit_price = stop_price if bar_open > stop_price else bar_open
                elif pt is not None:
                    pt_price = entry_price * (1 + entry_sigma * pt)
                    if bar_high >= pt_price:
                        reason = "pt"
                        exit_price = pt_price if bar_open < pt_price else bar_open
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
                if cur_day_sig > 0:
                    peak = bar_high
                    trough = None
                else:
                    trough = bar_low
                    peak = None
                step = (bar_close - entry_at) * qty * entry_side
                daily_bucket[bar_date] = daily_bucket.get(bar_date, 0.0) + step

        prev_close = bar_close
        prev_state = state
        prev_qty = entry_qty
        prev_side = entry_side

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
        "daily_bucket": daily_bucket,
    }


def _empty_result() -> dict:
    return {
        "n_trades": 0, "wins": 0, "win_rate": 0.0, "net": 0.0,
        "realized": 0.0, "open_unrealized": 0.0,
        "sharpe": float("nan"), "max_dd": 0.0, "ann": float("nan"),
        "trades": [], "daily_bucket": {},
    }


# ---------------------------------------------------------------------------
# Portfolio runner — asset-specific block windows.
# ---------------------------------------------------------------------------
def run_all_variants() -> dict:
    print("\n  Loading bars and pre-computing signals...")
    asset_data = {}
    for (asset, mode, trail, pt, dlib, dsym, ilib, isym) in CONFIGS:
        print(f"    {asset}...", flush=True)
        daily = load_daily(dlib, dsym)
        feat = add_per_asset_features(daily)
        sig = precompute_signal(asset, daily, mode)
        intra = load_intraday_rth(ilib, isym)
        intra = intra.loc[(intra.index >= OOS_START.tz_localize("UTC"))
                          & (intra.index <= OOS_END.tz_localize("UTC"))]
        asset_data[asset] = (daily, feat, sig, intra, trail, pt)

    results: dict = {}
    for (label, block_fn) in VARIANTS:
        print(f"\n  Variant: {label}", flush=True)
        results[label] = {}
        for (asset, mode, trail, pt, *_rest) in CONFIGS:
            daily, feat, sig, intra, trail_cfg, pt_cfg = asset_data[asset]
            window = block_fn(asset)
            b_start, b_end = (window if window is not None else (None, None))
            r = replay_intraday(
                daily, feat, sig, intra,
                trail_cfg, pt_cfg, COST_BP,
                b_start, b_end,
            )
            results[label][asset] = r
            wstr = (
                f"block={b_start//60:02d}:{b_start%60:02d}-{b_end//60:02d}:{b_end%60:02d}"
                if b_start is not None else "block=off"
            )
            print(
                f"    {asset:>5}: {wstr:<18} trades={r['n_trades']:>3}  "
                f"win%={r['win_rate']:>5.1f}  "
                f"net=${r['net']:>+10,.0f}  "
                f"sharpe={r['sharpe']:>+5.2f}",
                flush=True,
            )
    return results


def portfolio_sharpe(per: dict) -> float:
    """Portfolio-level Sharpe by summing per-asset daily P&L buckets."""
    daily_sum: dict = {}
    for r in per.values():
        for d, v in r.get("daily_bucket", {}).items():
            daily_sum[d] = daily_sum.get(d, 0.0) + v
    if not daily_sum or len(daily_sum) < 2:
        return float("nan")
    rets = np.asarray(list(daily_sum.values())) / NET_LIQ
    if rets.std() <= 0:
        return float("nan")
    return float(rets.mean() / rets.std() * math.sqrt(252))


def summarize(results: dict) -> None:
    print()
    print("=" * 130)
    print("  ASSET-SPECIFIC TIME-OF-DAY FILTER BAKE-OFF (OOS: 2024-05-20 → 2026-05-19)")
    print("=" * 130)

    print()
    print(f"  {'Variant':<32} {'Trades':>8}  {'WinRate':>8}  {'Net PnL':>14}  "
          f"{'Ret %':>8}  {'Ann %':>8}  {'PortSharpe':>11}  {'AvgAssetSharpe':>15}  "
          f"{'MaxDDsum':>11}  {'vs Ctrl':>12}")
    print("  " + "-" * 128)

    ctrl_net = None
    summary_rows = []
    for (label, _block_fn) in VARIANTS:
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

        port_sharpe = portfolio_sharpe(per)
        sharpes = [r["sharpe"] for r in per.values()
                   if r["n_trades"] > 0 and math.isfinite(r["sharpe"])]
        avg_asset_sharpe = float(np.mean(sharpes)) if sharpes else float("nan")
        max_dds = [r["max_dd"] for r in per.values()]
        max_dd_sum = float(np.sum(max_dds))

        if ctrl_net is None:
            ctrl_net = portfolio_net
            delta_str = "  (ctrl)"
        else:
            delta = portfolio_net - ctrl_net
            sign = "+" if delta >= 0 else ""
            delta_str = f"${sign}{delta:,.0f}"

        print(f"  {label:<32} {portfolio_trades:>8d}  {wr:>7.1f}%  "
              f"${portfolio_net:>+12,.0f}  {ret:>+6.2f}%  {ann:>+6.2f}%  "
              f"{port_sharpe:>+10.2f}  {avg_asset_sharpe:>+14.2f}  "
              f"${max_dd_sum:>+8,.0f}  {delta_str:>12s}")
        summary_rows.append({
            "label": label, "net": portfolio_net, "trades": portfolio_trades,
            "wr": wr, "ret": ret, "ann": ann,
            "port_sharpe": port_sharpe,
            "avg_asset_sharpe": avg_asset_sharpe,
        })

    print()
    print("=" * 130)

    ctrl = summary_rows[0]
    universal_30 = summary_rows[1]
    challengers = summary_rows[1:]
    best = max(challengers, key=lambda r: r["net"])

    print(f"\n  BEST CHALLENGER: {best['label']}")
    print(f"    Net = ${best['net']:+,.0f}  vs Universal 30-min (B) = ${universal_30['net']:+,.0f}  "
          f"(delta = ${best['net'] - universal_30['net']:+,.0f})")
    print(f"    Net vs NoFilter (A)   = ${best['net'] - ctrl['net']:+,.0f}")

    # Per-asset breakdown for every variant
    print()
    print("  PER-ASSET BREAKDOWN — net P&L per asset, all variants")
    header = "  " + f"{'Asset':<6}"
    for row in summary_rows:
        tag = row["label"].split()[0]
        header += f"  {tag:>14}"
    print(header)
    print("  " + "-" * (8 + 16 * len(summary_rows)))
    for asset in [c[0] for c in CONFIGS]:
        line = f"  {asset:<6}"
        for row in summary_rows:
            r = results[row["label"]][asset]
            line += f"  ${r['net']:>+12,.0f}"
        print(line)

    # Trade counts
    print()
    print("  PER-ASSET TRADE COUNT:")
    header = "  " + f"{'Asset':<6}"
    for row in summary_rows:
        tag = row["label"].split()[0]
        header += f"  {tag:>8}"
    print(header)
    print("  " + "-" * (8 + 10 * len(summary_rows)))
    for asset in [c[0] for c in CONFIGS]:
        line = f"  {asset:<6}"
        for row in summary_rows:
            r = results[row["label"]][asset]
            line += f"  {r['n_trades']:>8d}"
        print(line)

    # Per-asset Sharpe
    print()
    print("  PER-ASSET SHARPE:")
    header = "  " + f"{'Asset':<6}"
    for row in summary_rows:
        tag = row["label"].split()[0]
        header += f"  {tag:>8}"
    print(header)
    print("  " + "-" * (8 + 10 * len(summary_rows)))
    for asset in [c[0] for c in CONFIGS]:
        line = f"  {asset:<6}"
        for row in summary_rows:
            r = results[row["label"]][asset]
            s = r["sharpe"]
            line += f"  {s:>+8.2f}" if math.isfinite(s) else f"  {'nan':>8}"
        print(line)

    # Headline comparison: B vs C/D
    print()
    print("  HEADLINE: Asset-specific vs Universal 30-min (B)")
    for row in challengers:
        if row["label"] == universal_30["label"]:
            continue
        delta_pnl = row["net"] - universal_30["net"]
        delta_sharpe = row["port_sharpe"] - universal_30["port_sharpe"]
        sign = "BEATS" if delta_pnl > 0 else "loses to"
        print(f"    {row['label']:<32} {sign} B by ${delta_pnl:+,.0f} "
              f"(port-sharpe delta {delta_sharpe:+.3f})")

    # Reduction in exit count vs control
    print()
    print("  EXIT-COUNT REDUCTION vs NoFilter (control):")
    ctrl_trades = ctrl["trades"]
    for row in summary_rows:
        delta = row["trades"] - ctrl_trades
        print(f"    {row['label']:<32}  trades={row['trades']:>4d}  delta={delta:>+4d}")


def main() -> int:
    print()
    print("=" * 130)
    print("  98_asset_specific_tod_test — per-asset time-of-day block window bake-off")
    print(f"  Window: {OOS_START.date()} → {OOS_END.date()}  ({(OOS_END - OOS_START).days} days)")
    print(f"  Variants:")
    for label, _ in VARIANTS:
        print(f"      {label}")
    print("=" * 130)

    results = run_all_variants()
    summarize(results)

    print("\n" + "=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
