"""101_trail_optimization — constrained per-asset trail/PT sweep.

Goal: reduce bull-market drag in OOS_RECENT (2026-01 → 2026-05) by ≥30%
WITHOUT growing per-asset COVID or RATESHOCK max-drawdowns by more than 5%.

Targets: AAPL, MSFT, GOOGL, AMZN, NVDA (5 of the 8 v2 names).
Hold-fixed: META (lgbm), TSLA (macd), GLD (frozen baseline).

Sweep grid:
  trail σ ∈ {3.0, 3.5, 4.0, 4.5, 5.0}
  PT σ    ∈ {None, 3.0, 5.0, 7.0}
Per-asset = 20 cells × 3 windows = 60 evaluations per asset.

Because per-asset P&L is additive at the portfolio level (fixed per-name
notional, no rebalancing), the optimization decomposes into independent
per-asset problems. Per-asset constraint = per-asset max DD ≤ baseline × 1.05.
Portfolio drag reduction is then the sum of per-asset improvements.

Run:
    .venv/bin/python scripts/101_trail_optimization.py
    .venv/bin/python scripts/101_trail_optimization.py --apply
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from backtesting.fees import equity_tiered  # noqa: E402
from backtesting.slippage import slippage_bps_with_floor  # noqa: E402
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from scripts._storage import open_store  # noqa: E402


NET_LIQ = 100_000.0
N_ASSETS = 8
PER_NAME_NOTIONAL = NET_LIQ / N_ASSETS
WARMUP = 252

# Baseline config (signal_mode, trail_σ, PT_σ, arctic_lib, arctic_sym)
BASELINE_CONFIG = {
    "AAPL":  ("buy_and_hold",      3.0, 3.0,  "equity.daily_bars", "NASDAQ.AAPL"),
    "MSFT":  ("rsi2_mr_10_90",     1.0, 5.0,  "equity.daily_bars", "NASDAQ.MSFT"),
    "GOOGL": ("buy_and_hold",      2.0, None, "equity.daily_bars", "NASDAQ.GOOGL"),
    "AMZN":  ("buy_and_hold",      0.5, 5.0,  "equity.daily_bars", "NASDAQ.AMZN"),
    "META":  ("lgbm",              2.0, None, "equity.daily_bars", "NASDAQ.META"),
    "TSLA":  ("macd",              0.5, 3.0,  "equity.daily_bars", "NASDAQ.TSLA"),
    "NVDA":  ("buy_and_hold",      2.0, None, "equity.daily_bars", "NASDAQ.NVDA"),
    "GLD":   ("buy_and_hold",      2.0, 3.0,  "etf.daily_bars",    "ARCA.GLD"),
}

TARGET_ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
# Goal says "e.g., 3.0σ to 5.0σ" but for very-tight baselines (AMZN 0.5σ,
# MSFT 1.0σ) intermediate values are still "wider". Include them.
TRAIL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
PT_GRID = [None, 3.0, 5.0, 7.0, 10.0]

WINDOWS = {
    "OOS_RECENT": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-20")),
    "COVID":      (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
    "RATESHOCK":  (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30")),
}

MAX_DD_GROWTH = 1.05    # per-asset max DD may grow by ≤5%
DRAG_REDUCTION_TARGET = 0.30  # OOS_RECENT alpha drag must shrink ≥30%


# ---------------------------------------------------------------------------
# Data + signals (lifted from 100_continuous_learning_backtest)
# ---------------------------------------------------------------------------


def _load_arctic(lib: str, arctic_sym: str) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib).read(arctic_sym).data.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = df.index.normalize()
    df.index.name = "date"
    return df.sort_index()[["open", "high", "low", "close", "volume"]]


def _load_alpaca_gap(ticker: str) -> pd.DataFrame:
    """Pull recent gap-fill bars from Alpaca IEX."""
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
    )
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=60)
    req = StockBarsRequest(
        symbol_or_symbols=[ticker], timeframe=TimeFrame.Day,
        start=start, end=end, feed=DataFeed.IEX,
    )
    try:
        bars = client.get_stock_bars(req).df
    except Exception:
        return pd.DataFrame()
    if bars is None or bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    return out.sort_index()


def load_bars(asset: str) -> pd.DataFrame:
    _, _, _, lib, sym = BASELINE_CONFIG[asset]
    df = _load_arctic(lib, sym)
    # gap-fill recent
    last = df.index[-1].normalize()
    if (pd.Timestamp.utcnow().normalize().tz_localize(None) - last).days > 1:
        gap = _load_alpaca_gap(asset)
        if not gap.empty:
            gap = gap[~gap.index.isin(df.index)]
            df = pd.concat([df, gap]).sort_index()
    return df


def _signal_buy_and_hold(n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int8)
    out[WARMUP:] = 1
    return out


def _signal_rsi2_mr_10_90(feat: pd.DataFrame) -> np.ndarray:
    r = feat["rsi_2"].shift(1).values
    out = np.where(r < 10, 1, np.where(r > 90, -1, 0)).astype(np.int8)
    out[:WARMUP] = 0
    return out


def _signal_macd(feat: pd.DataFrame) -> np.ndarray:
    h = feat["macd_hist"].shift(1).values
    out = np.where(h > 0, 1, np.where(h < 0, -1, 0)).astype(np.int8)
    out[:WARMUP] = 0
    return out


def signal_for(asset: str, bars: pd.DataFrame, feat: pd.DataFrame) -> np.ndarray:
    mode, _, _, _, _ = BASELINE_CONFIG[asset]
    n = len(bars)
    if mode == "buy_and_hold":
        return _signal_buy_and_hold(n)
    if mode == "rsi2_mr_10_90":
        return _signal_rsi2_mr_10_90(feat)
    if mode == "macd":
        return _signal_macd(feat)
    # lgbm + others not in TARGET_ASSETS — return zeros (we don't sweep them)
    return np.zeros(n, dtype=np.int8)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def fill_cost_usd(qty: int, price: float, year: int,
                   daily_vol_bps: float, adv: float) -> float:
    if qty <= 0 or price <= 0:
        return 0.0
    commission = equity_tiered(float(qty), float(price)).total
    notional = qty * price
    slip = slippage_bps_with_floor(
        volume_traded=float(qty), adv=float(max(adv, 1.0)),
        daily_vol_bps=float(max(daily_vol_bps, 1.0)),
        asset_class="equity", k=0.1,
    )
    impact = notional * (slip.total_bps / 1e4)
    return commission + impact


# ---------------------------------------------------------------------------
# Replay — stripped down version of 100's replay, single asset
# ---------------------------------------------------------------------------


def replay_asset(bars: pd.DataFrame, feat: pd.DataFrame, sig: np.ndarray,
                  start_idx: int, end_idx: int, trail: float,
                  pt: float | None, notional: float) -> dict:
    closes = bars["close"].astype(float).values
    vol_21 = feat["vol_21"].values
    adv_series = bars["volume"].rolling(21).mean().fillna(method="bfill").values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    daily_bucket: dict = {}
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0
    realized = 0.0

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol_21[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig[i]) if i < len(sig) else 0
        close = float(closes[i])
        dk = bars.index[i].normalize()
        year = bars.index[i].year
        daily_vol_bps = sigma * math.sqrt(252) * 10_000.0
        adv = max(float(adv_series[i]) if math.isfinite(adv_series[i]) else 0.0, 1.0)

        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            daily_bucket[dk] = daily_bucket.get(dk, 0.0) + (close - prev_close) * prev_qty * prev_side

        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            stop = peak - entry_price * entry_sigma * trail
            reason = None
            if close <= stop:
                reason = "trail"
            elif pt is not None and close >= entry_price * (1 + entry_sigma * pt):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                gross = (close - entry_price) * entry_qty
                cost = fill_cost_usd(entry_qty, close, year, daily_vol_bps, adv)
                pnl = gross - cost
                realized += pnl
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            stop = trough + entry_price * entry_sigma * trail
            reason = None
            if close >= stop:
                reason = "trail"
            elif pt is not None and close <= entry_price * (1 - entry_sigma * pt):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                gross = (entry_price - close) * entry_qty
                cost = fill_cost_usd(entry_qty, close, year, daily_vol_bps, adv)
                pnl = gross - cost
                realized += pnl
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        if state == "FLAT" and sig_dir != 0:
            qty = int(notional // close) if close > 0 else 0
            if qty > 0:
                cost = fill_cost_usd(qty, close, year, daily_vol_bps, adv)
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
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
    if state != "FLAT" and entry_price is not None and end_idx > 0:
        last_close = float(closes[end_idx - 1])
        if state == "IN_LONG":
            open_unrz = (last_close - entry_price) * entry_qty
        else:
            open_unrz = (entry_price - last_close) * entry_qty

    net = realized + open_unrz
    max_dd = 0.0
    if daily_bucket:
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    return {"net": net, "max_dd": max_dd}


def bh_for_asset(bars: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                  notional: float) -> float:
    df = bars[(bars.index >= start_ts) & (bars.index <= end_ts)]
    if len(df) < 2:
        return 0.0
    entry = float(df["close"].iloc[0])
    exit_p = float(df["close"].iloc[-1])
    qty = int(notional // entry) if entry > 0 else 0
    if qty == 0:
        return 0.0
    year_start = df.index[0].year
    year_end = df.index[-1].year
    adv = float(df["volume"].iloc[:21].mean() if len(df) >= 21 else df["volume"].mean())
    entry_cost = fill_cost_usd(qty, entry, year_start, 100.0, adv)
    exit_cost = fill_cost_usd(qty, exit_p, year_end, 100.0, adv)
    return (exit_p - entry) * qty - entry_cost - exit_cost


# ---------------------------------------------------------------------------
# Per-asset sweep
# ---------------------------------------------------------------------------


def evaluate_per_asset(asset: str, bars: pd.DataFrame, feat: pd.DataFrame,
                        trail: float, pt: float | None) -> dict:
    """Evaluate (trail, pt) for one asset across the 3 windows. Returns
    {window: {net, max_dd}, "bh_oos": ...}."""
    sig = signal_for(asset, bars, feat)
    out = {}
    for wname, (start_ts, end_ts) in WINDOWS.items():
        s_idx = bars.index.get_indexer([start_ts], method="bfill")[0]
        e_idx = bars.index.get_indexer([end_ts], method="ffill")[0] + 1
        e_idx = min(e_idx, len(bars))
        if s_idx <= WARMUP or e_idx <= s_idx:
            out[wname] = {"net": 0.0, "max_dd": 0.0}
            continue
        out[wname] = replay_asset(bars, feat, sig, s_idx, e_idx, trail, pt, PER_NAME_NOTIONAL)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="If satisfying config found, write to live asset modules")
    args = ap.parse_args()

    print("=" * 130)
    print("  TRAIL OPTIMIZATION — constrained per-asset sweep")
    print("=" * 130)
    print(f"  Targets:   {TARGET_ASSETS}")
    print(f"  Trail grid: {TRAIL_GRID}")
    print(f"  PT grid:    {PT_GRID}")
    print(f"  Constraint: per-asset max DD growth ≤ {(MAX_DD_GROWTH-1)*100:.0f}% in COVID and RATESHOCK")
    print(f"  Target:    OOS_RECENT alpha-drag reduction ≥ {DRAG_REDUCTION_TARGET*100:.0f}%")
    print("=" * 130)

    # Load all 8 baseline assets (need them for B&H + non-target baseline)
    print("\n  Loading bars...")
    bars_cache: dict[str, pd.DataFrame] = {}
    feat_cache: dict[str, pd.DataFrame] = {}
    for asset in BASELINE_CONFIG.keys():
        try:
            bars = load_bars(asset)
            bars_cache[asset] = bars
            feat_cache[asset] = add_per_asset_features(bars)
            print(f"    {asset}: {len(bars)} bars")
        except Exception as exc:
            print(f"    {asset}: ERROR {exc}")

    # Baseline metrics per asset per window
    print("\n  Computing baselines...")
    baseline_per_asset: dict[str, dict] = {}
    bh_per_asset: dict[str, dict] = {}
    for asset in BASELINE_CONFIG.keys():
        if asset not in bars_cache:
            continue
        _, base_trail, base_pt, _, _ = BASELINE_CONFIG[asset]
        # Skip META — its lgbm signal would need re-computation each cell;
        # we hold it fixed at baseline anyway.
        if BASELINE_CONFIG[asset][0] == "lgbm":
            baseline_per_asset[asset] = {w: {"net": 0.0, "max_dd": 0.0} for w in WINDOWS}
        else:
            baseline_per_asset[asset] = evaluate_per_asset(
                asset, bars_cache[asset], feat_cache[asset], base_trail, base_pt)
        bh_per_asset[asset] = {
            w: bh_for_asset(bars_cache[asset], s_ts, e_ts, PER_NAME_NOTIONAL)
            for w, (s_ts, e_ts) in WINDOWS.items()
        }

    # Baseline portfolio OOS_RECENT alpha drag
    baseline_oos_bot = sum(baseline_per_asset[a]["OOS_RECENT"]["net"]
                            for a in baseline_per_asset)
    baseline_oos_bh = sum(bh_per_asset[a]["OOS_RECENT"]
                          for a in baseline_per_asset)
    baseline_drag = baseline_oos_bot - baseline_oos_bh
    print(f"\n  Baseline OOS_RECENT: bot=${baseline_oos_bot:+,.0f}  "
          f"B&H=${baseline_oos_bh:+,.0f}  drag=${baseline_drag:+,.0f}")

    # Per-asset sweep
    print("\n  Per-asset sweep (trail × PT grid)...")
    sweep_results: dict[str, list[dict]] = {}
    for asset in TARGET_ASSETS:
        if asset not in bars_cache:
            continue
        print(f"\n  === {asset} ===")
        base = baseline_per_asset[asset]
        bh = bh_per_asset[asset]
        cells = []
        for tr in TRAIL_GRID:
            for pt in PT_GRID:
                r = evaluate_per_asset(asset, bars_cache[asset], feat_cache[asset], tr, pt)
                # Constraint: COVID + RATESHOCK max DD growth ≤ 5% per asset
                covid_dd_ratio = (abs(r["COVID"]["max_dd"]) / abs(base["COVID"]["max_dd"])
                                   if base["COVID"]["max_dd"] != 0 else 1.0)
                ratesh_dd_ratio = (abs(r["RATESHOCK"]["max_dd"]) / abs(base["RATESHOCK"]["max_dd"])
                                    if base["RATESHOCK"]["max_dd"] != 0 else 1.0)
                passes = covid_dd_ratio <= MAX_DD_GROWTH and ratesh_dd_ratio <= MAX_DD_GROWTH
                oos_improve = r["OOS_RECENT"]["net"] - base["OOS_RECENT"]["net"]
                cells.append({
                    "trail": tr, "pt": pt, "passes": passes,
                    "oos_net": r["OOS_RECENT"]["net"],
                    "oos_improve": oos_improve,
                    "covid_dd": r["COVID"]["max_dd"],
                    "rateshock_dd": r["RATESHOCK"]["max_dd"],
                    "covid_dd_ratio": covid_dd_ratio,
                    "rateshock_dd_ratio": ratesh_dd_ratio,
                })
        # Per-asset best: passes DD constraint AND strictly improves OOS net.
        # If no cell strictly improves, keep baseline (do no harm).
        passing = [c for c in cells if c["passes"] and c["oos_improve"] > 0]
        if passing:
            best = max(passing, key=lambda c: c["oos_improve"])
        else:
            best = None
        sweep_results[asset] = cells
        if best:
            pt_str = f"{best['pt']:.1f}σ" if best["pt"] else "—"
            base_pt_str = f"{BASELINE_CONFIG[asset][2]:.1f}σ" if BASELINE_CONFIG[asset][2] else "—"
            print(f"    baseline: trail={BASELINE_CONFIG[asset][1]}σ pt={base_pt_str}  "
                  f"OOS_net=${base['OOS_RECENT']['net']:+,.0f}  "
                  f"COVID_dd=${base['COVID']['max_dd']:+,.0f}  RS_dd=${base['RATESHOCK']['max_dd']:+,.0f}")
            print(f"    BEST:     trail={best['trail']}σ pt={pt_str}  "
                  f"OOS_net=${best['oos_net']:+,.0f} (Δ ${best['oos_improve']:+,.0f})  "
                  f"COVID_dd×{best['covid_dd_ratio']:.2f} RS_dd×{best['rateshock_dd_ratio']:.2f}")
        else:
            print(f"    no cell passes constraint; keeping baseline")

    # Aggregate: portfolio-level drag reduction if we apply per-asset bests.
    # Only accept cells that STRICTLY improve OOS net (oos_improve > 0) AND
    # pass the DD constraint. Otherwise keep baseline (do no harm).
    new_oos_bot = 0.0
    new_covid_dd = 0.0
    new_rateshock_dd = 0.0
    candidate_config: dict[str, tuple[float, float | None]] = {}
    for asset in BASELINE_CONFIG.keys():
        base = baseline_per_asset.get(asset, {})
        if asset in TARGET_ASSETS:
            cells = sweep_results.get(asset, [])
            passing = [c for c in cells if c["passes"] and c["oos_improve"] > 0]
            if passing:
                best = max(passing, key=lambda c: c["oos_improve"])
                new_oos_bot += best["oos_net"]
                new_covid_dd += best["covid_dd"]
                new_rateshock_dd += best["rateshock_dd"]
                candidate_config[asset] = (best["trail"], best["pt"])
            else:
                new_oos_bot += base.get("OOS_RECENT", {}).get("net", 0.0)
                new_covid_dd += base.get("COVID", {}).get("max_dd", 0.0)
                new_rateshock_dd += base.get("RATESHOCK", {}).get("max_dd", 0.0)
                candidate_config[asset] = (BASELINE_CONFIG[asset][1], BASELINE_CONFIG[asset][2])
        else:
            new_oos_bot += base.get("OOS_RECENT", {}).get("net", 0.0)
            new_covid_dd += base.get("COVID", {}).get("max_dd", 0.0)
            new_rateshock_dd += base.get("RATESHOCK", {}).get("max_dd", 0.0)
            candidate_config[asset] = (BASELINE_CONFIG[asset][1], BASELINE_CONFIG[asset][2])

    new_drag = new_oos_bot - baseline_oos_bh
    drag_reduction = (abs(baseline_drag) - abs(new_drag)) / max(abs(baseline_drag), 1.0)
    baseline_covid_dd = sum(baseline_per_asset[a]["COVID"]["max_dd"] for a in baseline_per_asset)
    baseline_rateshock_dd = sum(baseline_per_asset[a]["RATESHOCK"]["max_dd"] for a in baseline_per_asset)
    covid_dd_ratio_port = abs(new_covid_dd) / abs(baseline_covid_dd) if baseline_covid_dd != 0 else 1.0
    rateshock_dd_ratio_port = abs(new_rateshock_dd) / abs(baseline_rateshock_dd) if baseline_rateshock_dd != 0 else 1.0

    print("\n" + "=" * 130)
    print("  AGGREGATE RESULT")
    print("=" * 130)
    print(f"  Baseline: OOS_drag=${baseline_drag:+,.0f}  "
          f"COVID_DD=${baseline_covid_dd:+,.0f}  RS_DD=${baseline_rateshock_dd:+,.0f}")
    print(f"  Optimized: OOS_drag=${new_drag:+,.0f}  "
          f"COVID_DD=${new_covid_dd:+,.0f}  RS_DD=${new_rateshock_dd:+,.0f}")
    print(f"  Drag reduction:   {drag_reduction*100:+.1f}%  (target ≥{DRAG_REDUCTION_TARGET*100:.0f}%)")
    print(f"  COVID DD growth:  {(covid_dd_ratio_port-1)*100:+.1f}%  (cap +5%)")
    print(f"  RS DD growth:     {(rateshock_dd_ratio_port-1)*100:+.1f}%  (cap +5%)")

    satisfies_drag = drag_reduction >= DRAG_REDUCTION_TARGET
    satisfies_covid = covid_dd_ratio_port <= MAX_DD_GROWTH
    satisfies_rs = rateshock_dd_ratio_port <= MAX_DD_GROWTH
    all_pass = satisfies_drag and satisfies_covid and satisfies_rs
    verdict = "OPTIMAL CONFIG FOUND" if all_pass else "FAILED to meet all constraints"
    print(f"\n  Verdict: {verdict}")

    print("\n  Recommended per-asset config (TARGETS only — others held fixed):")
    for asset in TARGET_ASSETS:
        if asset not in candidate_config:
            continue
        new_trail, new_pt = candidate_config[asset]
        old_trail, old_pt = BASELINE_CONFIG[asset][1], BASELINE_CONFIG[asset][2]
        new_pt_str = f"{new_pt:.1f}σ" if new_pt else "—"
        old_pt_str = f"{old_pt:.1f}σ" if old_pt else "—"
        delta_trail = f"{old_trail:.1f}σ→{new_trail:.1f}σ" if new_trail != old_trail else f"{old_trail:.1f}σ (same)"
        delta_pt = f"{old_pt_str}→{new_pt_str}" if new_pt != old_pt else f"{old_pt_str} (same)"
        print(f"    {asset}: trail={delta_trail}  pt={delta_pt}")

    # Persist the optimization report
    report_path = PROJECT_ROOT / "data" / "trail_optimization_report.csv"
    rows = []
    for asset, cells in sweep_results.items():
        for c in cells:
            rows.append({
                "asset": asset, "trail": c["trail"],
                "pt": c["pt"] if c["pt"] is not None else "None",
                "oos_net": c["oos_net"], "oos_improve": c["oos_improve"],
                "covid_dd": c["covid_dd"], "rateshock_dd": c["rateshock_dd"],
                "covid_dd_ratio": c["covid_dd_ratio"],
                "rateshock_dd_ratio": c["rateshock_dd_ratio"],
                "passes": c["passes"],
            })
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"\n  Full sweep report → {report_path}")

    if all_pass and args.apply:
        print("\n  Applying config to asset modules + reporting deploy diff...")
        _deploy_config(candidate_config)
        print("  DEPLOYED. Verify with the next runner fire.")
    elif all_pass and not args.apply:
        print("\n  Use --apply to write the config to live asset modules.")
    else:
        print("\n  Constraints not all satisfied — no deploy.")

    return 0 if all_pass else 1


def _deploy_config(config: dict[str, tuple[float, float | None]]) -> None:
    """Update each TARGET asset module's WINNER dict in-place.

    PAPER-SAFE: the runner reads this on next cold start (next 14:35 BST fire).
    """
    import re
    for asset in TARGET_ASSETS:
        if asset not in config:
            continue
        new_trail, new_pt = config[asset]
        old_trail, old_pt = BASELINE_CONFIG[asset][1], BASELINE_CONFIG[asset][2]
        if new_trail == old_trail and new_pt == old_pt:
            print(f"    {asset}: no change")
            continue
        module_path = PROJECT_ROOT / "src" / "ml_research" / "asset_specialists" / f"{asset.lower()}.py"
        if not module_path.exists():
            print(f"    {asset}: module {module_path} NOT FOUND — skipping")
            continue
        content = module_path.read_text()
        # Update trail_sigma
        content = re.sub(r'(["\']?trail_sigma["\']?\s*:\s*)[\d.]+',
                          lambda m: f"{m.group(1)}{new_trail}",
                          content)
        # Update profit_target_sigmas
        if new_pt is None:
            content = re.sub(r'(["\']?profit_target_sigmas["\']?\s*:\s*)[\d.None]+',
                              lambda m: f"{m.group(1)}None",
                              content)
        else:
            content = re.sub(r'(["\']?profit_target_sigmas["\']?\s*:\s*)[\d.None]+',
                              lambda m: f"{m.group(1)}{new_pt}",
                              content)
        module_path.write_text(content)
        print(f"    {asset}: wrote trail={new_trail} pt={new_pt} → {module_path.name}")


if __name__ == "__main__":
    sys.exit(main())
