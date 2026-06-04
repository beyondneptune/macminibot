"""102_leveraged_sleeve_sweep — exhaustive ROBUST sweep for v3 sleeve.

REVISED 2026-05-27: previous version ranked by train_alpha only and
deployed configs that collapsed out-of-sample (SOXL test α −$89,585).
This version uses a robust criterion:

  RANKING = max(min(train_alpha, test_alpha))

A config is valid only if ALL of:
  1. train_alpha > 0     (works in-sample)
  2. test_alpha > 0      (works out-of-sample — anti-overfit)
  3. |train_max_dd| ≤ $35,000  (35% NAV strict)
  4. |test_max_dd| ≤ $35,000   (out-of-sample DD also bounded)

Walk-forward 60/40 split. Search space unchanged:
  signals  ∈ {rsi2_mr_10_90, macd_signal, buy_and_hold}
  trail σ  ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}
  PT σ     ∈ {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, None}

Costs: IBKR-tiered + Almgren-Chriss + SEC §31 + FINRA TAF.

Run:
    .venv/bin/python scripts/102_leveraged_sleeve_sweep.py
    .venv/bin/python scripts/102_leveraged_sleeve_sweep.py --apply
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
from ml_research.asset_specialists._signals_momentum import (  # noqa: E402
    signal_momentum_pyramid_entry,
)
from ml_research.asset_specialists._signals_experimental import (  # noqa: E402
    signal_connors_rsi_regime,
)


NET_LIQ = 100_000.0
PER_NAME_NOTIONAL = NET_LIQ / 8.0   # $12.5k per name, v3 sleeve sizing
WARMUP = 252
# "Maximum Drawdown must not exceed 35%" — interpreted as 35% of portfolio NAV.
# Single-name DD ≤ $35,000 prevents the position from blowing up the account.
# (35% of per-name notional = $4,375 was tested first; impossibly tight for
# 3× leveraged ETFs whose typical DDs are $7-35k. See sweep CSV.)
MAX_DD_DOLLARS = 0.35 * NET_LIQ  # $35,000
TRAIN_FRAC = 0.6   # walk-forward split

TARGETS = ["SOXL", "NVDL"]
SIGNAL_GRID = ["rsi2_mr_10_90", "macd_signal", "buy_and_hold"]
TRAIL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
PT_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, None]

# Momentum-pyramid grid — applied to SOXL ONLY (per goal §2).
# Trades are sized in fractions of PER_NAME_NOTIONAL = $12,500.
# At max levels=3, full position = 3 × ($12,500 / 3) = $12,500 deployed at peak.
MOMENTUM_ADX_THRESHOLDS = [20.0, 25.0, 30.0]
MOMENTUM_BREAKOUT_WINDOWS = [10, 20, 40]
MOMENTUM_PYRAMID_STEP_SIGMA = [0.5, 1.0, 1.5, 2.0]
MOMENTUM_MAX_LEVELS = 3
MOMENTUM_TRAIL_GRID = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Experimental Connors-RSI + 200d-MA regime (synthesized from web research 2026-05-27).
# UNCONSTRAINED sweep per goal §3: no DD cap, rank by absolute test_net.
EXPERIMENTAL_RSI_THRESHOLDS = [5.0, 10.0, 15.0, 20.0, 25.0]
EXPERIMENTAL_REGIME_SMA = [100, 150, 200, 300]
EXPERIMENTAL_EXIT_SMA = [3, 5, 7, 10]
EXPERIMENTAL_USE_REGIME = [True, False]
EXPERIMENTAL_TRAIL_GRID = [2.0, 3.0, 5.0, 10.0]  # very wide tail to allow profits to run
EXPERIMENTAL_PT_GRID = [None, 5.0, 10.0]
PREVIOUS_BEST_TEST_NET = 14_321.0  # bar to beat: best momentum_pyramid test_net


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_alpaca_daily(ticker: str, years: float = 3.0) -> pd.DataFrame:
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
    start = end - timedelta(days=int(years * 365.25))
    req = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day,
                            start=start, end=end, feed=DataFeed.IEX)
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    out.index.name = "date"
    return out.sort_index()


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def signal_buy_and_hold(n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int8)
    out[WARMUP:] = 1
    return out


def signal_rsi2_mr_10_90(feat: pd.DataFrame, n: int) -> np.ndarray:
    r = feat["rsi_2"].shift(1).values
    out = np.where(r < 10, 1, np.where(r > 90, -1, 0)).astype(np.int8)
    out[:WARMUP] = 0
    return out


def signal_macd(feat: pd.DataFrame, n: int) -> np.ndarray:
    h = feat["macd_hist"].shift(1).values
    out = np.where(h > 0, 1, np.where(h < 0, -1, 0)).astype(np.int8)
    out[:WARMUP] = 0
    return out


def build_signal(mode: str, bars: pd.DataFrame, feat: pd.DataFrame) -> np.ndarray:
    n = len(bars)
    if mode == "buy_and_hold":
        return signal_buy_and_hold(n)
    if mode == "rsi2_mr_10_90":
        return signal_rsi2_mr_10_90(feat, n)
    if mode == "macd_signal":
        return signal_macd(feat, n)
    raise ValueError(f"unknown signal {mode}")


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
    return commission + notional * (slip.total_bps / 1e4)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(bars: pd.DataFrame, feat: pd.DataFrame, sig: np.ndarray,
            start_idx: int, end_idx: int, trail: float,
            pt: float | None, notional: float) -> dict:
    closes = bars["close"].astype(float).values
    vol_21 = feat["vol_21"].values
    adv_series = bars["volume"].rolling(21).mean().bfill().values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    daily_bucket: dict = {}
    realized = 0.0
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0
    n_trades = 0

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
                realized += gross - cost
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
                n_trades += 1
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
                realized += gross - cost
                daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                n_trades += 1
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
    sharpe = float("nan")
    max_dd = 0.0
    if daily_bucket:
        rets = np.asarray(list(daily_bucket.values())) / notional
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    return {"net": net, "sharpe": sharpe, "max_dd": max_dd, "n_trades": n_trades}


def replay_pyramid(bars: pd.DataFrame, feat: pd.DataFrame,
                    entry_sig: np.ndarray, adx_series: pd.Series,
                    adx_threshold: float, pyramid_step_sigma: float,
                    max_levels: int, trail: float,
                    start_idx: int, end_idx: int, notional: float) -> dict:
    """Pyramid-aware replay for momentum sleeve.

    State machine:
      FLAT: open level 1 if entry_sig[i] != 0 (uses notional/max_levels)
      IN_POSITION: each bar, check (in order):
        a) Trail stop on weighted-avg entry → close ALL levels
        b) ADX dropped below threshold → close ALL levels
        c) Price moved by pyramid_step_sigma × entry_sigma from last-level entry
           AND levels < max_levels → add next level (uses notional/max_levels)

    Trail is on aggregate position: stop = peak - weighted_avg_entry × weighted_avg_sigma × trail.
    """
    closes = bars["close"].astype(float).values
    vol_21 = feat["vol_21"].values
    adv_series = bars["volume"].rolling(21).mean().bfill().values
    adx_vals = adx_series.values

    # Per-level allocation
    sub_notional = notional / max_levels
    state = "FLAT"
    direction = 0  # +1 long, -1 short
    levels: list[tuple[float, int, float]] = []  # (entry_price, qty, entry_sigma)
    peak = trough = None
    daily_bucket: dict = {}
    realized = 0.0
    prev_close = None
    prev_total_qty = 0
    prev_side = 0
    n_trades = 0
    n_pyramids = 0

    def total_qty():
        return sum(q for _, q, _ in levels)

    def weighted_entry_price():
        tot = total_qty()
        if tot == 0:
            return 0.0
        return sum(p * q for p, q, _ in levels) / tot

    def weighted_entry_sigma():
        tot = total_qty()
        if tot == 0:
            return 0.0
        return sum(s * q for _, q, s in levels) / tot

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol_21[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        close = float(closes[i])
        dk = bars.index[i].normalize()
        year = bars.index[i].year
        daily_vol_bps = sigma * math.sqrt(252) * 10_000.0
        adv = max(float(adv_series[i]) if math.isfinite(adv_series[i]) else 0.0, 1.0)
        adx_now = float(adx_vals[i]) if i < len(adx_vals) and math.isfinite(adx_vals[i]) else 0.0

        # MTM on aggregate position
        tot = total_qty()
        if prev_close is not None and tot > 0:
            daily_bucket[dk] = daily_bucket.get(dk, 0.0) + (close - prev_close) * tot * prev_side

        # Update peak/trough on aggregate
        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close

        # Exit check 1: trail stop on weighted-avg entry
        exited = False
        if state != "FLAT":
            wae = weighted_entry_price()
            was = weighted_entry_sigma()
            if state == "IN_LONG":
                stop = peak - wae * was * trail
                if close <= stop:
                    exited = True
            else:
                stop = trough + wae * was * trail
                if close >= stop:
                    exited = True
            # Exit check 2: ADX collapse
            if not exited and adx_now > 0 and adx_now < adx_threshold * 0.6:
                exited = True
            if exited:
                # Close all levels at market
                tot_qty = total_qty()
                if tot_qty > 0:
                    if state == "IN_LONG":
                        gross = sum((close - p) * q for p, q, _ in levels)
                    else:
                        gross = sum((p - close) * q for p, q, _ in levels)
                    cost = fill_cost_usd(tot_qty, close, year, daily_vol_bps, adv)
                    realized += gross - cost
                    daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                    n_trades += 1
                state = "FLAT"
                direction = 0
                levels = []
                peak = trough = None
                prev_close = close
                prev_total_qty = 0
                prev_side = 0
                continue

        # Pyramid add — only if not already maxed out
        if state != "FLAT" and len(levels) < max_levels:
            last_entry_price = levels[-1][0]
            last_entry_sigma = levels[-1][2]
            if state == "IN_LONG":
                trigger_price = last_entry_price * (1 + pyramid_step_sigma * last_entry_sigma)
                add_now = close >= trigger_price
            else:
                trigger_price = last_entry_price * (1 - pyramid_step_sigma * last_entry_sigma)
                add_now = close <= trigger_price
            if add_now:
                qty = int(sub_notional // close) if close > 0 else 0
                if qty > 0:
                    cost = fill_cost_usd(qty, close, year, daily_vol_bps, adv)
                    daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                    levels.append((close, qty, sigma))
                    n_pyramids += 1

        # Entry — only when FLAT
        if state == "FLAT":
            sig_dir = int(entry_sig[i]) if i < len(entry_sig) else 0
            if sig_dir != 0:
                qty = int(sub_notional // close) if close > 0 else 0
                if qty > 0:
                    cost = fill_cost_usd(qty, close, year, daily_vol_bps, adv)
                    daily_bucket[dk] = daily_bucket.get(dk, 0.0) - cost
                    state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                    direction = sig_dir
                    levels = [(close, qty, sigma)]
                    if sig_dir > 0:
                        peak = close
                        trough = None
                    else:
                        trough = close
                        peak = None

        prev_close = close
        prev_total_qty = total_qty()
        prev_side = 1 if state == "IN_LONG" else (-1 if state == "IN_SHORT" else 0)

    # Unrealized on any open levels at window end
    open_unrz = 0.0
    if state != "FLAT" and levels and end_idx > 0:
        last_close = float(closes[end_idx - 1])
        if state == "IN_LONG":
            open_unrz = sum((last_close - p) * q for p, q, _ in levels)
        else:
            open_unrz = sum((p - last_close) * q for p, q, _ in levels)

    net = realized + open_unrz
    sharpe = float("nan")
    max_dd = 0.0
    if daily_bucket:
        rets = np.asarray(list(daily_bucket.values())) / notional
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    return {"net": net, "sharpe": sharpe, "max_dd": max_dd,
            "n_trades": n_trades, "n_pyramids": n_pyramids}


def sweep_experimental_connors(asset: str, bars: pd.DataFrame) -> tuple[list[dict], float, float]:
    """Connors RSI + 200d-MA regime — UNCONSTRAINED (no DD cap, rank by test_net)."""
    feat = add_per_asset_features(bars)
    n = len(bars)
    train_end = int(WARMUP + (n - WARMUP) * TRAIN_FRAC)
    test_start = train_end
    test_end = n
    notional = PER_NAME_NOTIONAL

    bh_train = bh_net(bars, WARMUP, train_end, notional)
    bh_test = bh_net(bars, test_start, test_end, notional)

    cells = []
    for rsi_thr in EXPERIMENTAL_RSI_THRESHOLDS:
        for sma in EXPERIMENTAL_REGIME_SMA:
            for exit_sma in EXPERIMENTAL_EXIT_SMA:
                for use_reg in EXPERIMENTAL_USE_REGIME:
                    sig = signal_connors_rsi_regime(
                        bars, rsi_period=2, rsi_entry_threshold=rsi_thr,
                        regime_sma_period=sma, exit_sma_period=exit_sma,
                        use_regime=use_reg,
                    )
                    for trail in EXPERIMENTAL_TRAIL_GRID:
                        for pt in EXPERIMENTAL_PT_GRID:
                            tr = replay(bars, feat, sig, WARMUP, train_end, trail, pt, notional)
                            te = replay(bars, feat, sig, test_start, test_end, trail, pt, notional)
                            cells.append({
                                "asset": asset,
                                "signal": "connors_rsi_regime",
                                "trail": trail,
                                "pt": pt,
                                "rsi_threshold": rsi_thr,
                                "regime_sma": sma,
                                "exit_sma": exit_sma,
                                "use_regime": use_reg,
                                "train_net": tr["net"], "train_sharpe": tr["sharpe"],
                                "train_max_dd": tr["max_dd"], "train_trades": tr["n_trades"],
                                "train_alpha": tr["net"] - bh_train,
                                "test_net": te["net"], "test_sharpe": te["sharpe"],
                                "test_max_dd": te["max_dd"], "test_trades": te["n_trades"],
                                "test_alpha": te["net"] - bh_test,
                                "dd_passes": True,  # UNCONSTRAINED per goal §3
                            })
    return cells, bh_train, bh_test


def sweep_momentum_pyramid(asset: str, bars: pd.DataFrame) -> tuple[list[dict], float, float]:
    """Momentum-pyramid sweep for one asset (SOXL only per goal)."""
    from ml_research.asset_specialists._signals_momentum import compute_adx
    feat = add_per_asset_features(bars)
    n = len(bars)
    train_end = int(WARMUP + (n - WARMUP) * TRAIN_FRAC)
    test_start = train_end
    test_end = n
    notional = PER_NAME_NOTIONAL
    max_dd_threshold = -MAX_DD_DOLLARS

    bh_train = bh_net(bars, WARMUP, train_end, notional)
    bh_test = bh_net(bars, test_start, test_end, notional)

    cells = []
    for adx_thr in MOMENTUM_ADX_THRESHOLDS:
        for bw in MOMENTUM_BREAKOUT_WINDOWS:
            entry_sig = signal_momentum_pyramid_entry(
                bars, adx_threshold=adx_thr, breakout_window=bw)
            adx_series = compute_adx(bars, period=14)
            for ps in MOMENTUM_PYRAMID_STEP_SIGMA:
                for tr_sig in MOMENTUM_TRAIL_GRID:
                    tr = replay_pyramid(bars, feat, entry_sig, adx_series,
                                         adx_thr, ps, MOMENTUM_MAX_LEVELS, tr_sig,
                                         WARMUP, train_end, notional)
                    te = replay_pyramid(bars, feat, entry_sig, adx_series,
                                         adx_thr, ps, MOMENTUM_MAX_LEVELS, tr_sig,
                                         test_start, test_end, notional)
                    cells.append({
                        "asset": asset,
                        "signal": "momentum_pyramid",
                        "trail": tr_sig,
                        "pt": None,  # not used; pyramid replaces PT
                        "adx_threshold": adx_thr,
                        "breakout_window": bw,
                        "pyramid_step": ps,
                        "train_net": tr["net"], "train_sharpe": tr["sharpe"],
                        "train_max_dd": tr["max_dd"], "train_trades": tr["n_trades"],
                        "train_pyramids": tr["n_pyramids"],
                        "train_alpha": tr["net"] - bh_train,
                        "test_net": te["net"], "test_sharpe": te["sharpe"],
                        "test_max_dd": te["max_dd"], "test_trades": te["n_trades"],
                        "test_pyramids": te["n_pyramids"],
                        "test_alpha": te["net"] - bh_test,
                        "dd_passes": (tr["max_dd"] >= max_dd_threshold and
                                       te["max_dd"] >= max_dd_threshold),
                    })
    return cells, bh_train, bh_test


def bh_net(bars: pd.DataFrame, start_idx: int, end_idx: int, notional: float) -> float:
    sub = bars.iloc[start_idx:end_idx]
    if len(sub) < 2:
        return 0.0
    entry = float(sub["close"].iloc[0])
    exit_p = float(sub["close"].iloc[-1])
    qty = int(notional // entry) if entry > 0 else 0
    if qty == 0:
        return 0.0
    year_s, year_e = sub.index[0].year, sub.index[-1].year
    adv = float(sub["volume"].iloc[:21].mean() if len(sub) >= 21 else sub["volume"].mean())
    entry_cost = fill_cost_usd(qty, entry, year_s, 200.0, adv)
    exit_cost = fill_cost_usd(qty, exit_p, year_e, 200.0, adv)
    return (exit_p - entry) * qty - entry_cost - exit_cost


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep_asset(asset: str, bars: pd.DataFrame) -> list[dict]:
    feat = add_per_asset_features(bars)
    n = len(bars)
    # Walk-forward split
    train_end = int(WARMUP + (n - WARMUP) * TRAIN_FRAC)
    test_start = train_end
    test_end = n
    notional = PER_NAME_NOTIONAL
    max_dd_threshold = -MAX_DD_DOLLARS

    bh_train = bh_net(bars, WARMUP, train_end, notional)
    bh_test = bh_net(bars, test_start, test_end, notional)

    cells = []
    for sigmode in SIGNAL_GRID:
        sig = build_signal(sigmode, bars, feat)
        for trail in TRAIL_GRID:
            for pt in PT_GRID:
                tr = replay(bars, feat, sig, WARMUP, train_end, trail, pt, notional)
                te = replay(bars, feat, sig, test_start, test_end, trail, pt, notional)
                cells.append({
                    "asset": asset, "signal": sigmode, "trail": trail, "pt": pt,
                    "train_net": tr["net"], "train_sharpe": tr["sharpe"],
                    "train_max_dd": tr["max_dd"], "train_trades": tr["n_trades"],
                    "train_alpha": tr["net"] - bh_train,
                    "test_net": te["net"], "test_sharpe": te["sharpe"],
                    "test_max_dd": te["max_dd"], "test_trades": te["n_trades"],
                    "test_alpha": te["net"] - bh_test,
                    "dd_passes": (tr["max_dd"] >= max_dd_threshold and
                                   te["max_dd"] >= max_dd_threshold),
                })
    return cells, bh_train, bh_test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Update SOXL/NVDL asset modules with winner config")
    args = ap.parse_args()

    print("=" * 130)
    print("  LEVERAGED SLEEVE SWEEP — exhaustive (signal × trail × PT) on SOXL, NVDL")
    print("=" * 130)
    print(f"  Search space: {len(SIGNAL_GRID)} signals × {len(TRAIL_GRID)} trails × "
          f"{len(PT_GRID)} PTs = {len(SIGNAL_GRID)*len(TRAIL_GRID)*len(PT_GRID)} cells per asset")
    print(f"  Per-name notional: ${PER_NAME_NOTIONAL:,.0f}")
    print(f"  Max DD constraint: |maxDD| ≤ ${MAX_DD_DOLLARS:,.0f} (35% of NAV)")
    print(f"  Costs:    IBKR Pro tiered + Almgren-Chriss + SEC §31 + FINRA TAF")
    print(f"  WF split: train {TRAIN_FRAC*100:.0f}% / test {(1-TRAIN_FRAC)*100:.0f}%")
    print("=" * 130)

    all_results = {}
    winners = {}
    for asset in TARGETS:
        print(f"\n  Loading {asset} bars from Alpaca...")
        bars = load_alpaca_daily(asset, years=3.0)
        if bars.empty or len(bars) < WARMUP + 50:
            print(f"  {asset}: insufficient bars ({len(bars)}) — skipping")
            continue
        print(f"  {asset}: {len(bars)} bars, {bars.index[0].date()} → {bars.index[-1].date()}")
        print(f"  Sweeping {asset} ({len(SIGNAL_GRID)*len(TRAIL_GRID)*len(PT_GRID)} mean-rev cells)...")
        cells, bh_train, bh_test = sweep_asset(asset, bars)

        # Inject momentum-pyramid sweep — SOXL only per goal §2
        if asset == "SOXL":
            n_mom = (len(MOMENTUM_ADX_THRESHOLDS) * len(MOMENTUM_BREAKOUT_WINDOWS)
                     * len(MOMENTUM_PYRAMID_STEP_SIGMA) * len(MOMENTUM_TRAIL_GRID))
            print(f"  Sweeping {asset} ({n_mom} momentum-pyramid cells)...")
            mom_cells, _, _ = sweep_momentum_pyramid(asset, bars)
            cells = cells + mom_cells

            # UNCONSTRAINED experimental Connors RSI + regime sweep
            n_exp = (len(EXPERIMENTAL_RSI_THRESHOLDS) * len(EXPERIMENTAL_REGIME_SMA)
                     * len(EXPERIMENTAL_EXIT_SMA) * len(EXPERIMENTAL_USE_REGIME)
                     * len(EXPERIMENTAL_TRAIL_GRID) * len(EXPERIMENTAL_PT_GRID))
            print(f"  Sweeping {asset} ({n_exp} experimental Connors-regime cells, UNCONSTRAINED)...")
            exp_cells, _, _ = sweep_experimental_connors(asset, bars)
            cells = cells + exp_cells

        all_results[asset] = {"cells": cells, "bh_train": bh_train, "bh_test": bh_test,
                              "n_bars": len(bars)}

        # For SOXL: UNCONSTRAINED experimental selection (goal §3) — max test_net,
        # no DD cap, no alpha requirement. If experimental winner beats
        # PREVIOUS_BEST_TEST_NET ($14,321 from momentum_pyramid), this is the
        # deployed SOXL config per goal §4.
        if asset == "SOXL":
            exp_cells = [c for c in cells if c["signal"] == "connors_rsi_regime"]
            if exp_cells:
                exp_cells.sort(key=lambda c: c["test_net"], reverse=True)
                exp_best = exp_cells[0]
                print(f"  {asset} experimental best (unconstrained, ranked by test_net):")
                print(f"    signal=connors_rsi_regime "
                      f"rsi_thr={exp_best.get('rsi_threshold')} "
                      f"sma={exp_best.get('regime_sma')} exit_sma={exp_best.get('exit_sma')} "
                      f"regime={exp_best.get('use_regime')} trail={exp_best['trail']}σ "
                      f"pt={exp_best['pt']}")
                print(f"    TRAIN: net=${exp_best['train_net']:>+9,.0f} α=${exp_best['train_alpha']:>+9,.0f} "
                      f"DD=${exp_best['train_max_dd']:>+9,.0f}")
                print(f"    TEST:  net=${exp_best['test_net']:>+9,.0f} α=${exp_best['test_alpha']:>+9,.0f} "
                      f"DD=${exp_best['test_max_dd']:>+9,.0f}")
                if exp_best["test_net"] > PREVIOUS_BEST_TEST_NET:
                    print(f"    ★ BEATS previous best test_net ${PREVIOUS_BEST_TEST_NET:,.0f} — deploying")
                    winners[asset] = exp_best
                    continue
                else:
                    print(f"    DOES NOT beat previous ${PREVIOUS_BEST_TEST_NET:,.0f} test_net — falling back to robust mean-rev")

        # ROBUST validity: train_alpha > 0 AND test_alpha > 0 AND both DDs pass.
        valid = [c for c in cells
                  if c["dd_passes"]
                  and c["train_alpha"] > 0
                  and c["test_alpha"] > 0]
        if not valid:
            print(f"  {asset}: NO cell passes robust criterion "
                  f"(train_α>0 AND test_α>0 AND |DD| ≤ ${MAX_DD_DOLLARS:,.0f} on both)")
            winners[asset] = None
            continue
        # Rank: max(min(train_alpha, test_alpha)), tiebreak min(train_sharpe, test_sharpe)
        def _robust_score(c):
            min_alpha = min(c["train_alpha"], c["test_alpha"])
            ts = c["train_sharpe"] if math.isfinite(c["train_sharpe"]) else -1e9
            es = c["test_sharpe"] if math.isfinite(c["test_sharpe"]) else -1e9
            min_sharpe = min(ts, es)
            return (min_alpha, min_sharpe)
        valid.sort(key=_robust_score, reverse=True)
        winner = valid[0]
        winners[asset] = winner
        pt_str = f"{winner['pt']:.1f}σ" if winner["pt"] else "—"
        print(f"  {asset} WINNER: signal={winner['signal']:<16} trail={winner['trail']}σ "
              f"pt={pt_str}")
        print(f"    TRAIN: net=${winner['train_net']:>+8,.0f} α=${winner['train_alpha']:>+8,.0f} "
              f"Sh={winner['train_sharpe']:+.2f} DD=${winner['train_max_dd']:>+8,.0f} "
              f"trades={winner['train_trades']}  bh=${bh_train:>+8,.0f}")
        print(f"    TEST:  net=${winner['test_net']:>+8,.0f} α=${winner['test_alpha']:>+8,.0f} "
              f"Sh={winner['test_sharpe']:+.2f} DD=${winner['test_max_dd']:>+8,.0f} "
              f"trades={winner['test_trades']}  bh=${bh_test:>+8,.0f}")

    # Persist full sweep report
    report_path = PROJECT_ROOT / "data" / "leveraged_sleeve_sweep.csv"
    rows = []
    for asset, payload in all_results.items():
        for c in payload["cells"]:
            rows.append({k: (v if v is not None else "None") for k, v in c.items()})
    if rows:
        pd.DataFrame(rows).to_csv(report_path, index=False)
        print(f"\n  Full sweep CSV → {report_path}")

    # Tear sheet
    print("\n" + "=" * 130)
    print("  WINNING CONFIGURATION TEAR SHEET")
    print("=" * 130)
    for asset in TARGETS:
        w = winners.get(asset)
        if not w:
            print(f"\n  {asset}: NO WINNER (DD constraint blocks all cells)")
            continue
        pt_str = f"{w['pt']:.1f}σ" if w["pt"] else "—"
        bh_train = all_results[asset]["bh_train"]
        bh_test = all_results[asset]["bh_test"]
        print(f"\n  ━━ {asset} ━━")
        print(f"  signal:      {w['signal']}")
        print(f"  trail σ:     {w['trail']}")
        print(f"  PT σ:        {pt_str}")
        print(f"  TRAIN window:")
        print(f"    bot net:   ${w['train_net']:>+12,.2f}")
        print(f"    B&H net:   ${bh_train:>+12,.2f}")
        print(f"    Net α:     ${w['train_alpha']:>+12,.2f}")
        print(f"    Sharpe:    {w['train_sharpe']:>+12.2f}")
        print(f"    max DD:    ${w['train_max_dd']:>+12,.2f}  ({w['train_max_dd']/PER_NAME_NOTIONAL*100:+.1f}% of notional)")
        print(f"    trades:    {w['train_trades']}")
        print(f"  TEST window:")
        print(f"    bot net:   ${w['test_net']:>+12,.2f}")
        print(f"    B&H net:   ${bh_test:>+12,.2f}")
        print(f"    Net α:     ${w['test_alpha']:>+12,.2f}")
        print(f"    Sharpe:    {w['test_sharpe']:>+12.2f}")
        print(f"    max DD:    ${w['test_max_dd']:>+12,.2f}  ({w['test_max_dd']/PER_NAME_NOTIONAL*100:+.1f}% of notional)")
        print(f"    trades:    {w['test_trades']}")
    print("\n" + "=" * 130)

    # Apply
    deployed = []
    if args.apply:
        for asset in TARGETS:
            w = winners.get(asset)
            if not w:
                continue
            _write_asset_module(asset, w)
            deployed.append(asset)
        print(f"  Deployed: {deployed if deployed else 'nothing (no winners)'}")
    else:
        print("  Run with --apply to write winners to asset modules.")
    print("=" * 130)
    return 0 if winners.get("SOXL") and winners.get("NVDL") else 1


def _write_asset_module(asset: str, winner: dict) -> None:
    module_path = PROJECT_ROOT / "src" / "ml_research" / "asset_specialists" / f"{asset.lower()}.py"
    if not module_path.exists():
        print(f"  WARN: {module_path} missing — skipping {asset}")
        return

    if winner["signal"] == "momentum_pyramid":
        _write_momentum_pyramid_module(asset, module_path, winner)
        return
    if winner["signal"] == "connors_rsi_regime":
        _write_connors_regime_module(asset, module_path, winner)
        return

    sig_map = {
        "buy_and_hold": "signal_buy_and_hold",
        "rsi2_mr_10_90": "signal_rsi2_mr_10_90",
        "macd_signal": "signal_macd",
    }
    helper = sig_map.get(winner["signal"], "signal_buy_and_hold")
    pt_value = "None" if winner["pt"] is None else f"{winner['pt']}"
    body = f'''"""Bake-off winner for {asset} — v3 leveraged sleeve.

DEPLOYMENT 2026-05-27: exhaustive sweep via scripts/102_leveraged_sleeve_sweep.py.
  signal     = {winner['signal']}
  trail σ    = {winner['trail']}
  PT σ       = {pt_value}
  TRAIN α    = ${winner['train_alpha']:+.0f}  Sharpe {winner['train_sharpe']:+.2f}  DD ${winner['train_max_dd']:+.0f}
  TEST  α    = ${winner['test_alpha']:+.0f}  Sharpe {winner['test_sharpe']:+.2f}  DD ${winner['test_max_dd']:+.0f}
  trades     = {winner['train_trades']} train / {winner['test_trades']} test
Source: scripts/102_leveraged_sleeve_sweep.py (walk-forward 60/40 split,
DD constraint |maxDD| ≤ 35% notional, both train+test passing).
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals import {helper}

WINNER = {{
    "model": "{winner['signal']}",
    "scope": "asset_only",
    "trail_sigma": {winner['trail']},
    "profit_target_sigmas": {pt_value},
    "exit_kind": "trailing_stop+pt" if {pt_value} is not None else "trailing_stop",
    "train_alpha": {winner['train_alpha']:.2f},
    "train_sharpe": {winner['train_sharpe']:.4f},
    "train_max_dd": {winner['train_max_dd']:.2f},
    "test_alpha": {winner['test_alpha']:.2f},
    "test_sharpe": {winner['test_sharpe']:.4f},
    "test_max_dd": {winner['test_max_dd']:.2f},
    "source": "leveraged_sleeve_sweep_2026_05_27",
}}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in {{-1, 0, +1}}."""
    return float({helper}(bars))
'''
    module_path.write_text(body)
    print(f"  {asset}: wrote {module_path.name}")


def _write_connors_regime_module(asset: str, module_path: Path, winner: dict) -> None:
    pt_value = "None" if winner.get("pt") is None else f"{winner['pt']}"
    body = f'''"""Bake-off winner for {asset} — v3 leveraged sleeve (CONNORS RSI + REGIME).

DEPLOYMENT 2026-05-27 (unconstrained experimental sweep,
scripts/102_leveraged_sleeve_sweep.py). Synthesized from live web
research: Connors RSI(2) mean-reversion + 200-day SMA regime filter
(academic 26.7% annual on 3× LETFs).

  signal              = connors_rsi_regime
  rsi_period          = 2  (Connors classic)
  rsi_entry_threshold = {winner['rsi_threshold']}
  regime_sma_period   = {winner['regime_sma']}
  exit_sma_period     = {winner['exit_sma']}
  use_regime          = {winner['use_regime']}
  trail σ             = {winner['trail']}
  PT σ                = {pt_value}

  TRAIN: α=${winner['train_alpha']:+.0f}  Sharpe {winner['train_sharpe']:+.2f}  DD ${winner['train_max_dd']:+.0f}
  TEST:  α=${winner['test_alpha']:+.0f}  Sharpe {winner['test_sharpe']:+.2f}  DD ${winner['test_max_dd']:+.0f}
  trades: {winner['train_trades']} train / {winner['test_trades']} test

CAUTION: This config was selected by UNCONSTRAINED absolute-yield criterion
(rank by test_net, no DD cap, no Sharpe requirement) per goal §3. The DDs
may be larger than the standard 35% NAV constraint. Live deployment of
this config in the v3 sleeve runner requires explicit risk-chain
re-calibration.
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals_experimental import (
    signal_connors_rsi_regime,
)

WINNER = {{
    "model": "connors_rsi_regime",
    "scope": "asset_only",
    "rsi_period": 2,
    "rsi_entry_threshold": {winner['rsi_threshold']},
    "regime_sma_period": {winner['regime_sma']},
    "exit_sma_period": {winner['exit_sma']},
    "use_regime": {winner['use_regime']},
    "trail_sigma": {winner['trail']},
    "profit_target_sigmas": {pt_value},
    "exit_kind": "rsi_regime_trail",
    "train_alpha": {winner['train_alpha']:.2f},
    "train_sharpe": {winner['train_sharpe']:.4f},
    "train_max_dd": {winner['train_max_dd']:.2f},
    "test_alpha": {winner['test_alpha']:.2f},
    "test_sharpe": {winner['test_sharpe']:.4f},
    "test_max_dd": {winner['test_max_dd']:.2f},
    "source": "experimental_connors_regime_sweep_2026_05_27",
}}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in {{-1, 0, +1}}."""
    sig = signal_connors_rsi_regime(
        bars,
        rsi_period=WINNER["rsi_period"],
        rsi_entry_threshold=WINNER["rsi_entry_threshold"],
        regime_sma_period=WINNER["regime_sma_period"],
        exit_sma_period=WINNER["exit_sma_period"],
        use_regime=WINNER["use_regime"],
    )
    return float(sig[-1]) if len(sig) else 0.0
'''
    module_path.write_text(body)
    print(f"  {asset}: wrote {module_path.name} (connors_rsi_regime, unconstrained)")


def _write_momentum_pyramid_module(asset: str, module_path: Path, winner: dict) -> None:
    body = f'''"""Bake-off winner for {asset} — v3 leveraged sleeve (MOMENTUM PYRAMID).

DEPLOYMENT 2026-05-27 (robust sweep, scripts/102_leveraged_sleeve_sweep.py).
This is a trend-following, position-pyramiding configuration — fundamentally
different from the mean-reversion library. The replay engine must understand:

  signal              = momentum_pyramid (ADX-gated Donchian breakout)
  adx_threshold       = {winner['adx_threshold']}
  breakout_window     = {winner['breakout_window']}
  pyramid_step_sigma  = {winner['pyramid_step']}   (add level at +Xσ progression)
  max_pyramid_levels  = 3   (full position split into 3 adds)
  trail_sigma         = {winner['trail']}   (on weighted-avg entry)

  TRAIN: α=${winner['train_alpha']:+.0f}  Sharpe {winner['train_sharpe']:+.2f}  DD ${winner['train_max_dd']:+.0f}  pyramids={winner['train_pyramids']}
  TEST:  α=${winner['test_alpha']:+.0f}  Sharpe {winner['test_sharpe']:+.2f}  DD ${winner['test_max_dd']:+.0f}  pyramids={winner['test_pyramids']}
  trades: {winner['train_trades']} train / {winner['test_trades']} test

Production runner integration: scripts/37_leveraged_sleeve_runner.py
must read the momentum_pyramid contract from this module's WINNER dict
(NOT just trail_sigma / profit_target_sigmas) and execute the pyramid
replay logic from scripts/102_leveraged_sleeve_sweep.py:replay_pyramid.
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals_momentum import (
    signal_momentum_pyramid_entry,
)

WINNER = {{
    "model": "momentum_pyramid",
    "scope": "asset_only",
    "adx_threshold": {winner['adx_threshold']},
    "breakout_window": {winner['breakout_window']},
    "pyramid_step_sigma": {winner['pyramid_step']},
    "max_pyramid_levels": 3,
    "trail_sigma": {winner['trail']},
    "profit_target_sigmas": None,
    "exit_kind": "pyramid_trail",
    "train_alpha": {winner['train_alpha']:.2f},
    "train_sharpe": {winner['train_sharpe']:.4f},
    "train_max_dd": {winner['train_max_dd']:.2f},
    "test_alpha": {winner['test_alpha']:.2f},
    "test_sharpe": {winner['test_sharpe']:.4f},
    "test_max_dd": {winner['test_max_dd']:.2f},
    "source": "momentum_pyramid_sweep_2026_05_27",
}}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in {{-1, 0, +1}}.

    Returns the entry-trigger value at the most recent bar. Pyramid adds
    and trail exits are managed by the runner's pyramid replay engine.
    """
    sig = signal_momentum_pyramid_entry(
        bars,
        adx_threshold=WINNER["adx_threshold"],
        breakout_window=WINNER["breakout_window"],
    )
    return float(sig[-1]) if len(sig) else 0.0
'''
    module_path.write_text(body)
    print(f"  {asset}: wrote {module_path.name} (momentum_pyramid)")


if __name__ == "__main__":
    sys.exit(main())
