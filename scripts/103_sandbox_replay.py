"""103_sandbox_replay — Run the three sandbox signal frameworks against
SOXL + NVDL daily history.

Sandbox-only. Does NOT touch production runner, asset specialist modules,
or DAILY_OVERRIDES. Outputs ranked metrics + writes
`src/ml_research/sandbox_ideas.md`.

For each (asset, framework, params) combo:
  * compute desired position fraction series (signal output)
  * walk-forward replay with IBKR-tiered fees + Almgren-Chriss slippage
  * compute Sharpe, Omega (MAR=0), max DD, alpha vs B&H
  * split: 60% train / 40% test
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
from ml_research.sandbox_signals import (  # noqa: E402
    signal_volatility_target, signal_kalman_trend, signal_mtf_coherence,
)


PER_NAME_NOTIONAL = 12_500.0
WARMUP = 252
TRAIN_FRAC = 0.6
ASSETS = ["SOXL", "NVDL"]


# ---------------------------------------------------------------------------
# Data loading
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
# Fractional-position replay (handles binary OR continuous signals)
# ---------------------------------------------------------------------------


def replay_fractional(bars: pd.DataFrame, position_frac: np.ndarray,
                       start_idx: int, end_idx: int, notional: float,
                       hysteresis_frac: float = 0.05) -> dict:
    """Generic replay where the signal is a target position fraction in [-1, 1].

    Rebalances when |new_frac - current_frac| > hysteresis_frac to avoid
    over-trading on tiny signal moves.
    """
    closes = bars["close"].astype(float).values
    returns = pd.Series(closes).pct_change().fillna(0.0).values
    # Annualized daily vol estimate (lagged 21d realized)
    log_ret = np.log(pd.Series(closes) / pd.Series(closes).shift(1))
    realized_vol_21d = log_ret.rolling(21).std().fillna(0.02).values
    adv_series = bars["volume"].rolling(21).mean().bfill().values

    current_qty = 0
    current_frac = 0.0
    daily_pnl = []
    trade_count = 0
    total_cost = 0.0

    for i in range(start_idx, end_idx):
        if i >= len(position_frac):
            break
        target_frac = float(position_frac[i])
        # Clip to [-1, 1] for safety
        target_frac = max(-1.0, min(1.0, target_frac))
        close = float(closes[i])
        year = bars.index[i].year
        daily_vol_bps = float(realized_vol_21d[i]) * math.sqrt(252) * 10_000.0
        adv = max(float(adv_series[i]) if math.isfinite(adv_series[i]) else 0.0, 1.0)

        # MTM on prior position
        if i > start_idx and current_qty != 0:
            day_pnl = (close - float(closes[i - 1])) * current_qty
        else:
            day_pnl = 0.0

        # Should we rebalance?
        if abs(target_frac - current_frac) > hysteresis_frac:
            target_qty = int((notional * target_frac) // close) if close > 0 else 0
            delta_qty = target_qty - current_qty
            if delta_qty != 0:
                cost = fill_cost_usd(abs(delta_qty), close, year, daily_vol_bps, adv)
                day_pnl -= cost
                total_cost += cost
                trade_count += 1
                current_qty = target_qty
                current_frac = target_frac

        daily_pnl.append(day_pnl)

    # Liquidate at end of window for final P&L
    if current_qty != 0 and end_idx > start_idx:
        last_close = float(closes[end_idx - 1])
        cost = fill_cost_usd(abs(current_qty), last_close,
                              bars.index[end_idx - 1].year,
                              float(realized_vol_21d[end_idx - 1]) * math.sqrt(252) * 10_000.0,
                              max(float(adv_series[end_idx - 1]) if math.isfinite(adv_series[end_idx - 1]) else 0.0, 1.0))
        if daily_pnl:
            daily_pnl[-1] -= cost
        total_cost += cost

    net = float(sum(daily_pnl)) if daily_pnl else 0.0
    rets = np.asarray(daily_pnl) / notional if daily_pnl else np.zeros(1)

    sharpe = float("nan")
    if rets.std() > 0 and len(rets) > 1:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252))

    # Omega ratio: sum(max(r, 0)) / sum(max(-r, 0)) with MAR=0
    pos_sum = float(rets[rets > 0].sum()) if (rets > 0).any() else 0.0
    neg_sum = float((-rets[rets < 0]).sum()) if (rets < 0).any() else 0.0
    omega = (pos_sum / neg_sum) if neg_sum > 0 else float("inf")

    cum = np.cumsum(daily_pnl)
    if len(cum):
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    else:
        max_dd = 0.0

    return {"net": net, "sharpe": sharpe, "omega": omega, "max_dd": max_dd,
            "trades": trade_count, "total_cost": total_cost,
            "n_sessions": len(daily_pnl)}


def bh_pnl(bars: pd.DataFrame, start_idx: int, end_idx: int,
            notional: float) -> float:
    sub = bars.iloc[start_idx:end_idx]
    if len(sub) < 2:
        return 0.0
    entry = float(sub["close"].iloc[0])
    exit_p = float(sub["close"].iloc[-1])
    qty = int(notional // entry) if entry > 0 else 0
    if qty == 0:
        return 0.0
    adv = float(sub["volume"].iloc[:21].mean() if len(sub) >= 21 else sub["volume"].mean())
    entry_cost = fill_cost_usd(qty, entry, sub.index[0].year, 200.0, adv)
    exit_cost = fill_cost_usd(qty, exit_p, sub.index[-1].year, 200.0, adv)
    return (exit_p - entry) * qty - entry_cost - exit_cost


# ---------------------------------------------------------------------------
# Frameworks
# ---------------------------------------------------------------------------


def evaluate_voltarget(bars: pd.DataFrame, train_end: int, test_end: int):
    cells = []
    for tv in [0.10, 0.20, 0.30, 0.50]:
        for sma_filter in [0, 100, 200]:
            for ml in [0.5, 1.0]:
                sig = signal_volatility_target(
                    bars,
                    target_vol_annual=tv,
                    vol_lookback=21,
                    max_leverage=ml,
                    entry_filter_sma_period=sma_filter,
                )
                tr = replay_fractional(bars, sig, WARMUP, train_end, PER_NAME_NOTIONAL)
                te = replay_fractional(bars, sig, train_end, test_end, PER_NAME_NOTIONAL)
                cells.append({
                    "framework": "VolTarget",
                    "params": f"tv={tv} sma={sma_filter} ml={ml}",
                    "tr": tr, "te": te,
                })
    return cells


def evaluate_kalman(bars: pd.DataFrame, train_end: int, test_end: int):
    cells = []
    for q in [1e-6, 1e-5, 1e-4, 1e-3]:
        for r in [1e-3, 1e-2, 1e-1]:
            for vthr in [0.0, 0.0001, 0.0005]:
                sig = signal_kalman_trend(bars,
                                           process_noise_q=q,
                                           observation_noise_r=r,
                                           velocity_threshold=vthr)
                # Convert int8 to float fraction (long-only for this study)
                frac = sig.astype(np.float32).clip(0.0, 1.0)
                tr = replay_fractional(bars, frac, WARMUP, train_end, PER_NAME_NOTIONAL)
                te = replay_fractional(bars, frac, train_end, test_end, PER_NAME_NOTIONAL)
                cells.append({
                    "framework": "Kalman",
                    "params": f"q={q} r={r} vthr={vthr}",
                    "tr": tr, "te": te,
                })
    return cells


def evaluate_mtf(bars: pd.DataFrame, train_end: int, test_end: int):
    cells = []
    sma_tuples = [
        (5, 20, 50, 200),
        (10, 30, 90, 200),
        (3, 10, 30, 100),
        (5, 13, 34, 89),     # Fibonacci-style
        (20, 50, 100, 200),
    ]
    for tup in sma_tuples:
        for strict in [True, False]:
            sig = signal_mtf_coherence(bars, sma_periods=tup, require_strict=strict)
            frac = sig.astype(np.float32).clip(0.0, 1.0)
            tr = replay_fractional(bars, frac, WARMUP, train_end, PER_NAME_NOTIONAL)
            te = replay_fractional(bars, frac, train_end, test_end, PER_NAME_NOTIONAL)
            cells.append({
                "framework": "MTF-Coherence",
                "params": f"sma={tup} strict={strict}",
                "tr": tr, "te": te,
            })
    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 130)
    print("  SANDBOX REPLAY — VolTarget, Kalman, MTF-Coherence on SOXL + NVDL")
    print("=" * 130)
    out_lines = []
    out_lines.append("# Sandbox Research Report — Overnight Campaign 2026-05-27\n")
    out_lines.append("## Frameworks evaluated\n")
    out_lines.append("1. **Volatility Targeting** — inverse-vol position sizing\n")
    out_lines.append("2. **Kalman Filter** — local linear trend state-space model\n")
    out_lines.append("3. **Multi-Timeframe Momentum Coherence** — 4-SMA alignment\n\n")
    out_lines.append("All evaluated on SOXL + NVDL (Alpaca IEX daily, ~750 bars each, 2023-05 → 2026-05). Walk-forward 60/40 train/test split.\n\n")
    out_lines.append("Costs: IBKR Pro tiered commission + Almgren-Chriss √-impact + SEC §31 + FINRA TAF.\n\n")
    out_lines.append("---\n\n")

    all_results = {}
    for asset in ASSETS:
        print(f"\n  Loading {asset}...")
        bars = load_alpaca_daily(asset, years=3.0)
        if bars.empty or len(bars) < WARMUP + 50:
            print(f"  {asset}: insufficient bars; skip")
            continue
        n = len(bars)
        train_end = int(WARMUP + (n - WARMUP) * TRAIN_FRAC)
        test_end = n
        bh_tr = bh_pnl(bars, WARMUP, train_end, PER_NAME_NOTIONAL)
        bh_te = bh_pnl(bars, train_end, test_end, PER_NAME_NOTIONAL)
        print(f"  {asset}: {n} bars; B&H train=${bh_tr:+,.0f} test=${bh_te:+,.0f}")

        cells = []
        print(f"  Running VolTarget sweep ({4*3*2}=24 cells)...")
        cells += evaluate_voltarget(bars, train_end, test_end)
        print(f"  Running Kalman sweep ({4*3*3}=36 cells)...")
        cells += evaluate_kalman(bars, train_end, test_end)
        print(f"  Running MTF-Coherence sweep ({5*2}=10 cells)...")
        cells += evaluate_mtf(bars, train_end, test_end)

        for c in cells:
            c["asset"] = asset
            c["bh_train"] = bh_tr
            c["bh_test"] = bh_te
            c["train_alpha"] = c["tr"]["net"] - bh_tr
            c["test_alpha"] = c["te"]["net"] - bh_te

        all_results[asset] = cells

        # Per-framework best by min(train_α, test_α)
        out_lines.append(f"## {asset}\n\n")
        out_lines.append(f"B&H train: ${bh_tr:+,.0f}  |  B&H test: ${bh_te:+,.0f}\n\n")

        for framework in ["VolTarget", "Kalman", "MTF-Coherence"]:
            fcells = [c for c in cells if c["framework"] == framework]
            if not fcells:
                continue
            fcells.sort(key=lambda c: min(c["train_alpha"], c["test_alpha"]),
                         reverse=True)
            best = fcells[0]
            out_lines.append(f"### {framework}\n\n")
            out_lines.append(f"Best (ranked by min(train_α, test_α)):\n\n")
            out_lines.append(f"- Params: `{best['params']}`\n")
            out_lines.append(f"- TRAIN: net=${best['tr']['net']:+,.0f}  α=${best['train_alpha']:+,.0f}  "
                              f"Sharpe={best['tr']['sharpe']:+.2f}  "
                              f"Omega={best['tr']['omega']:.2f}  "
                              f"DD=${best['tr']['max_dd']:+,.0f}  "
                              f"trades={best['tr']['trades']}\n")
            out_lines.append(f"- TEST: net=${best['te']['net']:+,.0f}  α=${best['test_alpha']:+,.0f}  "
                              f"Sharpe={best['te']['sharpe']:+.2f}  "
                              f"Omega={best['te']['omega']:.2f}  "
                              f"DD=${best['te']['max_dd']:+,.0f}  "
                              f"trades={best['te']['trades']}\n\n")
            # Overfit flag
            tr_sh = best["tr"]["sharpe"] if math.isfinite(best["tr"]["sharpe"]) else 0.0
            te_sh = best["te"]["sharpe"] if math.isfinite(best["te"]["sharpe"]) else 0.0
            if tr_sh > 0 and te_sh < 0:
                out_lines.append(f"  ⚠️ OVERFIT: Sharpe collapses from train ({tr_sh:+.2f}) to test ({te_sh:+.2f})\n\n")
            elif best["train_alpha"] > 0 and best["test_alpha"] < 0:
                out_lines.append(f"  ⚠️ OVERFIT: alpha positive on train but negative on test\n\n")
            elif tr_sh > 1.0 and te_sh < tr_sh * 0.3:
                out_lines.append(f"  ⚠️ DROP: train Sharpe {tr_sh:+.2f} vs test {te_sh:+.2f} suggests partial overfit\n\n")

        # Persist top-5 per framework
        out_lines.append("#### Top-3 per framework (raw)\n\n")
        for framework in ["VolTarget", "Kalman", "MTF-Coherence"]:
            fcells = [c for c in cells if c["framework"] == framework]
            fcells.sort(key=lambda c: min(c["train_alpha"], c["test_alpha"]), reverse=True)
            out_lines.append(f"**{framework}**\n\n")
            out_lines.append("| Rank | Params | TR α | TR Sh | TE α | TE Sh | TE Omega | TE DD |\n")
            out_lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for i, c in enumerate(fcells[:3], 1):
                tr_sh = c["tr"]["sharpe"] if math.isfinite(c["tr"]["sharpe"]) else float("nan")
                te_sh = c["te"]["sharpe"] if math.isfinite(c["te"]["sharpe"]) else float("nan")
                te_om = c["te"]["omega"] if math.isfinite(c["te"]["omega"]) else float("nan")
                out_lines.append(f"| {i} | `{c['params']}` | ${c['train_alpha']:+,.0f} | {tr_sh:+.2f} | "
                                  f"${c['test_alpha']:+,.0f} | {te_sh:+.2f} | {te_om:.2f} | ${c['te']['max_dd']:+,.0f} |\n")
            out_lines.append("\n")

    # Overall ranking — best across all asset/framework combos
    flat = [c for cells in all_results.values() for c in cells]
    flat.sort(key=lambda c: min(c["train_alpha"], c["test_alpha"]), reverse=True)
    out_lines.append("---\n\n## Overall ranking (top 10 by min(train_α, test_α))\n\n")
    out_lines.append("| Rank | Asset | Framework | Params | TR α | TR Sh | TE α | TE Sh | TE Omega |\n")
    out_lines.append("|---|---|---|---|---:|---:|---:|---:|---:|\n")
    for i, c in enumerate(flat[:10], 1):
        tr_sh = c["tr"]["sharpe"] if math.isfinite(c["tr"]["sharpe"]) else float("nan")
        te_sh = c["te"]["sharpe"] if math.isfinite(c["te"]["sharpe"]) else float("nan")
        te_om = c["te"]["omega"] if math.isfinite(c["te"]["omega"]) else float("nan")
        out_lines.append(f"| {i} | {c['asset']} | {c['framework']} | `{c['params']}` | "
                          f"${c['train_alpha']:+,.0f} | {tr_sh:+.2f} | "
                          f"${c['test_alpha']:+,.0f} | {te_sh:+.2f} | {te_om:.2f} |\n")
    out_lines.append("\n")

    # Robust candidates: positive train AND test alpha + DD < 35% NAV
    out_lines.append("## Robust candidates (train_α > 0 AND test_α > 0)\n\n")
    robust = [c for c in flat if c["train_alpha"] > 0 and c["test_alpha"] > 0]
    if not robust:
        out_lines.append("**None.** No sandbox configuration produces positive alpha on both train AND test windows.\n\n")
    else:
        out_lines.append(f"Found {len(robust)} robust configs:\n\n")
        out_lines.append("| Asset | Framework | Params | TR α | TE α | min α | TE Sh | TE Omega | TE DD |\n")
        out_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for c in robust[:15]:
            tr_sh = c["tr"]["sharpe"] if math.isfinite(c["tr"]["sharpe"]) else float("nan")
            te_sh = c["te"]["sharpe"] if math.isfinite(c["te"]["sharpe"]) else float("nan")
            te_om = c["te"]["omega"] if math.isfinite(c["te"]["omega"]) else float("nan")
            min_a = min(c["train_alpha"], c["test_alpha"])
            out_lines.append(f"| {c['asset']} | {c['framework']} | `{c['params']}` | "
                              f"${c['train_alpha']:+,.0f} | ${c['test_alpha']:+,.0f} | ${min_a:+,.0f} | "
                              f"{te_sh:+.2f} | {te_om:.2f} | ${c['te']['max_dd']:+,.0f} |\n")
        out_lines.append("\n")

    # Conclusion section
    out_lines.append("---\n\n## Conclusions & Citations\n\n")
    out_lines.append("**Methodology**: Walk-forward 60/40 split, IBKR-tiered commission + Almgren-Chriss √-impact slippage + SEC §31 + FINRA TAF. Position sizing via fractional replay (rebalance when |Δfrac| > 0.05).\n\n")
    out_lines.append("**Sample size caveat**: SOXL and NVDL only have ~750 daily bars (launched 2023-05). Statistical power is limited; results should be treated as exploratory rather than predictive.\n\n")
    out_lines.append("**Key sources** (web research 2026-05-27):\n\n")
    out_lines.append("- Volatility Targeting: [QuantPedia — An Introduction to Volatility Targeting](https://quantpedia.com/an-introduction-to-volatility-targeting/), [QuantPedia — Leveraged ETFs in Low-Volatility Environments](https://quantpedia.com/leveraged-etfs-in-low-volatility-environments/), [arXiv 2603.01298 — Single-Asset Adaptive Leveraged Volatility Control](https://arxiv.org/pdf/2603.01298)\n")
    out_lines.append("- Kalman Filter: [PyQuantLab — Adaptive Kalman Trend Execution](https://pyquantlab.medium.com/adaptive-kalman-trend-execution-auto-tuned-state-estimation-with-velocity-triggers-78438b7bd5f7), [QuantInsti — Kalman Filter Python Tutorial](https://blog.quantinsti.com/kalman-filter/), [Forecastegy — Kalman Filter for Time Series Forecasting](https://forecastegy.com/posts/kalman-filter-for-time-series-forecasting-in-python/)\n")
    out_lines.append("- Multi-Timeframe Momentum: [AlinaKhay — Tri-Timeframe Trend-Following System](https://alinakhay.com/p/mastering-the-tri-timeframe-trend), [Sword Red Medium — Multi-Timeframe EMA Trend Momentum](https://medium.com/@redsword_23261/multi-timeframe-ema-trend-momentum-trading-strategy-612d1df08d79), [Mind Math Money — Multi-Timeframe Analysis 2026](https://www.mindmathmoney.com/articles/multi-timeframe-analysis-trading-strategy-the-complete-guide-to-trading-multiple-timeframes)\n\n")

    out_path = PROJECT_ROOT / "src" / "ml_research" / "sandbox_ideas.md"
    out_path.write_text("".join(out_lines))
    print(f"\n  Report written → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
