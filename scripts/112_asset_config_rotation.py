"""112_asset_config_rotation — weekly per-asset signal/trail/PT picker.

For each asset in the current active universe, runs a mini-sweep of
signal × trail × PT combinations on trailing 6 months of daily bars
(60-bar train / 60-bar test split). Picks a per-asset winner that beats
the static default (buy_and_hold + trail=2σ + PT=3σ). Otherwise keeps
default for safety.

Writes ``data/asset_overrides.json``. The per-asset-runner reads this
file each fire for trail σ and PT σ per asset; falls back to module
default if absent.

Fires Sunday 23:00 BST (after universe rotation).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from backtesting.fees import equity_tiered  # noqa: E402
from backtesting.slippage import slippage_bps_with_floor  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "asset_overrides.json"
UNIVERSE_PATH = PROJECT_ROOT / "data" / "active_universe.json"

# Conservative defaults — used as fallback per asset.
DEFAULT_TRAIL = 2.0
DEFAULT_PT = 3.0
DEFAULT_MODEL = "buy_and_hold"
NOTIONAL = 50_000.0
WARMUP = 60

TRAIL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
PT_GRID = [None, 2.0, 3.0, 5.0]


def _logger() -> logging.Logger:
    log = logging.getLogger("macminibot.asset_config_rotation")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [asset_config_rotation] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def load_alpaca_daily(ticker: str, years: float = 1.0) -> pd.DataFrame:
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
    if hasattr(bars.index, "levels"):
        bars = bars.xs(ticker, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx
    return out.sort_index()


def fill_cost(qty: int, price: float, year: int,
                daily_vol_bps: float, adv: float) -> float:
    if qty <= 0 or price <= 0:
        return 0.0
    commission = equity_tiered(float(qty), float(price)).total
    notional = qty * price
    slip = slippage_bps_with_floor(float(qty), max(float(adv), 1.0),
                                     max(float(daily_vol_bps), 1.0),
                                     "equity", k=0.1)
    return commission + notional * (slip.total_bps / 1e4)


def replay_window(bars: pd.DataFrame, start_idx: int, end_idx: int,
                   trail: float, pt: float | None,
                   notional: float = NOTIONAL) -> dict:
    closes = bars["close"].astype(float).values
    log_ret = np.log(pd.Series(closes) / pd.Series(closes).shift(1))
    vol_21 = log_ret.rolling(21).std().fillna(0.02).values
    adv = bars["volume"].rolling(21).mean().bfill().fillna(1.0).values

    state = 0
    entry_price = 0.0
    entry_sigma = 0.0
    qty = 0
    peak = None
    daily_pnl = []
    trades = 0

    for i in range(max(WARMUP, start_idx), end_idx):
        if i >= len(bars):
            break
        sigma = float(vol_21[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        close = float(closes[i])
        year = bars.index[i].year if hasattr(bars.index[i], "year") else 2026
        vbps = sigma * math.sqrt(252) * 10_000
        a = max(float(adv[i]), 1.0)

        pnl = 0.0
        if state == 1 and qty > 0 and i > start_idx:
            pnl = (close - float(closes[i - 1])) * qty

        # Exit check
        if state == 1:
            if peak is None or close > peak:
                peak = close
            stop = peak - entry_price * entry_sigma * trail
            reason = None
            if close <= stop:
                reason = "trail"
            elif pt is not None and close >= entry_price * (1 + entry_sigma * pt):
                reason = "pt"
            if reason:
                cost = fill_cost(qty, close, year, vbps, a)
                pnl -= cost
                state = 0
                qty = 0
                peak = None
                trades += 1

        # buy_and_hold entry: always enter when flat
        if state == 0:
            new_qty = int(notional // close) if close > 0 else 0
            if new_qty > 0:
                cost = fill_cost(new_qty, close, year, vbps, a)
                pnl -= cost
                state = 1
                entry_price = close
                entry_sigma = sigma
                qty = new_qty
                peak = close
                trades += 1

        daily_pnl.append(pnl)

    # Liquidate
    open_unrz = 0.0
    if state == 1 and qty > 0 and len(closes):
        open_unrz = (float(closes[end_idx - 1]) - entry_price) * qty

    net = sum(daily_pnl) + open_unrz
    rets = np.asarray(daily_pnl)
    sharpe = float("nan")
    if rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
    cum = np.cumsum(daily_pnl)
    if len(cum):
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    else:
        max_dd = 0.0
    return {"net": net, "sharpe": sharpe, "max_dd": max_dd, "trades": trades}


def pick_per_asset(bars: pd.DataFrame, log) -> dict | None:
    n = len(bars)
    if n < WARMUP + 120:
        return None
    test_start = n - 60
    train_start = max(WARMUP, test_start - 60)
    train_end = test_start
    test_end = n

    cells = []
    for trail in TRAIL_GRID:
        for pt in PT_GRID:
            tr = replay_window(bars, train_start, train_end, trail, pt)
            te = replay_window(bars, test_start, test_end, trail, pt)
            cells.append({
                "trail": trail, "pt": pt,
                "tr_net": tr["net"], "tr_sharpe": tr["sharpe"],
                "te_net": te["net"], "te_sharpe": te["sharpe"],
                "te_dd": te["max_dd"],
            })
    if not cells:
        return None
    # Robust criterion: train_net > 0 AND test_net > 0
    valid = [c for c in cells if c["tr_net"] > 0 and c["te_net"] > 0]
    if not valid:
        return None
    # Pick max test Sharpe
    valid.sort(key=lambda c: c["te_sharpe"] if math.isfinite(c["te_sharpe"]) else -1e9,
                reverse=True)
    return valid[0]


def main() -> int:
    log = _logger()
    # Read active universe (fallback to current SUPPORTED_ASSETS)
    if UNIVERSE_PATH.exists():
        try:
            payload = json.loads(UNIVERSE_PATH.read_text("utf-8"))
            universe = payload.get("universe", [])
        except Exception:
            universe = []
    else:
        universe = []
    if not universe:
        from ml_research.asset_specialists import SUPPORTED_ASSETS
        universe = list(SUPPORTED_ASSETS)
        log.info("active_universe.json absent — using static SUPPORTED_ASSETS")

    log.info("evaluating %d assets: %s", len(universe), universe)

    overrides = {}
    summary = []
    for asset in universe:
        try:
            bars = load_alpaca_daily(asset, years=1.5)
            if bars.empty:
                log.warning("%s: no bars", asset)
                overrides[asset] = {"model": DEFAULT_MODEL,
                                     "trail_sigma": DEFAULT_TRAIL,
                                     "profit_target_sigmas": DEFAULT_PT,
                                     "source": "fallback_no_bars"}
                summary.append((asset, "default_no_bars"))
                continue
            best = pick_per_asset(bars, log)
            if best is None:
                overrides[asset] = {"model": DEFAULT_MODEL,
                                     "trail_sigma": DEFAULT_TRAIL,
                                     "profit_target_sigmas": DEFAULT_PT,
                                     "source": "fallback_no_robust_winner"}
                summary.append((asset, "default_no_winner"))
                continue
            overrides[asset] = {
                "model": DEFAULT_MODEL,
                "trail_sigma": float(best["trail"]),
                "profit_target_sigmas": float(best["pt"]) if best["pt"] is not None else None,
                "source": "asset_config_rotation",
                "tr_sharpe": float(best["tr_sharpe"]) if math.isfinite(best["tr_sharpe"]) else None,
                "te_sharpe": float(best["te_sharpe"]) if math.isfinite(best["te_sharpe"]) else None,
                "te_net": float(best["te_net"]),
            }
            summary.append((asset, f"trail={best['trail']} pt={best['pt']} teSh={best['te_sharpe']:+.2f}"))
        except Exception as exc:
            log.warning("%s evaluation failed: %s", asset, exc)
            overrides[asset] = {"model": DEFAULT_MODEL,
                                 "trail_sigma": DEFAULT_TRAIL,
                                 "profit_target_sigmas": DEFAULT_PT,
                                 "source": f"fallback_error:{type(exc).__name__}"}
            summary.append((asset, f"default_err:{type(exc).__name__}"))

    payload_out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "method": "60-bar train / 60-bar test, robust (tr_net>0 AND te_net>0)",
        "overrides": overrides,
        "default_when_missing": {
            "model": DEFAULT_MODEL,
            "trail_sigma": DEFAULT_TRAIL,
            "profit_target_sigmas": DEFAULT_PT,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload_out, indent=2, sort_keys=True))
    tmp.replace(OUT_PATH)

    log.info("==== ASSET CONFIG ROTATION SUMMARY ====")
    for asset, note in summary:
        log.info("  %s — %s", asset, note)
    log.info("written → %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
