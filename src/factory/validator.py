"""factory.validator — the Crucible.

Imports a generated strategy file dynamically, runs it against the last
7 days of bars for the target symbol, and enforces:

  PASS gates:
    * Sharpe (annualised, daily-bar) ≥ 2.0
    * |Max Drawdown| / per-name notional ≤ 10%
    * At least 3 trades (avoid degenerate "always-flat" passes)

Failed scripts are DELETED automatically (per goal §2). Passed scripts
are left in place for the router to pick up.

Validation uses IBKR-tiered fees + Almgren-Chriss slippage, same cost
model as the production backtest engine.

Data substrate:
  * "tick data" the goal mentions is approximated by 5-min minute_bars
    from ArcticDB for the 8 v2 assets (substrate available; see
    project_intraday_substrate.md).
  * For SOXL/NVDL/other names without minute-bar coverage, falls back
    to daily bars from Alpaca and reports the substitution.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PER_NAME_NOTIONAL = 12_500.0
MIN_SHARPE = 2.0
MAX_DD_PCT_NOTIONAL = 0.10
MIN_TRADES = 3


@dataclass
class ValidationResult:
    file_path: Path
    strategy_id: str
    passed: bool
    sharpe: float
    max_dd_pct: float
    net_pnl: float
    trades: int
    reasons_failed: list[str]
    substrate: str  # "minute_bars" | "daily_bars"


def _import_strategy(path: Path):
    """Dynamic import the strategy file → returns its LiveAnomalyStrategy class."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "LiveAnomalyStrategy"):
        raise AttributeError(f"{path.name}: missing LiveAnomalyStrategy class")
    return module.LiveAnomalyStrategy


def _load_bars_minute_or_daily(symbol: str, days: int = 7) -> tuple[pd.DataFrame, str]:
    """Load last N days of bars. Returns (df, substrate_kind).

    Priority: minute_bars from ArcticDB (5-min for v2 assets), else
    Alpaca daily fallback.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from _storage import open_store  # type: ignore
        store = open_store()
        for lib in ("equity.minute_bars", "etf.minute_bars"):
            try:
                arctic_sym = _arctic_lookup(symbol, lib)
                if arctic_sym is None:
                    continue
                df = store.get_library(lib).read(arctic_sym).data.copy()
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                df = df.sort_index()
                cutoff = df.index[-1] - pd.Timedelta(days=days)
                df = df[df.index >= cutoff][["open", "high", "low", "close", "volume"]]
                if len(df) > 10:
                    return df, "minute_bars"
            except Exception:
                continue
    except Exception:
        pass
    # Fallback: Alpaca daily
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts._bars import open_store  # noqa: F401, just to keep sys.path bootstrap
    except Exception:
        pass
    import os
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import datetime, timedelta
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
    )
    end = datetime.utcnow()
    start = end - timedelta(days=days + 5)
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Day,
                            start=start, end=end, feed=DataFeed.IEX)
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame(), "none"
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx
    return out.sort_index(), "daily_bars"


def _arctic_lookup(symbol: str, lib: str) -> str | None:
    table = {
        "AAPL": "NASDAQ.AAPL", "MSFT": "NASDAQ.MSFT", "GOOGL": "NASDAQ.GOOGL",
        "AMZN": "NASDAQ.AMZN", "META": "NASDAQ.META", "TSLA": "NASDAQ.TSLA",
        "NVDA": "NASDAQ.NVDA", "GLD": "ARCA.GLD",
    }
    sym = table.get(symbol)
    if not sym:
        return None
    if lib == "etf.minute_bars" and not sym.startswith("ARCA"):
        return None
    if lib == "equity.minute_bars" and sym.startswith("ARCA"):
        return None
    return sym


def _replay_signal(bars: pd.DataFrame, strategy_obj,
                    notional: float = PER_NAME_NOTIONAL) -> dict:
    """Walk the bars; call strategy.compute_signal(bars[:i+1]) per step.

    Computes simple long/short P&L with realistic costs from
    backtesting.fees + slippage modules.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from backtesting.fees import equity_tiered
    from backtesting.slippage import slippage_bps_with_floor

    closes = bars["close"].astype(float).values
    vol_series = bars["close"].pct_change().rolling(21).std().fillna(0.02).values
    adv_series = bars["volume"].rolling(21).mean().bfill().fillna(1.0).values

    state = 0  # -1 / 0 / +1
    entry_price = 0.0
    qty = 0
    daily_pnl = []
    trade_count = 0
    cum_cost = 0.0

    for i in range(21, len(bars)):
        sig = int(strategy_obj.compute_signal(bars.iloc[:i + 1]))
        close = float(closes[i])
        bar_vol_bps = float(vol_series[i]) * math.sqrt(252) * 10_000
        adv = max(float(adv_series[i]), 1.0)
        year = bars.index[i].year if hasattr(bars.index[i], "year") else 2026

        # MTM
        if i > 21 and state != 0 and qty > 0:
            pnl = (close - float(closes[i - 1])) * qty * state
        else:
            pnl = 0.0

        # State change
        if sig != state:
            # Close existing position
            if state != 0 and qty > 0:
                impact_bps = slippage_bps_with_floor(
                    float(qty), adv, bar_vol_bps, "equity").total_bps
                cost = equity_tiered(float(qty), close).total + qty * close * impact_bps / 1e4
                pnl -= cost
                cum_cost += cost
                trade_count += 1
            # Open new position
            if sig != 0:
                new_qty = int(notional // close)
                if new_qty > 0:
                    impact_bps = slippage_bps_with_floor(
                        float(new_qty), adv, bar_vol_bps, "equity").total_bps
                    cost = equity_tiered(float(new_qty), close).total + new_qty * close * impact_bps / 1e4
                    pnl -= cost
                    cum_cost += cost
                    trade_count += 1
                    state = sig
                    entry_price = close
                    qty = new_qty
                else:
                    state = 0
                    qty = 0
            else:
                state = 0
                qty = 0
        daily_pnl.append(pnl)

    # Final liquidation MTM
    open_unrz = 0.0
    if state != 0 and qty > 0 and len(closes):
        open_unrz = (float(closes[-1]) - entry_price) * qty * state

    net = sum(daily_pnl) + open_unrz
    arr = np.asarray(daily_pnl)
    sharpe = float("nan")
    if arr.std() > 0:
        sharpe = float(arr.mean() / arr.std() * math.sqrt(252))
    cum = np.cumsum(arr)
    if len(cum):
        peak = np.maximum.accumulate(cum)
        max_dd = float((cum - peak).min())
    else:
        max_dd = 0.0
    return {"net": net, "sharpe": sharpe, "max_dd": max_dd,
            "trades": trade_count, "cum_cost": cum_cost}


def validate_strategy(file_path: Path, *,
                       delete_on_fail: bool = True,
                       min_sharpe: float = MIN_SHARPE,
                       max_dd_pct: float = MAX_DD_PCT_NOTIONAL,
                       min_trades: int = MIN_TRADES) -> ValidationResult:
    """The Crucible. Import, backtest, gate, optionally delete.

    Production defaults: Sharpe ≥ 2.0, |DD| ≤ 10% notional, ≥3 trades.
    Override (e.g., demo runs) by passing custom thresholds.
    """
    cls = _import_strategy(file_path)
    obj = cls()
    symbol = obj.PARAMS.get("symbol", "AAPL")
    bars, substrate = _load_bars_minute_or_daily(symbol, days=7)
    if bars.empty or len(bars) < 25:
        result = ValidationResult(
            file_path=file_path, strategy_id=obj.STRATEGY_ID,
            passed=False, sharpe=float("nan"), max_dd_pct=float("nan"),
            net_pnl=0.0, trades=0,
            reasons_failed=["insufficient_data"], substrate=substrate,
        )
        if delete_on_fail:
            file_path.unlink(missing_ok=True)
        return result
    metrics = _replay_signal(bars, obj)
    dd_pct = abs(metrics["max_dd"]) / PER_NAME_NOTIONAL
    reasons = []
    sharpe_ok = math.isfinite(metrics["sharpe"]) and metrics["sharpe"] >= min_sharpe
    dd_ok = dd_pct <= max_dd_pct
    trades_ok = metrics["trades"] >= min_trades
    if not sharpe_ok:
        reasons.append(f"sharpe_below_threshold ({metrics['sharpe']:+.2f} < {min_sharpe})")
    if not dd_ok:
        reasons.append(f"max_dd_over_cap ({dd_pct*100:.1f}% > {max_dd_pct*100:.0f}%)")
    if not trades_ok:
        reasons.append(f"too_few_trades ({metrics['trades']} < {min_trades})")
    passed = sharpe_ok and dd_ok and trades_ok
    result = ValidationResult(
        file_path=file_path, strategy_id=obj.STRATEGY_ID,
        passed=passed, sharpe=metrics["sharpe"], max_dd_pct=dd_pct,
        net_pnl=metrics["net"], trades=metrics["trades"],
        reasons_failed=reasons, substrate=substrate,
    )
    if not passed and delete_on_fail:
        file_path.unlink(missing_ok=True)
    return result


__all__ = ["ValidationResult", "validate_strategy"]
