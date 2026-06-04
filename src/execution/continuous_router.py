"""execution.continuous_router — hot-swap router for factory strategies.

Reads ``src/factory/live_strategies/*.py``, dynamically imports any that
have passed the Crucible, allocates a STRICT 5% maximum of available
buying power per strategy, and SIMULATES execution.

By design this module DOES NOT submit orders to the live Alpaca client
unless the operator explicitly passes ``commit=True`` AND the env var
``MACMINIBOT_FACTORY_LIVE=1`` is set. The default mode prints DRY-RUN
intents and writes them to ``data/factory/dry_run_log.jsonl``.

Per durable feedback policy (``feedback_full_automation_auth.md``):
auto-deploying generated code to a live broker is in the "live-money
or external scope" carve-out and requires explicit confirmation. The
two flags above are the explicit confirmation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "src" / "factory" / "live_strategies"
LOG_PATH = PROJECT_ROOT / "data" / "factory" / "dry_run_log.jsonl"
MAX_ALLOCATION_PCT = 0.05  # strict 5% buying power per factory strategy


@dataclass
class RouterDecision:
    """One router decision per (strategy, symbol) pair."""

    strategy_id: str
    file_path: str
    symbol: str
    signal: int                  # -1 / 0 / +1
    side: str | None             # "BUY" | "SELL" | None
    allocated_usd: float         # bounded ≤ 5% of buying power
    qty: int
    mark: float
    dry_run: bool
    submitted_coid: str | None
    notes: str


def _import_strategy(file_path: Path):
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    spec.loader.exec_module(module)
    return module.LiveAnomalyStrategy


def _get_buying_power_and_mark(symbol: str) -> tuple[float, float]:
    """Pull paper-account buying power + a current mark for the symbol.

    Read-only: only ``get_account()`` + ``StockLatestQuoteRequest``.
    """
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    tc = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
        paper=True,
    )
    acct = tc.get_account()
    bp = float(getattr(acct, "buying_power", acct.cash) or acct.cash)
    data = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
    )
    q = data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=[symbol]))
    quote = q.get(symbol)
    bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
    ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
    mark = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)
    return bp, mark


def _load_bars_for_signal(symbol: str, days: int = 30) -> Any:
    """Pull recent daily bars to feed the strategy's compute_signal."""
    import pandas as pd
    from datetime import datetime, timedelta
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
    )
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Day,
                            start=start, end=end, feed=DataFeed.IEX)
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame()
    if hasattr(bars.index, "levels"):
        bars = bars.xs(symbol, level=0)
    return bars[["open", "high", "low", "close", "volume"]].copy()


def route_passed_strategies(*, commit: bool = False) -> list[RouterDecision]:
    """Iterate over all live-strategies, run their signal, allocate ≤5%.

    Returns the list of decisions. If ``commit=True`` AND env
    ``MACMINIBOT_FACTORY_LIVE=1`` is set, submits market orders to Alpaca
    paper account. Otherwise everything is dry-run logged.
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(LIVE_DIR.glob("*.py"))
    files = [f for f in files if not f.name.startswith("__")]
    decisions: list[RouterDecision] = []
    live_authorised = (commit and os.environ.get("MACMINIBOT_FACTORY_LIVE") == "1")
    for f in files:
        try:
            cls = _import_strategy(f)
            obj = cls()
            symbol = obj.PARAMS.get("symbol", "AAPL")
            bars = _load_bars_for_signal(symbol)
            if bars is None or bars.empty:
                continue
            sig = int(obj.compute_signal(bars))
            buying_power, mark = _get_buying_power_and_mark(symbol)
            allocated = round(buying_power * MAX_ALLOCATION_PCT, 2)
            qty = int(allocated // mark) if mark > 0 else 0
            side = "BUY" if sig > 0 else ("SELL" if sig < 0 else None)
            coid = None
            note = "dry_run"
            if sig != 0 and qty > 0 and live_authorised:
                # Live submission path. Routed via the standard AlpacaClient
                # so the journal + risk hooks are honoured.
                try:
                    sys.path.insert(0, str(PROJECT_ROOT / "src"))
                    from decimal import Decimal
                    from execution.alpaca_client import AlpacaClient
                    from shared_core.contracts import OrderIntent, Side, Symbol, Venue
                    sym = Symbol(ticker=symbol, venue=Venue.NASDAQ, currency="USD")
                    intent = OrderIntent(
                        intent_id=f"factory-{obj.STRATEGY_ID}-{int(time.time())}",
                        symbol=sym,
                        side=Side.BUY if side == "BUY" else Side.SELL,
                        delta_qty=Decimal(qty),
                        mark=Decimal(str(round(mark, 2))),
                        batch_id=f"factory-{obj.STRATEGY_ID}",
                        decision_id=f"factory-{obj.STRATEGY_ID}-{int(time.time())}",
                    )
                    coid = AlpacaClient.from_env().submit_order(
                        intent, account="PAPER",
                        strategy_sha=f"factory.{obj.STRATEGY_ID}",
                    )
                    note = "live_submitted"
                except Exception as exc:
                    note = f"live_submit_failed: {exc}"
            decision = RouterDecision(
                strategy_id=obj.STRATEGY_ID,
                file_path=str(f.relative_to(PROJECT_ROOT)),
                symbol=symbol, signal=sig, side=side,
                allocated_usd=allocated, qty=qty, mark=mark,
                dry_run=not live_authorised,
                submitted_coid=coid, notes=note,
            )
            decisions.append(decision)
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts_unix": time.time(), **asdict(decision)}) + "\n")
        except Exception as exc:
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts_unix": time.time(),
                                     "file": str(f), "error": str(exc)}) + "\n")
    return decisions


__all__ = ["RouterDecision", "MAX_ALLOCATION_PCT", "route_passed_strategies"]
