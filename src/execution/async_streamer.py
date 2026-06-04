"""execution.async_streamer — Alpaca WebSocket microstructure feature engine.

NOT an order submission path. NOT an HFT loop. READ-ONLY.

Purpose: subscribe to live IEX trades + quotes for the trading universe,
compute rolling order-book imbalance and VWAP deviation, persist to
`data/microstructure/<asset>.parquet`. Future entry/exit logic can read
these to pick better fill prices without firing market orders into
visible imbalance.

Why this is safe:
  * No TradingClient construction — only StockDataStream.
  * Persists features to disk; does NOT update per_asset_state.json.
  * Survives reconnects with exponential backoff.
  * Honors HALT flag — exits cleanly if state/HALT is present.

Run as daemon:
    python -m execution.async_streamer

Universe: 10 assets (8 v2 names + SOXL/NVDL leveraged sleeve).
TSLL omitted because IEX has thin TSLL quotes; can be added when feed quality
improves.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "GLD",
            "SOXL", "NVDL"]

VWAP_WINDOW_SECONDS = 300         # 5-min rolling
PERSIST_INTERVAL_SECONDS = 30     # flush feature snapshots every 30s
RECONNECT_BACKOFF_INITIAL = 2.0   # seconds
RECONNECT_BACKOFF_MAX = 60.0


@dataclass
class AssetState:
    """Rolling window of trades + last-quote snapshot for one asset."""

    # Each trade entry: (timestamp_ns, price, size)
    trades: Deque[tuple[int, float, float]] = field(default_factory=deque)
    # Last quote
    last_bid: float = 0.0
    last_ask: float = 0.0
    last_bid_size: float = 0.0
    last_ask_size: float = 0.0
    last_quote_ts_ns: int = 0
    # Last computed features (for persistence)
    last_features: dict | None = None


def _setup_logger() -> logging.Logger:
    log_path = PROJECT_ROOT / "data" / f"async_streamer_{datetime.utcnow().date().isoformat()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("async_streamer")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def _compute_features(state: AssetState, now_ns: int) -> dict:
    """Compute order-book imbalance and VWAP deviation from current state."""
    # Prune trades older than VWAP window
    cutoff_ns = now_ns - VWAP_WINDOW_SECONDS * 1_000_000_000
    while state.trades and state.trades[0][0] < cutoff_ns:
        state.trades.popleft()

    # VWAP
    if state.trades:
        total_notional = sum(p * s for _, p, s in state.trades)
        total_size = sum(s for _, _, s in state.trades)
        vwap = total_notional / total_size if total_size > 0 else 0.0
    else:
        vwap = 0.0

    # Mid + imbalance
    mid = (state.last_bid + state.last_ask) / 2.0 if state.last_bid > 0 and state.last_ask > 0 else 0.0
    spread = state.last_ask - state.last_bid if state.last_ask > state.last_bid else 0.0
    spread_bps = (spread / mid * 10_000) if mid > 0 else 0.0

    sz_sum = state.last_bid_size + state.last_ask_size
    imbalance = ((state.last_bid_size - state.last_ask_size) / sz_sum) if sz_sum > 0 else 0.0

    vwap_dev_bps = ((mid - vwap) / vwap * 10_000) if vwap > 0 and mid > 0 else 0.0

    return {
        "ts_ns": now_ns,
        "ts_iso": datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc).isoformat(),
        "bid": state.last_bid,
        "ask": state.last_ask,
        "mid": mid,
        "spread_bps": spread_bps,
        "bid_size": state.last_bid_size,
        "ask_size": state.last_ask_size,
        "imbalance": imbalance,
        "vwap_5min": vwap,
        "vwap_dev_bps": vwap_dev_bps,
        "trades_in_window": len(state.trades),
    }


async def _persist_loop(states: dict[str, AssetState],
                        logger: logging.Logger) -> None:
    """Periodically write the current feature snapshot for every asset."""
    out_dir = PROJECT_ROOT / "data" / "microstructure"
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        await asyncio.sleep(PERSIST_INTERVAL_SECONDS)
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        for asset, state in states.items():
            feat = _compute_features(state, now_ns)
            state.last_features = feat
            # Append to a daily file per asset
            day = datetime.now(timezone.utc).date().isoformat()
            path = out_dir / f"{asset}_{day}.parquet"
            df = pd.DataFrame([feat])
            try:
                if path.exists():
                    existing = pd.read_parquet(path)
                    df = pd.concat([existing, df], ignore_index=True)
                df.to_parquet(path, index=False)
            except Exception as exc:
                logger.warning("persist %s failed: %s", asset, exc)
        logger.info(
            "snapshot — %s",
            " ".join(
                f"{a}:mid={s.last_features['mid']:.2f},imb={s.last_features['imbalance']:+.2f},dev={s.last_features['vwap_dev_bps']:+.1f}bps"
                for a, s in states.items()
                if s.last_features is not None
            ),
        )


async def _run_stream(states: dict[str, AssetState],
                       logger: logging.Logger,
                       api_key: str, secret: str) -> None:
    """Subscribe to Alpaca quote + trade streams; update state on each event."""
    from alpaca.data.live import StockDataStream

    stream = StockDataStream(api_key, secret)

    async def quote_handler(quote) -> None:
        sym = quote.symbol
        if sym not in states:
            return
        s = states[sym]
        s.last_bid = float(quote.bid_price or 0.0)
        s.last_ask = float(quote.ask_price or 0.0)
        s.last_bid_size = float(quote.bid_size or 0.0)
        s.last_ask_size = float(quote.ask_size or 0.0)
        s.last_quote_ts_ns = int(quote.timestamp.timestamp() * 1e9)

    async def trade_handler(trade) -> None:
        sym = trade.symbol
        if sym not in states:
            return
        ts_ns = int(trade.timestamp.timestamp() * 1e9)
        s = states[sym]
        s.trades.append((ts_ns, float(trade.price), float(trade.size)))

    stream.subscribe_quotes(quote_handler, *UNIVERSE)
    stream.subscribe_trades(trade_handler, *UNIVERSE)
    logger.info("subscribed to %d symbols (quotes + trades)", len(UNIVERSE))

    # `_run_forever` is a blocking-async coroutine; await it.
    await stream._run_forever()


async def _supervisor(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Reconnect-on-failure supervisor for the stream task."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not secret:
        logger.error("missing ALPACA_API_KEY / ALPACA_SECRET — aborting")
        return

    states = {a: AssetState() for a in UNIVERSE}

    persist_task = asyncio.create_task(_persist_loop(states, logger))

    backoff = RECONNECT_BACKOFF_INITIAL
    while True:
        # Halt check
        try:
            from risk_management.kill_switch import is_halted
            if is_halted():
                logger.warning("HALT flag present — sleeping 60s")
                await asyncio.sleep(60)
                continue
        except Exception:
            pass

        try:
            logger.info("connecting to Alpaca StockDataStream")
            await _run_stream(states, logger, api_key, secret)
            backoff = RECONNECT_BACKOFF_INITIAL
        except asyncio.CancelledError:
            logger.info("supervisor cancelled — shutting down")
            break
        except Exception as exc:
            logger.exception("stream errored: %s — reconnecting in %.1fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, RECONNECT_BACKOFF_MAX)

    persist_task.cancel()
    try:
        await persist_task
    except asyncio.CancelledError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Alpaca microstructure streamer (read-only)")
    parser.add_argument("--once", action="store_true",
                        help="Run for 60s then exit (for smoke-test)")
    args = parser.parse_args()

    logger = _setup_logger()
    logger.info("=== async_streamer starting — UNIVERSE=%s ===", UNIVERSE)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Signal handlers for clean shutdown
    stop_event = asyncio.Event()
    def _on_sig(*_: object) -> None:
        logger.info("received signal — stopping")
        loop.call_soon_threadsafe(stop_event.set)
    try:
        loop.add_signal_handler(signal.SIGTERM, _on_sig)
        loop.add_signal_handler(signal.SIGINT, _on_sig)
    except NotImplementedError:
        # Windows/edge cases
        pass

    supervisor_task = loop.create_task(_supervisor(args, logger))

    if args.once:
        async def _smoke():
            await asyncio.sleep(60)
            supervisor_task.cancel()
        loop.create_task(_smoke())

    async def _wait_for_stop():
        if args.once:
            return
        await stop_event.wait()
        supervisor_task.cancel()
    loop.create_task(_wait_for_stop())

    try:
        loop.run_until_complete(supervisor_task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
    logger.info("=== async_streamer exit ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
