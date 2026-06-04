"""Live tick stream + dollar-bar derivation → ArcticDB.

Subscribes to tick-by-tick data for the 40-contract starter universe via
reqTickByTickData. Each tick is run through a per-contract DollarBarAccumulator;
when the notional threshold crosses, the bar is appended to the ArcticDB library
`<asset_class>.dollar_bars` under symbol `<exchange>.<name>`.

For contracts where the account has TRADES permissions we subscribe to 'Last' ticks
(real trades with size). For everything else we use 'MidPoint' and degrade to
tick-count bars (size unknown, threshold = N updates).

Designed to run forever as a daemon. Handles SIGINT/SIGTERM cleanly: flushes any
in-flight bars and disconnects. Bars are batched in memory and flushed to ArcticDB
every BATCH_FLUSH_SEC seconds OR when the per-contract in-memory list hits
BATCH_FLUSH_BARS — whichever first. This keeps ArcticDB writes infrequent.

Run from project root:

    PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/05_live_stream.py \\
        > data/05_live_stream.log 2>&1 &

Read-only on IBKR (readonly=True). No order code paths.
"""

from __future__ import annotations

import signal
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ib_async import IB, Contract, util

sys.path.insert(0, str(Path(__file__).parent))
from _bars import DollarBarAccumulator  # noqa: E402
from _storage import ensure_library, library_name, open_store, symbol_name  # noqa: E402
from _thresholds import threshold_for  # noqa: E402

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4001
CLIENT_ID = 20
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.db"

BATCH_FLUSH_SEC = 30.0      # flush bars to ArcticDB at least this often
BATCH_FLUSH_BARS = 50       # or whenever any one symbol has this many pending bars
HEARTBEAT_SEC = 60.0        # log status every minute

util.logToConsole(level="WARNING")


# Module-level shutdown flag — set by signal handlers
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    print(f"\n[signal] received {signal.Signals(signum).name}, shutting down...", flush=True)
    _shutdown = True


def load_universe_rows() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("SELECT * FROM universe"))
    conn.close()
    return [dict(r) for r in rows]


def reconstruct_contract(row: dict) -> Contract:
    return Contract(
        secType=row["sec_type"],
        conId=row["con_id"],
        symbol=row["local_symbol"] or row["name"],
        exchange=row["exchange"],
        primaryExchange=row["primary_exchange"] or "",
        currency=row["currency"],
        localSymbol=row["local_symbol"] or "",
    )


class StreamingApp:
    def __init__(self, ib: IB, rows: list[dict]):
        self.ib = ib
        self.rows = rows
        self.store = open_store()

        # Per-contract state, keyed by reqId returned by reqTickByTickData
        self.accumulators: dict[int, DollarBarAccumulator] = {}
        self.tick_types: dict[int, str] = {}   # 'Last' | 'MidPoint'
        self.row_by_reqid: dict[int, dict] = {}
        self.ticker_by_reqid: dict[int, "Ticker"] = {}

        # Pending bars per symbol, flushed in batches
        self.pending: dict[str, list[dict]] = defaultdict(list)
        self.last_flush = time.time()
        self.last_heartbeat = time.time()
        self.bars_written = 0
        self.ticks_received = 0
        self.last_tick_ts = {row["name"]: None for row in rows}

    def subscribe_all(self) -> None:
        for row in self.rows:
            try:
                self._subscribe_one(row)
            except Exception as exc:
                print(f"  {row['name']:8s}  SUBSCRIBE ERROR: {type(exc).__name__}: {exc}", flush=True)

    def _subscribe_one(self, row: dict) -> None:
        contract = reconstruct_contract(row)
        threshold, is_midpoint = threshold_for(row["name"], row["asset_class"])

        # Pick tickType based on permissions / asset class
        if row["asset_class"] == "fx":
            tick_type = "MidPoint"
        elif is_midpoint:
            tick_type = "MidPoint"
        else:
            tick_type = "Last"

        ticker = self.ib.reqTickByTickData(
            contract, tickType=tick_type, numberOfTicks=0, ignoreSize=False
        )
        # ib_async returns a Ticker — reqId is on the ticker's parent state, not exposed
        # directly. We key by id(ticker) instead, which is stable per subscription.
        key = id(ticker)
        self.accumulators[key] = DollarBarAccumulator(threshold=threshold, is_midpoint=is_midpoint)
        self.tick_types[key] = tick_type
        self.row_by_reqid[key] = row
        self.ticker_by_reqid[key] = ticker

        thresh_disp = f"${threshold:,.0f}" if not is_midpoint else f"{int(threshold)} ticks"
        print(
            f"  {row['name']:8s}  {tick_type:9s}  threshold={thresh_disp}",
            flush=True,
        )

    def on_pending_tickers(self, tickers: set) -> None:
        """Called by ib_async whenever any subscribed ticker has new data."""
        for ticker in tickers:
            key = id(ticker)
            if key not in self.accumulators:
                continue
            acc = self.accumulators[key]
            tick_type = self.tick_types[key]
            row = self.row_by_reqid[key]

            # Process only the NEW ticks since last call
            if tick_type in ("Last", "AllLast"):
                if not ticker.tickByTicks:
                    continue
                for tick in ticker.tickByTicks:
                    bar = acc.on_trade(tick.time, float(tick.price), float(tick.size))
                    self.ticks_received += 1
                    self.last_tick_ts[row["name"]] = tick.time
                    if bar:
                        self._queue_bar(row, bar)
                ticker.tickByTicks.clear()
            else:  # MidPoint
                if not ticker.tickByTicks:
                    continue
                for tick in ticker.tickByTicks:
                    # MidPoint tick has .midPoint instead of price/size
                    mid = float(getattr(tick, "midPoint", getattr(tick, "price", 0)))
                    if mid <= 0:
                        continue
                    bar = acc.on_midpoint(tick.time, mid)
                    self.ticks_received += 1
                    self.last_tick_ts[row["name"]] = tick.time
                    if bar:
                        self._queue_bar(row, bar)
                ticker.tickByTicks.clear()

    def _queue_bar(self, row: dict, bar: dict) -> None:
        sym = symbol_name(row["primary_exchange"] or row["exchange"], row["name"])
        bar["symbol"] = sym
        bar["what_to_show"] = self.tick_types[id(self.ticker_by_reqid_for(row))]
        self.pending[sym].append(bar)
        if len(self.pending[sym]) >= BATCH_FLUSH_BARS:
            self._flush_one(row["asset_class"], sym)

    def ticker_by_reqid_for(self, row: dict):
        # reverse lookup — slow but only called per bar (not per tick)
        for k, r in self.row_by_reqid.items():
            if r is row:
                return self.ticker_by_reqid[k]
        return None

    def _flush_one(self, asset_class: str, sym: str) -> None:
        bars = self.pending.pop(sym, [])
        if not bars:
            return
        lib_name = library_name(asset_class, "dollar_bars")
        lib = ensure_library(self.store, lib_name)
        df = pd.DataFrame(bars)
        # Use end_time as index
        df = df.set_index(pd.to_datetime(df["end_time"], utc=True).dt.tz_localize(None))
        df.index.name = "time"
        df = df.drop(columns=["end_time"])
        try:
            # Append if exists, else write
            if sym in lib.list_symbols():
                lib.append(sym, df)
            else:
                lib.write(sym, df, metadata={"created_at": datetime.now(timezone.utc).isoformat()})
            self.bars_written += len(bars)
        except Exception as exc:
            # Re-queue on failure so we don't lose data
            self.pending[sym].extend(bars)
            print(f"  {sym}  WRITE FAIL: {type(exc).__name__}: {exc}", flush=True)

    def flush_all(self) -> None:
        # Walk pending and flush per symbol (we know asset_class via row lookup)
        for sym in list(self.pending.keys()):
            # Find asset_class by reverse-looking the row
            for k, r in self.row_by_reqid.items():
                if symbol_name(r["primary_exchange"] or r["exchange"], r["name"]) == sym:
                    self._flush_one(r["asset_class"], sym)
                    break
        self.last_flush = time.time()

    def heartbeat(self) -> None:
        live_syms = sum(1 for ts in self.last_tick_ts.values() if ts is not None)
        pending_count = sum(len(v) for v in self.pending.values())
        print(
            f"[heartbeat] ticks={self.ticks_received:,}  bars_written={self.bars_written:,}  "
            f"pending={pending_count}  active_symbols={live_syms}/{len(self.rows)}",
            flush=True,
        )
        self.last_heartbeat = time.time()


def main() -> int:
    rows = load_universe_rows()
    if not rows:
        print("universe.db is empty — run scripts/02_head_timestamps.py first")
        return 1

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    ib = IB()
    print(f"connecting to IBKR Gateway @ {GATEWAY_HOST}:{GATEWAY_PORT} clientId={CLIENT_ID}", flush=True)
    try:
        ib.connect(GATEWAY_HOST, GATEWAY_PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    except Exception as exc:
        print(f"connect failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("connected (readonly). subscribing to live ticks...\n", flush=True)

    app = StreamingApp(ib, rows)
    ib.pendingTickersEvent += app.on_pending_tickers
    app.subscribe_all()
    print(f"\nsubscribed to {len(app.accumulators)} contracts. streaming...\n", flush=True)

    try:
        while not _shutdown:
            ib.waitOnUpdate(timeout=1.0)
            now = time.time()
            if now - app.last_flush >= BATCH_FLUSH_SEC:
                app.flush_all()
            if now - app.last_heartbeat >= HEARTBEAT_SEC:
                app.heartbeat()
    except KeyboardInterrupt:
        pass

    print("\n[shutdown] flushing pending bars...", flush=True)
    app.flush_all()
    app.heartbeat()
    ib.disconnect()
    print(f"[shutdown] clean. total bars written: {app.bars_written:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
