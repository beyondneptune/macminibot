"""Dollar-bar accumulator — pure functions over an event stream.

Per López de Prado's *Advances in Financial ML* Ch. 2: a dollar bar closes when the
cumulative notional ($-volume) crosses a per-contract threshold. For products with
no notional (MIDPOINT/FX), we degrade to a *tick bar* — emit every N updates with
synthetic notional = N.

The DollarBarAccumulator is contract-scoped and stateful but exposes a small surface:

    acc = DollarBarAccumulator(threshold=5_000_000)
    bar = acc.on_trade(ts, price, size)     # for TRADES streams
    bar = acc.on_midpoint(ts, mid)          # for MIDPOINT/BIDASK streams
    # bar is a dict OR None — None means accumulator absorbed the tick; dict means
    # threshold crossed and a fresh bar is ready to write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DollarBarAccumulator:
    threshold: float  # notional threshold for TRADES; or tick-count for MIDPOINT
    is_midpoint: bool = False  # True for MIDPOINT/BIDASK streams (no size info)

    _open_time: datetime | None = field(default=None, init=False)
    _open: float = field(default=0.0, init=False)
    _high: float = field(default=float("-inf"), init=False)
    _low: float = field(default=float("inf"), init=False)
    _close: float = field(default=0.0, init=False)
    _volume: float = field(default=0.0, init=False)
    _notional: float = field(default=0.0, init=False)
    _tick_count: int = field(default=0, init=False)

    def _reset(self) -> None:
        self._open_time = None
        self._open = 0.0
        self._high = float("-inf")
        self._low = float("inf")
        self._close = 0.0
        self._volume = 0.0
        self._notional = 0.0
        self._tick_count = 0

    def _accumulate(self, ts: datetime, price: float, size: float, notional_add: float) -> None:
        if self._open_time is None:
            self._open_time = ts
            self._open = price
        if price > self._high:
            self._high = price
        if price < self._low:
            self._low = price
        self._close = price
        self._volume += size
        self._notional += notional_add
        self._tick_count += 1
        self._last_ts = ts

    def _emit(self) -> dict:
        bar = {
            "start_time": self._open_time,
            "end_time": self._last_ts,
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "volume": self._volume,
            "notional": self._notional,
            "tick_count": self._tick_count,
        }
        self._reset()
        return bar

    def on_trade(self, ts: datetime, price: float, size: float) -> dict | None:
        if self.is_midpoint:
            raise ValueError("accumulator is midpoint-mode; use on_midpoint()")
        notional_add = price * size
        self._accumulate(ts, price, size, notional_add)
        if self._notional >= self.threshold:
            return self._emit()
        return None

    def on_midpoint(self, ts: datetime, mid: float) -> dict | None:
        if not self.is_midpoint:
            raise ValueError("accumulator is trade-mode; use on_trade()")
        self._accumulate(ts, mid, 1.0, 1.0)
        if self._tick_count >= self.threshold:
            return self._emit()
        return None
