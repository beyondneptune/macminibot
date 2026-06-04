"""backtesting tentacle — walk-forward replay with IBKR-conservative
fee/slippage models, DSR/PBO gates, and the same trade-journal schema
the live ``execution`` tentacle writes.

See ``.octogent/tentacles/backtesting/CONTEXT.md`` (ratified through 4
loop cycles) for the architecture. This package is the source-of-truth
implementation of:

* :mod:`backtesting.fees`     — IBKR Pro Tiered fee model (equity / ES future / IDEALPRO FX)
* :mod:`backtesting.slippage` — Almgren-Chriss sqrt impact + spread-half floor
* :mod:`backtesting.metrics`  — Sharpe / Sortino / Calmar / MaxDD / PSR / DSR / PBO
* :mod:`backtesting.driver`   — ``WalkForwardDriver`` (signal-generator → equity curve)
* :mod:`backtesting.journal`  — pyarrow Schema + ``write_fill`` (live/backtest parity)
"""

from backtesting import fees, journal, metrics, slippage  # noqa: F401
from backtesting.driver import WalkForwardDriver  # noqa: F401

__all__ = [
    "WalkForwardDriver",
    "fees",
    "journal",
    "metrics",
    "slippage",
]
