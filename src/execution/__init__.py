"""execution — Alpaca paper-trading adapter (greenfield).

This package owns the OMS code path for paper trading. The eventual IBKR
client (per CONTEXT.md) will sit alongside the Alpaca client; today we
need the paper trade and journal writer for the critical-path session.

Public surface (re-exported for caller convenience):
  - ``AlpacaClient`` — thin wrapper over ``alpaca-py`` TradingClient
  - ``write_fill`` / ``read_fills`` — minimal journal parquet writer/reader
  - ``execute_batch`` / ``ExecutionResult`` — top-level runner
  - ``derive_client_order_id`` — Contract-5-style deterministic id

Re-imports are kept lazy via star-export to avoid a hard ``alpaca-py``
import at package import time (tests that monkeypatch the client mustn't
trigger network init via dotenv).
"""

from __future__ import annotations

__all__ = [
    "AlpacaClient",
    "ExecutionResult",
    "OrderOutcome",
    "derive_client_order_id",
    "execute_batch",
    "read_fills",
    "write_fill",
]


def __getattr__(name: str):  # noqa: D401 — lazy re-export
    if name in ("AlpacaClient",):
        from .alpaca_client import AlpacaClient

        return AlpacaClient
    if name in ("write_fill", "read_fills"):
        from . import journal as _journal

        return getattr(_journal, name)
    if name in ("execute_batch", "ExecutionResult", "OrderOutcome", "derive_client_order_id"):
        from . import runner as _runner

        return getattr(_runner, name)
    raise AttributeError(f"module 'execution' has no attribute {name!r}")
