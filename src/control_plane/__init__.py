"""control_plane — operator CLI (HALT/UNHALT/status/positions/kill).

Minimum viable surface for the human operator. Wraps:
  - ``risk_management.kill_switch`` for HALT-flag toggling and heartbeat
  - Alpaca paper TradingClient for live positions + cancel-all

See ``.octogent/tentacles/control-plane/CONTEXT.md`` for the broader
design (HTTP API + UI come later — v1 is terminal-only).
"""

from __future__ import annotations

__all__: list[str] = []
