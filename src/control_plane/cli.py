"""control_plane.cli — operator CLI for HALT/UNHALT/status/positions/kill.

Usage:
    python -m control_plane.cli halt --reason "circuit broke"
    python -m control_plane.cli unhalt --reason "all clear"
    python -m control_plane.cli status
    python -m control_plane.cli positions
    python -m control_plane.cli kill

State (HALT flag, heartbeat, audit log) lives in
``$MACMINIBOT_STATE_DIR`` (default ``~/macminibot/state``) and is the
same flag the rest of the bot reads — see
``risk_management.kill_switch``.

Alpaca paper credentials are loaded from the project ``.env`` file
(or the process environment) — ``ALPACA_API_KEY`` + ``ALPACA_SECRET``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from risk_management.kill_switch import (
    HEARTBEAT_STALENESS_LIMIT_SECONDS,
    audit_log_path,
    halt as _halt,
    heartbeat_age_seconds,
    is_halted,
    unhalt as _unhalt,
)


# --------------------------------------------------------------------------
# env / client wiring
# --------------------------------------------------------------------------


def _load_env_if_present() -> None:
    """Best-effort load ``.env`` from project root if python-dotenv is available."""

    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    # Walk up looking for a .env: cwd, then this file's parents.
    candidates = [Path.cwd() / ".env"]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / ".env")
    for p in candidates:
        if p.is_file():
            load_dotenv(p)
            return


def _trading_client() -> Any:
    """Return an Alpaca paper ``TradingClient``.

    Prefers ``execution.alpaca_client.get_trading_client()`` if the
    execution tentacle has shipped; otherwise constructs one directly
    from env credentials.
    """

    _load_env_if_present()
    try:
        from execution import alpaca_client  # type: ignore[import-not-found]

        getter = getattr(alpaca_client, "get_trading_client", None)
        if callable(getter):
            return getter()
    except ImportError:
        pass

    try:
        from alpaca.trading.client import TradingClient  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "alpaca-py is not installed; cannot reach the broker. "
            "Install with `pip install alpaca-py`."
        ) from exc

    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET")
    if not api_key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET not set in environment "
            "(check .env in the project root)."
        )
    return TradingClient(api_key, secret, paper=True)


def _cancel_all_orders() -> Any:
    """Cancel every open order. Prefer the execution-tentacle helper."""

    _load_env_if_present()
    try:
        from execution import alpaca_client  # type: ignore[import-not-found]

        canceller = getattr(alpaca_client, "cancel_all_orders", None)
        if callable(canceller):
            return canceller()
    except ImportError:
        pass
    client = _trading_client()
    return client.cancel_orders()


# --------------------------------------------------------------------------
# read-side helpers
# --------------------------------------------------------------------------


def _last_audit_line() -> dict[str, Any] | None:
    p = audit_log_path()
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    last = ""
    for line in text.splitlines():
        if line.strip():
            last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"raw": last}


def _fetch_positions(client: Any) -> list[dict[str, Any]]:
    raw = client.get_all_positions()
    out: list[dict[str, Any]] = []
    for pos in raw:
        out.append(
            {
                "symbol": getattr(pos, "symbol", None),
                "qty": str(getattr(pos, "qty", "")),
                "avg_entry_price": str(getattr(pos, "avg_entry_price", "")),
                "market_value": str(getattr(pos, "market_value", "")),
                "unrealized_pl": str(getattr(pos, "unrealized_pl", "")),
                "side": str(getattr(pos, "side", "")),
            }
        )
    return out


# --------------------------------------------------------------------------
# command implementations (return shell exit codes)
# --------------------------------------------------------------------------


def cmd_halt(reason: str) -> int:
    _halt(reason)
    print(f"HALT set (reason: {reason})")
    return 0


def cmd_unhalt(reason: str) -> int:
    _unhalt(reason)
    print(f"HALT cleared (reason: {reason})")
    return 0


def cmd_status() -> int:
    halted = is_halted()
    hb_age = heartbeat_age_seconds()
    hb_stale = hb_age > HEARTBEAT_STALENESS_LIMIT_SECONDS
    print("=== macminibot status ===")
    print(f"  halted         : {halted}")
    if hb_age == float("inf"):
        print("  heartbeat_age  : inf (no heartbeat file)")
    else:
        print(f"  heartbeat_age  : {hb_age:.1f}s (stale={hb_stale})")
    last = _last_audit_line()
    if last is None:
        print("  last_audit     : (none)")
    else:
        print(f"  last_audit     : {json.dumps(last, sort_keys=True)}")

    # Best-effort broker view — may fail if creds/network down. Don't
    # let it block the local status read.
    try:
        client = _trading_client()
        positions = _fetch_positions(client)
        print(f"  positions      : {len(positions)} open")
        for pos in positions:
            print(
                f"    - {pos['symbol']:<6s} qty={pos['qty']:>10s}  "
                f"mv={pos['market_value']:>12s}  upl={pos['unrealized_pl']:>10s}"
            )
    except Exception as exc:  # noqa: BLE001 - operator wants to see ANY failure here
        print(f"  positions      : UNAVAILABLE ({type(exc).__name__}: {exc})")
    return 0


def cmd_positions() -> int:
    client = _trading_client()
    positions = _fetch_positions(client)
    if not positions:
        print("(no open positions)")
        return 0
    print(f"=== {len(positions)} open paper position(s) ===")
    for pos in positions:
        print(
            f"  {pos['symbol']:<6s} side={pos['side']:<5s} qty={pos['qty']:>10s}  "
            f"avg={pos['avg_entry_price']:>10s}  mv={pos['market_value']:>12s}  "
            f"upl={pos['unrealized_pl']:>10s}"
        )
    return 0


def cmd_kill() -> int:
    result = _cancel_all_orders()
    # Alpaca returns a list of CancelOrderResponse objects. Tolerate any shape.
    try:
        n = len(result)  # type: ignore[arg-type]
        print(f"cancel-all dispatched ({n} order(s))")
    except TypeError:
        print("cancel-all dispatched")
    return 0


# --------------------------------------------------------------------------
# argparse plumbing
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bot",
        description="macminibot operator CLI — HALT, status, positions, kill.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_halt = sub.add_parser("halt", help="Set the HALT flag (stops new orders).")
    p_halt.add_argument("--reason", required=True, help="Free-text reason (audit-logged).")

    p_unhalt = sub.add_parser("unhalt", help="Clear the HALT flag.")
    p_unhalt.add_argument("--reason", required=True, help="Free-text reason (audit-logged).")

    sub.add_parser("status", help="Show HALT state, heartbeat age, positions, last audit.")
    sub.add_parser("positions", help="List current Alpaca paper positions.")
    sub.add_parser("kill", help="Cancel ALL open Alpaca orders.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "halt":
        return cmd_halt(args.reason)
    if args.command == "unhalt":
        return cmd_unhalt(args.reason)
    if args.command == "status":
        return cmd_status()
    if args.command == "positions":
        return cmd_positions()
    if args.command == "kill":
        return cmd_kill()
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main())
