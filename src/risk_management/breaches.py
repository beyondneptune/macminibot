"""breaches — typed event emitter (Epic 8 / CONTEXT.md I7a).

A breach is a typed event, not just a log line:
    shared_core.BreachEvent (contract wire):
        {breach_id, rule_id, observed, threshold, mode, ts_ns,
         intent_id, snapshot_hash, would_halt}
    risk-management JSONL-audit extension (serialized alongside, NOT on
    the contract type):
        {config_id, symbol, ts_unix, extra}

LIVE  mode writes to ``risk/breaches.jsonl`` (append-only).
PAPER mode writes to ``risk/paper/breaches.jsonl``.
WHATIF mode does NOT touch the filesystem — caller receives the event in-memory.

Sink directory resolution:
  1. ``MACMINIBOT_RISK_DIR`` env var
  2. ``~/macminibot/risk``

Epic 26 migration (cycle-7):
  - ``BreachEvent`` is now re-exported from :mod:`shared_core.contracts`.
  - The shared_core schema dropped ``config_id``/``symbol``/``extra``/
    ``ts_unix`` (audit-context fields that have no cross-tentacle meaning).
  - The JSONL writer attaches those fields back at serialization time via
    an in-memory ``_audit_ctx`` mapping keyed by ``breach_id``. The
    in-memory ``BreachEvent`` instance carries the shared_core canonical
    shape only — replay-from-JSONL reconstructs the audit context. See
    shared-core/todo.md proposed-extensions section for a path to fold
    these into the contract if cross-tentacle audit needs ever arise.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from shared_core.contracts import BreachEvent, ChainMode

LIVE_BREACHES_FILENAME = "breaches.jsonl"
PAPER_SUBDIR = "paper"


# ---------------------------------------------------------------------------
# Audit-context sidecar (Epic 26)
# ---------------------------------------------------------------------------
#
# shared_core ``BreachEvent`` is the cross-tentacle wire type; it omits
# ``config_id``/``symbol``/``extra``/``ts_unix`` (those are audit-only
# fields with no contract significance). We keep them out-of-band, keyed
# by ``breach_id``, and merge them into the JSONL record at write time.
#
# Lifecycle: ``make_breach`` allocates a ``breach_id`` and registers the
# sidecar. ``emit_breach`` reads + clears it. WHATIF callers can drain
# ``_AUDIT_CTX`` manually if they want to introspect (rarely needed).


_AUDIT_CTX: dict[str, dict[str, Any]] = {}


def _resolve_risk_dir() -> Path:
    env = os.environ.get("MACMINIBOT_RISK_DIR")
    if env:
        return Path(env)
    return Path.home() / "macminibot" / "risk"


def risk_dir() -> Path:
    d = _resolve_risk_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stringify(v: Any) -> str:
    if isinstance(v, Decimal):
        return format(v, "f")  # avoid scientific notation
    return str(v)


def make_breach(
    *,
    rule_id: str,
    observed: Any,
    threshold: Any,
    intent_id: str,
    snapshot_hash: str,
    mode: "ChainMode | str",
    ts_ns: int,
    would_halt: bool,
    config_id: str = "",
    symbol: str = "",
    extra: dict[str, Any] | None = None,
    ts_unix: float | None = None,
    breach_id: str | None = None,
) -> BreachEvent:
    """Construct a shared_core ``BreachEvent`` + register audit-context sidecar.

    ``mode`` accepts the enum or its value. ``ts_ns`` is required (KD2 —
    int64 nanoseconds UTC, the canonical timestamp shape). ``would_halt``
    is required so the contract record is self-describing — the chain
    knows which rules touch HALT and passes it in.
    """

    mode_enum: ChainMode
    if isinstance(mode, ChainMode):
        mode_enum = mode
    else:
        mode_enum = ChainMode(str(mode))

    if breach_id is None:
        breach_id = uuid.uuid4().hex

    extra_dict = dict(extra or {})
    ev = BreachEvent(
        breach_id=breach_id,
        rule_id=rule_id,
        observed=observed,
        threshold=threshold,
        mode=mode_enum,
        ts_ns=ts_ns,
        intent_id=intent_id or None,
        snapshot_hash=snapshot_hash,
        would_halt=would_halt,
        config_id=config_id,
        symbol=symbol,
        extra=extra_dict,
    )

    _AUDIT_CTX[breach_id] = {
        "config_id": config_id,
        "symbol": symbol,
        "extra": extra_dict,
        "ts_unix": ts_unix if ts_unix is not None else time.time(),
    }
    return ev


def _sink_path_for(mode_value: str) -> Path | None:
    """Return the JSONL sink path for the given mode, or None for whatif."""

    if mode_value == "whatif":
        return None
    base = risk_dir()
    if mode_value == "paper":
        d = base / PAPER_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d / LIVE_BREACHES_FILENAME
    # live (and any other value defaults to live path so contract is loud)
    return base / LIVE_BREACHES_FILENAME


def _serialize_for_jsonl(event: BreachEvent) -> dict[str, Any]:
    """Merge the shared_core ``BreachEvent`` + audit sidecar into one JSON record."""

    audit = _AUDIT_CTX.pop(event.breach_id, {})
    rec: dict[str, Any] = {
        "breach_id": event.breach_id,
        "rule_id": event.rule_id,
        "observed": _stringify(event.observed),
        "threshold": _stringify(event.threshold),
        "intent_id": event.intent_id or "",
        "snapshot_hash": event.snapshot_hash,
        "mode": event.mode.value,
        "ts_ns": event.ts_ns,
        "would_halt": event.would_halt,
        # Audit-context sidecar (merged in for backward-compat with replay
        # tooling that expects the legacy field set):
        "config_id": audit.get("config_id", ""),
        "symbol": audit.get("symbol", ""),
        "extra": audit.get("extra", {}),
        "ts_unix": audit.get("ts_unix", time.time()),
    }
    return rec


def emit_breach(event: BreachEvent) -> None:
    """Append a breach event to the appropriate JSONL sink.

    WHATIF mode is a no-op: the caller collected the event in-memory.
    The audit sidecar is still drained so it doesn't accumulate.
    """

    sink = _sink_path_for(event.mode.value)
    rec = _serialize_for_jsonl(event)
    if sink is None:
        return
    line = json.dumps(rec, sort_keys=True, default=_stringify)
    with sink.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


__all__ = [
    "BreachEvent",
    "emit_breach",
    "make_breach",
    "risk_dir",
]
