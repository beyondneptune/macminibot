"""kill_switch — file-flag HALT + heartbeat reader (Epic 1).

Single source of truth:
  - presence of ``$STATE_DIR/HALT`` flips the system to halt
  - ``$STATE_DIR/heartbeat.json`` carries ``{pid, ts_unix, version}``;
    staleness over 300s alone halts trading.
  - ``$STATE_DIR/halt_audit.log`` is the append-only audit trail
    (who, when, reason) for every halt and unhalt.

State directory resolution (in order):
  1. ``MACMINIBOT_STATE_DIR`` env var
  2. ``~/macminibot/state``
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

HEARTBEAT_STALENESS_LIMIT_SECONDS: Final[float] = 300.0
HALT_FILENAME: Final[str] = "HALT"
HEARTBEAT_FILENAME: Final[str] = "heartbeat.json"
AUDIT_LOG_FILENAME: Final[str] = "halt_audit.log"


def _resolve_state_dir() -> Path:
    """Resolve the state directory (env override → home default)."""

    env = os.environ.get("MACMINIBOT_STATE_DIR")
    if env:
        return Path(env)
    return Path.home() / "macminibot" / "state"


def state_dir() -> Path:
    """Return the resolved state directory, creating it if absent."""

    d = _resolve_state_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def halt_path() -> Path:
    return state_dir() / HALT_FILENAME


def heartbeat_path() -> Path:
    return state_dir() / HEARTBEAT_FILENAME


def audit_log_path() -> Path:
    return state_dir() / AUDIT_LOG_FILENAME


def is_halted() -> bool:
    """Return True if a HALT flag file exists.

    Pure file-stat check — no audit-log read, no broker call. Cheap enough
    for ``execution`` to call per-order per the CONTEXT.md I5 contract.
    """

    return halt_path().exists()


def heartbeat_age_seconds() -> float:
    """Return seconds since the heartbeat file was last written.

    Reads ``ts_unix`` from ``heartbeat.json``. If the file is missing or
    unreadable, returns ``+inf`` so the caller (execution) refuses to
    send orders rather than failing open.
    """

    p = heartbeat_path()
    try:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return float("inf")
    ts = payload.get("ts_unix")
    if not isinstance(ts, (int, float)):
        return float("inf")
    return max(0.0, time.time() - float(ts))


def write_heartbeat(*, pid: int | None = None, version: str = "0") -> None:
    """Write a heartbeat record. Used by ``risk_heartbeat.py`` launchd job."""

    payload = {
        "pid": pid if pid is not None else os.getpid(),
        "ts_unix": time.time(),
        "version": version,
    }
    tmp = heartbeat_path().with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(heartbeat_path())


def _append_audit(action: str, reason: str) -> None:
    """Append ``{ts, action, reason}`` line to the halt audit log."""

    line = json.dumps(
        {
            "ts_unix": time.time(),
            "action": action,
            "reason": reason,
        },
        sort_keys=True,
    )
    p = audit_log_path()
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


@dataclass(frozen=True)
class HaltResult:
    halted: bool
    reason: str


def halt(reason: str) -> HaltResult:
    """Create the HALT flag file and append an audit-log entry.

    Idempotent — if HALT already exists, only the audit-log line is
    appended (so re-halting with a different reason is recorded).
    """

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("halt(reason=...) requires a non-empty reason")
    p = halt_path()
    p.touch(exist_ok=True)
    _append_audit("HALT", reason)
    return HaltResult(halted=True, reason=reason)


def unhalt(reason: str) -> HaltResult:
    """Remove the HALT flag file and append an audit-log entry.

    If HALT is not present, the audit line is still written so the
    "operator pressed unhalt redundantly" event is visible in the digest.
    """

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("unhalt(reason=...) requires a non-empty reason")
    p = halt_path()
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    _append_audit("UNHALT", reason)
    return HaltResult(halted=False, reason=reason)


__all__ = [
    "HEARTBEAT_STALENESS_LIMIT_SECONDS",
    "HaltResult",
    "audit_log_path",
    "halt",
    "halt_path",
    "heartbeat_age_seconds",
    "heartbeat_path",
    "is_halted",
    "state_dir",
    "unhalt",
    "write_heartbeat",
]
