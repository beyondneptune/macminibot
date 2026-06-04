"""execution.runner — top-level batch executor.

Wires together:
  - per-order HALT + heartbeat gating (Contract 2)
  - ChainMode dispatch (Contract 4): LIVE / PAPER → submit, WHATIF → no-op
  - submission via :class:`execution.alpaca_client.AlpacaClient`
  - journal write via :func:`execution.journal.write_fill`

Today's critical-path scope:
  - the Alpaca paper API is the only broker (so LIVE and PAPER both go
    through the same AlpacaClient and both write to the journal; this is
    fine for first-paper-trade and will fork once we wire IBKR LIVE).
  - WHATIF returns the planned envelope without filesystem side-effects.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from shared_core.contracts import ChainMode, OrderIntent
from shared_core.types import now_ns

from .alpaca_client import AlpacaClient, derive_client_order_id
from .journal import FillRecord, fill_record_from_intent, write_fill

__all__ = [
    "ExecutionResult",
    "OrderOutcome",
    "derive_client_order_id",
    "execute_batch",
]


# ---------------------------------------------------------------------------
# State / heartbeat resolution (mirrors risk_management.kill_switch helpers
# but local-imported here so this module stays independent at import time).
# ---------------------------------------------------------------------------


HEARTBEAT_STALENESS_LIMIT_SECONDS: float = 300.0
"""5-minute window per Contract 2."""


def _state_dir() -> Path:
    env = os.environ.get("MACMINIBOT_STATE_DIR")
    if env:
        return Path(env)
    return Path.home() / "macminibot" / "state"


def _halt_present() -> bool:
    return (_state_dir() / "HALT").exists()


def _heartbeat_age_seconds() -> float:
    """Seconds since heartbeat.json last wrote (``+inf`` if missing)."""

    import json

    p = _state_dir() / "heartbeat.json"
    try:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return float("inf")
    ts = payload.get("ts_unix")
    if not isinstance(ts, (int, float)):
        return float("inf")
    return max(0.0, time.time() - float(ts))


def _heartbeat_check_disabled() -> bool:
    """Allow tests / first-day runs to bypass the heartbeat freshness gate.

    Set ``MACMINIBOT_DISABLE_HEARTBEAT_CHECK=1`` to skip the staleness check
    while still honouring HALT. The check itself remains synchronous in
    production. Day-0 runs that don't yet have ``risk_heartbeat`` wired
    can opt out here rather than papering over the contract.
    """

    return os.environ.get("MACMINIBOT_DISABLE_HEARTBEAT_CHECK") == "1"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderOutcome:
    """Per-order result returned by :func:`execute_batch`."""

    intent_id: str
    status: str
    client_order_id: str | None = None
    reason: str | None = None
    record: FillRecord | None = None


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    outcomes: tuple[OrderOutcome, ...] = field(default_factory=tuple)

    @property
    def submitted(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "SUBMITTED")

    @property
    def refused(self) -> int:
        return sum(1 for o in self.outcomes if o.status.startswith("REFUSED"))

    @property
    def whatif(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "WHATIF")


# ---------------------------------------------------------------------------
# Single-intent driver
# ---------------------------------------------------------------------------


def _check_gates() -> tuple[bool, str | None]:
    """Return ``(ok, reason)`` — synchronous HALT + heartbeat gate."""

    if _halt_present():
        return False, "HALT"
    if not _heartbeat_check_disabled():
        age = _heartbeat_age_seconds()
        if age > HEARTBEAT_STALENESS_LIMIT_SECONDS:
            return False, f"HEARTBEAT_STALE:{int(age)}s"
    return True, None


def _execute_one(
    intent: OrderIntent,
    *,
    mode: ChainMode,
    client: AlpacaClient | None,
    journal_root: Path | None,
) -> OrderOutcome:
    # WHATIF — pure function, no broker call, no journal write
    if mode is ChainMode.WHATIF:
        coid = derive_client_order_id(intent.intent_id, account="WHATIF")
        return OrderOutcome(
            intent_id=intent.intent_id,
            status="WHATIF",
            client_order_id=coid,
            reason="WHATIF: no broker call, no journal write",
        )

    # LIVE + PAPER both go through the gate (Contract 4)
    ok, reason = _check_gates()
    if not ok:
        return OrderOutcome(
            intent_id=intent.intent_id,
            status=f"REFUSED:{reason}",
            client_order_id=None,
            reason=reason,
        )

    if client is None:
        raise RuntimeError(
            "execute_batch: AlpacaClient is required for LIVE/PAPER mode"
        )

    submitted_at_ns = now_ns()
    try:
        coid = client.submit_order(intent, account=mode.value.upper())
        status = "SUBMITTED"
        reject_reason: str | None = None
    except Exception as exc:  # noqa: BLE001 — surface broker errors as REJECTED
        coid = derive_client_order_id(intent.intent_id, account=mode.value.upper())
        status = "REJECTED"
        reject_reason = f"{type(exc).__name__}: {exc}"

    record = fill_record_from_intent(
        intent,
        client_order_id=coid,
        submitted_at_ns=submitted_at_ns,
        status=status,
        mode=mode,
    )
    write_fill(record, root=journal_root)
    return OrderOutcome(
        intent_id=intent.intent_id,
        status=status,
        client_order_id=coid,
        reason=reject_reason,
        record=record,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute_batch(
    intents: Sequence[OrderIntent],
    mode: ChainMode,
    *,
    client: AlpacaClient | None = None,
    journal_root: Path | None = None,
) -> ExecutionResult:
    """Execute a batch of OrderIntents under a given ChainMode.

    Parameters
    ----------
    intents:
        Sequence of post-risk OrderIntents.
    mode:
        ChainMode for the entire batch. WHATIF makes the function pure.
    client:
        Required for LIVE / PAPER; ignored for WHATIF.
    journal_root:
        Override for the journal output root (used by tests).
    """

    if not isinstance(mode, ChainMode):
        raise TypeError(f"mode must be ChainMode (got {type(mode).__name__})")

    outcomes: list[OrderOutcome] = []
    for intent in intents:
        outcomes.append(
            _execute_one(
                intent,
                mode=mode,
                client=client,
                journal_root=journal_root,
            )
        )
    return ExecutionResult(outcomes=tuple(outcomes))
