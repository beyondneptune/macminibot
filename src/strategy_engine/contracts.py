"""Strategy-engine contract layer — wraps shared_core canonicals.

Cycle 7 migration (2026-05-17): ``Signal``, ``NavSnapshot``, and ``ChainMode``
are now re-exported directly from ``shared_core.contracts``. ``TargetPosition``
and ``TargetPositionBatch`` remain strategy-engine-local envelopes because the
shared_core forms drop the reasoning-trace fields that CONTEXT.md declares
load-bearing for compliance and operator post-mortems. Conversion helpers
(``to_shared_core``) lower the envelopes to the canonical shared_core types
at the cross-tentacle boundary.

Cycle 8 promotion (shared-core owner): ``decision_id``, ``mode``, and
``solver_status`` were lifted from this local envelope into the canonical
shared_core ``TargetPositionBatch`` as REQUIRED fields. They are still
carried here (the strategy-engine pipeline needs them pre-lowering) but the
``to_shared_core()`` lowering now forwards all three.

Cycle 9 (strategy-engine owner, 2026-05-18): ``SolverStatus`` standardizes
the string-typed ``solver_status`` value set so consumers (risk-management
replay, observability dashboards) can switch on a finite vocabulary instead
of regex-matching arbitrary strings. The shared_core form still types
``solver_status: str`` (kept open for forward-compat with future solver
tiers) but all strategy-engine pipeline emissions now go through
``SolverStatus.<name>.value`` rather than ad-hoc f-strings.

Field-name reconciliation (vs the cycle-6 local stubs):
- ``Signal.symbol`` was bare ``str`` → now ``shared_core.contracts.Symbol``.
- ``Signal.confidence`` was ``Decimal`` → now ``float`` (shared_core canonical;
  it is a unitless [0,1] ratio, not money — see shared_core T6 docstring).
- ``NavSnapshot.nav`` → ``net_liq``; ``as_of_ns`` → ``snapshot_ns``;
  ``snapshot_version`` is now sha256 hex (64 chars); ``positions`` and
  ``buying_power`` are now required fields.
- ``ChainMode`` enum values are lowercase (``"live"`` etc.) in shared_core,
  not uppercase. Name-only access (``ChainMode.LIVE``) is unchanged.

Extension proposals for shared_core (filed in shared-core/todo.md, not edited
directly here): the local ``TargetPosition`` and ``TargetPositionBatch`` carry
fields that the shared_core canonicals lack. Until shared-core ratifies them,
the local envelopes remain authoritative for the strategy-engine pipeline; the
``to_shared_core`` lowering produces the canonical batch by dropping the
reasoning-trace fields into ``TargetPosition.reason_trace: dict[str, Any]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, List, Optional

from shared_core.contracts import (
    ChainMode,
    NavSnapshot,
    Signal,
    Symbol,
)
from shared_core.contracts import TargetPosition as _SharedTargetPosition
from shared_core.contracts import TargetPositionBatch as _SharedTargetPositionBatch


class SolverStatus(str, Enum):
    """Standardized ``solver_status`` vocabulary (Cycle-9 ratification).

    Cycle-8 promoted ``solver_status`` to a required field on the canonical
    ``shared_core.TargetPositionBatch``; cycle-9 nails down the value set so
    the cross-tentacle replay / dashboarding code can switch on a closed
    enum rather than parse free-form strings.

    Members and when they fire:

    - ``OPTIMAL`` — cvxpy ECOS/OSQP returned ``optimal`` (or
      ``optimal_inaccurate``; both are healthy from our viewpoint and roll
      up to this single value at the boundary).
    - ``HRP_OK`` — Stage-3 HRP was selected by tier (``n > 500``) and
      allocated cleanly. Not a fallback; this is the planned path.
    - ``FALLBACK_HRP`` — cvxpy Stage-1/2 failed (``infeasible``,
      ``unbounded``, solver error, etc.) and the pipeline downgraded to
      HRP for the session. Replay should flag these for the morning huddle.
    - ``FALLBACK_DIAGONAL`` — Σ shrinkage hit the diagonal fallback
      (``ledoit_wolf_shrunk.diagonal_fallback=True``). Solver itself may
      have been OPTIMAL but covariance estimation degraded; reported here
      so the diagnostic surface for the day rolls up to one status string.
    - ``INFEASIBLE`` — solver returned a status we explicitly classify as
      infeasible *and* the HRP fallback also could not run (e.g., empty
      universe after validation). Currently emitted as a precondition for
      ``HALTED_NO_UNIVERSE`` but kept as a distinct value for future
      degenerate-cov paths.
    - ``ZERO_SIGNAL_DECAY`` — every accepted signal was below the
      ``is_zero_signal_day`` threshold; the pipeline emitted a 20% decay
      toward cash rather than calling the optimizer. This is a *trade*
      decision (not a halt) and downstream rules must apply.
    - ``HALTED_NO_UNIVERSE`` — signal validation removed every name; the
      batch carries zero targets. Risk-management must not synthesize any
      ``OrderIntent`` from this batch.

    String values are stable; they appear in the rebalance ledger and in
    replay tooling. **Do not rename without a shared-core cycle bump.**
    """

    OPTIMAL = "optimal"
    HRP_OK = "hrp_ok"
    FALLBACK_HRP = "fallback_hrp"
    FALLBACK_DIAGONAL = "fallback_diagonal"
    INFEASIBLE = "infeasible"
    ZERO_SIGNAL_DECAY = "zero_signal_decay"
    HALTED_NO_UNIVERSE = "halted_no_universe"


# Mapping from raw optimizer/cvxpy status strings to the standardized vocabulary.
# Kept here (not in optimizer.py) so the optimizer stays a pure numerics module
# and the canonical-status translation is owned by the contract layer.
_CVXPY_OPTIMAL_STATUSES: frozenset[str] = frozenset(
    {"optimal", "optimal_inaccurate"}
)


def normalize_solver_status(raw: str) -> SolverStatus:
    """Translate a raw optimizer status string to the standardized enum.

    Accepts both cvxpy statuses (``optimal``, ``optimal_inaccurate``,
    ``infeasible``, ``unbounded``, …) and the strategy-engine pipeline
    statuses (``hrp_ok``, ``fallback_hrp_from_<status>``,
    ``zero_signal_decay``, ``halted_no_universe``). Unknown strings collapse
    to ``FALLBACK_HRP`` (the safe-default tier) rather than raising — the
    pipeline must never crash at the lowering boundary because of a new
    solver status that hasn't yet been mapped.
    """
    if raw in _CVXPY_OPTIMAL_STATUSES:
        return SolverStatus.OPTIMAL
    if raw == SolverStatus.HRP_OK.value:
        return SolverStatus.HRP_OK
    if raw == SolverStatus.ZERO_SIGNAL_DECAY.value:
        return SolverStatus.ZERO_SIGNAL_DECAY
    if raw == SolverStatus.HALTED_NO_UNIVERSE.value:
        return SolverStatus.HALTED_NO_UNIVERSE
    if raw.startswith("fallback_hrp"):
        return SolverStatus.FALLBACK_HRP
    if raw == "infeasible" or raw == "unbounded":
        return SolverStatus.INFEASIBLE
    return SolverStatus.FALLBACK_HRP


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """Strategy-engine target with full reasoning trace (Improvement #6 + #12).

    The shared_core canonical ``TargetPosition`` carries only
    ``(symbol, target_qty, hold_reason, reason_trace)``. We keep this richer
    envelope because the scheduler-time pipeline needs ``target_weight``
    (risk-units, pre-lowering), ``r_hat``/``confidence`` (for post-mortem
    attribution), ``w_prev`` (delta for risk-mgmt), and ``sigma_contribution``
    (marginal variance contribution for the trace). ``contracts`` is the
    Alpaca lot-rounded share count populated by the emitter epic
    (Improvement #11); ``None`` is fine for pre-emitter pipeline tests but
    must be non-None for any LIVE/PAPER batch handed to risk-management.

    ``to_shared_core()`` collapses the trace into the canonical envelope.
    """

    symbol: Symbol
    target_weight: float  # risk-units; emitter lowers to contracts at boundary
    r_hat: float
    confidence: float
    w_prev: float
    sigma_contribution: float
    hold_reason: Optional[str] = None  # NULL for optimized names; rule-id for frozen
    contracts: Optional[Decimal] = None  # lot-rounded shares (fractional or integer per ContractSpec)

    def to_shared_core(self) -> _SharedTargetPosition:
        """Lower to the shared_core canonical form for cross-tentacle hand-off.

        ``target_qty`` is ``contracts`` if the emitter has lowered, else
        ``Decimal(0)`` as a placeholder — risk-management will reject any
        batch where pipeline-only weights leak past the boundary.
        """
        qty = self.contracts if self.contracts is not None else Decimal(0)
        trace: dict[str, Any] = {
            "target_weight": self.target_weight,
            "r_hat": self.r_hat,
            "confidence": self.confidence,
            "w_prev": self.w_prev,
            "sigma_contribution": self.sigma_contribution,
        }
        return _SharedTargetPosition(
            symbol=self.symbol,
            target_qty=qty,
            hold_reason=self.hold_reason,
            reason_trace=trace,
        )


@dataclass(frozen=True, slots=True)
class TargetPositionBatch:
    """Strategy-engine batch envelope (ratified CONTEXT key decision).

    Carries the full reasoning surface (``decision_id``, ``solver_status``,
    ``sigma_portfolio_annualized``, ``edge_case_rules_fired``, ``mode``,
    ``nav_snapshot`` + ``nav_snapshot_source``) that the shared_core
    canonical batch drops. ``to_shared_core()`` lowers to the shared_core
    form for cross-tentacle hand-off; the ``batch_id`` of the lowered form
    is the ``decision_id`` (deterministic hash), the ``generated_at_ns`` is
    ``as_of_ns``, and the reasoning fields are dropped (their values live
    in each ``TargetPosition.reason_trace`` and in the strategy-engine
    rebalance ledger, not on the wire).
    """

    decision_id: str
    as_of_ns: int
    session_close_ns: int
    valid_until_ns: int
    nav_snapshot: Decimal
    nav_snapshot_source: str
    targets: List[TargetPosition]
    targets_count: int
    mode: ChainMode
    supersedes: Optional[str] = None
    solver_status: str = "unknown"
    sigma_portfolio_annualized: float = 0.0
    edge_case_rules_fired: List[str] = field(default_factory=list)

    def to_shared_core(
        self,
        *,
        nav_snapshot_version: str,
        batch_id: Optional[str] = None,
    ) -> _SharedTargetPositionBatch:
        """Lower to ``shared_core.TargetPositionBatch`` for risk-management.

        ``nav_snapshot_version`` is the sha256 hash from the ``NavSnapshot``
        the scheduler consumed; it is **not** the strategy-engine-local
        ``nav_snapshot_source`` string. Passed explicitly so the strategy
        engine never has to reach back into the ``NavSnapshot`` instance
        once the batch is built.

        Cycle-8 promotion: the canonical batch now REQUIRES ``decision_id``,
        ``mode``, and ``solver_status``. They are forwarded bit-for-bit from
        the strategy-engine envelope. ``batch_id`` defaults to
        ``decision_id`` for single-emission sessions (the shared_core
        contract permits equality); pass an explicit ``batch_id`` (a ULID,
        per shared_core docstring) for re-emission flows that need to
        distinguish the wire envelope from the reproducibility key.
        """
        canonical_targets = tuple(t.to_shared_core() for t in self.targets)
        # solver_status is forwarded verbatim — the shared_core form types it
        # as ``str`` to stay forward-compatible with new solver tiers, but the
        # strategy-engine pipeline now always emits a SolverStatus enum value
        # (see scheduler._assemble_batch). ``.value`` is a no-op for plain str.
        status_str = (
            self.solver_status.value
            if isinstance(self.solver_status, SolverStatus)
            else self.solver_status
        )
        return _SharedTargetPositionBatch(
            batch_id=batch_id if batch_id is not None else self.decision_id,
            decision_id=self.decision_id,
            mode=self.mode,
            solver_status=status_str,
            generated_at_ns=self.as_of_ns,
            supersedes=self.supersedes,
            targets_count=self.targets_count,
            targets=canonical_targets,
            nav_snapshot_version=nav_snapshot_version,
        )


__all__ = [
    "ChainMode",
    "NavSnapshot",
    "Signal",
    "SolverStatus",
    "Symbol",
    "TargetPosition",
    "TargetPositionBatch",
    "normalize_solver_status",
]
