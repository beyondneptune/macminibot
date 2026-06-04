"""risk_management — pre-trade check chain, kill switch, limits, breach stream.

Single approval gate between strategy-engine and execution. See
.octogent/tentacles/risk-management/CONTEXT.md for the ratified design.

Epic 26 (cycle-7) migration: cross-tentacle types now live in
:mod:`shared_core.contracts`. ``OrderIntent``, ``Side``, ``TargetPosition``,
``TargetPositionBatch``, ``ChainMode``, ``BreachEvent`` are re-exported
through this package for ergonomic call sites. Chain-local fields that
were previously bolted onto ``OrderIntent`` (``current_qty``,
``target_qty``) now live on :class:`risk_management.diff.IntentContext`,
stored on ``ChainContext.intent_contexts``.
"""

from __future__ import annotations

# Re-export canonical contract types from shared_core (Epic 26).
from shared_core.contracts import (
    BreachEvent,
    ChainMode,
    OrderIntent,
    Side,
    TargetPosition,
    TargetPositionBatch,
)

from risk_management.breaches import emit_breach
from risk_management.calibration import (
    DrawdownCalibration,
    calibrate_drawdown_thresholds,
)
from risk_management.chain import (
    Approved,
    ChainContext,
    Rejected,
    run_check_chain,
)
from risk_management.diff import (
    IntentContext,
    diff_targets_to_intents,
)
from risk_management.kill_switch import (
    halt,
    heartbeat_age_seconds,
    is_halted,
    unhalt,
)
from risk_management.limits import (
    LimitChange,
    Limits,
    LimitsDiff,
    LimitsLoadError,
    diff_limits,
    load_limits,
)
from risk_management.live_divergence import (
    LiveDivergenceVerdict,
    evaluate_live_divergence,
)
from risk_management.regime_gate import (
    VIX_HIGH_RULE_ID,
    VIX_OUT_OF_BAND_RULE_ID,
    VixBandGateDecision,
    VixGateDecision,
    VixHighGate,
    VixOutOfBandGate,
)
from risk_management.metrics_emitter import (
    METRIC_NAMES,
    MetricSample,
    MetricsSampler,
    MetricsValidationError,
    SCHEMA as METRICS_SCHEMA,
    new_sampler,
)
from risk_management.snapshot import NavSnapshot, build_snapshot

__all__ = [
    "Approved",
    "BreachEvent",
    "ChainContext",
    "ChainMode",
    "DrawdownCalibration",
    "IntentContext",
    "LimitChange",
    "Limits",
    "LimitsDiff",
    "LimitsLoadError",
    "LiveDivergenceVerdict",
    "METRICS_SCHEMA",
    "METRIC_NAMES",
    "MetricSample",
    "MetricsSampler",
    "MetricsValidationError",
    "NavSnapshot",
    "OrderIntent",
    "Rejected",
    "Side",
    "TargetPosition",
    "TargetPositionBatch",
    "VIX_HIGH_RULE_ID",
    "VIX_OUT_OF_BAND_RULE_ID",
    "VixBandGateDecision",
    "VixGateDecision",
    "VixHighGate",
    "VixOutOfBandGate",
    "build_snapshot",
    "calibrate_drawdown_thresholds",
    "diff_limits",
    "diff_targets_to_intents",
    "emit_breach",
    "evaluate_live_divergence",
    "halt",
    "heartbeat_age_seconds",
    "is_halted",
    "load_limits",
    "new_sampler",
    "run_check_chain",
    "unhalt",
]
