"""metrics_emitter — Epic 21 risk-metric stream to JSONL (CONTEXT.md I7e).

The CONTEXT specifies an 11-metric schema (`bot_risk_chain_runs_total`,
`bot_risk_check_rejections_total`, ...) for observability dashboards/alerts.
Until ``observability`` lands its metrics library, this module provides the
JSONL fallback prescribed in I7e:

    "fallback when observability is unavailable is to append a
    ``MetricSample`` JSON line to ``risk/metrics.jsonl`` so post-hoc
    replay is possible."

The fallback is the contract surface — when observability ships, it will
drain ``risk/metrics.jsonl`` (or be wired into the same emit calls). Keeping
the schema codified here means the observability library can be written
against a frozen set of metric names, types, and labels.

Mode discipline (I7a):
  - LIVE     -> ``risk/metrics.jsonl``
  - PAPER    -> ``risk/paper/metrics.jsonl``
  - WHATIF   -> no filesystem writes (caller may read via the in-memory
    sampler list returned by ``new_sampler``)

Every sample carries ``mode`` and ``config_id`` labels by construction
(CONTEXT.md I7e: "All metrics carry the labels ``mode``, ``config_id``").
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from shared_core.contracts import ChainMode

from risk_management.breaches import risk_dir


METRICS_FILENAME = "metrics.jsonl"
PAPER_SUBDIR = "paper"


# ---------------------------------------------------------------------------
# Schema — the 11 metrics from CONTEXT.md I7e (frozen contract)
# ---------------------------------------------------------------------------


MetricType = Literal["counter", "gauge", "histogram"]


@dataclass(frozen=True)
class MetricSpec:
    """Static schema entry — name, type, required labels."""

    name: str
    type: MetricType
    required_labels: frozenset[str]
    description: str


# Required across every metric per CONTEXT.md I7e.
COMMON_LABELS: frozenset[str] = frozenset({"mode", "config_id"})


def _spec(name: str, type_: MetricType, extra_labels: frozenset[str], desc: str) -> MetricSpec:
    return MetricSpec(
        name=name,
        type=type_,
        required_labels=COMMON_LABELS | extra_labels,
        description=desc,
    )


SCHEMA: dict[str, MetricSpec] = {
    spec.name: spec
    for spec in (
        _spec(
            "bot_risk_chain_runs_total",
            "counter",
            frozenset({"outcome"}),
            "Total check_chain.run invocations; outcome=approved|rejected.",
        ),
        _spec(
            "bot_risk_check_rejections_total",
            "counter",
            frozenset({"rule_id", "symbol", "severity"}),
            "Per-rule rejection count; severity wired in by I10.",
        ),
        _spec(
            "bot_risk_intent_notional_usd",
            "histogram",
            frozenset({"symbol", "side"}),
            "Notional USD of every intent reaching the chain.",
        ),
        _spec(
            "bot_risk_limit_utilization_ratio",
            "gauge",
            frozenset({"rule_id", "scope", "symbol"}),
            "Limit utilization in [0,1+]; scope=account|symbol.",
        ),
        _spec(
            "bot_risk_halt_active",
            "gauge",
            frozenset(),
            "1 if HALT (or PAPER_HALT) exists, else 0.",
        ),
        _spec(
            "bot_risk_halt_total",
            "counter",
            frozenset({"rule_id"}),
            "Incremented when a HALT is created (not re-sampled).",
        ),
        _spec(
            "bot_risk_unhalt_cooldown_remaining_seconds",
            "gauge",
            frozenset({"rule_id"}),
            "Nonzero while a HALT cooldown is active; drives resume alerts.",
        ),
        _spec(
            "bot_risk_heartbeat_age_seconds",
            "gauge",
            frozenset(),
            "Seconds since last heartbeat write; SLO breach >120s.",
        ),
        _spec(
            "bot_risk_drawdown_pct",
            "gauge",
            frozenset({"window"}),
            "Drawdown fraction from equity_curve; window=daily|weekly|ptt.",
        ),
        _spec(
            "bot_risk_snapshot_age_seconds",
            "gauge",
            frozenset(),
            "now - snapshot.ts during a chain run; alert if >600s.",
        ),
        _spec(
            "bot_risk_config_reload_total",
            "counter",
            frozenset({"outcome"}),
            "limits.toml reload attempts; outcome=accepted|rejected_loosen|rejected_invalid.",
        ),
    )
}


METRIC_NAMES: frozenset[str] = frozenset(SCHEMA.keys())


# ---------------------------------------------------------------------------
# Sample dataclass + emitter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSample:
    """A single observation. ``value`` is float-coerced at emit time."""

    name: str
    value: float
    labels: dict[str, str]
    ts_unix: float
    type: MetricType

    def to_jsonl(self) -> str:
        rec = {
            "name": self.name,
            "value": self.value,
            "labels": dict(self.labels),
            "ts_unix": self.ts_unix,
            "type": self.type,
        }
        return json.dumps(rec, sort_keys=True)


class MetricsValidationError(ValueError):
    """Raised when a sample doesn't match SCHEMA."""


def _coerce_value(v: Any, name: str) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        # JSON spec forbids NaN/Inf — coerce to None at the caller, not here.
        return f
    raise MetricsValidationError(
        f"metric {name!r}: value must be numeric (got {type(v).__name__}={v!r})"
    )


def _normalize_mode(mode: ChainMode | str) -> str:
    if isinstance(mode, ChainMode):
        return mode.value
    return str(mode)


def _sink_path_for(mode_value: str) -> Path | None:
    """JSONL sink path for the given mode. None = whatif (no write)."""

    if mode_value == "whatif":
        return None
    base = risk_dir()
    if mode_value == "paper":
        d = base / PAPER_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d / METRICS_FILENAME
    return base / METRICS_FILENAME


@dataclass
class MetricsSampler:
    """Stateful emitter — holds mode + config_id so callers don't repeat them.

    Sampler instances are *cheap*; build one per chain run (or per
    heartbeat tick). The in-memory ``buffer`` is always populated so
    WHATIF callers can introspect; LIVE/PAPER additionally flush to the
    JSONL sink at ``flush()`` time.
    """

    mode: ChainMode
    config_id: str
    buffer: list[MetricSample] = field(default_factory=list)

    def emit(
        self,
        name: str,
        value: Any,
        *,
        labels: dict[str, str] | None = None,
        ts_unix: float | None = None,
    ) -> MetricSample:
        """Validate, record, and return a sample.

        Schema enforcement: ``name`` must be in SCHEMA; supplied ``labels``
        plus the auto-injected ``mode`` / ``config_id`` must cover the
        spec's ``required_labels``. Unknown label keys are accepted (they
        flow through to the sink) — useful for ad-hoc dashboards — but
        missing required keys raise so the call site fails loudly.
        """

        spec = SCHEMA.get(name)
        if spec is None:
            raise MetricsValidationError(
                f"unknown metric {name!r}; allowed: {sorted(METRIC_NAMES)}"
            )

        full_labels: dict[str, str] = {
            "mode": _normalize_mode(self.mode),
            "config_id": self.config_id,
        }
        if labels:
            for k, v in labels.items():
                full_labels[str(k)] = str(v)

        missing = spec.required_labels - set(full_labels)
        if missing:
            raise MetricsValidationError(
                f"metric {name!r}: missing required labels {sorted(missing)}"
            )

        sample = MetricSample(
            name=name,
            value=_coerce_value(value, name),
            labels=full_labels,
            ts_unix=ts_unix if ts_unix is not None else time.time(),
            type=spec.type,
        )
        self.buffer.append(sample)
        return sample

    def flush(self) -> int:
        """Append buffered samples to the JSONL sink and clear the buffer.

        Returns the count of samples written. WHATIF mode is a no-op and
        leaves ``buffer`` intact so the caller can drain it manually
        (replay harnesses often want the full sample stream in-memory).
        """

        mode_value = _normalize_mode(self.mode)
        sink = _sink_path_for(mode_value)
        if sink is None:
            return 0

        if not self.buffer:
            return 0

        with sink.open("a", encoding="utf-8") as fh:
            for sample in self.buffer:
                fh.write(sample.to_jsonl() + "\n")
        n = len(self.buffer)
        self.buffer.clear()
        return n


def new_sampler(mode: ChainMode, config_id: str) -> MetricsSampler:
    """Convenience constructor — pinned to a (mode, config_id) pair."""

    return MetricsSampler(mode=mode, config_id=config_id)


__all__ = [
    "COMMON_LABELS",
    "METRIC_NAMES",
    "METRICS_FILENAME",
    "MetricSample",
    "MetricSpec",
    "MetricsSampler",
    "MetricsValidationError",
    "SCHEMA",
    "new_sampler",
]
