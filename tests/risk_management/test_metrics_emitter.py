"""Tests for risk_management.metrics_emitter (Epic 21 schema + JSONL fallback).

Schema is the contract surface for observability. These tests pin:
  - the 11-metric name set,
  - common-label injection (mode + config_id always present),
  - required-label enforcement (missing label => loud error),
  - mode dispatch (LIVE writes risk/metrics.jsonl, PAPER writes paper/, WHATIF no-op),
  - JSONL line shape so a downstream consumer can be written against frozen fields.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from shared_core.contracts import ChainMode

from risk_management.metrics_emitter import (
    METRIC_NAMES,
    METRICS_FILENAME,
    PAPER_SUBDIR,
    MetricsSampler,
    MetricsValidationError,
    SCHEMA,
    new_sampler,
)


def test_schema_has_11_metrics():
    # CONTEXT.md I7e says 11; today's epic 21 enumerates exactly 11 names.
    assert len(METRIC_NAMES) == 11


def test_schema_includes_each_documented_metric():
    expected = {
        "bot_risk_chain_runs_total",
        "bot_risk_check_rejections_total",
        "bot_risk_intent_notional_usd",
        "bot_risk_limit_utilization_ratio",
        "bot_risk_halt_active",
        "bot_risk_halt_total",
        "bot_risk_unhalt_cooldown_remaining_seconds",
        "bot_risk_heartbeat_age_seconds",
        "bot_risk_drawdown_pct",
        "bot_risk_snapshot_age_seconds",
        "bot_risk_config_reload_total",
    }
    assert METRIC_NAMES == expected


def test_every_metric_requires_mode_and_config_id():
    for spec in SCHEMA.values():
        assert "mode" in spec.required_labels
        assert "config_id" in spec.required_labels


def test_unknown_metric_name_rejected():
    s = new_sampler(ChainMode.WHATIF, "test")
    with pytest.raises(MetricsValidationError, match="unknown metric"):
        s.emit("bot_risk_made_up", 1.0)


def test_missing_required_label_rejected():
    s = new_sampler(ChainMode.WHATIF, "test")
    # bot_risk_check_rejections_total needs rule_id + symbol + severity.
    with pytest.raises(MetricsValidationError, match="missing required labels"):
        s.emit("bot_risk_check_rejections_total", 1, labels={"rule_id": "X"})


def test_mode_and_config_id_auto_injected():
    s = new_sampler(ChainMode.LIVE, "2026-05-17.a")
    sample = s.emit("bot_risk_halt_active", 1)
    assert sample.labels["mode"] == "live"
    assert sample.labels["config_id"] == "2026-05-17.a"


def test_bool_value_coerced_to_float():
    s = new_sampler(ChainMode.WHATIF, "test")
    sample = s.emit("bot_risk_halt_active", True)
    assert sample.value == 1.0
    sample = s.emit("bot_risk_halt_active", False)
    assert sample.value == 0.0


def test_decimal_value_coerced_to_float():
    s = new_sampler(ChainMode.WHATIF, "test")
    sample = s.emit(
        "bot_risk_drawdown_pct",
        Decimal("-0.025"),
        labels={"window": "daily"},
    )
    assert isinstance(sample.value, float)
    assert sample.value == pytest.approx(-0.025)


def test_non_numeric_value_rejected():
    s = new_sampler(ChainMode.WHATIF, "test")
    with pytest.raises(MetricsValidationError, match="must be numeric"):
        s.emit("bot_risk_halt_active", "not-a-number")


def test_whatif_mode_does_not_write_file(_isolate_fs: Path):
    s = new_sampler(ChainMode.WHATIF, "test")
    s.emit("bot_risk_halt_active", 1)
    written = s.flush()
    assert written == 0
    # No file should exist anywhere under the temp risk dir.
    risk_root = _isolate_fs / "risk"
    assert not any(risk_root.rglob(METRICS_FILENAME))
    # Buffer retained for WHATIF callers.
    assert len(s.buffer) == 1


def test_live_mode_writes_to_risk_metrics_jsonl(_isolate_fs: Path):
    s = new_sampler(ChainMode.LIVE, "test")
    s.emit("bot_risk_halt_active", 1)
    s.emit(
        "bot_risk_chain_runs_total", 3, labels={"outcome": "approved"}
    )
    written = s.flush()
    assert written == 2
    path = _isolate_fs / "risk" / METRICS_FILENAME
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    # Each line round-trips through json.
    for line in lines:
        rec = json.loads(line)
        assert "name" in rec and "value" in rec and "labels" in rec
        assert "ts_unix" in rec and "type" in rec


def test_paper_mode_writes_under_paper_subdir(_isolate_fs: Path):
    s = new_sampler(ChainMode.PAPER, "test")
    s.emit("bot_risk_halt_active", 0)
    s.flush()
    path = _isolate_fs / "risk" / PAPER_SUBDIR / METRICS_FILENAME
    assert path.exists()
    # And NOT in the live path.
    live_path = _isolate_fs / "risk" / METRICS_FILENAME
    assert not live_path.exists()


def test_flush_clears_buffer_for_live_mode(_isolate_fs: Path):
    s = new_sampler(ChainMode.LIVE, "test")
    s.emit("bot_risk_halt_active", 1)
    s.flush()
    assert s.buffer == []
    # Second flush is a no-op.
    assert s.flush() == 0


def test_flush_appends_not_truncates(_isolate_fs: Path):
    s1 = new_sampler(ChainMode.LIVE, "test")
    s1.emit("bot_risk_halt_active", 1)
    s1.flush()
    s2 = new_sampler(ChainMode.LIVE, "test")
    s2.emit("bot_risk_halt_active", 0)
    s2.flush()
    path = _isolate_fs / "risk" / METRICS_FILENAME
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_extra_label_keys_pass_through():
    """Unknown labels are allowed (ad-hoc dashboard support) so long as
    the required ones are present."""

    s = new_sampler(ChainMode.WHATIF, "test")
    sample = s.emit(
        "bot_risk_halt_active",
        1,
        labels={"experiment": "shadow"},
    )
    assert sample.labels["experiment"] == "shadow"


def test_chain_mode_string_value_accepted_as_mode():
    # Sampler is typed as ChainMode but `mode` label coerces via .value.
    s = MetricsSampler(mode=ChainMode.LIVE, config_id="test")
    sample = s.emit("bot_risk_halt_active", 0)
    assert sample.labels["mode"] == "live"
