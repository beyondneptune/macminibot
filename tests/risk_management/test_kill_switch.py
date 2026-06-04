"""Tests for the file-flag kill switch."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from risk_management.kill_switch import (
    HEARTBEAT_STALENESS_LIMIT_SECONDS,
    audit_log_path,
    halt,
    halt_path,
    heartbeat_age_seconds,
    heartbeat_path,
    is_halted,
    unhalt,
    write_heartbeat,
)


def test_halt_then_check_returns_true() -> None:
    assert is_halted() is False
    halt("manual smoke test")
    assert is_halted() is True
    assert halt_path().exists()


def test_unhalt_then_check_returns_false() -> None:
    halt("first halt")
    assert is_halted() is True
    unhalt("operator resumed after review")
    assert is_halted() is False
    assert not halt_path().exists()


def test_halt_and_unhalt_audited() -> None:
    halt("kill_switch_active_test")
    unhalt("operator review complete")
    lines = audit_log_path().read_text(encoding="utf-8").splitlines()
    actions = [json.loads(ln)["action"] for ln in lines]
    assert "HALT" in actions
    assert "UNHALT" in actions


def test_heartbeat_age_zero_after_write() -> None:
    write_heartbeat(pid=12345, version="test")
    age = heartbeat_age_seconds()
    assert age < 1.0


def test_heartbeat_age_inf_when_file_missing() -> None:
    heartbeat_path().unlink(missing_ok=True)
    assert heartbeat_age_seconds() == float("inf")


def test_heartbeat_age_inf_when_file_malformed() -> None:
    heartbeat_path().write_text("not json", encoding="utf-8")
    assert heartbeat_age_seconds() == float("inf")


def test_heartbeat_age_computed_from_ts_unix() -> None:
    past = time.time() - 600.0
    heartbeat_path().write_text(
        json.dumps({"pid": 1, "ts_unix": past, "version": "x"}),
        encoding="utf-8",
    )
    age = heartbeat_age_seconds()
    assert age > HEARTBEAT_STALENESS_LIMIT_SECONDS


def test_halt_requires_nonempty_reason() -> None:
    with pytest.raises(ValueError):
        halt("")
    with pytest.raises(ValueError):
        halt("   ")
