"""Tests for scripts/heartbeat_writer.py (the every-60s liveness file)."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEARTBEAT_PATH = PROJECT_ROOT / "scripts" / "heartbeat_writer.py"


@pytest.fixture(scope="module")
def heartbeat_module():
    """Import scripts/heartbeat_writer.py as a module (numeric prefix safe)."""
    spec = importlib.util.spec_from_file_location("heartbeat_writer", HEARTBEAT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_writes_valid_json_with_required_fields(tmp_path, heartbeat_module):
    target = heartbeat_module.write_heartbeat(state_dir=tmp_path)
    assert target == tmp_path / "heartbeat.json"
    payload = json.loads(target.read_text())

    assert "ts_unix" in payload
    assert "pid" in payload
    assert "version" in payload

    # ts_unix is recent (within 5s of now).
    assert abs(payload["ts_unix"] - time.time()) < 5.0
    assert isinstance(payload["pid"], int) and payload["pid"] > 0
    assert payload["version"].startswith("heartbeat-")


def test_overwrites_atomically_on_repeat_call(tmp_path, heartbeat_module):
    first = heartbeat_module.write_heartbeat(state_dir=tmp_path)
    ts1 = json.loads(first.read_text())["ts_unix"]
    time.sleep(0.01)
    second = heartbeat_module.write_heartbeat(state_dir=tmp_path)
    ts2 = json.loads(second.read_text())["ts_unix"]
    assert first == second
    assert ts2 >= ts1
    # No leftover tmp file.
    assert not (tmp_path / "heartbeat.json.tmp").exists()


def test_module_has_main_returning_zero(tmp_path, heartbeat_module, monkeypatch):
    monkeypatch.setenv("MACMINIBOT_STATE_DIR", str(tmp_path))
    rc = heartbeat_module.main()
    assert rc == 0
    assert (tmp_path / "heartbeat.json").exists()
