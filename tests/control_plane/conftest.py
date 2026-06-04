"""Shared fixtures for control_plane tests.

Isolates state dir per test so HALT/audit-log writes never touch the
operator's real ``~/macminibot/state``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MACMINIBOT_STATE_DIR", str(state_dir))
    return state_dir
