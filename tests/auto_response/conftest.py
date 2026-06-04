"""Isolate state + risk + data dirs per test so the live bot is untouched."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_fs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state_dir = tmp_path / "state"
    risk_dir = tmp_path / "risk"
    data_dir = tmp_path / "data"
    state_dir.mkdir(parents=True, exist_ok=True)
    risk_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MACMINIBOT_STATE_DIR", str(state_dir))
    monkeypatch.setenv("MACMINIBOT_RISK_DIR", str(risk_dir))
    monkeypatch.setenv("MACMINIBOT_DATA_DIR", str(data_dir))
    # Force the daemon's curve path env override too — the production
    # path is the project's data/ dir, but tests must not write there.
    return tmp_path
