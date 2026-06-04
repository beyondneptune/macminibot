"""Tests for the three launchd plists that automate the paper bot.

We assert:

* every plist parses as a valid Apple property-list XML document,
* every plist has the right Label,
* the paper-runner plist points at the real ``30_paper_trade_runner.py``
  and fires Mon-Fri at 14:35 LOCAL (= 09:35 ET while UK is on BST and
  US is on EDT),
* the heartbeat plist runs the real ``heartbeat_writer.py`` every 60s
  and is RunAtLoad=true,
* the dashboard plist runs Streamlit on 0.0.0.0:8501 with
  KeepAlive=true and RunAtLoad=true,
* all three plists declare WorkingDirectory = project root, and
  stdout/stderr go under ``data/``.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHD_DIR = PROJECT_ROOT / "scripts" / "launchd"

PAPER_RUNNER_PLIST = LAUNCHD_DIR / "com.macminibot.paper-runner.plist"
HEARTBEAT_PLIST = LAUNCHD_DIR / "com.macminibot.heartbeat.plist"
DASHBOARD_PLIST = LAUNCHD_DIR / "com.macminibot.dashboard.plist"

ALL_PLISTS = [PAPER_RUNNER_PLIST, HEARTBEAT_PLIST, DASHBOARD_PLIST]


def _load(path: Path) -> dict:
    with path.open("rb") as fh:
        return plistlib.load(fh)


# ──────────────────────────────────────────────────────────────────────
# Generic checks that apply to all three plists
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("plist_path", ALL_PLISTS, ids=lambda p: p.name)
def test_plist_file_exists(plist_path: Path) -> None:
    assert plist_path.is_file(), f"plist missing: {plist_path}"


@pytest.mark.parametrize("plist_path", ALL_PLISTS, ids=lambda p: p.name)
def test_plist_parses_as_valid_xml(plist_path: Path) -> None:
    """plistlib.load raises on any XML / schema problem."""
    data = _load(plist_path)
    assert isinstance(data, dict)
    assert "Label" in data
    assert "ProgramArguments" in data


@pytest.mark.parametrize(
    "plist_path,expected_label",
    [
        (PAPER_RUNNER_PLIST, "com.macminibot.paper-runner"),
        (HEARTBEAT_PLIST, "com.macminibot.heartbeat"),
        (DASHBOARD_PLIST, "com.macminibot.dashboard"),
    ],
    ids=["paper-runner", "heartbeat", "dashboard"],
)
def test_plist_label_matches_filename(plist_path: Path, expected_label: str) -> None:
    data = _load(plist_path)
    assert data["Label"] == expected_label


@pytest.mark.parametrize("plist_path", ALL_PLISTS, ids=lambda p: p.name)
def test_working_directory_is_project_root(plist_path: Path) -> None:
    data = _load(plist_path)
    assert data.get("WorkingDirectory") == str(PROJECT_ROOT)


@pytest.mark.parametrize("plist_path", ALL_PLISTS, ids=lambda p: p.name)
def test_stdout_and_stderr_under_data_dir(plist_path: Path) -> None:
    data = _load(plist_path)
    out = data.get("StandardOutPath", "")
    err = data.get("StandardErrorPath", "")
    data_dir = str(PROJECT_ROOT / "data") + "/"
    assert out.startswith(data_dir), out
    assert err.startswith(data_dir), err
    assert out.endswith(".log")
    assert err.endswith(".err")


# ──────────────────────────────────────────────────────────────────────
# Paper runner specifics
# ──────────────────────────────────────────────────────────────────────
def test_paper_runner_targets_30_paper_trade_runner() -> None:
    data = _load(PAPER_RUNNER_PLIST)
    args = data["ProgramArguments"]
    expected_script = str(PROJECT_ROOT / "scripts" / "30_paper_trade_runner.py")
    assert expected_script in args, args
    assert Path(expected_script).is_file(), "real runner script missing"
    # Make sure we run it under our venv interpreter.
    expected_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    assert args[0] == expected_python


def test_paper_runner_fires_mon_to_fri_at_1435_local() -> None:
    data = _load(PAPER_RUNNER_PLIST)
    sched = data["StartCalendarInterval"]
    assert isinstance(sched, list), "weekday schedule must be an array of dicts"
    weekdays = sorted(entry["Weekday"] for entry in sched)
    assert weekdays == [1, 2, 3, 4, 5], weekdays
    for entry in sched:
        # 14:35 LOCAL Europe/London == 09:35 ET when both DST regimes agree.
        assert entry["Hour"] == 14, entry
        assert entry["Minute"] == 35, entry


def test_paper_runner_does_not_run_at_load() -> None:
    data = _load(PAPER_RUNNER_PLIST)
    assert data.get("RunAtLoad") is False


# ──────────────────────────────────────────────────────────────────────
# Heartbeat specifics
# ──────────────────────────────────────────────────────────────────────
def test_heartbeat_runs_real_writer_script() -> None:
    data = _load(HEARTBEAT_PLIST)
    args = data["ProgramArguments"]
    expected_script = str(PROJECT_ROOT / "scripts" / "heartbeat_writer.py")
    assert expected_script in args
    assert Path(expected_script).is_file(), "heartbeat writer script missing"


def test_heartbeat_fires_every_60_seconds() -> None:
    data = _load(HEARTBEAT_PLIST)
    assert data.get("StartInterval") == 60
    # And we want it up immediately after install.
    assert data.get("RunAtLoad") is True


# ──────────────────────────────────────────────────────────────────────
# Dashboard specifics
# ──────────────────────────────────────────────────────────────────────
def test_dashboard_runs_streamlit_on_0_0_0_0_8501() -> None:
    data = _load(DASHBOARD_PLIST)
    args = data["ProgramArguments"]
    streamlit_path = str(PROJECT_ROOT / ".venv" / "bin" / "streamlit")
    dashboard_script = str(PROJECT_ROOT / "scripts" / "dashboard.py")

    assert args[0] == streamlit_path
    assert "run" in args
    assert dashboard_script in args
    assert "--server.address" in args
    assert args[args.index("--server.address") + 1] == "0.0.0.0"
    assert "--server.port" in args
    assert args[args.index("--server.port") + 1] == "8501"


def test_dashboard_keep_alive_and_run_at_load() -> None:
    data = _load(DASHBOARD_PLIST)
    assert data.get("KeepAlive") is True
    assert data.get("RunAtLoad") is True


# ──────────────────────────────────────────────────────────────────────
# Install / uninstall scripts
# ──────────────────────────────────────────────────────────────────────
def test_install_script_exists_and_is_executable() -> None:
    install_sh = PROJECT_ROOT / "scripts" / "install_launchd.sh"
    assert install_sh.is_file()
    import os

    assert os.access(install_sh, os.X_OK), "install_launchd.sh must be executable"
    contents = install_sh.read_text()
    # We promised we'd PRINT the bootstrap commands, not run them.
    assert "launchctl bootstrap" in contents
    for plist in ALL_PLISTS:
        assert plist.name in contents


def test_uninstall_script_exists_and_is_executable() -> None:
    uninstall_sh = PROJECT_ROOT / "scripts" / "uninstall_launchd.sh"
    assert uninstall_sh.is_file()
    import os

    assert os.access(uninstall_sh, os.X_OK), "uninstall_launchd.sh must be executable"
    contents = uninstall_sh.read_text()
    assert "launchctl bootout" in contents
    for plist in ALL_PLISTS:
        assert plist.name in contents
