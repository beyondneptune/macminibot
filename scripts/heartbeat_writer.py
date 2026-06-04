"""heartbeat_writer — emit a tiny liveness file once per invocation.

Designed to be fired every 60 seconds by launchd
(``com.macminibot.heartbeat.plist``). One write, one exit. No deps.

Writes to ``~/macminibot/state/heartbeat.json`` (override via
``MACMINIBOT_STATE_DIR`` env var, matching the dashboard convention).

Atomic via ``os.replace`` so the dashboard never sees a half-written file.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HEARTBEAT_VERSION = "heartbeat-v1"


def _state_dir() -> Path:
    override = os.environ.get("MACMINIBOT_STATE_DIR")
    if override:
        return Path(override)
    return Path.home() / "macminibot" / "state"


def write_heartbeat(state_dir: Path | None = None) -> Path:
    """Write a heartbeat JSON file and return its path."""
    target_dir = state_dir if state_dir is not None else _state_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "heartbeat.json"
    tmp = target.with_suffix(".json.tmp")

    payload = {
        "ts_unix": time.time(),
        "pid": os.getpid(),
        "version": HEARTBEAT_VERSION,
        "source": "launchd-heartbeat-writer",
    }
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, target)
    return target


def main() -> int:
    try:
        path = write_heartbeat()
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"heartbeat_writer failed: {exc}\n")
        return 1
    sys.stdout.write(f"heartbeat ok -> {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
