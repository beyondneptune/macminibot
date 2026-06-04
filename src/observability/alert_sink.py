"""observability.alert_sink — pluggable alert backend.

The auto-halt daemon and EOD summary publish AlertEvents to a configured
set of sinks. v0 ships three backends:

- ``file://<path>``         : append a JSONL record to ``<path>``.
- ``desktop://``            : macOS notification via ``osascript``.
- ``webhook://<url>``       : POST JSON to ``<url>``.

The sink URI list is read from ``data/config/alerts.toml``:

    # data/config/alerts.toml
    sinks = ["file://data/alerts.log", "desktop://"]

Tests can construct sinks directly. The daemon is responsible for
catching backend exceptions so a flaky webhook never blocks a halt.

Idempotency
-----------
Sinks themselves are stateless — they emit on every call. The daemon is
responsible for de-duping (see ``auto_halt_daemon._already_alerted``).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.parse import urlparse

__all__ = [
    "AlertEvent",
    "AlertSeverity",
    "AlertSink",
    "DesktopAlertSink",
    "FileAlertSink",
    "MultiSink",
    "WebhookAlertSink",
    "build_sink_from_uri",
    "load_sinks_from_config",
]


# ---------------------------------------------------------------------------
# Event payload
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class AlertEvent:
    """One alert. ``rule_id`` lets sinks de-dupe by rule across a run."""

    title: str
    body: str
    severity: AlertSeverity = AlertSeverity.WARN
    rule_id: str = ""
    ts_unix: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


# ---------------------------------------------------------------------------
# Sink protocol + concrete backends
# ---------------------------------------------------------------------------


class AlertSink(Protocol):
    """Minimal write-only contract every backend must satisfy."""

    def emit(self, event: AlertEvent) -> None: ...


@dataclass
class FileAlertSink:
    """Append one JSONL record per alert to ``path``.

    Uses an atomic append (``"a"`` mode with one ``write`` call) — safe
    even if the file is rotated mid-run because each line is self-contained.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: AlertEvent) -> None:
        line = json.dumps(event.to_dict(), sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


@dataclass
class DesktopAlertSink:
    """macOS desktop notification via ``osascript``.

    Best-effort — on non-Darwin platforms (or if osascript is missing)
    the call is a silent no-op rather than a hard failure. The auto-halt
    daemon must NEVER fail because the OS can't show a banner.
    """

    timeout: float = 5.0

    def emit(self, event: AlertEvent) -> None:
        # osascript wants double-quoted strings; escape any inner quotes.
        title = event.title.replace('"', "'")
        body = event.body.replace('"', "'")
        script = f'display notification "{body}" with title "{title}"'
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                timeout=self.timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            # Don't let a missing/broken osascript brick the daemon.
            pass


@dataclass
class WebhookAlertSink:
    """POST the AlertEvent (JSON body) to a URL.

    Designed for Slack-compatible incoming-webhook endpoints and generic
    JSON receivers. We send the full AlertEvent dict plus a ``text`` key
    so a Slack webhook renders out-of-the-box without a custom transform.
    """

    url: str
    timeout: float = 5.0

    def emit(self, event: AlertEvent) -> None:
        payload = event.to_dict()
        payload["text"] = f"[{event.severity.value}] {event.title}\n{event.body}"
        data = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            # The daemon must not crash if the webhook is unreachable; the
            # FileAlertSink (which is always included) preserves the record.
            return


@dataclass
class MultiSink:
    """Fan out one ``emit`` call to every wrapped sink.

    Errors from individual sinks are swallowed so a single broken backend
    never silences the rest. Use ``FileAlertSink`` as a guaranteed-local
    record-of-truth in any production config.
    """

    sinks: list[AlertSink] = field(default_factory=list)

    def emit(self, event: AlertEvent) -> None:
        for s in self.sinks:
            try:
                s.emit(event)
            except Exception:
                # Per-sink failures must not propagate. The whole point of
                # a multisink is fault tolerance.
                continue


# ---------------------------------------------------------------------------
# URI -> sink factory + TOML loader
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    # src/observability/alert_sink.py → repo root is parents[2].
    return Path(__file__).resolve().parents[2]


def build_sink_from_uri(uri: str) -> AlertSink:
    """Parse a sink URI and return a concrete backend.

    Recognised schemes: ``file://`` (path can be relative — resolved
    against the project root), ``desktop://``, ``webhook://``.
    """

    if not isinstance(uri, str) or not uri.strip():
        raise ValueError(f"invalid sink uri: {uri!r}")

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme == "file":
        # urlparse("file://data/alerts.log") puts "data" in netloc and
        # "/alerts.log" in path. Rejoin them so a leading-./relative path
        # behaves intuitively.
        raw = (parsed.netloc + parsed.path) or parsed.path
        p = Path(raw)
        if not p.is_absolute():
            p = _project_root() / p
        return FileAlertSink(path=p)

    if scheme == "desktop":
        return DesktopAlertSink()

    if scheme in ("webhook", "http", "https"):
        # ``webhook://example.com/path`` → swap the scheme to https so
        # urllib accepts it. Users who want an explicit scheme pass
        # ``http://`` / ``https://`` directly.
        if scheme == "webhook":
            target = "https://" + parsed.netloc + parsed.path
            if parsed.query:
                target += "?" + parsed.query
        else:
            target = uri
        return WebhookAlertSink(url=target)

    raise ValueError(f"unsupported sink scheme: {scheme!r} (uri={uri!r})")


def _load_toml(path: Path) -> dict[str, Any]:
    """Read TOML from ``path`` using stdlib ``tomllib`` (Python 3.11+)."""

    import tomllib

    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_sinks_from_config(
    config_path: Path | None = None,
    *,
    default_uris: Iterable[str] = ("file://data/alerts.log",),
) -> MultiSink:
    """Read sink URIs from ``data/config/alerts.toml`` and build a MultiSink.

    Schema::

        sinks = ["file://data/alerts.log", "desktop://"]

    If the file is missing or malformed, fall back to ``default_uris`` —
    the daemon ALWAYS gets at least a file sink so halt decisions remain
    auditable. Env override: ``MACMINIBOT_ALERTS_CONFIG``.
    """

    if config_path is None:
        env_override = os.environ.get("MACMINIBOT_ALERTS_CONFIG")
        if env_override:
            config_path = Path(env_override)
        else:
            config_path = _project_root() / "data" / "config" / "alerts.toml"

    config_path = Path(config_path)
    uris: list[str] = []

    if config_path.exists():
        try:
            doc = _load_toml(config_path)
            raw = doc.get("sinks")
            if isinstance(raw, list):
                uris = [str(u) for u in raw if isinstance(u, str) and u.strip()]
        except Exception:
            uris = []

    if not uris:
        uris = list(default_uris)

    backends: list[AlertSink] = []
    for u in uris:
        try:
            backends.append(build_sink_from_uri(u))
        except ValueError:
            # Skip unknown schemes but keep the rest; never crash the daemon.
            continue

    return MultiSink(sinks=backends)
