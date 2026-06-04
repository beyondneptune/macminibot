"""Tests for observability.alert_sink — backend behaviour + config loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observability.alert_sink import (
    AlertEvent,
    AlertSeverity,
    DesktopAlertSink,
    FileAlertSink,
    MultiSink,
    WebhookAlertSink,
    build_sink_from_uri,
    load_sinks_from_config,
)


def test_alert_event_to_dict_roundtrips_severity() -> None:
    ev = AlertEvent(title="t", body="b", severity=AlertSeverity.CRITICAL, rule_id="R1")
    d = ev.to_dict()
    assert d["severity"] == "CRITICAL"
    assert d["title"] == "t"
    assert d["rule_id"] == "R1"


def test_file_sink_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "alerts.log"
    sink = FileAlertSink(path=path)
    sink.emit(AlertEvent(title="a", body="b", rule_id="R1"))
    sink.emit(AlertEvent(title="c", body="d", rule_id="R2"))
    lines = path.read_text("utf-8").splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert rec0["rule_id"] == "R1"
    assert rec1["rule_id"] == "R2"


def test_file_sink_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "deep" / "alerts.log"
    sink = FileAlertSink(path=nested)
    sink.emit(AlertEvent(title="x", body="y"))
    assert nested.exists()


def test_desktop_sink_swallows_osascript_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing osascript must not crash the daemon."""

    import subprocess as sp

    def boom(*args, **kwargs):  # noqa: ANN001
        raise FileNotFoundError("osascript not installed")

    monkeypatch.setattr(sp, "run", boom)
    DesktopAlertSink().emit(AlertEvent(title="t", body="b"))  # must not raise


def test_webhook_sink_swallows_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    def boom(*args, **kwargs):  # noqa: ANN001
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    WebhookAlertSink(url="http://127.0.0.1:1").emit(AlertEvent(title="t", body="b"))


def test_multisink_continues_on_inner_failure(tmp_path: Path) -> None:
    file_sink = FileAlertSink(path=tmp_path / "ok.log")

    class Boom:
        def emit(self, event):
            raise RuntimeError("nope")

    multi = MultiSink(sinks=[Boom(), file_sink])
    multi.emit(AlertEvent(title="t", body="b"))
    assert (tmp_path / "ok.log").exists()
    assert len((tmp_path / "ok.log").read_text("utf-8").splitlines()) == 1


def test_build_sink_from_uri_file(tmp_path: Path) -> None:
    abs_uri = "file://" + str(tmp_path / "x.log")
    sink = build_sink_from_uri(abs_uri)
    assert isinstance(sink, FileAlertSink)
    assert sink.path == tmp_path / "x.log"


def test_build_sink_from_uri_desktop() -> None:
    assert isinstance(build_sink_from_uri("desktop://"), DesktopAlertSink)


def test_build_sink_from_uri_webhook() -> None:
    s = build_sink_from_uri("webhook://hooks.example.com/abc")
    assert isinstance(s, WebhookAlertSink)
    assert s.url.startswith("https://")


def test_build_sink_from_uri_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError):
        build_sink_from_uri("smtp://example.com")


def test_load_sinks_from_config_reads_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "alerts.toml"
    cfg.write_text('sinks = ["file://%s"]\n' % (tmp_path / "out.log"))
    multi = load_sinks_from_config(cfg)
    assert isinstance(multi, MultiSink)
    assert len(multi.sinks) == 1
    assert isinstance(multi.sinks[0], FileAlertSink)


def test_load_sinks_from_config_fallback_when_missing(tmp_path: Path) -> None:
    multi = load_sinks_from_config(tmp_path / "does-not-exist.toml")
    # Should fall back to default — at least one sink.
    assert len(multi.sinks) >= 1


def test_load_sinks_from_config_skips_invalid_schemes(tmp_path: Path) -> None:
    cfg = tmp_path / "alerts.toml"
    cfg.write_text(
        'sinks = ["file://%s", "smtp://nope.example.com"]\n' % (tmp_path / "ok.log")
    )
    multi = load_sinks_from_config(cfg)
    # The smtp:// entry is silently dropped; file sink survives.
    assert len(multi.sinks) == 1
    assert isinstance(multi.sinks[0], FileAlertSink)
