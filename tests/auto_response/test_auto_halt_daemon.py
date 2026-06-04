"""Tests for risk_management.auto_halt_daemon.

We isolate state/risk/data dirs per-test (see conftest) and inject a
fake broker probe so no Alpaca calls happen. The daemon's pure helpers
(equity-curve maths, reconciliation) are tested directly; the
``run_once`` end-to-end test uses an injected stub sink and probe.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from observability.alert_sink import AlertEvent, FileAlertSink, MultiSink
from risk_management import auto_halt_daemon as ah


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_append_equity_point_persists_and_trims(tmp_path: Path) -> None:
    curve = tmp_path / "ec.parquet"
    now = time.time()
    rows = ah.append_equity_point(Decimal("100000"), now - 86400, curve_path=curve)
    rows = ah.append_equity_point(Decimal("99000"), now, curve_path=curve)
    assert len(rows) == 2
    assert rows[-1][1] == Decimal("99000")


def test_append_equity_point_drops_old_rows(tmp_path: Path) -> None:
    curve = tmp_path / "ec.parquet"
    now = time.time()
    ah.append_equity_point(
        Decimal("100"),
        now - 200 * 86400,  # 200 days old
        curve_path=curve,
        lookback_days=60,
    )
    rows = ah.append_equity_point(
        Decimal("90"), now, curve_path=curve, lookback_days=60
    )
    # The 200-day-old point should be dropped on the second write.
    assert len(rows) == 1
    assert rows[0][1] == Decimal("90")


def test_peak_to_trough_finds_worst_drawdown() -> None:
    curve = [
        (1.0, Decimal("100")),
        (2.0, Decimal("110")),  # new peak
        (3.0, Decimal("90")),   # -18.18% from peak
        (4.0, Decimal("95")),
    ]
    dd = ah.peak_to_trough(curve)
    assert dd < Decimal("-0.18")
    assert dd > Decimal("-0.19")


def test_peak_to_trough_empty_returns_zero() -> None:
    assert ah.peak_to_trough([]) == Decimal("0")
    assert ah.peak_to_trough([(1.0, Decimal("100"))]) == Decimal("0")


def test_weekly_pnl_pct_computes_over_window() -> None:
    now = time.time()
    curve = [
        (now - 6 * 86400, Decimal("100")),
        (now - 3 * 86400, Decimal("95")),
        (now, Decimal("94")),
    ]
    pct = ah.weekly_pnl_pct(curve, now_unix=now, lookback_days=7)
    # 94 vs 100 == -6%
    assert pct < Decimal("-0.05")


def test_reconcile_detects_mismatch() -> None:
    diffs = ah.reconcile_positions(
        broker={"AAPL": Decimal("100"), "MSFT": Decimal("50")},
        journal={"AAPL": Decimal("80"), "MSFT": Decimal("50")},
    )
    assert len(diffs) == 1
    assert diffs[0].symbol == "AAPL"
    assert diffs[0].delta == Decimal("20")


def test_reconcile_clean_when_equal() -> None:
    diffs = ah.reconcile_positions(
        broker={"AAPL": Decimal("100")},
        journal={"AAPL": Decimal("100")},
    )
    assert diffs == []


# ---------------------------------------------------------------------------
# Rule evaluation (pure)
# ---------------------------------------------------------------------------


def _broker_state(
    *,
    equity="100000",
    last_equity="100000",
    trading_blocked=False,
    status="ACTIVE",
    positions=None,
) -> ah.BrokerState:
    return ah.BrokerState(
        equity=Decimal(equity),
        last_equity=Decimal(last_equity),
        cash=Decimal("50000"),
        trading_blocked=trading_blocked,
        account_status=status,
        positions=positions or {},
        ts_unix=time.time(),
    )


def test_evaluate_rules_all_clean() -> None:
    broker = _broker_state()
    decisions = ah.evaluate_rules(
        broker=broker, curve=[(time.time(), broker.equity)], reconcile=[], now_unix=time.time()
    )
    assert not any(d.fired for d in decisions)


def test_evaluate_rules_fires_daily_drawdown() -> None:
    broker = _broker_state(equity="97000", last_equity="100000")  # -3%
    decisions = ah.evaluate_rules(
        broker=broker, curve=[(time.time(), broker.equity)], reconcile=[], now_unix=time.time()
    )
    fired = [d for d in decisions if d.fired]
    assert any(d.rule_id == "DD_DAILY" for d in fired)


def test_evaluate_rules_fires_trading_blocked() -> None:
    broker = _broker_state(trading_blocked=True)
    decisions = ah.evaluate_rules(
        broker=broker, curve=[], reconcile=[], now_unix=time.time()
    )
    assert any(d.rule_id == "BROKER_TRADING_BLOCKED" and d.fired for d in decisions)


def test_evaluate_rules_fires_account_inactive() -> None:
    broker = _broker_state(status="RESTRICTED")
    decisions = ah.evaluate_rules(
        broker=broker, curve=[], reconcile=[], now_unix=time.time()
    )
    assert any(d.rule_id == "BROKER_ACCOUNT_INACTIVE" and d.fired for d in decisions)


def test_evaluate_rules_fires_reconcile() -> None:
    broker = _broker_state()
    diffs = [ah.ReconcileDiff(symbol="AAPL", broker_qty=Decimal("100"), journal_qty=Decimal("80"))]
    decisions = ah.evaluate_rules(
        broker=broker, curve=[], reconcile=diffs, now_unix=time.time()
    )
    assert any(d.rule_id == "POSITION_RECONCILE" and d.fired for d in decisions)


def test_evaluate_rules_fires_peak_to_trough() -> None:
    now = time.time()
    broker = _broker_state(equity="80000", last_equity="80000")
    curve = [(now - 2 * 86400, Decimal("100000")), (now, Decimal("80000"))]
    decisions = ah.evaluate_rules(broker=broker, curve=curve, reconcile=[], now_unix=now)
    assert any(d.rule_id == "DD_PEAK_TO_TROUGH" and d.fired for d in decisions)


# ---------------------------------------------------------------------------
# End-to-end run_once
# ---------------------------------------------------------------------------


class _FakeProbe:
    def __init__(self, state: ah.BrokerState) -> None:
        self._state = state

    def fetch_state(self) -> ah.BrokerState:
        return self._state


def _build_daemon(
    tmp_path: Path,
    state: ah.BrokerState,
    *,
    sink: MultiSink | None = None,
) -> tuple[ah.AutoHaltDaemon, Path]:
    alerts_path = tmp_path / "alerts.log"
    if sink is None:
        sink = MultiSink(sinks=[FileAlertSink(path=alerts_path)])
    daemon = ah.AutoHaltDaemon(
        probe=_FakeProbe(state),
        sink=sink,
        state_dir=tmp_path / "state",
        journal_root=tmp_path / "data" / "journal",
        curve_path=tmp_path / "data" / "equity_curve.parquet",
        project_root=tmp_path,
    )
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "journal").mkdir(parents=True, exist_ok=True)
    return daemon, alerts_path


def test_run_once_clean_does_not_halt(tmp_path: Path) -> None:
    state = _broker_state()
    daemon, alerts_path = _build_daemon(tmp_path, state)
    result = daemon.run_once()
    assert result["halted"] is False
    assert not (tmp_path / "state" / "HALT").exists()
    # An empty alert log is the expected "happy path" — daemon only
    # emits on firing rules.
    assert not alerts_path.exists() or alerts_path.read_text("utf-8") == ""


def test_run_once_drawdown_halts_and_alerts(tmp_path: Path) -> None:
    state = _broker_state(equity="97000", last_equity="100000")
    daemon, alerts_path = _build_daemon(tmp_path, state)
    result = daemon.run_once()
    assert result["halted"] is True
    assert "DD_DAILY" in result["fired_rules"]
    assert (tmp_path / "state" / "HALT").exists()
    # An alert was written to the file sink.
    assert alerts_path.exists()
    records = [json.loads(ln) for ln in alerts_path.read_text("utf-8").splitlines()]
    assert any(r["rule_id"] == "DD_DAILY" for r in records)
    assert any(r["severity"] == "CRITICAL" for r in records)


def test_run_once_idempotent_no_double_alert(tmp_path: Path) -> None:
    """Re-running the same cycle on the same day must not double-alert."""

    state = _broker_state(equity="97000", last_equity="100000")
    daemon, alerts_path = _build_daemon(tmp_path, state)
    daemon.run_once()
    first_count = len(alerts_path.read_text("utf-8").splitlines())
    daemon.run_once()
    second_count = len(alerts_path.read_text("utf-8").splitlines())
    assert second_count == first_count, "second cycle must not double-alert"
    # HALT still present (idempotent flag).
    assert (tmp_path / "state" / "HALT").exists()


def test_run_once_reconcile_halt(tmp_path: Path) -> None:
    """Live broker qty != journal qty must trigger POSITION_RECONCILE."""

    state = _broker_state(positions={"AAPL": Decimal("100")})
    daemon, alerts_path = _build_daemon(tmp_path, state)
    # Write a fake journal parquet with qty=80 / status=FILLED on AAPL.
    import pyarrow as pa
    import pyarrow.parquet as pq

    journal_dir = tmp_path / "data" / "journal" / "paper" / "dt=2026-05-18"
    journal_dir.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(
        [
            {
                "client_order_id": "c1",
                "intent_id": "i1",
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 80,
                "fill_qty": 80,
                "fill_price": 150.0,
                "status": "FILLED",
                "mode": "paper",
                "submitted_at_ns": 1,
                "filled_at_ns": 1,
            }
        ]
    )
    pq.write_table(tbl, journal_dir / "part-1.parquet")

    result = daemon.run_once()
    assert result["halted"] is True
    assert "POSITION_RECONCILE" in result["fired_rules"]


def test_run_once_writes_breach_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A halt-firing run must persist a typed BreachEvent so dashboard sees it."""

    monkeypatch.setenv("MACMINIBOT_RISK_DIR", str(tmp_path / "risk"))
    state = _broker_state(trading_blocked=True)
    daemon, _ = _build_daemon(tmp_path, state)
    daemon.run_once()
    breach_log = tmp_path / "risk" / "paper" / "breaches.jsonl"
    assert breach_log.exists()
    lines = breach_log.read_text("utf-8").splitlines()
    assert any("BROKER_TRADING_BLOCKED" in ln for ln in lines)
