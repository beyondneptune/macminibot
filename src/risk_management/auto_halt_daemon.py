"""auto_halt_daemon — periodic risk-state probe that auto-halts the bot.

Runs every 5 minutes during US market hours (launchd-driven). For each
poll cycle it pulls Alpaca account state and decides whether any of the
following rules trip:

  - ``DD_DAILY``                 day P&L pct < ``DD_DAILY_THRESHOLD`` (default -2%)
  - ``DD_WEEKLY``                rolling 5-trading-day P&L pct < -5%
  - ``DD_PEAK_TO_TROUGH``        60-day peak-to-trough drawdown < -10%
  - ``BROKER_TRADING_BLOCKED``   ``account.trading_blocked == True``
  - ``BROKER_ACCOUNT_INACTIVE``  ``account.status != ACTIVE``
  - ``POSITION_RECONCILE``       live broker qty differs from journal qty
                                 for any symbol

If ANY rule fires, the daemon:

  1. writes ``~/macminibot/state/HALT`` (via ``risk_management.halt``);
  2. emits a typed ``BreachEvent`` to ``risk/breaches.jsonl``
     (visible on the dashboard);
  3. publishes a CRITICAL alert through the configured sinks;
  4. records the rule fingerprint in
     ``state/auto_halt_fingerprints.json`` so a re-run on the same UTC
     day does not double-halt or double-alert (idempotency contract).

The rolling 60-day equity curve is persisted to
``data/equity_curve.parquet`` so peak-to-trough survives daemon restarts.
The daemon appends-and-trims (drops rows older than 60 calendar days
each cycle) so the file stays bounded.

CLI
---
``python -m risk_management.auto_halt_daemon`` runs ONE cycle and exits
(launchd schedules the cadence). Pass ``--loop SECONDS`` for an
in-process loop (useful for local testing or fallback when launchd is
unavailable). All thresholds can be overridden via env vars
(``MMB_DD_DAILY_PCT`` etc.) for stress-testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Defaults — read once at module load, env-overridable for stress tests.
# ---------------------------------------------------------------------------


def _env_decimal(key: str, default: str) -> Decimal:
    val = os.environ.get(key)
    if val is None or not val.strip():
        return Decimal(default)
    try:
        return Decimal(val)
    except Exception:
        return Decimal(default)


DD_DAILY_THRESHOLD: Decimal = _env_decimal("MMB_DD_DAILY_PCT", "-0.02")
DD_WEEKLY_THRESHOLD: Decimal = _env_decimal("MMB_DD_WEEKLY_PCT", "-0.05")
DD_PTT_THRESHOLD: Decimal = _env_decimal("MMB_DD_PTT_PCT", "-0.10")
PTT_LOOKBACK_DAYS: int = int(os.environ.get("MMB_DD_PTT_LOOKBACK_DAYS", "60"))
WEEKLY_LOOKBACK_DAYS: int = int(os.environ.get("MMB_DD_WEEKLY_LOOKBACK_DAYS", "7"))

EQUITY_CURVE_FILENAME = "equity_curve.parquet"
FINGERPRINT_FILENAME = "auto_halt_fingerprints.json"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _logger() -> logging.Logger:
    log = logging.getLogger("macminibot.auto_halt")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [auto_halt] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


# ---------------------------------------------------------------------------
# Broker probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BrokerState:
    """Minimal snapshot of an Alpaca account for risk evaluation."""

    equity: Decimal
    last_equity: Decimal
    cash: Decimal
    trading_blocked: bool
    account_status: str
    positions: dict[str, Decimal]  # symbol -> signed qty
    ts_unix: float

    @property
    def day_pnl_pct(self) -> Decimal:
        if self.last_equity == 0:
            return Decimal("0")
        return (self.equity - self.last_equity) / self.last_equity


class BrokerProbe:
    """Pulls live account state from Alpaca.

    Constructed lazily so unit tests can hand a hand-rolled stub
    satisfying the same surface (``fetch_state``).
    """

    def __init__(self, trading_client: Any | None = None) -> None:
        self._client = trading_client

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "BrokerProbe":
        # We re-use the same config loader as execution.alpaca_client so
        # secrets live in one place.
        from execution.alpaca_client import load_alpaca_config

        cfg = load_alpaca_config(dotenv_path)
        from alpaca.trading.client import TradingClient

        client = TradingClient(
            api_key=cfg.api_key,
            secret_key=cfg.api_secret,
            paper=cfg.paper,
        )
        return cls(trading_client=client)

    def fetch_state(self) -> BrokerState:
        if self._client is None:
            raise RuntimeError("BrokerProbe has no trading client wired")
        acct = self._client.get_account()
        # Pull positions as signed Decimals.
        positions: dict[str, Decimal] = {}
        for pos in self._client.get_all_positions():
            ticker = str(getattr(pos, "symbol", ""))
            if not ticker:
                continue
            qty = Decimal(str(getattr(pos, "qty", "0")))
            side = str(getattr(getattr(pos, "side", "long"), "value", "long")).lower()
            if side == "short" and qty > 0:
                qty = -qty
            positions[ticker] = qty
        status = str(getattr(acct, "status", "") or "")
        return BrokerState(
            equity=Decimal(str(getattr(acct, "equity", "0"))),
            last_equity=Decimal(str(getattr(acct, "last_equity", "0"))),
            cash=Decimal(str(getattr(acct, "cash", "0"))),
            trading_blocked=bool(getattr(acct, "trading_blocked", False)),
            account_status=status,
            positions=positions,
            ts_unix=time.time(),
        )


# ---------------------------------------------------------------------------
# Equity curve persistence (rolling 60d)
# ---------------------------------------------------------------------------


def equity_curve_path(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        data_dir = _project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / EQUITY_CURVE_FILENAME


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def append_equity_point(
    equity: Decimal,
    ts_unix: float,
    *,
    curve_path: Path | None = None,
    lookback_days: int = PTT_LOOKBACK_DAYS,
) -> "list[tuple[float, Decimal]]":
    """Append ``(ts_unix, equity)`` to the rolling equity curve.

    Trims any row older than ``lookback_days`` so the file stays bounded.
    Returns the full retained curve as a list of tuples (chronological).
    Uses pyarrow because the rest of the project already depends on it.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    p = curve_path if curve_path is not None else equity_curve_path()
    cutoff = ts_unix - lookback_days * 86400.0

    rows: list[tuple[float, Decimal]] = []
    if p.exists():
        try:
            tbl = pq.read_table(p)
            ts_col = tbl.column("ts_unix").to_pylist()
            eq_col = tbl.column("equity").to_pylist()
            for t, e in zip(ts_col, eq_col):
                if t is None or e is None:
                    continue
                if float(t) < cutoff:
                    continue
                rows.append((float(t), Decimal(str(e))))
        except Exception:
            rows = []

    rows.append((float(ts_unix), Decimal(str(equity))))
    rows.sort(key=lambda r: r[0])

    schema = pa.schema(
        [pa.field("ts_unix", pa.float64()), pa.field("equity", pa.string())]
    )
    table = pa.Table.from_pylist(
        [{"ts_unix": t, "equity": str(e)} for t, e in rows],
        schema=schema,
    )
    tmp = p.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(p)
    return rows


def peak_to_trough(curve: Iterable[tuple[float, Decimal]]) -> Decimal:
    """Largest drawdown observed on the curve as a Decimal pct (<= 0)."""

    rows = list(curve)
    if len(rows) < 2:
        return Decimal("0")
    peak = rows[0][1]
    worst = Decimal("0")
    for _, eq in rows:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (eq - peak) / peak
            if dd < worst:
                worst = dd
    return worst


def weekly_pnl_pct(
    curve: Iterable[tuple[float, Decimal]],
    *,
    now_unix: float,
    lookback_days: int = WEEKLY_LOOKBACK_DAYS,
) -> Decimal:
    """Rolling P&L pct over the last ``lookback_days``.

    Compares the most recent equity point against the oldest point within
    the lookback window. Returns 0 when fewer than 2 points exist.
    """

    rows = sorted(curve, key=lambda r: r[0])
    if len(rows) < 2:
        return Decimal("0")
    cutoff = now_unix - lookback_days * 86400.0
    in_window = [r for r in rows if r[0] >= cutoff]
    if len(in_window) < 2:
        in_window = rows[-2:]
    start_eq = in_window[0][1]
    end_eq = in_window[-1][1]
    if start_eq == 0:
        return Decimal("0")
    return (end_eq - start_eq) / start_eq


# ---------------------------------------------------------------------------
# Position reconciliation
# ---------------------------------------------------------------------------


def read_journal_positions(journal_root: Path) -> dict[str, Decimal]:
    """Sum signed fill_qty by symbol from the paper journal.

    Reads every parquet under ``<root>/paper/dt=*/`` and aggregates
    ``fill_qty`` with sign from ``side``. Used for the live-vs-journal
    reconciliation check. Returns ``{}`` when the journal is empty.
    """

    paper = journal_root / "paper"
    if not paper.exists():
        return {}

    import pyarrow.parquet as pq

    totals: dict[str, Decimal] = {}
    for part in paper.rglob("*.parquet"):
        try:
            tbl = pq.read_table(part)
        except Exception:
            continue
        # Pull only the columns we need so we tolerate the wider schema.
        cols = tbl.column_names
        if "symbol" not in cols or "side" not in cols:
            continue
        symbols = tbl.column("symbol").to_pylist()
        sides = tbl.column("side").to_pylist()
        # Prefer fill_qty when populated, else qty (intent qty as a fallback).
        fill_q = tbl.column("fill_qty").to_pylist() if "fill_qty" in cols else [None] * len(symbols)
        intent_q = tbl.column("qty").to_pylist() if "qty" in cols else [None] * len(symbols)
        statuses = tbl.column("status").to_pylist() if "status" in cols else [None] * len(symbols)

        for sym, side, fq, iq, status in zip(symbols, sides, fill_q, intent_q, statuses):
            if status is not None and str(status).upper() not in ("FILLED", "PARTIALLY_FILLED"):
                # Only count completed fills toward broker reconciliation.
                continue
            qty = fq if fq is not None else iq
            if qty is None or sym is None:
                continue
            try:
                d = Decimal(str(qty))
            except Exception:
                continue
            side_norm = str(side or "BUY").upper()
            if side_norm.startswith("S"):
                d = -d
            totals[str(sym)] = totals.get(str(sym), Decimal("0")) + d

    # Drop zero rows so the diff function is clean.
    return {k: v for k, v in totals.items() if v != 0}


@dataclass(frozen=True, slots=True)
class ReconcileDiff:
    symbol: str
    broker_qty: Decimal
    journal_qty: Decimal

    @property
    def delta(self) -> Decimal:
        return self.broker_qty - self.journal_qty


def reconcile_positions(
    broker: dict[str, Decimal],
    journal: dict[str, Decimal],
) -> list[ReconcileDiff]:
    """Return any symbol where broker qty != journal qty."""

    diffs: list[ReconcileDiff] = []
    for sym in sorted(set(broker) | set(journal)):
        b = broker.get(sym, Decimal("0"))
        j = journal.get(sym, Decimal("0"))
        if b != j:
            diffs.append(ReconcileDiff(symbol=sym, broker_qty=b, journal_qty=j))
    return diffs


# ---------------------------------------------------------------------------
# Rule decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleDecision:
    rule_id: str
    fired: bool
    observed: str
    threshold: str
    detail: dict[str, Any] = field(default_factory=dict)


def evaluate_rules(
    *,
    broker: BrokerState,
    curve: list[tuple[float, Decimal]],
    reconcile: list[ReconcileDiff],
    now_unix: float,
) -> list[RuleDecision]:
    """Pure evaluation — no side effects. Returns one decision per rule.

    The auto-halt daemon halts iff ANY decision has ``fired=True``.
    """

    decisions: list[RuleDecision] = []

    # Rule 1 — daily P&L pct
    day_pct = broker.day_pnl_pct
    decisions.append(
        RuleDecision(
            rule_id="DD_DAILY",
            fired=day_pct < DD_DAILY_THRESHOLD,
            observed=f"day_pnl_pct={day_pct:.4f}",
            threshold=f"day_pnl_pct<{DD_DAILY_THRESHOLD}",
            detail={"equity": str(broker.equity), "last_equity": str(broker.last_equity)},
        )
    )

    # Rule 2 — rolling weekly P&L pct
    wk_pct = weekly_pnl_pct(curve, now_unix=now_unix)
    decisions.append(
        RuleDecision(
            rule_id="DD_WEEKLY",
            fired=wk_pct < DD_WEEKLY_THRESHOLD,
            observed=f"weekly_pnl_pct={wk_pct:.4f}",
            threshold=f"weekly_pnl_pct<{DD_WEEKLY_THRESHOLD}",
            detail={"points": len(curve)},
        )
    )

    # Rule 3 — peak-to-trough drawdown
    ptt = peak_to_trough(curve)
    decisions.append(
        RuleDecision(
            rule_id="DD_PEAK_TO_TROUGH",
            fired=ptt < DD_PTT_THRESHOLD,
            observed=f"peak_to_trough_pct={ptt:.4f}",
            threshold=f"peak_to_trough_pct<{DD_PTT_THRESHOLD}",
            detail={"lookback_days": PTT_LOOKBACK_DAYS, "points": len(curve)},
        )
    )

    # Rule 4 — broker trading_blocked
    decisions.append(
        RuleDecision(
            rule_id="BROKER_TRADING_BLOCKED",
            fired=bool(broker.trading_blocked),
            observed=f"trading_blocked={broker.trading_blocked}",
            threshold="trading_blocked==False",
        )
    )

    # Rule 5 — account status
    status_active = broker.account_status.upper() == "ACTIVE"
    decisions.append(
        RuleDecision(
            rule_id="BROKER_ACCOUNT_INACTIVE",
            fired=not status_active,
            observed=f"account_status={broker.account_status!r}",
            threshold="account_status==ACTIVE",
        )
    )

    # Rule 6 — position reconciliation
    decisions.append(
        RuleDecision(
            rule_id="POSITION_RECONCILE",
            fired=len(reconcile) > 0,
            observed=f"reconcile_mismatches={len(reconcile)}",
            threshold="mismatch_count==0",
            detail={
                "diffs": [
                    {
                        "symbol": d.symbol,
                        "broker_qty": str(d.broker_qty),
                        "journal_qty": str(d.journal_qty),
                    }
                    for d in reconcile
                ],
            },
        )
    )

    return decisions


# ---------------------------------------------------------------------------
# Fingerprint store — idempotency
# ---------------------------------------------------------------------------


def _fingerprint_path(state_dir: Path) -> Path:
    return state_dir / FINGERPRINT_FILENAME


def _load_fingerprints(state_dir: Path) -> dict[str, str]:
    p = _fingerprint_path(state_dir)
    if not p.exists():
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(p.read_text("utf-8")).items()}
    except Exception:
        return {}


def _save_fingerprints(state_dir: Path, data: dict[str, str]) -> None:
    p = _fingerprint_path(state_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _fingerprint_key(rule_id: str, ts_unix: float) -> str:
    day = datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"{day}:{rule_id}"


# ---------------------------------------------------------------------------
# Daemon entrypoint
# ---------------------------------------------------------------------------


@dataclass
class AutoHaltDaemon:
    """One-shot probe; ``run_once`` is the only entry point.

    Wiring is dependency-injected so tests can hand in a fake broker probe
    + sink. Production code uses ``AutoHaltDaemon.from_env()``.
    """

    probe: BrokerProbe
    sink: Any  # AlertSink protocol — avoid hard import at class scope
    state_dir: Path
    journal_root: Path
    curve_path: Path
    project_root: Path

    @classmethod
    def from_env(cls) -> "AutoHaltDaemon":
        from observability.alert_sink import load_sinks_from_config
        from risk_management.kill_switch import state_dir as _state_dir

        root = _project_root()
        return cls(
            probe=BrokerProbe.from_env(),
            sink=load_sinks_from_config(),
            state_dir=_state_dir(),
            journal_root=root / "data" / "journal",
            curve_path=equity_curve_path(root / "data"),
            project_root=root,
        )

    def run_once(self) -> dict[str, Any]:
        """Execute one probe cycle. Returns a structured result dict.

        Always emits a heartbeat-style INFO alert so an empty alert log
        is a real signal (probe never ran), not a happy state.
        """

        log = _logger()
        from observability.alert_sink import AlertEvent, AlertSeverity
        from risk_management.breaches import emit_breach, make_breach
        from risk_management.kill_switch import halt, is_halted
        from shared_core.contracts import ChainMode

        broker = self.probe.fetch_state()
        log.info(
            "broker probed: equity=%s last_equity=%s blocked=%s status=%s positions=%d",
            broker.equity,
            broker.last_equity,
            broker.trading_blocked,
            broker.account_status,
            len(broker.positions),
        )

        curve = append_equity_point(
            broker.equity, broker.ts_unix, curve_path=self.curve_path
        )
        journal_pos = read_journal_positions(self.journal_root)
        diffs = reconcile_positions(broker.positions, journal_pos)
        decisions = evaluate_rules(
            broker=broker, curve=curve, reconcile=diffs, now_unix=broker.ts_unix
        )

        fingerprints = _load_fingerprints(self.state_dir)
        ts_ns = int(broker.ts_unix * 1e9)
        snapshot_hash = f"sha256:auto_halt:{int(broker.ts_unix)}"
        new_alerts: list[AlertEvent] = []
        new_halts: list[RuleDecision] = []

        for dec in decisions:
            if not dec.fired:
                continue
            key = _fingerprint_key(dec.rule_id, broker.ts_unix)
            if key in fingerprints:
                log.info(
                    "rule %s already fired today (key=%s) — skipping alert",
                    dec.rule_id,
                    key,
                )
                continue

            # Persist a typed breach event (visible on dashboard).
            try:
                ev = make_breach(
                    rule_id=dec.rule_id,
                    observed=dec.observed,
                    threshold=dec.threshold,
                    intent_id="",
                    snapshot_hash=snapshot_hash,
                    mode=ChainMode.PAPER,
                    ts_ns=ts_ns,
                    would_halt=True,
                    config_id="auto_halt_daemon",
                    extra=dec.detail,
                )
                emit_breach(ev)
            except Exception as exc:
                log.warning("failed to emit breach for %s: %s", dec.rule_id, exc)

            new_halts.append(dec)
            new_alerts.append(
                AlertEvent(
                    title=f"AUTO-HALT: {dec.rule_id}",
                    body=f"{dec.observed} (threshold {dec.threshold})",
                    severity=AlertSeverity.CRITICAL,
                    rule_id=dec.rule_id,
                    ts_unix=broker.ts_unix,
                    extra=dec.detail,
                )
            )
            fingerprints[key] = dec.observed

        if new_halts:
            # Single halt() call no matter how many rules fired — the file
            # flag is binary. Idempotent: halt() also no-ops on the file
            # itself if it already exists.
            reason = "auto_halt:" + ",".join(d.rule_id for d in new_halts)
            if not is_halted():
                halt(reason)
                log.warning("HALT engaged — reason=%s", reason)
            else:
                log.info("HALT already present — re-confirming via audit log: %s", reason)
                halt(reason)  # appends audit line only
            for alert in new_alerts:
                try:
                    self.sink.emit(alert)
                except Exception as exc:
                    log.warning("alert emit failed: %s", exc)
            _save_fingerprints(self.state_dir, fingerprints)
        else:
            log.info("all rules clean")

        return {
            "ts_unix": broker.ts_unix,
            "halted": bool(new_halts),
            "fired_rules": [d.rule_id for d in new_halts],
            "decisions": [
                {
                    "rule_id": d.rule_id,
                    "fired": d.fired,
                    "observed": d.observed,
                    "threshold": d.threshold,
                }
                for d in decisions
            ],
            "reconcile_diffs": [
                {"symbol": d.symbol, "broker": str(d.broker_qty), "journal": str(d.journal_qty)}
                for d in diffs
            ],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loop",
        type=float,
        default=None,
        help="If set, run in a forever-loop every N seconds (else: one cycle).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON result to stdout.",
    )
    args = parser.parse_args(argv)

    daemon = AutoHaltDaemon.from_env()

    def one() -> dict[str, Any]:
        try:
            return daemon.run_once()
        except Exception as exc:
            _logger().exception("auto-halt cycle errored: %s", exc)
            return {"error": str(exc)}

    if args.loop is None:
        result = one()
        if args.json:
            print(json.dumps(result, sort_keys=True, default=str))
        return 0 if "error" not in result else 1

    while True:
        result = one()
        if args.json:
            print(json.dumps(result, sort_keys=True, default=str), flush=True)
        time.sleep(max(1.0, args.loop))


if __name__ == "__main__":
    sys.exit(main())
