"""41_end_of_day_summary — markdown daily summary, 16:30 ET (Mon-Fri).

Generates ``data/reports/YYYY-MM-DD.md`` with:

  - Account: starting equity, ending equity, day P&L %.
  - Per-symbol P&L (live broker view).
  - Orders submitted vs filled (today).
  - Breach events from ``risk/breaches.jsonl`` for today.
  - VIX regime classification for the day.
  - GREEN / YELLOW / RED top-line status.

GREEN  — no breaches today, day P&L >= 0.
YELLOW — minor: at least one breach but day P&L >= -1%.
RED    — day P&L < -1% OR any HALT-triggering breach.

Run manually:
    .venv/bin/python scripts/41_end_of_day_summary.py
    .venv/bin/python scripts/41_end_of_day_summary.py --date 2026-05-18

Launchd schedule: 21:30 BST = 16:30 ET on weekdays.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Today helpers
# ---------------------------------------------------------------------------


def _today_utc_date(now: datetime | None = None) -> str:
    n = now or datetime.now(timezone.utc)
    return n.strftime("%Y-%m-%d")


def _day_window_unix(day_iso: str) -> tuple[float, float]:
    """[start, end) unix-seconds for the given YYYY-MM-DD in UTC."""

    start = datetime.strptime(day_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


@dataclass
class AccountSection:
    starting_equity: Decimal
    ending_equity: Decimal
    day_pnl: Decimal
    day_pnl_pct: Decimal
    trading_blocked: bool
    account_status: str

    def render(self) -> str:
        lines = [
            "## Account",
            "",
            f"- Starting equity: **${self.starting_equity:,.2f}**",
            f"- Ending equity:   **${self.ending_equity:,.2f}**",
            f"- Day P&L:         **${self.day_pnl:+,.2f}**  ({self.day_pnl_pct:+.2%})",
            f"- Trading blocked: {self.trading_blocked}",
            f"- Account status:  {self.account_status}",
        ]
        return "\n".join(lines)


@dataclass
class PerSymbolSection:
    rows: list[dict[str, Any]]  # symbol, qty, mark, unrealized_pl, unrealized_pct

    def render(self) -> str:
        if not self.rows:
            return "## Per-symbol P&L\n\n_no open positions_"
        out = ["## Per-symbol P&L", "", "| Symbol | Qty | Mark | Unrealized P&L | Unrealized % |", "| --- | ---: | ---: | ---: | ---: |"]
        for r in self.rows:
            out.append(
                f"| {r['symbol']} | {r['qty']:+,.0f} | ${r['mark']:,.4f} | "
                f"${r['unrealized_pl']:+,.2f} | {r['unrealized_pct']:+.2%} |"
            )
        return "\n".join(out)


@dataclass
class OrdersSection:
    submitted: int
    filled: int
    cancelled: int
    rejected: int

    def render(self) -> str:
        return (
            "## Orders\n"
            "\n"
            f"- Submitted: **{self.submitted}**\n"
            f"- Filled:    **{self.filled}**\n"
            f"- Cancelled: {self.cancelled}\n"
            f"- Rejected:  {self.rejected}"
        )


@dataclass
class BreachSection:
    breaches: list[dict[str, Any]]
    halt_count: int

    def render(self) -> str:
        if not self.breaches:
            return "## Breach events\n\n_none today_"
        out = ["## Breach events", "", f"_{len(self.breaches)} total, {self.halt_count} halt-triggering_", ""]
        # Group by rule_id with a count.
        by_rule: dict[str, int] = {}
        for b in self.breaches:
            r = str(b.get("rule_id", "UNKNOWN"))
            by_rule[r] = by_rule.get(r, 0) + 1
        out.append("| Rule | Count |")
        out.append("| --- | ---: |")
        for rule, n in sorted(by_rule.items(), key=lambda x: -x[1]):
            out.append(f"| {rule} | {n} |")
        # First 5 raw examples.
        out.append("")
        out.append("First 5 events:")
        out.append("")
        for b in self.breaches[:5]:
            ts = b.get("ts_unix", 0)
            ts_str = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%H:%M:%S") if ts else "?"
            out.append(
                f"- `{ts_str}Z` **{b.get('rule_id', '?')}** "
                f"obs={b.get('observed', '?')} threshold={b.get('threshold', '?')}"
            )
        return "\n".join(out)


@dataclass
class VixSection:
    regime: str
    detail: str

    def render(self) -> str:
        return f"## VIX regime\n\n- Regime: **{self.regime}**\n- {self.detail}"


@dataclass
class Summary:
    day_iso: str
    account: AccountSection
    per_symbol: PerSymbolSection
    orders: OrdersSection
    breaches: BreachSection
    vix: VixSection
    status: str  # GREEN | YELLOW | RED
    status_reason: str

    def render(self) -> str:
        return "\n\n".join(
            [
                f"# Daily summary — {self.day_iso}",
                f"## Status: **{self.status}**",
                f"_{self.status_reason}_",
                self.account.render(),
                self.per_symbol.render(),
                self.orders.render(),
                self.breaches.render(),
                self.vix.render(),
            ]
        )


# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------


def collect_account(trading_client: Any) -> AccountSection:
    acct = trading_client.get_account()
    equity = Decimal(str(getattr(acct, "equity", "0")))
    last_eq = Decimal(str(getattr(acct, "last_equity", "0")))
    pnl = equity - last_eq
    pct = (pnl / last_eq) if last_eq != 0 else Decimal("0")
    return AccountSection(
        starting_equity=last_eq,
        ending_equity=equity,
        day_pnl=pnl,
        day_pnl_pct=pct,
        trading_blocked=bool(getattr(acct, "trading_blocked", False)),
        account_status=str(getattr(acct, "status", "")),
    )


def collect_positions(trading_client: Any) -> PerSymbolSection:
    rows: list[dict[str, Any]] = []
    for p in trading_client.get_all_positions():
        try:
            qty = float(getattr(p, "qty", 0) or 0)
            mark = float(getattr(p, "current_price", 0) or 0)
            unr = float(getattr(p, "unrealized_pl", 0) or 0)
            unr_pct = float(getattr(p, "unrealized_plpc", 0) or 0)
            rows.append(
                {
                    "symbol": str(getattr(p, "symbol", "?")),
                    "qty": qty,
                    "mark": mark,
                    "unrealized_pl": unr,
                    "unrealized_pct": unr_pct,
                }
            )
        except Exception:
            continue
    rows.sort(key=lambda r: -abs(r["unrealized_pl"]))
    return PerSymbolSection(rows=rows)


def collect_orders(trading_client: Any, day_iso: str) -> OrdersSection:
    """Count today's order activity by status.

    Uses the Alpaca ``get_orders`` API with no date filter then bucketizes
    locally — the API's date filter requires careful timezone handling
    and we already have a UTC window.
    """

    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        # ALL statuses for today.
        orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)
        )
    except Exception:
        return OrdersSection(submitted=0, filled=0, cancelled=0, rejected=0)

    start_u, end_u = _day_window_unix(day_iso)
    submitted = filled = cancelled = rejected = 0
    for o in orders:
        ts = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
        if ts is None:
            continue
        try:
            t = ts.timestamp()
        except Exception:
            continue
        if not (start_u <= t < end_u):
            continue
        submitted += 1
        status = str(getattr(getattr(o, "status", ""), "value", getattr(o, "status", "")))
        s = status.lower()
        if "filled" in s and "partial" not in s:
            filled += 1
        elif "cancel" in s:
            cancelled += 1
        elif "reject" in s:
            rejected += 1
    return OrdersSection(submitted=submitted, filled=filled, cancelled=cancelled, rejected=rejected)


def collect_breaches(day_iso: str, breach_paths: list[Path]) -> BreachSection:
    """Read breach JSONL files and keep only events from the target day."""

    start_u, end_u = _day_window_unix(day_iso)
    out: list[dict[str, Any]] = []
    halt_count = 0
    for p in breach_paths:
        if not p.exists():
            continue
        for line in p.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("ts_unix")
            if ts is None and "ts_ns" in rec:
                try:
                    ts = float(rec["ts_ns"]) / 1e9
                except Exception:
                    ts = None
            if ts is None:
                continue
            try:
                t = float(ts)
            except Exception:
                continue
            if not (start_u <= t < end_u):
                continue
            out.append(rec)
            if bool(rec.get("would_halt", False)):
                halt_count += 1
    return BreachSection(breaches=out, halt_count=halt_count)


def classify_vix(day_iso: str) -> VixSection:
    """Best-effort VIX regime read.

    The project's regime gate classifies via ``VIX_HIGH``/``VIX_MID``/
    ``VIX_LOW`` bands fed from a daily VIX-proxy series. The runner
    persists today's regime decision in ``data/30_paper_run_<date>.log``
    as a line ``vix_regime=<TIER>`` or similar. We grep that log; if
    absent we report ``UNKNOWN``.
    """

    log = PROJECT_ROOT / "data" / f"30_paper_run_{day_iso}.log"
    if not log.exists():
        return VixSection(regime="UNKNOWN", detail="no paper-run log for the day")

    regime = "UNKNOWN"
    detail = "not found in log"
    try:
        for ln in log.read_text("utf-8").splitlines():
            low = ln.lower()
            for marker in ("vix_regime=", "vix regime:", "vix_tier="):
                idx = low.find(marker)
                if idx >= 0:
                    val = ln[idx + len(marker):].strip().split()[0]
                    regime = val.upper().strip(",.;:")
                    detail = f"classified from `{log.name}`"
                    break
            if regime != "UNKNOWN":
                break
    except Exception as exc:
        detail = f"log read error: {exc}"
    return VixSection(regime=regime, detail=detail)


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------


def classify_status(
    account: AccountSection, breaches: BreachSection
) -> tuple[str, str]:
    """Top-line GREEN | YELLOW | RED traffic light."""

    reasons: list[str] = []
    if account.trading_blocked:
        reasons.append("trading_blocked=True")
    if account.account_status and account.account_status.upper() not in ("ACTIVE", ""):
        reasons.append(f"account_status={account.account_status}")
    if breaches.halt_count > 0:
        reasons.append(f"{breaches.halt_count} halt-triggering breaches")
    if account.day_pnl_pct < Decimal("-0.01"):
        reasons.append(f"day P&L {account.day_pnl_pct:+.2%}")

    if reasons:
        return "RED", "; ".join(reasons)

    if breaches.breaches or account.day_pnl_pct < Decimal("0"):
        bits = []
        if breaches.breaches:
            bits.append(f"{len(breaches.breaches)} non-halt breaches")
        if account.day_pnl_pct < 0:
            bits.append(f"day P&L {account.day_pnl_pct:+.2%}")
        return "YELLOW", "; ".join(bits)

    return "GREEN", f"clean session, day P&L {account.day_pnl_pct:+.2%}"


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_summary(
    *,
    day_iso: str,
    trading_client: Any,
    breach_paths: list[Path] | None = None,
) -> Summary:
    if breach_paths is None:
        breach_paths = [
            Path.home() / "macminibot" / "risk" / "breaches.jsonl",
            Path.home() / "macminibot" / "risk" / "paper" / "breaches.jsonl",
            PROJECT_ROOT / "risk" / "breaches.jsonl",
            PROJECT_ROOT / "risk" / "paper" / "breaches.jsonl",
        ]

    account = collect_account(trading_client)
    per_symbol = collect_positions(trading_client)
    orders = collect_orders(trading_client, day_iso)
    breaches = collect_breaches(day_iso, breach_paths)
    vix = classify_vix(day_iso)
    status, reason = classify_status(account, breaches)

    return Summary(
        day_iso=day_iso,
        account=account,
        per_symbol=per_symbol,
        orders=orders,
        breaches=breaches,
        vix=vix,
        status=status,
        status_reason=reason,
    )


def write_summary(summary: Summary, *, reports_dir: Path | None = None) -> Path:
    if reports_dir is None:
        reports_dir = PROJECT_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{summary.day_iso}.md"
    tmp = out.with_suffix(".md.tmp")
    tmp.write_text(summary.render() + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _trading_client_from_env() -> Any:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    from execution.alpaca_client import load_alpaca_config
    from alpaca.trading.client import TradingClient

    cfg = load_alpaca_config()
    return TradingClient(cfg.api_key, cfg.api_secret, paper=cfg.paper)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD (UTC). Defaults to today.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the rendered markdown to stdout.",
    )
    args = parser.parse_args(argv)

    day_iso = args.date or _today_utc_date()
    client = _trading_client_from_env()

    # Emit an INFO alert that EOD ran — operator sees "we generated EOD"
    # in the alert log even on a flat day.
    try:
        from observability.alert_sink import AlertEvent, AlertSeverity, load_sinks_from_config

        sink = load_sinks_from_config()
    except Exception:
        sink = None

    summary = build_summary(day_iso=day_iso, trading_client=client)
    out_path = write_summary(summary)
    print(f"[eod] wrote {out_path} (status={summary.status})")

    if args.stdout:
        print()
        print(summary.render())

    if sink is not None:
        try:
            severity = {
                "GREEN": AlertSeverity.INFO,
                "YELLOW": AlertSeverity.WARN,
                "RED": AlertSeverity.CRITICAL,
            }.get(summary.status, AlertSeverity.INFO)
            sink.emit(
                AlertEvent(
                    title=f"EOD {summary.status} {day_iso}",
                    body=f"{summary.status_reason} | day P&L {summary.account.day_pnl_pct:+.2%} | report={out_path}",
                    severity=severity,
                    rule_id="EOD_SUMMARY",
                )
            )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
