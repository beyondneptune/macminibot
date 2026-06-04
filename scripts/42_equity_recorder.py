"""42_equity_recorder — append one NetLiq sample to the equity curve.

Fired every 5 minutes during the US regular session by launchd
(``com.macminibot.equity-recorder.plist``). One IBKR account-summary
fetch, one append, one exit.

Trading-window gate
-------------------
The recorder only writes when it's a US trading hour (09:30-16:00 ET,
Mon-Fri, ignoring holidays). Outside that window we exit 0 quietly so
the launchd interval doesn't fill the curve with weekend zeros.

IBKR availability
-----------------
This is the LIVE bot — TWS is already running for the paper-trade
runner. We connect with a unique ``CLIENT_ID`` so we never collide
with the main runner's session. If TWS isn't reachable we log a
warning and exit 0 (this is a recorder, not a hard dependency for
trading).

Override env vars (for local replay/tests):
    MACMINIBOT_EQUITY_CURVE_PATH     parquet sink path
    MACMINIBOT_EQUITY_RECORDER_HOST  ib host (default 127.0.0.1)
    MACMINIBOT_EQUITY_RECORDER_PORT  ib port (default 7497)
    MACMINIBOT_EQUITY_RECORDER_CID   client id (default 42)
    MACMINIBOT_EQUITY_FAKE_NETLIQ    if set, skip IBKR and use this value
                                     (used by the launchd smoke-test and
                                      offline development)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Ensure src/ is importable when launchd runs us directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observability.equity_curve import record  # noqa: E402


NY = ZoneInfo("America/New_York")
SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)


def _in_us_session(now_utc: datetime | None = None) -> bool:
    """True iff we're inside 09:30-16:00 ET on a weekday (no holiday calendar)."""

    now = now_utc if now_utc is not None else datetime.now(tz=timezone.utc)
    et = now.astimezone(NY)
    if et.weekday() >= 5:  # Sat=5, Sun=6
        return False
    return SESSION_OPEN <= et.timetz().replace(tzinfo=None) <= SESSION_CLOSE


def _fetch_account_summary_via_ib() -> dict[str, Any]:
    """Fetch NetLiq + friends from IBKR. Lazy import so test paths skip it."""

    from ib_async import IB  # type: ignore[import-not-found]

    host = os.environ.get("MACMINIBOT_EQUITY_RECORDER_HOST", "127.0.0.1")
    port = int(os.environ.get("MACMINIBOT_EQUITY_RECORDER_PORT", "7497"))
    cid = int(os.environ.get("MACMINIBOT_EQUITY_RECORDER_CID", "42"))

    ib = IB()
    ib.connect(host, port, clientId=cid, timeout=10)
    try:
        rows = ib.accountSummary()
    finally:
        ib.disconnect()

    wanted = {"NetLiquidation", "AvailableFunds", "GrossPositionValue", "Cushion"}
    out: dict[str, Any] = {}
    for r in rows:
        tag = getattr(r, "tag", None)
        val = getattr(r, "value", None)
        if tag in wanted and val is not None:
            out[tag] = val
    return out


def _account_summary() -> dict[str, Any]:
    """Pluggable account-summary source: env override for tests, else IBKR."""

    fake = os.environ.get("MACMINIBOT_EQUITY_FAKE_NETLIQ")
    if fake is not None:
        return {
            "NetLiquidation": fake,
            "AvailableFunds": fake,
            "GrossPositionValue": "0",
            "Cushion": "1",
        }
    return _fetch_account_summary_via_ib()


def main() -> int:
    if not _in_us_session():
        sys.stdout.write(
            f"[equity_recorder] outside US session ({datetime.now(tz=NY):%Y-%m-%d %H:%M %Z}); skip\n"
        )
        return 0

    try:
        summary = _account_summary()
    except Exception as exc:  # pragma: no cover - network path
        sys.stderr.write(f"[equity_recorder] IBKR fetch failed: {exc}\n")
        return 0  # soft fail — never break launchd schedule

    if "NetLiquidation" not in summary:
        sys.stderr.write(
            f"[equity_recorder] account summary missing NetLiquidation: {summary!r}\n"
        )
        return 0

    df = record(summary, time.time_ns())
    sys.stdout.write(
        f"[equity_recorder] wrote row "
        f"net_liq={summary['NetLiquidation']} rows={len(df)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
