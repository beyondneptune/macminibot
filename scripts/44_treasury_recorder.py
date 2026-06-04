"""44_treasury_recorder — periodic Alpaca balance snapshot + reconciliation.

Fired every 15 minutes by ``com.macminibot.treasury-recorder.plist``
during paper-trading hours. One pass:

  1. Pull ``trading.get_account()`` and ``trading.get_all_positions()``.
  2. Append the account row to ``data/treasury/balances.parquet``.
  3. Read internal positions from the execution journal
     (``data/journal/paper/positions.json`` if present; empty otherwise)
     and reconcile against Alpaca's reported positions.
  4. Any breaks → ``data/treasury/breaks.jsonl`` (one JSON record per
     break, one line each).
  5. Also writes a tiny ``data/treasury/last_snapshot.json`` for the
     dashboard ("treasury last ran Ns ago, N breaks").

Skips writes outside market hours (uses ``trading.get_clock().is_open``)
so weekend / overnight runs are a no-op — but still emits the heartbeat
file so the dashboard can show "skipped: market closed".

Failure isolation: any exception is caught, logged to stderr, and the
script returns 0 so launchd doesn't ratchet up backoff. We rely on the
dashboard's staleness check (snapshot age > 30 min during RTH) to flag
real outages.
"""

from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

# Make ``src/`` importable when invoked by launchd (no editable install
# guarantee in production). This mirrors the trick in conftest.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from treasury.balance_ledger import BalanceLedger, reconcile  # noqa: E402


DEFAULT_BALANCES = _REPO_ROOT / "data" / "treasury" / "balances.parquet"
DEFAULT_BREAKS = _REPO_ROOT / "data" / "treasury" / "breaks.jsonl"
DEFAULT_HEARTBEAT = _REPO_ROOT / "data" / "treasury" / "last_snapshot.json"
INTERNAL_POSITIONS = _REPO_ROOT / "data" / "journal" / "paper" / "positions.json"


def _load_internal_positions() -> dict[str, Decimal]:
    """Read internal positions from the execution journal, or return {}.

    Schema expected: a JSON object ``{ticker: qty_as_str_or_int}``. The
    journal module may not have written it yet on first run; we treat
    the missing-file case as "no internal positions" (which will then
    reconcile cleanly against an empty Alpaca account, or surface every
    Alpaca position as a break if the venue holds shares — both are
    correct outcomes).
    """
    if not INTERNAL_POSITIONS.exists():
        return {}
    try:
        raw = json.loads(INTERNAL_POSITIONS.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.stderr.write(
            f"treasury_recorder: cannot parse {INTERNAL_POSITIONS}: {exc}\n"
        )
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Decimal] = {}
    for ticker, qty in raw.items():
        if isinstance(qty, bool):
            continue
        try:
            out[str(ticker)] = Decimal(str(qty))
        except Exception:  # pragma: no cover - defensive
            continue
    return out


def _write_heartbeat(
    path: Path,
    *,
    ts_ns: int,
    breaks_count: int,
    skipped_reason: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ts_ns": ts_ns,
        "breaks_count": breaks_count,
        "skipped": skipped_reason,
        "source": "44_treasury_recorder",
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def run_once(
    *,
    balances_path: Path = DEFAULT_BALANCES,
    breaks_path: Path = DEFAULT_BREAKS,
    heartbeat_path: Path = DEFAULT_HEARTBEAT,
    skip_if_market_closed: bool = True,
) -> int:
    """One snapshot pass. Returns 0 on success or no-op, 1 on hard failure."""

    from shared_core.types import now_ns

    ts_ns = now_ns()

    try:
        from execution.alpaca_client import AlpacaClient
    except Exception as exc:
        sys.stderr.write(f"treasury_recorder: cannot import AlpacaClient: {exc}\n")
        return 1

    try:
        client = AlpacaClient.from_env()
    except Exception as exc:
        sys.stderr.write(f"treasury_recorder: AlpacaClient init failed: {exc}\n")
        _write_heartbeat(
            heartbeat_path, ts_ns=ts_ns, breaks_count=0, skipped_reason="init_failed"
        )
        return 1

    # Market-hours gating. We still write the heartbeat so the dashboard
    # can show "skipped: market closed" rather than "stale".
    if skip_if_market_closed:
        try:
            clock = client.client.get_clock()
            if not bool(getattr(clock, "is_open", True)):
                _write_heartbeat(
                    heartbeat_path,
                    ts_ns=ts_ns,
                    breaks_count=0,
                    skipped_reason="market_closed",
                )
                sys.stdout.write("treasury_recorder: skip (market closed)\n")
                return 0
        except Exception as exc:  # pragma: no cover - defensive
            sys.stderr.write(
                f"treasury_recorder: get_clock failed, proceeding anyway: {exc}\n"
            )

    # 1. Snapshot the account.
    try:
        account = client.client.get_account()
    except Exception as exc:
        sys.stderr.write(f"treasury_recorder: get_account failed: {exc}\n")
        _write_heartbeat(
            heartbeat_path, ts_ns=ts_ns, breaks_count=0, skipped_reason="api_error"
        )
        return 1

    ledger = BalanceLedger(parquet_path=balances_path)
    row = ledger.record_snapshot(ts_ns, account)
    sys.stdout.write(
        f"treasury_recorder: snapshot ts_ns={ts_ns} equity={row.get('equity')} "
        f"cash={row.get('cash')}\n"
    )

    # 2. Reconcile against internal journal positions.
    try:
        venue_positions = client.get_positions()
    except Exception as exc:
        sys.stderr.write(f"treasury_recorder: get_positions failed: {exc}\n")
        _write_heartbeat(
            heartbeat_path,
            ts_ns=ts_ns,
            breaks_count=0,
            skipped_reason="positions_api_error",
        )
        return 1

    internal = _load_internal_positions()
    breaks = reconcile(internal, venue_positions)
    if breaks:
        ledger.write_breaks(ts_ns, breaks, breaks_path=breaks_path)
        sys.stdout.write(
            f"treasury_recorder: {len(breaks)} breaks written to {breaks_path}\n"
        )
        for br in breaks:
            sys.stdout.write(
                f"  BREAK {br.ticker}: internal={br.internal_qty} "
                f"venue={br.venue_qty} diff={br.diff}\n"
            )
    else:
        sys.stdout.write("treasury_recorder: 0 breaks (reconciled cleanly)\n")

    _write_heartbeat(
        heartbeat_path,
        ts_ns=ts_ns,
        breaks_count=len(breaks),
        skipped_reason=None,
    )
    return 0


def main() -> int:
    try:
        return run_once()
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"treasury_recorder: unhandled exception: {exc!r}\n")
        # Return 0 so launchd doesn't backoff; we rely on staleness detection.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
