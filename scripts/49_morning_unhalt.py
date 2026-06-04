"""49_morning_unhalt — pre-market auto-unhalt for routine overnight halts.

Fires Mon-Fri 14:00 BST (35 min before the daily per-asset-runner). For
the halt to clear, ALL of these must hold:

  1. ``state/HALT`` exists (otherwise no-op, clean exit).
  2. The MOST RECENT ``HALT`` entry in ``state/halt_audit.log`` has a
     reason matching the whitelist (subset of:
     ``BROKER_ACCOUNT_INACTIVE``, ``POSITION_RECONCILE``). Any other
     halt reason — DD, manual operator halt, KILL_SWITCH outside the
     whitelist, fat-finger, etc. — is REFUSED and the operator must
     intervene.
  3. Alpaca account is reachable AND ``account.status == ACTIVE``.
  4. Broker positions == journal positions (POSITION_RECONCILE clean).

If any condition fails the script exits non-zero, leaves HALT in place,
and writes a structured "REFUSED" line to the audit log so the operator
sees it.

Whitelist rationale: the BROKER_ACCOUNT_INACTIVE auto-halt has fired
every overnight for 8+ days (Alpaca paper account briefly flips
INACTIVE during overnight maintenance). POSITION_RECONCILE often
co-fires because the reconcile check runs on the same poll. Both are
routine, harmless, and require manual unhalt in the morning — that's
what this script automates.

Run manually:
    PYTHONPATH=src .venv/bin/python scripts/49_morning_unhalt.py
    PYTHONPATH=src .venv/bin/python scripts/49_morning_unhalt.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from risk_management.auto_halt_daemon import (  # noqa: E402
    BrokerProbe, read_journal_positions, reconcile_positions,
)
from risk_management.kill_switch import (  # noqa: E402
    audit_log_path, halt_path, is_halted, unhalt,
)


# Whitelist of halt reason FRAGMENTS that this script is permitted to
# clear. The most recent HALT line in halt_audit.log must contain ONLY
# fragments from this set (i.e., its rule list is a subset).
WHITELIST_RULES = {"BROKER_ACCOUNT_INACTIVE", "POSITION_RECONCILE"}


def _logger() -> logging.Logger:
    log = logging.getLogger("macminibot.morning_unhalt")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [morning_unhalt] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def _latest_halt_reason(audit_path: Path) -> tuple[str | None, dict | None]:
    """Return the reason of the most recent ``action=HALT`` line, or None."""
    if not audit_path.exists():
        return None, None
    try:
        lines = audit_path.read_text("utf-8").strip().split("\n")
    except Exception:
        return None, None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue
        if evt.get("action") == "HALT":
            return str(evt.get("reason", "")), evt
    return None, None


def _parse_auto_halt_rules(reason: str) -> set[str]:
    """Extract rule names from an auto_halt reason string.

    Format from auto_halt_daemon: ``"auto_halt:RULE1,RULE2"``. Anything
    that doesn't start with ``auto_halt:`` returns an empty set so the
    whitelist subset check fails (= we refuse to clear).
    """
    if not reason.startswith("auto_halt:"):
        return set()
    payload = reason[len("auto_halt:"):]
    parts = {p.strip() for p in payload.split(",") if p.strip()}
    return parts


def _append_audit(action: str, reason: str) -> None:
    line = json.dumps({"action": action, "reason": reason,
                        "ts_unix": time.time()}, sort_keys=True)
    with audit_log_path().open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def evaluate(*, dry_run: bool) -> dict[str, Any]:
    """One pass of the auto-unhalt logic. Returns structured result."""
    log = _logger()
    load_dotenv(PROJECT_ROOT / ".env")

    if not is_halted():
        log.info("HALT file absent — nothing to do")
        return {"action": "no_op", "reason": "not_halted"}

    halt_reason, evt = _latest_halt_reason(audit_log_path())
    if halt_reason is None:
        log.warning("HALT file present but no HALT entry in audit log — REFUSING")
        _append_audit("UNHALT_REFUSED", "no_audit_entry_found")
        return {"action": "refused", "reason": "no_audit_entry"}

    log.info("latest halt reason: %s", halt_reason)
    rules_in_reason = _parse_auto_halt_rules(halt_reason)
    if not rules_in_reason:
        log.warning("halt reason is non-auto_halt (manual or unknown) — REFUSING")
        _append_audit("UNHALT_REFUSED",
                       f"non_auto_halt_reason: {halt_reason}")
        return {"action": "refused", "reason": "non_auto_halt",
                "raw_reason": halt_reason}

    if not rules_in_reason.issubset(WHITELIST_RULES):
        non_whitelisted = rules_in_reason - WHITELIST_RULES
        log.warning("halt reason contains non-whitelisted rules %s — REFUSING",
                     sorted(non_whitelisted))
        _append_audit("UNHALT_REFUSED",
                       f"non_whitelisted_rules: {sorted(non_whitelisted)} "
                       f"in {halt_reason}")
        return {"action": "refused", "reason": "non_whitelisted_rules",
                "non_whitelisted": sorted(non_whitelisted)}

    # Probe broker
    try:
        probe = BrokerProbe.from_env()
        state = probe.fetch_state()
    except Exception as exc:
        log.error("broker probe failed: %s — REFUSING", exc)
        _append_audit("UNHALT_REFUSED", f"broker_probe_failed: {exc}")
        return {"action": "refused", "reason": "broker_probe_failed",
                "error": str(exc)}

    log.info("broker status=%s positions=%d", state.account_status,
              len(state.positions))

    # alpaca-py returns AccountStatus enum stringified as 'AccountStatus.ACTIVE'.
    # Normalise: take last dot-segment and compare uppercase.
    status_norm = state.account_status.split(".")[-1].upper()
    if status_norm != "ACTIVE":
        log.warning("broker still inactive (status=%s norm=%s) — REFUSING",
                     state.account_status, status_norm)
        _append_audit("UNHALT_REFUSED",
                       f"broker_still_inactive: status={state.account_status}")
        return {"action": "refused", "reason": "broker_still_inactive",
                "status": state.account_status}

    # Reconcile positions. Production runners write fills to ``data/paper/dt=*/``,
    # so the journal_root for read_journal_positions (which appends "/paper")
    # must be ``data/``, NOT ``data/journal/``. The auto_halt_daemon's stored
    # path is misaligned — that's why POSITION_RECONCILE fires every overnight
    # even when nothing has actually diverged.
    journal_pos = read_journal_positions(PROJECT_ROOT / "data")
    diffs = reconcile_positions(state.positions, journal_pos)
    if diffs:
        diff_summary = [
            {"symbol": d.symbol, "broker_qty": str(d.broker_qty),
              "journal_qty": str(d.journal_qty)}
            for d in diffs
        ]
        log.warning("position reconcile FAILED — %d mismatch(es): %s",
                     len(diffs), diff_summary)
        _append_audit("UNHALT_REFUSED",
                       f"position_reconcile_failed: {len(diffs)} mismatches")
        return {"action": "refused", "reason": "position_reconcile_failed",
                "diffs": diff_summary}

    # All checks pass → unhalt
    if dry_run:
        log.info("DRY-RUN: would clear halt (all conditions pass)")
        return {"action": "would_clear", "reason": "dry_run"}

    summary = (f"auto-cleared at pre-market — original reason was "
               f"{halt_reason}; broker=ACTIVE; "
               f"positions reconciled ({len(state.positions)} symbols)")
    result = unhalt(summary)
    log.info("HALT CLEARED — %s", summary)
    return {"action": "cleared", "reason": summary, "unhalt_result": str(result)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate all gates but do not actually unhalt")
    ap.add_argument("--json", action="store_true",
                    help="print result as JSON")
    args = ap.parse_args()

    result = evaluate(dry_run=args.dry_run)
    if args.json:
        print(json.dumps(result, default=str, sort_keys=True))
    # Exit 0 on successful clear/no_op, 1 on refusal
    return 0 if result["action"] in {"cleared", "no_op", "would_clear"} else 1


if __name__ == "__main__":
    sys.exit(main())
