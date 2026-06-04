"""45_security_check — daily credential + dependency audit.

Runs every check in the :mod:`security` tentacle and prints a human-
readable report. Exit status:

    0  — no CRITICAL findings (INFO/WARN allowed).
    1  — at least one CRITICAL finding (broken creds, .env leak, downgrade).

Intended for launchd. The schedule (06:00 BST = 05:00 UTC = 01:00 ET)
fires BEFORE the trading day so a credential rot is caught in time to
fix it. The check is read-only and side-effect free — it cannot perturb
the live bot.

Run manually:
    .venv/bin/python scripts/45_security_check.py
    .venv/bin/python scripts/45_security_check.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from security import audit as audit_credentials  # noqa: E402
from security import scan_dependencies  # noqa: E402
from security.findings import Finding, Severity  # noqa: E402


def _render_text(findings: list[Finding]) -> str:
    if not findings:
        return "[OK] security check: no findings."

    by_sev: dict[Severity, list[Finding]] = {s: [] for s in Severity}
    for f in findings:
        by_sev[f.severity].append(f)

    lines: list[str] = ["macminibot security check"]
    for sev in (Severity.CRITICAL, Severity.WARN, Severity.INFO):
        items = by_sev[sev]
        if not items:
            continue
        lines.append("")
        lines.append(f"## {sev.value} ({len(items)})")
        for f in items:
            lines.append(f"  [{f.code}] {f.message}")
    return "\n".join(lines)


def _render_json(findings: list[Finding]) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="macminibot daily security audit")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    findings.extend(audit_credentials(PROJECT_ROOT))

    # Dep scan may fail if pip is broken — never let that block the
    # credential audit. Surface the failure as a WARN.
    try:
        findings.extend(scan_dependencies())
    except Exception as exc:  # pragma: no cover - defensive
        findings.append(
            Finding(
                code="DEP_SCAN_ERROR",
                severity=Severity.WARN,
                message=f"dep_scanner failed: {exc!r}",
            )
        )

    if args.json:
        print(_render_json(findings))
    else:
        print(_render_text(findings))

    return 1 if any(f.is_high() for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
