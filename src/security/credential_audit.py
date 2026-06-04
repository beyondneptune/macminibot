"""security.credential_audit — verify the bot's .env credentials hygiene.

What we check:

1. ``.env`` exists at the project root.
2. Every required key (`ALPACA_API_KEY`, `ALPACA_SECRET`,
   `ALPACA_BASE_URL`, `ACCOUNT_EQUITY_INITIAL`) is present and has a
   non-empty value.
3. ``.env`` file permissions are ``0600`` (owner-only read/write).
4. ``.env`` is listed in ``.gitignore`` (or a parent ``.gitignore``).

The :func:`mask` helper redacts secrets that may otherwise leak into log
files. It keeps the first 2 and last 2 characters of the input and
replaces the middle with ``****`` so we can still verify a key by sight
without exposing it.

Public API
----------
``audit()`` returns a ``list[Finding]``. The function is side-effect free
— it never writes, never mutates env vars, and never reaches the network.
This is mandatory: the live bot is running and must not be perturbed.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Iterable

from security.findings import Finding, Severity

__all__ = [
    "REQUIRED_ENV_KEYS",
    "audit",
    "audit_credentials",
    "mask",
]

REQUIRED_ENV_KEYS: tuple[str, ...] = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET",
    "ALPACA_BASE_URL",
    "ACCOUNT_EQUITY_INITIAL",
)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def mask(value: str | None, *, keep: int = 2, placeholder: str = "****") -> str:
    """Redact a secret-like string for log output.

    Strategy: keep the first ``keep`` and last ``keep`` characters intact
    and replace the middle with ``placeholder``. Very short strings
    (<= 2*keep+1 chars) are fully masked so we never accidentally leak
    the entire value.

    >>> mask("PKU3TRVIFSSMCXN64WBOAADTHF")
    'PK****HF'
    >>> mask("short")
    '****'
    """

    if value is None:
        return ""
    s = str(value)
    if len(s) <= (2 * keep) + 1:
        return placeholder
    return f"{s[:keep]}{placeholder}{s[-keep:]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    # src/security/credential_audit.py → repo root is parents[2].
    return Path(__file__).resolve().parents[2]


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal dotenv parser — KEY=VALUE lines, ignore comments / blanks.

    We do NOT pull in python-dotenv; the live bot already has its own
    loader path and the audit must remain dependency-free so it can run
    even if the venv is degraded.
    """

    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


def _check_env_file_exists(env_path: Path) -> list[Finding]:
    if not env_path.is_file():
        return [
            Finding(
                code="ENV_MISSING",
                severity=Severity.CRITICAL,
                message=f".env file not found at {env_path}",
            )
        ]
    return []


def _check_required_keys(env: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    for key in REQUIRED_ENV_KEYS:
        if key not in env:
            findings.append(
                Finding(
                    code="ENV_KEY_MISSING",
                    severity=Severity.CRITICAL,
                    message=f"required key missing from .env: {key}",
                    extra={"key": key},
                )
            )
            continue
        if not env[key]:
            findings.append(
                Finding(
                    code="ENV_KEY_EMPTY",
                    severity=Severity.CRITICAL,
                    message=f"required key has empty value in .env: {key}",
                    extra={"key": key},
                )
            )
    return findings


def _check_file_permissions(env_path: Path) -> list[Finding]:
    """``.env`` must be owner-only (chmod 600). Group/other reads = leak."""

    try:
        mode = stat.S_IMODE(env_path.stat().st_mode)
    except FileNotFoundError:
        return []  # covered by _check_env_file_exists
    if mode != 0o600:
        return [
            Finding(
                code="ENV_PERMISSIONS",
                severity=Severity.CRITICAL,
                message=(
                    f".env permissions are {oct(mode)} but must be 0o600 — "
                    f"run: chmod 600 {env_path}"
                ),
                extra={"actual_mode": oct(mode), "required_mode": "0o600"},
            )
        ]
    return []


def _find_gitignore_files(root: Path) -> Iterable[Path]:
    """Yield the project ``.gitignore`` plus any in parents (up to /)."""

    project_ignore = root / ".gitignore"
    if project_ignore.is_file():
        yield project_ignore
    # Walk up — covers monorepo / nested checkouts.
    parent = root.parent
    seen: set[Path] = {root}
    while parent not in seen and parent != parent.parent:
        candidate = parent / ".gitignore"
        if candidate.is_file():
            yield candidate
        seen.add(parent)
        parent = parent.parent


def _check_gitignore(root: Path) -> list[Finding]:
    """Make sure ``.env`` is excluded from git."""

    for path in _find_gitignore_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Accept the obvious patterns: ".env", "/.env", "*.env".
            if line in (".env", "/.env", "./.env") or line == "*.env":
                return []
    return [
        Finding(
            code="ENV_NOT_GITIGNORED",
            severity=Severity.CRITICAL,
            message=".env is not listed in any .gitignore — risk of commit leak",
        )
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit(project_root: Path | None = None) -> list[Finding]:
    """Run every credential-hygiene check and return findings.

    Pure function: never writes, never mutates env vars, never networks.
    Suitable for invocation from a launchd-driven daily job.
    """

    root = Path(project_root) if project_root else _project_root()
    env_path = root / ".env"

    findings: list[Finding] = []
    findings.extend(_check_env_file_exists(env_path))

    # Only parse / inspect contents + perms if the file exists.
    if env_path.is_file():
        env = _parse_env_file(env_path)
        findings.extend(_check_required_keys(env))
        findings.extend(_check_file_permissions(env_path))

    findings.extend(_check_gitignore(root))
    return findings


# Convenience alias to match the spec doc.
audit_credentials = audit
