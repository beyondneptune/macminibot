"""security.dep_scanner — diff installed packages vs a known-good frozen set.

We pin the bot's transitive dependency surface in
``data/security/known_deps.json`` (a list of ``{"name": ..., "version": ...}``
records — the same shape ``pip list --format=json`` emits). Every audit
run shells out to ``pip list --format=json`` in the project venv and
reports:

- ``NEW_PACKAGE``    : a package present now that wasn't pinned.
- ``VERSION_CHANGE`` : version differs from the pinned one
                       (downgrades are CRITICAL, upgrades are WARN).
- ``MISSING_PACKAGE``: pinned package no longer installed (WARN).

The scanner never modifies state. To accept a new baseline, the human
calls :func:`write_known_deps` explicitly (e.g. after a deliberate
``pip install`` and a green test run). It is intentionally NOT triggered
by the daily security check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from security.findings import Finding, Severity

__all__ = [
    "DepDiff",
    "load_known_deps",
    "scan_dependencies",
    "write_known_deps",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_known_path() -> Path:
    return _project_root() / "data" / "security" / "known_deps.json"


def _default_python() -> str:
    """Path to the venv interpreter that runs the bot."""
    venv_py = _project_root() / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _normalise_name(name: str) -> str:
    # PEP-503 normalisation: lowercase + dashes vs underscores collapsed.
    return name.strip().lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------


def _parse_version(v: str) -> tuple[Any, ...]:
    """Return a coarsely comparable tuple — falls back to string compare.

    We avoid pulling in ``packaging`` to keep this module dependency-free.
    For semver-ish versions this works; for exotic schemes the comparison
    degrades to lexicographic, which is fine for our "did it change?"
    use-case (we only need ordering for downgrade detection).
    """

    parts: list[Any] = []
    for chunk in v.replace("-", ".").replace("+", ".").split("."):
        if chunk.isdigit():
            parts.append((0, int(chunk)))
        else:
            # Non-numeric (e.g. "rc1", "post3") sorts AFTER plain ints.
            parts.append((1, chunk))
    return tuple(parts)


def _is_downgrade(old: str, new: str) -> bool:
    try:
        return _parse_version(new) < _parse_version(old)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pip integration
# ---------------------------------------------------------------------------


def _run_pip_list(python_exe: str | None = None, timeout: float = 30.0) -> list[dict[str, str]]:
    """Shell out to ``<python> -m pip list --format=json``.

    Returns a list of ``{"name": str, "version": str}`` records.
    """

    python_exe = python_exe or _default_python()
    proc = subprocess.run(
        [python_exe, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    data = json.loads(proc.stdout)
    if not isinstance(data, list):
        raise ValueError("pip list returned non-list payload")
    return [
        {"name": str(d["name"]), "version": str(d["version"])}
        for d in data
        if isinstance(d, dict) and "name" in d and "version" in d
    ]


# ---------------------------------------------------------------------------
# Known-good baseline I/O
# ---------------------------------------------------------------------------


def load_known_deps(path: Path | None = None) -> dict[str, str]:
    """Read the pinned baseline into a ``{name: version}`` dict.

    Missing file returns an empty dict — the first invocation of the
    scanner is expected to be a no-op + bootstrap via ``write_known_deps``.
    """

    p = Path(path) if path else _default_known_path()
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{p} must contain a JSON array")
    return {_normalise_name(d["name"]): str(d["version"]) for d in raw if "name" in d}


def write_known_deps(
    packages: Iterable[dict[str, str]] | None = None,
    path: Path | None = None,
    *,
    python_exe: str | None = None,
) -> Path:
    """Persist the current ``pip list`` (or the supplied snapshot) as baseline.

    Caller-supplied snapshots are used by tests; production seeding pulls
    from the live venv. This is the ONLY function in this module that
    writes to disk.
    """

    p = Path(path) if path else _default_known_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    snapshot = (
        list(packages)
        if packages is not None
        else _run_pip_list(python_exe=python_exe)
    )
    # Sort for deterministic diffs.
    snapshot_sorted = sorted(snapshot, key=lambda d: _normalise_name(d["name"]))
    p.write_text(json.dumps(snapshot_sorted, indent=2, sort_keys=True), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DepDiff:
    """Structured diff for human-readable reports."""

    new_packages: list[tuple[str, str]] = field(default_factory=list)
    removed_packages: list[tuple[str, str]] = field(default_factory=list)
    upgraded: list[tuple[str, str, str]] = field(default_factory=list)
    downgraded: list[tuple[str, str, str]] = field(default_factory=list)


def _compute_diff(
    current: list[dict[str, str]], known: dict[str, str]
) -> DepDiff:
    current_by_name = {_normalise_name(d["name"]): str(d["version"]) for d in current}

    new_packages: list[tuple[str, str]] = []
    upgraded: list[tuple[str, str, str]] = []
    downgraded: list[tuple[str, str, str]] = []
    removed_packages: list[tuple[str, str]] = []

    for name, ver in current_by_name.items():
        if name not in known:
            new_packages.append((name, ver))
        elif known[name] != ver:
            if _is_downgrade(known[name], ver):
                downgraded.append((name, known[name], ver))
            else:
                upgraded.append((name, known[name], ver))

    for name, ver in known.items():
        if name not in current_by_name:
            removed_packages.append((name, ver))

    return DepDiff(
        new_packages=sorted(new_packages),
        removed_packages=sorted(removed_packages),
        upgraded=sorted(upgraded),
        downgraded=sorted(downgraded),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_dependencies(
    *,
    known_path: Path | None = None,
    current_packages: list[dict[str, str]] | None = None,
    python_exe: str | None = None,
) -> list[Finding]:
    """Compare current ``pip list`` against the pinned baseline.

    ``current_packages`` is injectable for tests; in production the
    function shells out to pip itself.
    """

    known = load_known_deps(known_path)

    if not known:
        # First-ever run: tell the operator to seed the baseline.
        return [
            Finding(
                code="DEP_BASELINE_MISSING",
                severity=Severity.WARN,
                message=(
                    "no known_deps.json baseline — run "
                    "`python -c 'from security.dep_scanner import write_known_deps; "
                    "write_known_deps()'` to seed it"
                ),
                extra={"path": str(known_path or _default_known_path())},
            )
        ]

    current = (
        current_packages
        if current_packages is not None
        else _run_pip_list(python_exe=python_exe)
    )
    diff = _compute_diff(current, known)

    findings: list[Finding] = []
    for name, ver in diff.new_packages:
        findings.append(
            Finding(
                code="DEP_NEW_PACKAGE",
                severity=Severity.WARN,
                message=f"new package not in baseline: {name}=={ver}",
                extra={"name": name, "version": ver},
            )
        )
    for name, old, new in diff.downgraded:
        findings.append(
            Finding(
                code="DEP_DOWNGRADE",
                severity=Severity.CRITICAL,
                message=f"package downgraded: {name} {old} -> {new}",
                extra={"name": name, "old": old, "new": new},
            )
        )
    for name, old, new in diff.upgraded:
        findings.append(
            Finding(
                code="DEP_UPGRADE",
                severity=Severity.WARN,
                message=f"package version changed: {name} {old} -> {new}",
                extra={"name": name, "old": old, "new": new},
            )
        )
    for name, ver in diff.removed_packages:
        findings.append(
            Finding(
                code="DEP_MISSING_PACKAGE",
                severity=Severity.WARN,
                message=f"baseline package no longer installed: {name}=={ver}",
                extra={"name": name, "version": ver},
            )
        )
    return findings
