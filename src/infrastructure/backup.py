"""infrastructure.backup — daily rsync of ArcticDB, journal, and state.

Three entry points, one per data domain:

* :func:`backup_arctic`  — ``data/arctic`` → ``backup/arctic-YYYY-MM-DD``
* :func:`backup_journal` — ``data/journal`` → ``backup/journal-YYYY-MM-DD``
* :func:`backup_state`   — ``state``        → ``backup/state-YYYY-MM-DD``

Each call shells out to ``rsync -a`` (archive mode, preserves perms /
mtimes / symlinks) into a freshly dated directory. ``rsync`` is used over
``shutil.copytree`` because it handles partial-update incrementality
cheaply for an Arctic store that may already weigh tens of gigabytes.

After every successful backup we call :func:`prune_backups` which keeps:

    - last 7 daily backups
    - last 4 weekly backups (Sundays)
    - last 12 monthly backups (1st of month)

Everything outside those windows is removed. The pruner is a pure
function over the directory listing; tests inject a synthetic ``today``
to verify retention boundaries.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

__all__ = [
    "BackupResult",
    "backup_arctic",
    "backup_journal",
    "backup_state",
    "prune_backups",
]


# ---------------------------------------------------------------------------
# Result + roots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackupResult:
    """Outcome of one rsync call."""

    source: Path
    destination: Path
    ok: bool
    skipped: bool = False
    message: str = ""
    pruned: list[Path] = field(default_factory=list)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _backup_root(root: Path | None = None) -> Path:
    p = (Path(root) if root else _project_root()) / "backup"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Pruning policy
# ---------------------------------------------------------------------------


def _parse_dated_dir(name: str, prefix: str) -> date | None:
    """Extract YYYY-MM-DD from ``f"{prefix}-YYYY-MM-DD"`` or return None."""

    if not name.startswith(prefix + "-"):
        return None
    suffix = name[len(prefix) + 1 :]
    try:
        return datetime.strptime(suffix, "%Y-%m-%d").date()
    except ValueError:
        return None


def _classify_keep(d: date, today: date) -> bool:
    """Return True iff backup dated ``d`` should be kept under our policy.

    Policy (relative to ``today``):
        - 7 most recent daily backups (d within last 7 days inclusive)
        - 4 most recent weekly backups (Sundays within last 28 days)
        - 12 most recent monthly backups (first-of-month within last
          ~366 days)
    """

    days_old = (today - d).days
    if days_old < 0:
        return True  # future-dated (clock skew) — keep, never silently drop
    if days_old <= 6:
        return True
    # weekday: Mon=0..Sun=6. Treat Sunday as the weekly anchor.
    if d.weekday() == 6 and days_old <= 7 * 4:
        return True
    if d.day == 1 and days_old <= 366:
        return True
    return False


def prune_backups(
    prefix: str,
    *,
    root: Path | None = None,
    today: date | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Apply the 7/4/12 retention policy to ``backup/<prefix>-YYYY-MM-DD/``.

    Returns the list of directories that were (or would be) removed.
    """

    backup_dir = _backup_root(root)
    today = today or datetime.now().date()
    removed: list[Path] = []
    for child in sorted(backup_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_dated_dir(child.name, prefix)
        if parsed is None:
            continue
        if _classify_keep(parsed, today):
            continue
        removed.append(child)
        if not dry_run:
            shutil.rmtree(child, ignore_errors=True)
    return removed


# ---------------------------------------------------------------------------
# Core rsync wrapper
# ---------------------------------------------------------------------------


def _today_str(today: date | None = None) -> str:
    return (today or datetime.now().date()).strftime("%Y-%m-%d")


def _rsync(source: Path, destination: Path, timeout: float = 3600.0) -> None:
    """Run ``rsync -a --delete`` from source/ to destination/.

    Trailing slash on source is critical — it copies the *contents* into
    destination so re-running on the same day overwrites cleanly.
    """

    destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-a",
        "--delete",
        str(source).rstrip("/") + "/",
        str(destination).rstrip("/") + "/",
    ]
    subprocess.run(cmd, check=True, timeout=timeout)


def _do_backup(
    source: Path,
    prefix: str,
    *,
    root: Path | None,
    today: date | None,
    rsync_func,
) -> BackupResult:
    if not source.exists():
        return BackupResult(
            source=source,
            destination=_backup_root(root) / f"{prefix}-{_today_str(today)}",
            ok=False,
            skipped=True,
            message=f"source missing: {source}",
        )

    backup_dir = _backup_root(root)
    destination = backup_dir / f"{prefix}-{_today_str(today)}"

    try:
        rsync_func(source, destination)
    except subprocess.CalledProcessError as exc:
        return BackupResult(
            source=source,
            destination=destination,
            ok=False,
            message=f"rsync exit {exc.returncode}",
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as exc:
        return BackupResult(
            source=source,
            destination=destination,
            ok=False,
            message=f"rsync error: {exc!r}",
        )

    pruned = prune_backups(prefix, root=root, today=today)
    return BackupResult(
        source=source,
        destination=destination,
        ok=True,
        message="rsync ok",
        pruned=pruned,
    )


# ---------------------------------------------------------------------------
# Public per-domain wrappers
# ---------------------------------------------------------------------------


def backup_arctic(
    *,
    project_root: Path | None = None,
    today: date | None = None,
    rsync_func=None,
) -> BackupResult:
    """Snapshot ``data/arctic`` into ``backup/arctic-YYYY-MM-DD``."""

    root = Path(project_root) if project_root else _project_root()
    return _do_backup(
        source=root / "data" / "arctic",
        prefix="arctic",
        root=root,
        today=today,
        rsync_func=rsync_func or _rsync,
    )


def backup_journal(
    *,
    project_root: Path | None = None,
    today: date | None = None,
    rsync_func=None,
) -> BackupResult:
    """Snapshot ``data/journal`` into ``backup/journal-YYYY-MM-DD``."""

    root = Path(project_root) if project_root else _project_root()
    return _do_backup(
        source=root / "data" / "journal",
        prefix="journal",
        root=root,
        today=today,
        rsync_func=rsync_func or _rsync,
    )


def backup_state(
    *,
    project_root: Path | None = None,
    today: date | None = None,
    rsync_func=None,
) -> BackupResult:
    """Snapshot ``state`` (the heartbeat / runtime breadcrumbs)."""

    root = Path(project_root) if project_root else _project_root()
    return _do_backup(
        source=root / "state",
        prefix="state",
        root=root,
        today=today,
        rsync_func=rsync_func or _rsync,
    )


# Re-export for typing convenience.
def __all_iter() -> Iterable[str]:  # pragma: no cover
    return iter(__all__)
