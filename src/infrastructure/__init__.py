"""infrastructure — operational hygiene tentacle.

v0 surface:
    backup_arctic, backup_journal, backup_state, prune_backups
    rotate_logs
    health_check()              : returns a markdown report
    HealthReport                : dataclass with structured status

The check + backup + rotation routines are all idempotent: re-running
them produces no harmful side effect. Backups are append-only (a fresh
dated directory each day); pruning is the only destructive operation,
and it operates by retention policy (7 daily + 4 weekly + 12 monthly).
"""

from infrastructure.backup import (
    BackupResult,
    backup_arctic,
    backup_journal,
    backup_state,
    prune_backups,
)
from infrastructure.health_check import (
    HealthReport,
    HealthStatus,
    health_check,
)
from infrastructure.log_rotation import (
    RotationResult,
    prune_old_rotations,
    rotate_logs,
)

__all__ = [
    "BackupResult",
    "HealthReport",
    "HealthStatus",
    "RotationResult",
    "backup_arctic",
    "backup_journal",
    "backup_state",
    "health_check",
    "prune_backups",
    "prune_old_rotations",
    "rotate_logs",
]
