"""Common finding + severity types shared by credential & dependency audits.

Kept in a separate module so :mod:`security.credential_audit` and
:mod:`security.dep_scanner` can both import them without a circular
dependency at package-init time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["Finding", "Severity"]


class Severity(str, Enum):
    """Severity tiers. CRITICAL is the only level that fails launchd jobs."""

    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class Finding:
    """One audit result. ``code`` is a stable string ID for grep/diff."""

    code: str
    severity: Severity
    message: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "extra": dict(self.extra),
        }

    def is_high(self) -> bool:
        """High-severity = CRITICAL. Launchd wrapper exits non-zero on these."""
        return self.severity is Severity.CRITICAL
