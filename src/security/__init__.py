"""security — credential & dependency audit tentacle.

v0 surface:
    Finding, Severity        : structured audit results
    audit_credentials()      : verify .env keys, masking, perms, gitignore
    scan_dependencies()      : diff `pip list` vs known-good frozen set
    mask()                   : redact a secret-like string for log output

This package does NOT mutate any system state — every routine is read-only
and side-effect free. The live bot must never be perturbed by a security
check. Findings carry a Severity (INFO/WARN/CRITICAL) so the launchd
wrapper script can exit non-zero only on real issues.
"""

from security.credential_audit import (
    REQUIRED_ENV_KEYS,
    audit,
    audit_credentials,
    mask,
)
from security.dep_scanner import (
    DepDiff,
    load_known_deps,
    scan_dependencies,
    write_known_deps,
)
from security.findings import Finding, Severity

__all__ = [
    "DepDiff",
    "Finding",
    "REQUIRED_ENV_KEYS",
    "Severity",
    "audit",
    "audit_credentials",
    "load_known_deps",
    "mask",
    "scan_dependencies",
    "write_known_deps",
]
