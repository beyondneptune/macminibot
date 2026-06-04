# infrastructure

Deployment, low-latency tuning, CI/CD, time sync, secrets backend, audit logging.

## Status
Greenfield — no source code exists yet. The notes below are a **starting proposal**, not ratified design. The target host is a Mac mini (project name: `macminibot`), so this dept's bias is toward a simple single-host setup, not Kubernetes.

## Scope
- Project bootstrap: language toolchain(s), build system, dependency manager, linters, formatters
- CI/CD: tests on push, type checks, format checks, optional deploy-to-host pipeline
- Process supervision on the Mac mini (launchd, supervisord, or a single Python/Rust supervisor process)
- Time synchronization (NTP/chrony) — accurate clocks matter for `ts_local` and audit
- Secrets backend (macOS Keychain, sops-encrypted files, or a small vault) — never plaintext API keys in repo
- Audit log (append-only) for risk overrides, manual commands, key access
- Backup strategy for tick archive, model registry, trade journal

## Boundaries
- **Owns:** repo bootstrap, CI config, supervisor config, secrets backend, audit-log storage, host runbooks
- **Consumes from:** every dept (their deploy needs, secrets, runtime requirements)
- **Provides to:** every dept (secrets, time, log infra, deploy mechanics)

## Key Decisions (to be made)
- **Language stack** — biggest decision in the project. Probably Python for `ml-research`/`backtesting`; hot-path (`market-data`/`strategy-engine`/`execution`) could be Python (simpler), Rust (faster), or a Python+Rust split via PyO3. Document the call here.
- Monorepo vs. multi-repo
- Build/dep manager (uv, poetry, cargo, just, make)
- CI host (GitHub Actions on this repo, locally on Mac mini via act, none for now)
- Whether to containerize (Docker on macOS has caveats) or run natively on macOS
- Backup target (Time Machine, rclone to S3, restic to B2)

## Conventions (proposed)
- All secrets accessed via a single helper that reads from the chosen backend — never `os.environ` directly in app code
- Every commit runs typecheck + lint + tests in CI; failing builds block merge
- Long-running processes are managed by `launchd` plist files checked into the repo
- Time-sync health is itself a monitored metric — clock drift > N ms triggers an alert
- Audit-log entries are append-only and signed/hashed to detect tampering
