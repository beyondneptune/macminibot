# Todo

Bootstrap backlog for `security`. Each item is a self-contained agent-sized epic.

- [ ] **Threat model v1** — Write the initial threat model: enumerated scenarios (lost laptop, malicious dependency, leaked API key, compromised venue, exposed log) with mitigations and detection signals for each. This is the doc every other epic refers back to.
- [ ] **Key inventory + manifest** — Define the schema for an API-key entry `(venue, env, scope, owner, created_at, rotates_at)` and the file/store it lives in. Includes a CLI that lists keys and flags overdue rotations.
- [ ] **Secrets-helper integration** — Define the `security` → `infrastructure` API: `load_key(venue, env, scope)` resolves through the secrets backend and writes an audit event. No other key-read path exists.
- [ ] **Order-signing key handling** — Generation procedure, at-rest storage choice, in-process loader that holds the key in memory only and clears on shutdown. Includes a test that the disk has no plaintext after a run.
- [ ] **Withdrawal-allowlist enforcement** — Codify the per-venue withdrawal address allowlist; add a CI check that any withdrawal-enabled key has a non-empty allowlist and that the allowlist is reviewed-and-signed.
- [ ] **Plaintext-secret pre-commit hook** — Pre-commit + CI scanner that fails on common secret patterns; ships with an explicit allowlist file for known false positives.
- [ ] **Dependency vulnerability gate** — Pick a scanner (e.g. trivy / osv-scanner), wire it into CI with a severity threshold defined in the threat model, and produce an SBOM artifact on every build.
- [ ] **Network isolation policy** — Document and (where mechanizable) enforce egress rules: which processes reach which venue endpoints; IP-allowlist registration steps per venue.
- [ ] **Rotation runbook + cadence** — Per-scope rotation cadence (trade-only / withdrawal / signing), and a runbook the operator can follow end-to-end. Include the post-compromise revocation procedure.
- [ ] **Key-access audit feed** — Subscribe to the audit log for key-access events and emit "unexpected key access" alerts to `observability` (e.g. trade key accessed outside the trading process).
