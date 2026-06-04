# security

Exchange API-key custody, order-signing keys, network isolation, dependency vulnerability scanning.

## Status
Greenfield — no source code exists yet. The notes below are a **starting proposal**, not ratified design. This dept is a thin **policy and key-material** layer that sits next to `infrastructure` — `infrastructure` owns the *mechanism* for storing a secret (Keychain, sops file, vault); `security` owns the *policy* for which keys exist, what permissions each has, when they rotate, and what blast radius a compromise has. The target host is a single Mac mini, so the threat model is biased toward host compromise and supply-chain attacks rather than multi-tenant cloud concerns.

## Scope
- API-key inventory: every venue key, its scope (read / trade / withdraw), its environment (testnet / prod), its owner, its last rotation date
- Key lifecycle: provisioning, rotation cadence, revocation, post-compromise procedure (named scripts that an oncall can run)
- Order-signing key custody for venues that require client-side signing (Ed25519 / ECDSA over the order payload) — generation, storage, in-process loading
- Network isolation policy: which processes are allowed to reach which venue endpoints; IP allowlist registration with venues; egress firewall rules
- Withdrawal address allowlists (where supported by the venue) — codified, reviewed, never auto-edited from the trading path
- Dependency vulnerability scanning: SBOM generation, CVE feed integration, an upgrade-or-mitigate workflow for each finding
- Threat model document: enumerated attack scenarios (lost laptop, malicious dependency, compromised venue, exposed log) with concrete mitigations and detection signals

## Boundaries
- **Owns:** the key inventory, rotation runbooks, order-signing key material spec, network isolation policy, withdrawal allowlist source-of-truth, SBOM and vuln-scan pipeline, threat model doc
- **Consumes from:** `infrastructure` (the actual secrets backend, host firewall primitives, audit log to record key access); `execution` (the list of *which* venue keys it needs, with required scopes); `compliance` (key-access events that need to land in the regulatory journal)
- **Provides to:** `infrastructure` (the policy that CI must enforce — e.g. no plaintext-secret commits, dependency-scan gate on PRs); `execution` (a key-loading API that returns the right key for `(venue, env, scope)`); operator (runbooks, threat model, rotation schedule)
- **Does not own:** the storage backend itself (`infrastructure`); regulatory record-keeping (`compliance`); pre-trade risk gating (`risk-management`). Trade-permission-only keys must *never* have withdrawal scope — this is a policy this dept enforces, not a mechanism it builds

## Key Decisions (to be made)
- Default key scope policy: trade-only keys should be the norm; withdrawal-enabled keys, if any, are exceptional and locked to a fixed allowlist
- Rotation cadence per scope (e.g. trade-only quarterly, withdrawal-enabled monthly, signing keys yearly + on suspicion)
- Where order-signing private keys live at rest (macOS Keychain, sops-encrypted file, hardware token) and how the trading process loads them at startup without leaving them on disk decrypted
- Whether to maintain a separate "ops" key per venue (read-only) for `treasury` snapshots vs. sharing the trade key
- Egress firewall enforcement (pf.conf on macOS, an outbound-proxy, or none — accept the risk) — pick one
- SBOM/vuln scanning: tool choice (e.g. trivy, grype, osv-scanner) and CI gate severity threshold

## Conventions (proposed)
- A withdrawal-enabled key without a corresponding allowlisted destination set is a **bug**. CI rejects the configuration
- Every key access goes through the `infrastructure` secrets helper *and* logs an access event to the audit log — there is no direct read path
- New venue integration includes: key provisioned, scope minimized to trade-only by default, rotation date recorded, threat model updated. PR is blocked without these
- Plaintext secret in the repo (even in a test fixture) is a **release-blocker** — a pre-commit hook enforces it
- Order-signing private keys are loaded once at process start and held in memory; never written to disk by the trading process; cleared on shutdown
- Dependency updates that introduce a new transitive dep require an SBOM diff review on the PR
- The threat model is a living document: every incident or near-miss results in a documented update — empty git history on the threat model after a year is itself a problem
