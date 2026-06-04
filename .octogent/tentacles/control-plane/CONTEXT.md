# control-plane

Operator dashboard/API, manual overrides, runtime config, kill-switch UI, strategy toggles.

## Status
Greenfield — no source code exists yet. The notes below are a **starting proposal**, not ratified design.

## Scope
- HTTP/gRPC API exposed to a human operator (and any front-end UI)
- Web UI (or terminal UI) showing live bot status: positions, P&L, open orders, feed health
- Operator commands: pause/resume strategies, flatten-all, cancel-all, trip kill switch, override limits
- Runtime config: surface and (where safe) live-edit configuration values
- Authentication for the API — even a single-user home deployment needs a token; do not ship an open endpoint

## Boundaries
- **Owns:** API server, UI app, command dispatcher, config-edit gateway
- **Consumes from:** `observability` (current state for display); `execution` (positions, orders); `risk-management` (limit utilization); all dept configs
- **Provides to:** `risk-management` (manual kill-switch trip, limit overrides); `strategy-engine` (pause/resume strategy); `execution` (flatten-all, cancel-all)

## Key Decisions (to be made)
- Stack — Python (FastAPI) + React, Rust + leptos/htmx, Go + templates, terminal-only TUI (textual/ratatui), or just a Telegram bot
- UI scope for v1 — full dashboard or read-only monitor + kill switch only
- AuthN/AuthZ — local token, OS keychain, OAuth, mTLS
- Where this runs — on the same Mac mini as the bot or on a separate machine
- How commands propagate to other depts (in-process for monolith, RPC for separate processes)

## Conventions (proposed)
- All commands are idempotent and audit-logged (`infrastructure` audit log) — same command applied twice does not double-act
- Read endpoints are derived from `observability` data, not parallel state — single source of truth
- Destructive commands (flatten-all, kill switch) require a confirmation step in the UI/API (typed confirmation token or two-step)
- Never bypass `risk-management` — operator overrides go *through* risk, recorded as a breach-override event
- API is the only way operator commands enter the system — no SSH-into-the-process patches
