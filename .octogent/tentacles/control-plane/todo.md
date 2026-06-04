# Todo

Bootstrap backlog for `control-plane`. Each item is a self-contained agent-sized epic.

- [ ] **Choose stack + scope for v1** — Pick backend framework and UI approach (full web dashboard vs. read-only monitor + kill switch vs. Telegram bot). Document in `CONTEXT.md`.
- [ ] **Read-only status API** — Endpoints for positions, open orders, P&L, feed health, kill-switch state, strategy state. Backed by `observability`/`execution` — no parallel state.
- [ ] **Auth layer** — Token-based authentication (or chosen alternative). Even a home deployment ships behind auth.
- [ ] **Kill-switch command** — Synchronous trip that propagates to `risk-management` and `execution`. Two-step confirmation. Audit-logged.
- [ ] **Flatten-all / cancel-all commands** — Issue cancel for all open orders and market-out remaining positions. Idempotent. Confirmation gated.
- [ ] **Strategy on/off toggles** — Pause/resume per-strategy through `strategy-engine`. Surfaced in the UI.
- [ ] **Operator UI v1** — Dashboard showing the read-only status plus the gated commands. Match whatever stack was chosen.
- [ ] **Runtime config viewer / safe-editor** — Display current config; allow edits limited to fields marked safe-to-hot-reload. Tightening risk limits OK, loosening requires breach-override flow.
