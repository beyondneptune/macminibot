# Todo

Bootstrap backlog for `treasury`. Each item is a self-contained agent-sized epic.

- [ ] **Balance ledger schema + writer** — Define per-venue, per-currency balance entries (decimal types from `shared-core`, no floats) and the storage layer. Includes a "current balances" query and a point-in-time query.
- [ ] **Venue snapshot pullers** — One puller per supported venue that fetches the authoritative balance + margin snapshot and feeds it to the reconciliation engine. Snapshot cadence is configurable per venue.
- [ ] **Reconciliation engine** — Pure function `(prior_ledger, fills_window, venue_snapshot) → (new_ledger, breaks)`. Unit-tested with at-tolerance, over-tolerance, and missing-fill scenarios.
- [ ] **Fee ledger + attribution** — Capture venue-reported fees from every fill, attribute to `(strategy, venue, fee_tier)`, expose totals to `observability` and `compliance`.
- [ ] **Margin and collateral tracker** — For each margined venue, track posted collateral, maintenance margin, and computed liquidation distance. Surface "liquidation distance < N%" as an alertable metric.
- [ ] **FX layer** — Cross-rate fetcher (from `market-data`) + helpers to convert any `(amount, currency)` to home currency for reporting. Reject any conversion older than a configured staleness.
- [ ] **Buying-power feed for `risk-management`** — Expose true available buying power per venue/account so pre-trade checks can use real headroom rather than estimated notional.
- [ ] **Break event stream + dashboard view** — Emit structured break events to `observability`; expose the current break log to `control-plane` for operator triage.
- [ ] **Low-balance + rebalance suggestions** — Configurable thresholds per venue; when buying power drops below, emit an alert and a suggested rebalance amount (no auto-rebalancing in v1).
