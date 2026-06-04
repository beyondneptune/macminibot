# treasury

Multi-venue balance reconciliation, cash/FX, margin and collateral, fee and rebate accounting.

## Status
Greenfield — no source code exists yet. The notes below are a **starting proposal**, not ratified design. This dept exists because `execution` is authoritative for *order-level* state (orders, fills, in-flight positions) but it is **not** the right place for *account-level* state (cash balances, fees actually charged, margin requirements, FX between quote currencies). Conflating the two has historically caused systems to "know" they are flat while the venue has actually charged them, lent to them, or held collateral they cannot find.

## Scope
- Per-venue, per-currency balance ledger maintained against authoritative venue snapshots
- Reconciliation: comparing `execution`'s position book + computed cash deltas against venue-reported balances; emitting breaks
- Fee and rebate accounting: capturing actual fees from fill reports, attributing to strategy / venue / fee tier
- Margin and collateral tracking for margined / futures / perp venues: posted collateral, maintenance margin, liquidation distance
- FX handling when a strategy trades pairs quoted in currencies other than the home currency (mark-to-home for reporting)
- Cash management: low-balance alerts, suggested-rebalance reports, withdrawal/deposit recording

## Boundaries
- **Owns:** the multi-venue balance ledger, the fee ledger, margin/collateral state, FX conversion helpers, the reconciliation engine and break log
- **Consumes from:** `execution` (fills with venue-reported fees, position deltas, account snapshots pulled from venues); `market-data` (FX cross-rates and venue mark prices for collateral valuation); `shared-core` (`Position`, `Fill`, `Venue`, decimal types); `compliance` (the journal, as a redundant source for fee attribution)
- **Provides to:** `risk-management` (true buying-power and liquidation-distance inputs that pre-trade checks should respect, separate from notional exposure); `observability` (P&L decomposed into gross / fees / FX / funding components); `control-plane` (current cash and margin view, break alerts, low-balance warnings); `compliance` (fee-paid totals for tax reports)
- **Does not own:** order or position state in flight (that's `execution`); pre-trade limit *enforcement* (that's `risk-management` — treasury *informs* those limits but does not gate orders)

## Key Decisions (to be made)
- "Home currency" for reporting (USD is conventional but can be USDT/EUR/etc.) — drives every FX conversion
- Reconciliation cadence per venue (every fill? every N seconds? once a minute? on every snapshot push?) — trade-off between API quota and detection latency
- Break-handling policy: auto-resolve small breaks under a tolerance, escalate above; or always escalate
- Whether to model funding payments for perp venues as a separate stream or as synthetic fills
- How to handle venue-side rebates that arrive on a delayed schedule (e.g. monthly maker-rebate credits)
- Margin model abstraction — every margined venue has its own; pick whether we maintain a generic model or one per venue

## Conventions (proposed)
- The venue's reported balance is **authoritative** for cash and collateral state; `execution`'s computed cash delta is a check, not the source of truth
- All monetary values use the decimal/fixed-point types from `shared-core` — never `float`, no exceptions in this dept
- Fees attached to fills are always the *venue-reported* fee on that fill, not a re-derived estimate; estimated fees are a separate field used only when the venue hasn't reported yet
- Every break is an event: `{venue, currency, expected, observed, tolerance, fill_window}`, surfaced to `observability` and stored for replay
- A reconciliation pass is a pure function of `(prior_ledger, fills_in_window, venue_snapshot) → (new_ledger, breaks)` — replayable and unit-testable
- Treasury is read-only with respect to the rest of the system: it never cancels orders or flips kill-switches. It reports state; `risk-management` and the operator act on it
