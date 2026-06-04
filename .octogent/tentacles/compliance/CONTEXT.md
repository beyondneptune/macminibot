# compliance

Regulatory trade reporting, order-record retention, tax-lot accounting, surveillance for wash/spoof patterns.

## Status
Greenfield — no source code exists yet. The notes below are a **starting proposal**, not ratified design. This dept exists because regulatory record-keeping has a *different* shape from operational audit logging (`infrastructure`) and from pre-trade risk gating (`risk-management`): it is written-always / read-rarely, has multi-year retention, and must remain queryable and exportable years after the live system has moved on.

## Scope
- Immutable order-and-event journal: every `OrderIntent`, `Order` state transition, `Fill`, cancel, and reject, with the timestamps a regulator cares about (decision, send, ack, fill)
- Trade-reporting exports: regulator-shaped extracts (CAT/MiFID II–style record formats) generated from the journal
- Tax-lot accounting: lot-by-lot cost basis, realized/unrealized gain attribution, FIFO/LIFO/specific-lot selection
- Internal surveillance: post-trade detection of wash trades, layering, spoofing, marking-the-close on our *own* order flow
- Retention policy enforcement and tamper-evidence (cryptographic chaining of journal entries)
- Compliance reports for the operator: end-of-day, end-of-month, ad-hoc date range

## Boundaries
- **Owns:** the regulatory journal store, report generators, tax-lot ledger, surveillance rules
- **Consumes from:** `execution` (every order state transition, every fill, raw venue req/resp); `market-data` (mark prices for unrealized P&L on tax-lot reports); `shared-core` (Order, Fill, Position types); `infrastructure` (retention/backup primitives, append-only storage)
- **Provides to:** `observability` (compliance health metrics: journal-write latency, lag from execution, missing-event alerts); `control-plane` (operator-facing report downloads and surveillance findings); regulators (via the operator, never directly)
- **Does not own:** pre-trade rejection of orders (that's `risk-management`); operational audit of human commands (that's `infrastructure`); P&L attribution for performance analysis (that's `observability`)

## Key Decisions (to be made)
- Which regulatory regimes apply (depends on entity domicile, asset class, venues) — drives the export schema
- Journal storage: append-only file with hash-chain, append-only DB table, or write-once object store with manifest — pick based on retention horizon and audit requirements
- Synchronous vs. asynchronous journaling — if asynchronous, define the bounded lag tolerated before a halt is triggered
- Tax-lot accounting method default (FIFO is common for crypto in many jurisdictions; equities often allow specific-lot)
- How to handle corrections / post-trade busts — append a correction entry, never mutate prior entries
- Wash-trade detection sensitivity (own-account self-cross is the obvious case; near-self-cross within N seconds is the harder one)

## Conventions (proposed)
- The journal is **append-only**. A correction is a new entry referencing the prior one — no in-place updates, ever
- Every journal entry carries `decision_id` and `client_order_id` (matching `shared-core` logging conventions) so it can be joined to any other log
- Reports are regenerated **from the journal** on demand — never stored as primary state; the journal is authoritative
- Surveillance rules are pure functions over a time window of journal entries — testable in isolation, replayable against history
- The journal-write path cannot share a failure domain with `execution`'s order path: an `execution` outage must not lose journal entries, and a journal outage must trip a halt rather than allow silent trading
- Tax-lot mutations are deterministic given the journal — re-running tax-lot accounting from scratch over the journal must yield bit-identical output
