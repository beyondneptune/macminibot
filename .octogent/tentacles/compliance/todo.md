# Todo

Bootstrap backlog for `compliance`. Each item is a self-contained agent-sized epic.

- [ ] **Regulatory regime scoping doc** — Capture which regimes apply given the operator's domicile, venue list, and asset classes. The output is a short doc that locks down the *fields* every other epic must produce; without it, journal/export schemas are guesses.
- [ ] **Append-only event journal** — Define the journal entry schema, the storage backend (file-with-hash-chain vs. append-only table), and the writer API. Include a "tamper check" CLI that verifies the hash chain end-to-end.
- [ ] **`execution` → journal wiring** — Subscribe to every order state transition, fill, cancel, and reject from `execution`; persist with bounded lag; trip a halt on backlog overflow. Includes integration tests with a fake execution feed.
- [ ] **Trade-report exporter v1** — Pick one regime from the scoping doc and implement an exporter that produces a regulator-shaped extract for an arbitrary date range. Output is a file the operator can hand off.
- [ ] **Tax-lot ledger** — Implement FIFO + specific-lot accounting over the journal; emit per-symbol realized/unrealized gain reports. Property test: re-running the ledger from scratch is bit-identical.
- [ ] **Surveillance rules v1** — Detectors for self-cross, layering, and marking-the-close on our own flow. Each rule is a pure function over a time window with both passing and triggering test cases.
- [ ] **Operator report endpoints** — Expose end-of-day, end-of-month, and ad-hoc date-range report generation to `control-plane`, with download links and a "what's in this report" preamble.
- [ ] **Compliance health metrics** — Emit journal-write latency, execution-to-journal lag, missing-event count, and surveillance-finding count to `observability`. Wire alerts for the unhealthy thresholds.
