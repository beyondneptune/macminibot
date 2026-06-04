# Todo

Bootstrap backlog for `execution`. Each item is a self-contained agent-sized epic. Cycle 1 epics that survived ratification are kept; new epics map directly to the five ratified contracts in `CONTEXT.md`.

## Ratified-contract epics (cycle-3, must land first — sibling tentacles depend on these)

- [ ] **C1: `OrderIntent` ingest from `risk-management/inbox/`** — Implement the atomic-file poller for `risk/inbox/YYYY-MM-DD.json` (per `risk-management` I7b: `.tmp` + fsync + atomic-rename is the "complete" signal). Validate the batch envelope (`targets_count` checksum, `batch_id` dedupe via `processed_batches` set, schema parity with `risk-management`'s emitter). On unknown `batch_id` for same date → raise `DuplicateRebalanceAttempt` and touch HALT. Timeout: no batch by 09:36 ET → emit nothing (risk-management itself HALTs with `STRATEGY_INTENTS_TIMEOUT`; we're a downstream observer). Replace any cycle-1 assumption that `execution` reads `TargetPosition` directly from `strategy-engine`.

- [ ] **C2: Per-order synchronous HALT + heartbeat gate** — Implement `execution.gating.preflight(intent) -> None | OrderRefused`. Called immediately before every `placeOrder` IBKR call. `stat` the HALT file, read+parse `heartbeat.json`, enforce `mtime/ts_unix < 300s`. No caching, no async, no batch-level memoization. Unit tests: (a) HALT file present → refused; (b) heartbeat 299s old → passes; (c) heartbeat 301s old → refused; (d) heartbeat.json missing → refused; (e) parallel batch with HALT touched between order 3 and 4 → orders 1–3 placed, order 4 refused, orders 5+ refused. Add property-based test asserting no code path to `placeOrder` exists that doesn't pass through `preflight`.

- [ ] **C3a: `execution.journal.SCHEMA` + `write_fill`** — Define the canonical `pyarrow.Schema` matching the 31-column table in `CONTEXT.md` (28 core fill columns + 3 audit columns). Implement `write_fill(fill_row, mode: ChainMode) -> None` with partition routing: LIVE → `live/trades/dt=YYYY-MM-DD/`, PAPER → `paper/trades/dt=YYYY-MM-DD/`, backtest → `runs/<run_id>/trades/fold=YYYY/`. Ensure `(client_order_id, fill_seq, backtest_run_id)` PK uniqueness via a per-partition dedupe map. Land `tests/contracts/test_journal_schema.py` — both this tentacle's CI AND `backtesting`'s CI must import `execution.journal.SCHEMA` and assert equality with the pinned spec.

- [ ] **C3b: Backtest-write contract surface** — Expose `execution.journal.write_fill(...)` as a stable public API for `backtesting`'s `ExecutionSimulator`. Document import path and stable kwargs. Add a smoke test where `backtesting` writes a fake fill row through this entry point and reads it back via `pyarrow.dataset`, asserting all 31 columns roundtrip with correct dtypes (especially `dictionary<string>` columns and nullable floats).

- [ ] **C4: `ChainMode` enforcement at OMS entry** — Add `mode: ChainMode` as a required field on `OrderIntent` ingest (no default — missing `mode` is a `TypeError`). Branch the OMS at the topmost dispatch: LIVE → real IBKR client; PAPER → simulated-fill stub at next bar open + write to `paper/trades/` with `backtest_run_id="PAPER-{YYYY-MM-DD}"`, `account="PAPER"`; WHATIF → return envelope to caller, write nothing, no IBKR connection opened. Defense-in-depth assertion: even with a misconfigured client, a `mode != LIVE` order CANNOT reach `placeOrder`. Test: monkeypatch IBKR client to raise on any call, run PAPER and WHATIF batches, assert no raise occurred.

- [ ] **C5: `client_order_id` derivation + idempotency** — Implement `derive_client_order_id(intent_id, strategy_sha, account, attempt_n) -> ULID` per the formula in CONTEXT.md Contract 5. Wire `attempt_n` into the IBKR retry path so a network-drop retry re-derives the same `client_order_id` (idempotency key: IBKR rejects duplicate → we learn the prior submission succeeded). Add unit tests: (a) determinism (same inputs → same id); (b) attempt_n monotonicity → distinct ids; (c) cross-session uniqueness via `intent_id` date component; (d) backtest path uses `run_id` instead of `intent_id`, never collides with live name-space. Document the `client_order_id → intent_id → decision_id → signal_row → model_version` join chain in a reconciliation test.

## Cycle-1 epics that survived ratification (refit to the new flow)

- [ ] **OMS state machine** — Build the order-lifecycle state machine (New → Ack → PartialFill → Filled / Canceled / Rejected / CanceledByReplace). Cover terminal-state transitions and illegal-transition guards. Add `CanceledByReplace` for cancel-and-replace days where `risk-management` issues a cancel before the new batch's `OrderIntent.replaces` reference is honored (per `strategy-engine` Improvement #4).

- [ ] **IBKR client (TWS / IB Gateway)** — Implement order placement, modify, cancel via `ib_insync` (or `ib_async`). Wire reconnect logic and replay of in-flight orders from the persistent OMS event log on `~/macminibot/state/oms/YYYY-MM-DD.jsonl`. Separate process from any market-data client (fault isolation).

- [ ] **Intent lowering** — Convert `OrderIntent` from `risk-management` (NOT `strategy-engine`) into IBKR-valid orders: tick-size rounding, lot rounding, TIF defaults (OPG for MOO, DAY for LMT), post-only flags where applicable, exchange routing. Unit-test against IBKR contract-spec corner cases (ES tick = 0.25, EURUSD tick = 0.00005, etc.).

- [ ] **Session-long TWAP/VWAP slicer** — The daily batch is handed over at ≈09:35 ET and has the entire session (≈09:35 → 15:55 ET, respecting half-day early close from the batch's `session_close` field) to fill. Implement a TWAP slicer as v1; VWAP follow-on. Honor per-name `c_max` participation rate. Pause/resume on HALT mid-session.

- [ ] **Position & cash book** — Authoritative position tracking that updates on fills and reconciles to IBKR on reconnect. Persist to the OMS event log so a restart doesn't lose state. Expose `current_positions()` for `risk-management`'s 09:30 ET snapshot input.

- [ ] **Reconciliation on startup/reconnect** — Pull IBKR open orders + positions via `reqOpenOrders` + `reqPositions`, reconcile against OMS event log replay. On disagreement >1 share: write `ReconciliationDelta` journal row, touch HALT, force operator resolution. Matches `risk-management` `POSITION_RECONCILES` check #9.

- [ ] **Cancel-all / flatten-all command path** — Synchronous handlers for `control-plane` commands. `cancel-all` cancels every working order; `flatten-all` adds market sell-to-zero on every non-zero position via a fresh internal OrderIntent batch (still goes through preflight HALT check — flatten doesn't bypass risk). Both commands write a `ControlPlaneEvent` row to the audit log.

- [ ] **Trade audit log** — Persist raw IBKR request/response per interaction for compliance and post-trade analysis. Append-only JSONL alongside the OMS event log. Rotation: weekly via launchd → `audit.YYYY-WW.jsonl.gz`.

## Out-of-cycle / open coordination items

- [ ] **Coordinate with `risk-management` to pin `OrderIntent.replaces` field** — Currently `risk-management/CONTEXT.md` I4 describes diff semantics but doesn't pin the `replaces: str | None` field on the `OrderIntent` dataclass surface. We need it to mark the prior working order as `CANCELED_BY_REPLACE` and link the journal rows. Open a cross-tentacle PR against `shared-core` to add the field; until then, treat absence as "no prior intent" and rely on `risk-management` to have cancelled the prior order itself.

- [ ] **Watch for `Signal` schema ratification in `shared-core`** — `strategy-engine` flagged drift on `as_of` vs `as_of_ns` and `float` vs `Decimal`. `decision_id` flows through to our journal; once `shared-core` ratifies, audit `ts_decision` parse path for any widened acceptance needed.

- [ ] **Coordinate snapshot-staleness window** — `strategy-engine` says >10 min stale, `risk-management` says 09:30–09:35 ET window. Both feed our preflight chain indirectly (via HALT triggers). Track resolution in `risk-management` I7c.2; no action on our side until they reconcile.
