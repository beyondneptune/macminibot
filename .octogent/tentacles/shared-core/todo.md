# Todo

Bootstrap backlog for `shared-core`. Each item is a self-contained agent-sized epic. The 4 sibling tentacles (ml-research, strategy-engine, risk-management, backtesting) are blocked on Epics 1-3 — those land first or the rest of the project stalls.

## Ratified 2026-05-17 (cycle 1)
Type specs in `CONTEXT.md` are the single source of truth. Each epic below implements one section.

---

- [ ] **Epic 1 — Decimal + timestamp primitives (KD1, KD2)**
  - [ ] `shared_core.decimal_ctx.install()`: set `getcontext().prec = 28`, `getcontext().rounding = ROUND_HALF_EVEN`. Called from every tentacle entrypoint.
  - [ ] CI test: `Decimal("0.1") + Decimal("0.2") == Decimal("0.3")` proves context is installed.
  - [ ] `shared_core.contracts._decimal_guard`: runtime check (and optional mypy plugin) that raises on `Decimal(float_value)` construction inside project code.
  - [ ] `shared_core.time`: helpers `from_iso`, `to_iso`, `from_datetime` (raises on naive), `to_datetime` (always tz-aware UTC). NO arithmetic helpers — int arithmetic is the API.
  - [ ] Convention test: grep for `float(` or `: float` in any contract file → fail (`confidence` is the only allowed `float` field; whitelist it).
  - [ ] Unit tests for round-trip ISO ↔ ns conversion at nanosecond precision (Python `datetime` only has microsecond precision — document the precision-loss boundary at the ISO conversion).

- [ ] **Epic 2 — Core enums + `Symbol` + `Venue` + `Side` (T1, T2, T3)**
  - [ ] Implement `Venue` as `StrEnum` with all IBKR venues listed in CONTEXT.md T2.
  - [ ] Implement `Side` as `StrEnum`.
  - [ ] Implement `Symbol` as `@dataclass(frozen=True, slots=True)` with `__post_init__` validation (ASCII ticker, ISO 4217 currency).
  - [ ] `__str__` returns `f"{ticker}@{venue.value}.{currency}"`.
  - [ ] CI test: `Symbol("AAPL", Venue.NASDAQ, "USD")` is hashable, equality-comparable, dict-keyable.
  - [ ] CI test: `Venue.SMART` rejected as primary listing in `Symbol`.

- [ ] **Epic 3 — Bar + DollarBar + Tick (T4, T5)**
  - [ ] `Bar` with OHLCV + validation (low ≤ open/close ≤ high, volume ≥ 0).
  - [ ] `DollarBar(Bar)` adds `notional` + `tick_count` with positivity checks.
  - [ ] `Tick` with disjoint-field discriminator on `TickType` (TRADES vs MIDPOINT); validation enforces the variant invariants.
  - [ ] Property tests (Hypothesis): random OHLCV → invalid combinations raise; valid ones round-trip through pyarrow.

- [ ] **Epic 4 — `Signal` dataclass (T6) + ml-research contract parity test**
  - [ ] Implement `Signal` mirroring `ml-research/CONTEXT.md` Signal spec byte-for-byte.
  - [ ] CI test (lives here, runs in both tentacles' CI): compare `dataclasses.fields(Signal)` against a frozen golden tuple `(("symbol", Symbol), ("r_hat_daily", Decimal), ("confidence", float), ("model_version", str), ("as_of_ns", int))`. Any drift breaks the build.
  - [ ] Validation: `|r_hat_daily| > 20%` raises (unit-error trap); `confidence` in `[0,1]`; `model_version` non-empty.

- [ ] **Epic 5 — `TargetPosition` + `TargetPositionBatch` (T7, T8)**
  - [ ] `TargetPosition` with `hold_reason: str | None` (strategy-engine Improvement #6).
  - [ ] `TargetPositionBatch` with `targets_count` checksum (risk-management I7b), `supersedes` for re-emission (strategy-engine Improvement #4), `nav_snapshot_version` hash.
  - [ ] Validation: duplicate-symbol rejection; checksum mismatch rejection.
  - [ ] CI test: round-trip through atomic-rename pattern (`tmp` → `fsync` → `rename`) produces an identical batch.

- [ ] **Epic 6 — `OrderIntent` + `NavSnapshot` (T9, T10)**
  - [ ] `OrderIntent` with unsigned `delta_qty` + direction-on-`side` convention (matches backtesting trade-journal).
  - [ ] `NavSnapshot` with `snapshot_version` sha256 hash; helper `compute_snapshot_version(snapshot_ns, net_liq, positions, buying_power, account_id) -> str`.
  - [ ] CI test: two `NavSnapshot` instances with same content have same `snapshot_version`; flipping any byte changes it.

- [ ] **Epic 7 — `ChainMode` + `BreachEvent` (T11, T12)**
  - [ ] `ChainMode` as `StrEnum` (LIVE/PAPER/WHATIF). NO default value.
  - [ ] `BreachEvent` with `mode` field carried into every event.
  - [ ] CI test: `check_chain.run(...)` signature (asserted via inspect) has `mode` as required keyword arg.
  - [ ] JSONL writer: render `observed`/`threshold` via `repr()` for the audit log; preserve in-memory types.

- [ ] **Epic 8 — `CorporateAction` (T13)**
  - [ ] `CorporateActionType` StrEnum with 10 action types per strategy-engine Improvement #5.
  - [ ] `CorporateAction` dataclass with `effective_date_ns`/`payable_date_ns`/`as_of_ns` naming (per KD2 — flags consumer-side rename for strategy-engine).
  - [ ] Validation: ratio positivity, multiplier-vs-ratio consistency, required `new_symbol` for SPINOFF/TICKER_CHANGE/MERGER_STOCK, required `cash_amount > 0` for CASH_DIVIDEND.

- [ ] **Epic 9 — `FoldSpec` (T14)**
  - [ ] `FoldSpec` dataclass with `numpy.ndarray` fields (`frozen=True` enforces no reassignment; arrays aren't hashed).
  - [ ] CI signature test (shared with backtesting via `tests/contracts/test_ml_research_imports.py`): `WalkForwardHarness.__init__` accepts `{train_min_years, train_max_years, val_days, embargo_days, purge_days, test_days, refit_cadence}` as kwargs; emits 5-tuple `FoldSpec`.

- [ ] **Epic 10 — Config loader**
  - [ ] Load + validate config files (TOML — matches `risk-management/CONTEXT.md` `risk/limits.toml`) into typed structures. Fails loudly on missing/malformed config.
  - [ ] No silent default fallbacks — every required field must be present in the file.

- [ ] **Epic 11 — Structured logger**
  - [ ] Logging wrapper emitting structured records with correlation ids (`decision_id`, `client_order_id`, `batch_id`, `intent_id`, `breach_id`).
  - [ ] Integrates with `observability`'s metrics where appropriate.
  - [ ] JSON output by default; pretty-printed in interactive shells.

- [ ] **Epic 12 — Typed error catalog**
  - [ ] Error types with `kind` discriminator (`ConnectionError`, `RateLimitError`, `RiskRejection`, `ParityViolation`, `SnapshotStale`, `BatchCorrupt`, `DuplicateBatch`, `ContractViolation`, etc.).
  - [ ] Callers branch on `kind`, not error string. Lint test forbids `str(e) ==` patterns in `except` blocks.

- [ ] **Epic 13 — Anti-circular-dep guard**
  - [ ] CI check (import-linter or equivalent) that fails if `shared_core/` imports from any other tentacle.
  - [ ] Plus a converse check: every other tentacle's `tests/contracts/test_consumer_imports.py` imports every public dataclass from `shared_core.contracts` to catch rename breakage immediately.

- [ ] **Epic 14 — Cross-tentacle contract test suite**
  - [ ] One test per consumer tentacle that imports every dataclass it uses and asserts fields/types via `dataclasses.fields` introspection.
  - [ ] Lives under `shared-core/tests/contracts/` and is also runnable from each consumer tentacle's CI.
  - [ ] Documents which fields each consumer reads so a renamer can see breakage scope before pushing.

- [ ] **Epic 15 — Mismatch reconciliation PRs (consumer-side updates noted in CONTEXT.md)**
  - [ ] `strategy-engine`: rename `CorporateAction.ex_date` → `effective_date_ns`, `payable_date` → `payable_date_ns`, `as_of` → `as_of_ns`. Type `action_type` as `CorporateActionType` enum.
  - [ ] `strategy-engine`: drop `nav_snapshot: Money` and `nav_snapshot_source: str` from `TargetPositionBatch` envelope; keep only `nav_snapshot_version: str`. Add note that consumers re-read the full `NavSnapshot` from risk-management using the version hash.
  - [ ] `risk-management`: change `TargetPosition.target_qty` reference from `int` to `Decimal`; lot-rounding happens at the broker boundary, not in the diff.
  - [ ] `backtesting`: document the `Decimal → float64` wire-format conversion at `execution.journal.write_fill(...)` and add the round-trip parity CI test.

- [x] **Epic 16 — `TargetPosition` / `TargetPositionBatch` reasoning-trace extension (proposed by strategy-engine cycle 7)** — RESOLVED cycle 8: HYBRID. Lifted only the three cross-tentacle observability fields (`decision_id`, `mode: ChainMode`, `solver_status: str`) onto `TargetPositionBatch`. Strategy-engine-internal numerics (`r_hat`, `sigma_contribution`, `target_weight`, `confidence`, `w_prev`, `sigma_portfolio_annualized`, `edge_case_rules_fired`) stay OUT of shared_core — they live in the strategy-engine envelope and/or `TargetPosition.reason_trace`. Epic 15's drop of `nav_snapshot: Decimal` + `nav_snapshot_source` stands. Strategy-engine can now collapse the `to_shared_core()` lowering down to the per-name trace fields only.

  **Proposal context.** `strategy-engine` cycle 7 migrated `Signal`, `NavSnapshot`, `ChainMode`, and `Symbol` to import from `shared_core.contracts` directly. `TargetPosition` and `TargetPositionBatch` were *not* migrated in place; the strategy-engine-local envelopes wrap the canonical ones via a `to_shared_core()` lowering. Reason: the canonical shared_core forms drop the reasoning-trace fields that `strategy-engine/CONTEXT.md` declares load-bearing for compliance and operator post-mortems. The trace is currently shoved into `TargetPosition.reason_trace: dict[str, Any]` at the boundary, which is lossy (no schema, no validation, no introspection).

  **Decision to ratify (shared-core owner — not strategy-engine).** Either (a) extend the canonical shared_core dataclasses with the trace fields below, OR (b) bless the sidecar pattern (strategy-engine keeps the rich envelope; canonical shared_core stays minimal; the dict-in-reason_trace remains the wire format). Both are defensible; the trade-off is schema-discoverability vs canonical-stability. Strategy-engine is unblocked either way.

  **Fields strategy-engine currently keeps on the local envelope (option (a) target schema):**

  `TargetPosition` extras (per-name reasoning trace):
  - [ ] `target_weight: float` — risk-units, pre-lowering (lives upstream of `contracts`).
  - [ ] `r_hat: float` — winsorized signal value used by the optimizer.
  - [ ] `confidence: float` — model-reported confidence used as a multiplicative tilt.
  - [ ] `w_prev: float` — prior-session weight on this name; risk-management uses for delta sizing.
  - [ ] `sigma_contribution: float` — marginal variance contribution `(Σ @ w_final)_i`.

  `TargetPositionBatch` extras (per-batch reasoning trace):
  - [ ] `decision_id: str` — reproducibility hash over `(as_of, model_version, signal_payload, w_prev, contract_spec_version, solver_status, snapshot_version, corporate_actions)`. CONTEXT.md "Conventions" makes this the **primary key** of the batch; canonical `batch_id` should either be `decision_id` or shared_core should ratify both fields with one as a deterministic function of the other.
  - [ ] `mode: ChainMode` — frozen at 09:35 wake (Improvement #7); risk-management I7a requires `mode` as a required kwarg with no default. Currently nowhere on the canonical batch.
  - [ ] `solver_status: str` — Stage-1/2/3 result + final status (`optimal | optimal_inaccurate | fallback_hrp_after_3_retries | zero_signal_decay | halted_no_universe`). Observability hard-depends on this.
  - [ ] `sigma_portfolio_annualized: float` — `sqrt(w' Σ w) * sqrt(252)`. Vol-target tracking-error metric source.
  - [ ] `edge_case_rules_fired: list[str]` — every rule the pipeline fired this session (e.g. `"signal.winsorized:S3"`, `"halt.luld"`, `"stale_bar.tier2_halve_target"`). Compliance reads this.
  - [ ] `nav_snapshot: Decimal` + `nav_snapshot_source: str` — currently on local envelope; shared_core Epic 15 says to drop both and keep only `nav_snapshot_version`. If shared_core ratifies that, strategy-engine drops them too (the `_assemble_batch` already has the version handy). If shared_core wants the full snapshot inlined for atomic auditability, keep both. **Decision needed.**
  - [ ] `as_of_ns: int` + `session_close_ns: int` + `valid_until_ns: int` — session timestamps. `generated_at_ns` on the canonical batch maps to `as_of_ns`; the other two are missing from canonical.

  **If shared-core picks (a) extend:** strategy-engine `contracts.py` collapses to a one-line re-export. The cycle-7 `to_shared_core()` lowering is deleted. Risk-management I-spec stays as-is because it already requires `mode` and `decision_id`.

  **If shared-core picks (b) sidecar:** strategy-engine documents the current `to_shared_core()` envelope in CONTEXT.md as the canonical lowering. No code changes required either side. The local `TargetPosition`/`TargetPositionBatch` are the strategy-engine source-of-truth; `shared_core` forms are the cross-tentacle wire format.

  **Greenfield project — no migration cost for either option.** Decision rests with shared-core owner.

- [x] **Epic 17 — `BreachEvent` audit-context + str-keyed snapshot bridges (proposed by risk-management cycle 7)** — RESOLVED cycle 8: SPLIT decision. (1) `BreachEvent` extended with `config_id: str = ""`, `symbol: str = ""` (ticker string, NOT `Symbol`), `extra: dict[str, Any] = field(default_factory=dict)`. Replay tooling now reconstructs from JSONL alone; the `_AUDIT_CTX` sidecar can be dropped. `ts_unix` deliberately NOT added — KD2 says ns is canonical, derive at I/O. (2) `NavSnapshot.positions: dict[Symbol, Decimal]` STAYS canonical. Migration to str-keyed rejected — Symbol is the cross-tentacle primary key (T1), and execution requires real venue/currency metadata. Risk-management keeps its local str-keyed cache and bridges via `IntentContext.ticker` as documented in risk-management/CONTEXT.md.

  **Proposal context.** `risk-management` cycle 7 migrated `OrderIntent`, `Side`, `TargetPosition`, `TargetPositionBatch`, `ChainMode`, and `BreachEvent` to import from `shared_core.contracts`. Chain-local sidecar fields (`current_qty`, `target_qty`, ticker-key) moved to a new `risk_management.diff.IntentContext` dataclass carried on `ChainContext.intent_contexts` (per Epic 26 in `risk-management/todo.md`). Two friction points emerged that the shared-core owner needs to decide on:

  **(1) `BreachEvent` audit-context fields.** The shared_core `BreachEvent` omits four fields the risk-management JSONL audit log carries today: `config_id`, `symbol` (ticker, for ops dashboards that index by name), `extra: dict[str, Any]` (per-rule context like `pct`/`net_liq` on MAX_SINGLE_NAME, `kelly_optimal`/`kelly_fraction` on KELLY_CAP, `broker`/`internal` on POSITION_RECONCILES), and `ts_unix: float` (replay-tooling convenience, since `ts_ns` is the canonical KD2 field). The cycle-7 implementation handles this via an in-memory `_AUDIT_CTX: dict[breach_id, dict]` sidecar in `risk_management.breaches` that the JSONL writer merges into the wire record. This works but is brittle (process-local state; can leak across tests if `emit_breach` isn't called).

  **Options to ratify (shared-core owner):**
  - (a) **Extend** `BreachEvent` with `config_id: str = ""`, `symbol: str = ""`, `extra: dict[str, Any] = field(default_factory=dict)`. Drop the local sidecar; the contract type carries everything the JSONL needs. `ts_unix` stays derived from `ts_ns` at I/O time (no need to add). Cost: `BreachEvent` is no longer minimal; consumers in other tentacles see four fields they don't use. Benefit: replay tooling reconstructs without out-of-band state.
  - (b) **Bless the sidecar pattern.** Document that audit context lives in `risk_management.breaches._AUDIT_CTX`, replay tooling is risk-management's responsibility, shared_core stays minimal. Strategy-engine cycle-7 sidecar (`to_shared_core()`) is the precedent.

  **(2) `NavSnapshot.positions` keying.** The shared_core `NavSnapshot.positions: dict[Symbol, Decimal]` (Symbol-keyed). The risk-management local `NavSnapshot.positions: dict[str, Decimal]` is ticker-keyed and still drives the chain (the chain dereferences `IntentContext.ticker` to look up positions/prev_close/feature_ok/broker_positions). The bridge introduces a footgun: `risk_management.diff._synth_symbol_from_ticker` synthesizes Symbols with hardcoded `NASDAQ`/`USD` for flatten-only positions (the snapshot doesn't know the real venue/currency). Downstream (execution) MUST receive intents with real Symbol metadata.

  **Options to ratify:**
  - (a) **Migrate** local `NavSnapshot` to `dict[Symbol, Decimal]` (matches shared_core) and ratify a single `Symbol` canonical store. Touches `risk_management.snapshot.build_snapshot`, all 6 dict-typed fields (`positions`, `broker_positions`, `prev_close`, `feature_ok`, plus market-data inputs), every test fixture, and the IBKR snapshot ingestion adapter. Substantial churn but eliminates the bridge.
  - (b) **Keep the str bridge** and document `IntentContext.ticker` as the canonical lookup key for chain-local snapshot reads; flatten-emitted intents carry placeholder Symbols that execution-tentacle re-resolves against its own symbology table.

  **Risk-management is unblocked either way.** Cycle-7 ships the sidecar+bridge implementation; if shared-core picks (a) on either question, a future risk-management cycle drops the local stubs.
