# execution

Order management, IBKR connectivity, fill tracking, trade-journal writer, position state. The **only** tentacle that talks to the broker for trading; the **owner** of the trade-journal schema for both live fills and backtest fills.

## Status
Greenfield — no source code yet. **Architecture ratified cross-tentacle, cycles 2 + 3.** This document supersedes the cycle-1 draft, which assumed direct `strategy-engine → execution` flow and a per-batch HALT check. Both were ratified away by sibling tentacles. The contracts below are honored, not proposed.

## Scope
- Order management system (OMS): lifecycle of every order (New → Ack → PartialFill → Filled / Canceled / Rejected)
- IBKR client for placing, modifying, canceling orders (TWS / IB Gateway over `ib_insync` or `ib_async`)
- Lowering of post-risk `OrderIntent` records into IBKR-valid orders (tick-size rounding, lot rounding, TIF defaults, exchange routing flags)
- Session-long TWAP/VWAP slicer that fills the day's batch across ≈09:35 → 15:55 ET
- Position and cash accounting (authoritative source of truth for "what do we hold right now")
- Reconciliation against IBKR state on reconnect or startup
- **Trade-journal writer** — single source of truth for the fill schema; serves both live trades and backtest fills via `execution.journal.write_fill(...)`
- ChainMode honoring: LIVE submits to IBKR, PAPER writes to journal sidecar, WHATIF emits/writes nothing
- Synchronous HALT + heartbeat check before **every** order submission (per-order, no caching)

## Boundaries

Ratified flow (cycle 2): `strategy-engine` (TargetPositionBatch) → `risk-management` (diff + 10-check chain) → `execution` (OrderIntent batch).

- **Owns:**
  - OMS state machine and IBKR client
  - The `execution.journal` module + `execution.journal.SCHEMA` (the pyarrow Schema for the trade-journal parquet)
  - `execution.journal.write_fill(...)` — the single writer for both live trades and backtest sim fills
  - Position book (authoritative; no other tentacle maintains parallel position state)
  - Client-order-id derivation from `intent_id`
  - Reconciliation logic on reconnect / startup
  - Session-long slicing (TWAP/VWAP) for the daily batch
- **Consumes from:**
  - `risk-management` — vetted `OrderIntent` batches post-chain (NOT `TargetPosition` and NOT direct from `strategy-engine`). Each `OrderIntent` carries `intent_id`, `decision_id` (upstream), `replaces` (for cancel-and-replace days), `ChainMode` (LIVE/PAPER/WHATIF).
  - `risk-management` — the kill-switch artifacts: `~/macminibot/state/HALT` (file flag) and `~/macminibot/state/heartbeat.json` (60s mtime). Read-only, file-based, no IPC.
  - `market-data` — marks for OMS internal pricing decisions (slicer cadence, marketable-limit logic), via the read API. Never the source of truth for the fill price — that comes from the broker.
  - `shared-core` — `OrderIntent`, `Order`, `Position`, `Side`, `Symbol`, `ContractSpec`, `Money`, `ChainMode` types.
  - `infrastructure` — IBKR connection secrets (`tws_user`, paper/live flag, gateway host/port).
- **Provides to:**
  - `risk-management` — live position + exposure feed; fill events for the 09:30 ET snapshot's `positions` field on the next session.
  - `strategy-engine` — end-of-day realized positions for tomorrow's `w_prev` computation (read via journal, not push).
  - `backtesting` — `execution.journal.SCHEMA` and `execution.journal.write_fill(...)` (imported and called by the `ExecutionSimulator` so backtest fills land in the same parquet definition as live).
  - `observability` — slippage realized vs modeled, fill rates, reject reasons, IBKR latency, partial-fill timing.
  - `control-plane` — `cancel-all`, `flatten-all` commands (synchronous; honored even mid-batch).

## Contracts honored

These five contracts are non-negotiable, externally ratified, and load-bearing for cross-tentacle correctness. Each maps to an epic in `todo.md`.

### Contract 1 — Input is `OrderIntent` from `risk-management`, never `TargetPosition` from `strategy-engine`
Cycle 2 ratified the flow `strategy-engine (TargetPositionBatch) → risk-management (diff_targets_to_intents + 10-check chain) → execution (OrderIntent batch)`. `execution` MUST NOT read `TargetPosition` records directly. The `TargetPosition → OrderIntent` lowering is owned by `risk-management` (see `risk-management/CONTEXT.md` I4); we receive the post-diff, post-chain output. Per `risk-management` I7b, the batch is delivered as a single atomic JSON file at `risk/inbox/YYYY-MM-DD.json` written via `.tmp` + fsync + atomic-rename. We poll `risk/inbox/` from 09:35 ET. **There is no streaming intent feed.** One batch per session (with optional cancel-and-replace mid-session per `strategy-engine` Improvement #4 — handled by `risk-management` cancel ownership; we receive a new batch whose `OrderIntent.replaces` field references the prior `intent_id`).

### Contract 2 — Per-order synchronous HALT + heartbeat check (no batch-level, no caching)
Per `risk-management` I5 ("Execution's synchronous HALT + heartbeat contract"), before submitting **every** IBKR order — not once per batch, not once per session, not cached from 10s ago — `execution` MUST:
1. `stat ~/macminibot/state/HALT` — if it exists, abort the order and emit `OrderRefused(reason="HALT")`.
2. Read `~/macminibot/state/heartbeat.json`, parse `ts_unix`, compare to `time.time()`. If `now - ts_unix > 300`, abort and emit `OrderRefused(reason="HEARTBEAT_STALE", age=<seconds>)`.
3. Both checks are synchronous, blocking, on the order-submission code path. No async, no in-memory cache, no "I checked it for the previous order".
4. Failure to perform these checks is a P0 bug. There is no `--skip-halt-check` flag and never will be.

The per-order requirement is load-bearing: a HALT can trip mid-batch (e.g., drawdown breaker on unrealized P&L from the first fill must block subsequent fills). The 5-minute heartbeat window is intentional — `risk_heartbeat.py` writes every 60s, so 5 minutes = 5 consecutive write failures = something is genuinely wrong upstream.

### Contract 3 — Trade-journal schema OWNERSHIP (`execution.journal.SCHEMA` is the single source of truth)
Per `backtesting/CONTEXT.md` Boundaries and Trade-journal schema sections, `execution` is the OWNER of the trade-journal parquet schema. Both live fills and backtest `ExecutionSimulator` fills write through `execution.journal.write_fill(...)`. No second writer, no second schema, no translation layer — this is what makes the live-vs-backtest parity report a straight `JOIN ... USING (client_order_id)` rather than a schema-reconciliation exercise.

**28-column schema (v1, ratified cycle 2):**

| # | Column | dtype | Nullable | Notes |
|---|---|---|---|---|
| 1 | `client_order_id` | `string` (ULID) | NO | **PK component**; deterministic in backtest = `hash(run_id, decision_id)`; reproducible from `intent_id` in live (Contract 5) |
| 2 | `backtest_run_id` | `string` | YES | NULL in LIVE; set in PAPER (`PAPER-{date}`) and backtest. **PK component** |
| 3 | `decision_id` | `string` (ULID) | NO | links to upstream signal row in `strategy-engine`'s ledger |
| 4 | `ts_decision` | `timestamp[ns, UTC]` | NO | when `strategy-engine` emitted intent |
| 5 | `ts_submit` | `timestamp[ns, UTC]` | NO | when OMS sent to broker |
| 6 | `ts_ack` | `timestamp[ns, UTC]` | YES | broker ack (NULL in backtest) |
| 7 | `ts_fill` | `timestamp[ns, UTC]` | NO | fill timestamp |
| 8 | `symbol` | `string` | NO | canonical (`AAPL`, `ESZ6`, `EUR.USD`) |
| 9 | `asset_class` | `dictionary<string>` | NO | `equity` / `future` / `fx` |
| 10 | `side` | `dictionary<string>` | NO | `BUY` / `SELL` |
| 11 | `qty` | `int64` | NO | unsigned; sign comes from `side` |
| 12 | `order_type` | `dictionary<string>` | NO | `MOO` / `LMT` / `MKT` |
| 13 | `tif` | `dictionary<string>` | NO | `OPG` / `DAY` / `IOC` |
| 14 | `limit_px` | `float64` | YES | NULL for MOO/MKT |
| 15 | `fill_px` | `float64` | NO | actual or simulated |
| 16 | `fill_qty` | `int64` | NO | partials allowed → multiple rows same `client_order_id` differing in `fill_seq` |
| 17 | `fill_seq` | `int32` | NO | **PK component**; `0` if single fill |
| 18 | `ref_px_close_tm1` | `float64` | NO | T-1 close used by signal (gap-PnL attribution; `backtesting` pitfall #4) |
| 19 | `ref_px_open_t` | `float64` | NO | T open used for fill |
| 20 | `commission` | `float64` | NO | IBKR side only |
| 21 | `exchange_fee` | `float64` | NO | venue pass-through |
| 22 | `reg_fee` | `float64` | NO | SEC §31 + TAF + FINRA |
| 23 | `clearing_fee` | `float64` | NO | NSCC etc. |
| 24 | `slippage_bps_modeled` | `float64` | YES | NULL in LIVE; populated by backtest sim |
| 25 | `slippage_bps_realized` | `float64` | NO | `(fill_px - ref_px_open_t) / ref_px_open_t * 1e4 * side_sign` |
| 26 | `gap_pnl` | `float64` | NO | `(open_t - close_tm1) * qty * side_sign` |
| 27 | `session_pnl` | `float64` | NO | intra-session PnL only |
| 28 | `venue` | `string` | YES | `NYSE` / `NSDQ` / `CME` / `IDEALPRO` |

Plus three audit columns inherited from the `backtesting` ratified schema and required for parity joins:

| # | Column | dtype | Nullable | Notes |
|---|---|---|---|---|
| 29 | `route` | `string` | YES | IBKR `SMART` / `DIRECT` |
| 30 | `account` | `string` | NO | IBKR account id in LIVE; `BACKTEST` literal in sim; `PAPER` in paper mode |
| 31 | `strategy_sha` | `string` | NO | for parity join across live/backtest |

**Authoritative count: 31 columns** (the "28-column" framing in the cycle directive counted the core fill row; columns 29–31 are the audit triple required for the parity contract `backtesting` depends on. Both tentacles' CI runs the same `tests/contracts/test_journal_schema.py` against `execution.journal.SCHEMA`, so drift breaks the build immediately.)

**Primary key:** `(client_order_id, fill_seq, backtest_run_id)` — `backtest_run_id` is part of the PK so the same `client_order_id` can appear in both a live row (PK = `(coid, 0, NULL)`) and N backtest rows (PK = `(coid, 0, run_id_X)`). Enables parity reporting without UID collisions.

**Partitioning:**
- LIVE: `live/trades/dt=YYYY-MM-DD/part-*.parquet`
- PAPER: `paper/trades/dt=YYYY-MM-DD/part-*.parquet`
- Backtest: `runs/<run_id>/trades/fold=YYYY/part-*.parquet`

Same schema across all three → `pyarrow.dataset.dataset([live_path, paper_path, bt_path]).to_table()` works without translation.

**Contract test:** `tests/contracts/test_journal_schema.py` imports `execution.journal.SCHEMA` (a `pyarrow.Schema`), asserts equality with the pinned columns/dtypes/nullability above. Runs in both `execution` and `backtesting` CIs.

### Contract 4 — `ChainMode` propagation (LIVE / PAPER / WHATIF)
`OrderIntent` carries a `ChainMode` enum (defined in `shared-core`, mirrors `risk-management` I7a):

```python
class ChainMode(Enum):
    LIVE   = "live"    # submit to IBKR; write to live/trades/
    PAPER  = "paper"   # do NOT submit to IBKR; simulate fill at next bar open; write to paper/trades/
                       # with backtest_run_id = "PAPER-YYYY-MM-DD"
    WHATIF = "whatif"  # emit nothing to broker; write nothing to journal;
                       # return the planned-order envelope to the caller for inspection
```

Rules:
- `mode` is a required field on every `OrderIntent`. No default. A missing `mode=` raises `TypeError` at the OMS entry point.
- `WHATIF` is a pure function: zero filesystem side-effects, zero IBKR calls. Returns the would-be order envelope for replay-harness / notebook callers.
- `PAPER` runs the full OMS state machine (so PAPER exercises the same code path as LIVE) but the IBKR client is swapped for a simulated-fill stub; the journal write goes to `paper/trades/` with `backtest_run_id="PAPER-{YYYY-MM-DD}"` and `account="PAPER"`.
- `LIVE` is the only mode that opens a real IBKR connection or writes to `live/trades/`. A `mode != LIVE` order can never reach the broker — defense in depth, not just a config flag.
- The per-order HALT + heartbeat check (Contract 2) is enforced in LIVE and PAPER (so PAPER faithfully exercises the gating); skipped in WHATIF (no order ever leaves the process).

### Contract 5 — Client-order-id scheme (unique across restarts AND reproducible from `intent_id`)
`client_order_id` is derived deterministically from the upstream `intent_id`:

```python
client_order_id = ulid_from_seed(
    sha256(f"{intent_id}|{strategy_sha}|{account}|{attempt_n}".encode()).digest()[:16]
)
```

Properties:
- **Unique across process restarts**: derived from inputs that already encode the session date (via `intent_id`, which `risk-management` derives from `decision_id` + symbol); a restart cannot collide with prior orders for the same session.
- **Reproducible**: given the same `intent_id`, `strategy_sha`, `account`, and `attempt_n` (retry counter for network-failure idempotency), the same `client_order_id` is regenerated. This is the **idempotency key** for the IBKR client — a retry after a network drop submits the same `client_order_id`; IBKR rejects the duplicate, we learn we succeeded, no double-fill.
- **Auditable join chain**: `client_order_id` → `intent_id` (via the reverse hash table in the journal manifest) → `decision_id` (via `risk-management`'s diff log) → `signal_row` (via `strategy-engine`'s ledger) → `model_version` (via `ml-research`'s registry). End-to-end traceable; one journal row reconstructs the entire decision lineage.
- **Backtest reproducibility**: backtest sim uses `client_order_id = ulid_from_seed(sha256(f"{run_id}|{decision_id}|{attempt_n}").digest()[:16])` — same construction, different seed inputs. Live `client_order_id`s are never reused by backtest runs because `intent_id` and `run_id` live in disjoint name-spaces.

`ULID` (not `UUID`) for lexicographic-sort = chronological-sort: parquet row-group pruning by `client_order_id` range becomes a date-range query for free.

## Conventions
- The OMS is the single source of truth for positions. No other tentacle maintains parallel position state. `risk-management`'s 09:30 ET snapshot reads from IBKR directly (its own concern); intraday position deltas come from this tentacle's journal.
- All IBKR interactions log raw request + response for audit (passed to `infrastructure`'s audit log).
- Trading client and market-data client are **separate processes**, even though IBKR permits sharing a session. Fault isolation > connection efficiency on a 16 GB Mac mini.
- Kill-switch from `risk-management` (HALT file) and `control-plane` (cancel-all/flatten-all) is honored synchronously — no in-flight order escapes the check.
- The OMS state machine is **persistent**: state lives in `~/macminibot/state/oms/YYYY-MM-DD.jsonl` (append-only event log). A process restart replays the log to reconstruct in-flight orders; reconciliation against IBKR open-orders happens immediately on startup.
- Reconciliation policy on disagreement: **IBKR is authoritative.** If our internal book disagrees with IBKR's `reqPositions` result by >1 share on any symbol, write the disagreement to the journal as a `ReconciliationDelta` row, then HALT (touch `~/macminibot/state/HALT`) and force operator to resolve. This matches `risk-management`'s `POSITION_RECONCILES` check #9.
- No NumPy random seeds or wall-clock-derived order suffixes — `client_order_id` derivation is pure function of inputs (Contract 5).
- All timestamps are int64 ns UTC, matching `market-data` and the journal schema.

## Cross-tentacle gaps spotted (read of sibling CONTEXT files)
- **Snapshot staleness window mismatch (already flagged in `risk-management` I7c.2)**: `strategy-engine/CONTEXT.md` line 66 says snapshot stale at ">10 min"; `risk-management` chain check #2 says "snapshot.ts within today 09:30–09:35 ET" (5-minute window). `execution` is downstream of both, so we inherit whichever is tighter once they reconcile. Action lives in those two tentacles; we don't pick.
- **`replaces` field on `OrderIntent`**: `strategy-engine` Improvement #4 (cancel-and-replace) says `risk-management` emits new `OrderIntent` records with `replaces: prior_intent_id` populated. `risk-management/CONTEXT.md` I4 documents `diff_targets_to_intents` but does not pin the `replaces` field in the `OrderIntent` dataclass surface. `execution` needs the field to be authoritative on which prior `client_order_id` to cancel before placing the replacement. Action: `risk-management` should pin `OrderIntent.replaces: str | None` in its next cycle; for now we treat absence as "no prior intent" and rely on `risk-management` to issue cancels itself (which I4 says it does — we just need the field to know which working order to mark `CANCELED_BY_REPLACE` in our state machine).
- **`Signal` schema field-name drift** (flagged in `strategy-engine` "Cross-tentacle gaps surfaced"): `as_of` vs `as_of_ns`, `float` vs `Decimal`. Doesn't touch `execution` directly but the `decision_id` flows through us into the journal — once `shared-core` ratifies the `Signal` dataclass, our `ts_decision` parse path may need to widen acceptance.
