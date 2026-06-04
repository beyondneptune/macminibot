# risk-management

Pre-trade check chain, position/exposure limits, drawdown circuit breakers, and the kill switch. This tentacle is the single approval gate between `strategy-engine` and `execution` — no `TargetPosition` reaches the broker without passing through here.

## Status
Greenfield. Framework ratified (see `todo.md` for the 10-epic build order). No source yet. Treated as a guardrail layer with its own correctness bar — small, auditable, aggressively tested. Friction is a feature: solo-operator, once-per-day 09:35 ET rebalance, no GUI, no auto-resume on HALT.

## Scope
- Pre-trade check chain (10 checks, fail-fast) gating every order intent before `execution` sees it.
- Account-state snapshot at 09:30 ET (NetLiq, AvailableFunds, GrossPositionValue, positions) — deterministic input to the chain, not live broker calls.
- Limit catalog (`risk/limits.toml`) — versioned, hot-reload for tightening only, loosening requires `--ack-loosen`.
- Daily P&L tracker + drawdown breakers: -2% daily / -5% weekly / -10% peak-to-trough off 60-day HWM.
- Position-sizing governor: quarter-Kelly default, 15% single-name notional ceiling regardless of Kelly.
- Sanity-and-staleness guards (ML NaN/inf, T-1 bar age, broker-vs-internal reconciliation, NTP clock drift).
- Kill switch: file-flag (`~/macminibot/state/HALT`) + 60s heartbeat (`heartbeat.json`), checked synchronously by `execution`.
- Breach event stream + 09:45 ET daily digest (rejected intents, HALTs, limits >80% utilized).
- Replay harness for limit changes; 30-day paper-trade soak gate before going live.

## Boundaries
- **Owns:** the limit catalog, the pre-trade check chain, kill-switch state, breach-event emitter, account snapshots, equity curve, drawdown state.
- **Consumes from:** `market-data` (marks for sizing and unrealized P&L); IBKR `AccountSummary` + positions (loaded by Epic 2 — this tentacle owns the snapshot); `strategy-engine` (`TargetPosition` intents + an `intents_complete` sentinel); `shared-core` (types).
- **Provides to:** `execution` (synchronous `Approved | Rejected(rule_id, observed, threshold)` + HALT flag); `observability` (breach events, utilization metrics, equity curve); `control-plane` (current limit utilization, HALT state, ack endpoints).
- **Gates (hard contract):** every `TargetPosition` from `strategy-engine` must pass through `check_chain.run(intent, snapshot, mark)`. Single entry point. No bypass paths, even for "manual" trades — manual rebalances submit synthetic intents through the same chain.

## Key Decisions (ratified)
- **In-process with execution, not a separate service.** v1 latency budget is "before 09:35 ET", not microseconds; in-process eliminates an entire class of IPC failure modes.
- **TOML config + JSONL breach log + Parquet equity curve.** No database. Flat files are inspectable with `cat`, version-controllable, and rsync-backupable.
- **Breaches require manual ack to resume.** No auto-resolve when utilization drops. A breach is a "stop and think" event for a solo operator, not a counter.
- **Single account, single strategy in v1.** Per-strategy and per-asset-class limits are explicitly out of scope until Epic 10 ships and second strategy is live.
- **Snapshot is authoritative, not live broker calls.** Every check reads from the 09:30 ET snapshot — if it's stale, HALT. Live calls during the chain run would make the system non-replayable.
- **Reconciliation: broker is authoritative.** If `POSITION_RECONCILES` fails, internal book is wrong; HALT and force operator to re-snapshot before resume.
- **No leverage in v1.** `MAX_GROSS_EXPOSURE` hard ceiling of 1.0 × NetLiq. The check is redundant with cash-only IBKR account settings, intentionally — defense in depth.

## Conventions
- Checks are pure functions of `(intent, snapshot, mark) → Approved | Rejected(rule_id, observed, threshold)`. Trivially unit-testable, golden-file friendly, replayable.
- A breach is a typed event, not just a log line — `{rule_id, observed, threshold, intent_id, ts, snapshot_hash}` — written to `risk/breaches.jsonl` (append-only).
- Kill switch flips a single file flag; `execution` checks `not exists(HALT) and heartbeat_age < 300s` synchronously before every order.
- New limits land with a unit test for both pass and reject paths AND a replay-harness run on the last 30 days of snapshots — refuse to merge if it would have rejected a "should-have-passed" intent.
- Limit-config tightening is hot-reloadable; loosening requires `--ack-loosen` with a diff to stderr and an entry in `halt_audit.log`.
- Filesystem layout:
  ```
  ~/macminibot/state/
    HALT                       # presence = halt
    heartbeat.json             # {pid, ts_unix, version}, written every 60s
    halt_audit.log             # append-only: who, when, reason
  risk/
    limits.toml                # versioned in git
    snapshots/YYYY-MM-DD.json  # 09:30 ET account state
    breaches.jsonl             # append-only event log
    equity_curve.parquet       # daily NetLiq, realized + unrealized P&L
  ```

## Framework Reference

### 1. Pre-trade check chain (run in order, fail-fast, single entry, no bypass)

| # | Check | Formula | Why |
|---|---|---|---|
| 1 | `KILL_SWITCH_ACTIVE` | `not exists(HALT) and heartbeat_age < 300s` | If anything is wrong elsewhere, the operator (or a prior breach) flipped HALT — respect it before doing any work. |
| 2 | `SNAPSHOT_FRESH` | `snapshot.ts within today 09:30–09:35 ET` | A stale snapshot means equity/positions are wrong; sizing decisions on yesterday's equity can blow through limits. |
| 3 | `INTENT_NONZERO` | `abs(target_qty - current_qty) * mark >= $50` | Don't pay commission + slippage on rounding-noise rebalances. Also catches strategy bugs that emit zero-deltas in a loop. |
| 4 | `PRICE_BAND_FATFINGER` | `0.85 * prev_close <= mark <= 1.15 * prev_close` | Bad mark data (split not adjusted, decimal error) sizes orders 10x too large. Reject and HALT — never trade on a suspect price. |
| 5 | `MAX_SINGLE_NAME_NOTIONAL` | `target_notional <= 0.15 * NetLiq` | Single-name blow-up (fraud, halt, gap) capped at 15% of equity = max ~7.5% loss on a 50% gap. Survivable. |
| 6 | `MAX_GROSS_EXPOSURE` | `sum(abs(target_notional)) <= 1.0 * NetLiq` | No leverage in v1. Hard ceiling at 100% gross prevents an off-by-one in sizing from accidentally levering up. |
| 7 | `KELLY_CAP` | `target_notional <= 0.25 * kelly_optimal_notional` | Quarter-Kelly: full Kelly maximizes long-run growth but with brutal drawdowns; fractional Kelly trades a small fraction of growth for dramatically lower vol-of-vol (Thorp 2006). |
| 8 | `DAILY_TURNOVER_CAP` | `sum(abs(delta_notional)) <= 0.5 * NetLiq` | Once-per-day cadence shouldn't churn the book. Hard cap catches a strategy that flips long/short on every signal flip. |
| 9 | `POSITION_RECONCILES` | `|broker_qty - internal_qty| <= 1 share` | If our book disagrees with IBKR, we don't know our real exposure. Reject all intents until reconciled. |
| 10 | `FEATURE_SANITY` | `all features finite and within historical [p0.1, p99.9]` | ML pipelines silently emit NaN/inf or out-of-distribution garbage; never size off a feature you wouldn't have seen in training. |

### 2. Drawdown breakers

| Window | Trigger | Resume |
|---|---|---|
| Daily | -2% of prior-day NetLiq | Manual ack next session |
| Weekly (rolling 5 sessions) | -5% | Manual ack |
| Peak-to-trough off 60-day HWM | -10% | Written 1-pager post-mortem + ack |

Grounding: Kaminski & Lo (2014) show stop-losses add value for strategies with positive serial correlation; the optimal band balances cost-of-missing-rebound vs. cost-of-riding-drawdown. For daily-compounded systematic strategies on liquid US equities, -2% daily / -10% PTT sits in the empirically supported band. Bot wakes once a day, so breakers trigger at session start off prior close — no intraday whipsaw.

### 3. Position-limit philosophy

Daily-compounding Sharpe is destroyed by drawdowns more than it is helped by peak returns: a -20% DD requires +25% to recover, ~3 months of edge at 8% annualized. Right limits are therefore **tighter than peak-return-optimal**. Default fractional Kelly = **0.25** (Thorp 2006; MacLean/Thorp/Ziemba 2010), with a **hard 15% single-name notional ceiling regardless of what Kelly says**. Half-Kelly is the upper bound, only justified with >3 years of OOS edge validation.

### 4. Kill-switch mechanics

- `execution` checks both `HALT` existence AND `heartbeat_age < 300s` synchronously before every order. Either failing = no orders sent.
- `risk_heartbeat.py` runs from `launchd` (more reliable than cron on a Mac mini) every 60s and writes `heartbeat.json`. If it can't write, staleness alone halts trading.
- Drawdown breakers and check-chain failures `touch HALT` and append to `halt_audit.log`.
- Recovery: `risk-mgmt unhalt --reason "<text>"` appends to audit log and removes `HALT`. No GUI, no web dashboard. Friction is the feature.
- No 24/7 on-call: bot only acts once a day at 09:35 ET. If something breaks at 14:00, fix by tomorrow morning.

### 5. Risk-specific failure modes (what no other tentacle catches)

| Failure | Detection | Response |
|---|---|---|
| ML model returns NaN/inf in target_qty | `FEATURE_SANITY` + `INTENT_NONZERO` + type-assert at chain entry | Reject intent, log `ModelOutputInvalid`, HALT if >1 symbol affected |
| IBKR reports a position we don't know about (manual trade, corporate action, stale internal book) | `POSITION_RECONCILES` at snapshot time | HALT; require operator to reconcile and re-snapshot before resume |
| System clock drift vs. NTP > 2s | `risk_heartbeat.py` runs `sntp -t 1 time.apple.com` each tick | HALT; bad clock means bar timestamps are wrong |
| Daily-bar update failed overnight, feature uses T-2 data | `SNAPSHOT_FRESH` + check `max(bar.ts)` per symbol < 36h | Reject intents for stale symbols; HALT if >50% of universe stale |
| Limits-config file edited mid-session loosening a limit | Hash check at chain entry vs. session-start hash; loosening requires `--ack-loosen` | Reject the chain run entirely; require explicit ack |
| Strategy emits the same intent in a tight loop | Dedupe by `(symbol, date)` per session; second emission rejected | Log `DuplicateIntent`, silent reject after first |
| Account equity dropped overnight via withdrawal/deposit, limits now mis-scaled | `NetLiq` delta vs. yesterday > 5% absent P&L explanation | HALT; force operator to ack cash movement |
| `strategy-engine` crashes mid-emission, only half the universe gets intents | Wait for `intents_complete` sentinel from `strategy-engine` before running chain; timeout = HALT | HALT; don't trade a partial book |

---

## Improvements over baseline

These are deliberate strengthenings of the ratified framework, captured here so they aren't lost when epics get implemented.

### I1. Add `BROKER_AVAILABILITY` precheck (new failure mode, not in baseline)

The baseline catches *our* failures and *data* failures but not **IBKR-side** failures that occur between 09:30 (snapshot) and 09:35 (rebalance). Concrete cases:
- TWS/IB Gateway disconnect or reauth prompt during the 5-min window.
- IBKR maintenance/outage (https://www.interactivebrokers.com/en/general/about/system-status.php).
- Margin-requirement change for a held symbol (rare but happens on volatile names; Reg-T eligibility flip).
- Account flag flip (PDT-locked, restricted, funds-on-hold from a recent deposit).

**Detection:** at 09:34 ET, immediately before the chain runs, call `reqAccountSummary` + `reqOpenOrders` and assert: (a) response within 2s, (b) `Cushion > 0.1`, (c) no `AccountReady=False`, (d) account status is `Funded` not `Restricted`. Failure = HALT, do not emit the day's orders.

This sits *before* check #1 because it's not about us — it's about whether IBKR can accept orders at all. Cheaper to skip a day than to send orders into a degraded broker session.

### I2. Reconsider the -2% daily threshold (better citation)

Kaminski & Lo (2014) is the right paper for *whether* stop-losses help, but their empirical work is on monthly/momentum bands, not daily systematic equity strategies. For the specific question "what daily drawdown should halt a daily-compounding strategy," the more rigorous reference is **Frazzini & Pedersen (2014), "Betting Against Beta"** (Section 5, drawdown behavior of leveraged low-beta strategies) combined with **Sortino & van der Meer (1991), "Downside Risk"** (which formalizes the asymmetric loss-vs-recovery cost that motivates the limit in the first place).

Action: at Epic 5, calibrate the -2% threshold against the strategy's actual backtest daily-return distribution — specifically, target a HALT-trigger rate that catches the worst ~1-2% of historical daily returns but not the worst ~10%. A strategy with a tighter daily vol may justify -1.5%; a wider one -2.5%. The point: **the threshold should be derived from the strategy's own distribution, not copied from a paper**. The -2% in the baseline is a starting prior, not a ratified number.

### I3. NAV delta detection must reconcile with `strategy-engine` sizing

The baseline failure-mode row "Account equity dropped overnight via withdrawal/deposit and limits are now mis-scaled" is detected here, but the integration with `strategy-engine` is undefined. Specifically: when this tentacle HALTs on a NAV delta, **does `strategy-engine` know its sizing inputs are wrong?**

Concrete gap: if the operator deposits $20k overnight, `strategy-engine` recomputes `target_qty` using NEW NetLiq, but this tentacle HALTs because the delta is unexplained. The operator acks, resumes, but `strategy-engine` may have already cached the day's intents using the new NetLiq while *this tentacle's snapshot* was taken before the cash hit.

**Fix:** the 09:30 ET snapshot from Epic 2 must be the **single source of truth for NetLiq** that `strategy-engine` reads. `strategy-engine` does NOT independently call IBKR for NetLiq — it reads `risk/snapshots/YYYY-MM-DD.json`. This makes the snapshot the contract between the two tentacles and eliminates the race. Document this in `strategy-engine/CONTEXT.md` as a hard dependency.

### I4. `TargetPosition` → `OrderIntent` diff (cross-tentacle contract)

`strategy-engine` emits a **batch** of `TargetPosition` records (declarative end-state: "I want 100 shares of AAPL, 0 of MSFT"). The pre-trade check chain operates on per-order `OrderIntent` records (imperative delta: "BUY 30 AAPL, SELL 50 MSFT"). The translation lives **inside this tentacle**, not strategy-engine — the diff must be deterministic and replayable, which is this tentacle's invariant.

**Contract:**
```
diff_targets_to_intents(
    targets: list[TargetPosition],   # from strategy-engine, after intents_complete sentinel
    snapshot: AccountSnapshot,       # 09:30 ET, owned here (Epic 2)
) -> list[OrderIntent]
```

Semantics:
- For every `(symbol, target_qty)` in `targets`: `delta = target_qty - snapshot.positions.get(symbol, 0)`. If `|delta| > 0`, emit one `OrderIntent(symbol, side=BUY if delta>0 else SELL, qty=abs(delta))`.
- For every symbol in `snapshot.positions` NOT in `targets`: emit a flatten intent (`SELL` to zero). Missing-from-targets means "exit", never "hold" — guards against strategy-engine silently dropping a symbol.
- Universe mismatch (target references a symbol with no mark in `market-data`) → reject the entire batch, HALT. Don't half-trade.
- The diff is **pre-chain**: each emitted `OrderIntent` is then passed through `check_chain.run(intent, snapshot, mark)`. Chain rejections are per-intent; the batch as a whole succeeds only if all 10 checks pass for every emitted intent. Partial-batch execution is forbidden (atomicity — see failure-mode row 8).

This is owned by **Epic 3** (chain entry point); make the diff function the public API alongside `check_chain.run`.

### I5. Execution's synchronous HALT + heartbeat contract (what `execution` MUST honor)

This tentacle cannot enforce execution behavior — but it specifies the contract execution must meet, and it provides the artifacts (`HALT` file, `heartbeat.json`) that make compliance trivial.

**REQUIREMENT on `execution` (non-negotiable, document in `execution/CONTEXT.md`):**

Before placing **every** order (not once per batch, not once per session — every order), `execution` MUST:
1. `stat ~/macminibot/state/HALT` — if it exists, abort the order and emit `OrderRefused(reason="HALT")`.
2. Read `~/macminibot/state/heartbeat.json`, parse `ts_unix`, compare to `time.time()`. If `now - ts_unix > 300`, abort and emit `OrderRefused(reason="HEARTBEAT_STALE", age=<seconds>)`.
3. Both checks are synchronous, blocking, and on the order-submission code path. No caching, no "checked 10s ago", no async.
4. Failure to perform these checks is a P0 bug in `execution`, not a configuration option. There is no `--skip-halt-check` flag and never will be.

Why per-order, not per-batch: a HALT triggered mid-batch (e.g., drawdown breaker tripping on the unrealized P&L from the first fill) must stop subsequent orders. The 5-minute heartbeat staleness window is intentionally generous — `risk_heartbeat.py` writes every 60s, so 5 minutes = 5 consecutive write failures = something is genuinely wrong.

This is the *only* hard cross-tentacle contract this tentacle imposes. Every other interaction is read-only file consumption.

### I6. Unhalt cooldown (prevent oncall trigger-finger)

Baseline allows immediate unhalt: `risk-mgmt unhalt --reason "..."`. For a solo operator at 09:35 ET who just saw a HALT, the temptation to type a one-line reason and resume in 90 seconds is real — and exactly when a post-mortem would have caught the root cause.

**Rule:** unhalt is rate-limited by HALT severity:

| HALT cause | Minimum cooldown before unhalt accepted |
|---|---|
| `KILL_SWITCH_MANUAL` (operator `touch HALT`) | 0 (operator-controlled, trust them) |
| Single-intent reject (e.g., `PRICE_BAND_FATFINGER`, `FEATURE_SANITY`) | 1 hour |
| Drawdown breaker, daily (-2%) | 24 hours (next session) |
| Drawdown breaker, weekly (-5%) | 72 hours |
| Drawdown breaker, peak-to-trough (-10%) | 168 hours (1 week) + written post-mortem committed to `risk/postmortems/YYYY-MM-DD.md` |
| `POSITION_RECONCILES` failure | 0 (operator must re-snapshot; the act of fixing the book is the cooldown) |
| `BROKER_AVAILABILITY` failure | 1 hour (let IBKR's status settle; transient disconnects often re-trip) |

Implementation: `halt_audit.log` records the rule_id and timestamp of the HALT; `risk-mgmt unhalt` reads the latest HALT entry, looks up the cooldown, and rejects with `CooldownNotElapsed(remaining_seconds=N)` if too soon. Override exists (`--force --reason "..."`) but writes a loud `UNHALT_FORCE_OVERRIDE` event to `breaches.jsonl` for the daily digest — visibility, not prevention.

Cost of false positives (waiting an hour you didn't need to) is one missed session at 09:35 ET. Cost of false negatives (resuming into a broken system) is unbounded. Asymmetric, so default toward friction.

### I7a. `WHATIF` mode flag — backtest/replay must not touch prod state

Cycle 2 flagged: the same `check_chain.run` is used by Epic 9 replay harness, Epic 10 paper soak, and live trading. Today nothing prevents a replay run from appending to `risk/breaches.jsonl`, touching `~/macminibot/state/HALT`, or writing to `halt_audit.log`. A backtest sweep over 5 years of snapshots could trip the live drawdown breaker and HALT the production bot mid-day — a catastrophic cross-contamination.

**Contract:** every entry point that runs the chain (`check_chain.run`, `diff_targets_to_intents`, drawdown evaluator, NAV-delta detector) takes a required `mode: ChainMode` argument:

```
class ChainMode(Enum):
    LIVE      = "live"        # writes breaches.jsonl, can touch HALT, updates equity_curve
    PAPER     = "paper"       # writes paper-suffixed sidecars; touches state/PAPER_HALT (separate file)
    WHATIF    = "whatif"      # pure function; ZERO filesystem side-effects beyond caller-provided sinks
```

Rules:
- `WHATIF` returns the `Approved | Rejected` decisions and a `BreachEvent[]` list to the caller; the chain itself opens no files for write. Caller (replay harness, notebook) decides what to do with the list.
- `PAPER` writes to `risk/paper/breaches.jsonl`, `state/PAPER_HALT`, `risk/paper/equity_curve.parquet` — fully parallel filesystem, never collides with live paths.
- `LIVE` is the only mode that can flip the real `HALT` flag. A `mode != LIVE` chain run that triggers what would be a HALT in live merely tags the returned breach with `would_halt=true`.
- Default value: there is no default. Every callsite picks explicitly; a missing `mode=` is a type error. Removes the "I forgot to set the flag" footgun.
- `mode` is logged inside every `BreachEvent` (`mode: "live" | "paper" | "whatif"`) so the daily digest can filter and so post-mortem analysis can never confuse a replay event with a real one.

This is owned by **Epic 3** (chain entry point) at construction, not bolted on later. Epic 9 (replay) and Epic 10 (paper soak) consume it; Epic 3's signature must accept it on day one.

### I7b. `intents_complete` is the batch envelope, not a sentinel file

Cycle 2 flagged "how does `strategy-engine` signal done emitting?" — the answer changed in cycle 2 when both tentacles ratified `TargetPositionBatch` as the wire format. **There is no separate sentinel.** The batch IS the sentinel: a single in-memory `TargetPositionBatch` object (or its JSON serialization at `risk/inbox/YYYY-MM-DD.json`) is delivered atomically. Partial batches are structurally impossible — you have the batch or you don't.

**Contract update (supersedes failure-mode row 8's "wait for `intents_complete` sentinel" wording):**
- `strategy-engine` writes `risk/inbox/YYYY-MM-DD.json.tmp`, fsyncs, then atomic-renames to `risk/inbox/YYYY-MM-DD.json`. The atomic rename is the "complete" signal — POSIX guarantees no reader sees a half-written file.
- This tentacle polls `risk/inbox/` from 09:35 ET; if no batch by 09:36 ET → HALT with `rule_id=STRATEGY_INTENTS_TIMEOUT`.
- Batch schema (mirrors `strategy-engine/CONTEXT.md` line 50) includes `batch_id`, `as_of_ns`, `nav_snapshot`, `nav_snapshot_source`, `targets: list[TargetPosition]`, AND a `targets_count: int` checksum field. Loader asserts `len(targets) == targets_count` — guards against fsync-but-no-rename race on exotic filesystems.
- Dedupe: this tentacle records `processed_batches` set keyed by `batch_id`. Replaying the same batch is a no-op idempotent ack (returns the original decisions); a NEW `batch_id` for the same date raises `DuplicateRebalanceAttempt` and HALTs.

Action: update failure-mode row 8 in this CONTEXT.md and in `risk-management/todo.md` Epic 13 to reference the batch envelope, not a sentinel file. Drop the "`intents_complete` sentinel" wording — it's stale terminology from cycle 1.

### I7c. Cross-tentacle inconsistency: `execution` consumes via risk, not direct from strategy-engine

Reading sibling CONTEXT.md files surfaced two real contradictions to flag (action: open a cross-tentacle PR to fix the inconsistent side, do NOT silently change either file from this tentacle):

1. **`execution/CONTEXT.md` line 18** says `execution` "Consumes from: `strategy-engine` (intents)". **`strategy-engine/CONTEXT.md` line 32** explicitly says "`execution` does NOT receive directly from strategy-engine. execution receives `OrderIntent` batches from risk-management". Execution's CONTEXT is stale (cycle 1 terminology). The ratified flow is `strategy-engine → risk-management (diff + chain) → execution`. Fix: edit `execution/CONTEXT.md` to read "Consumes from: `risk-management` (vetted `OrderIntent` batches post-chain)".
2. **Snapshot staleness window mismatch.** `strategy-engine/CONTEXT.md` line 66 says missing/stale snapshot is ">10 min old → halt rebalance". `risk-management/CONTEXT.md` chain check #2 (`SNAPSHOT_FRESH`) says "snapshot.ts within today 09:30–09:35 ET" — a 5-minute window. If the snapshot lands at 09:30:00 ET and the chain runs at 09:39:00 ET, strategy-engine would proceed but risk-management would reject. Fix: ratify ONE window (recommend the tighter risk-side bound: `snapshot.ts within [09:30:00, 09:35:00] ET` AND `now - snapshot.ts < 10 min` — both must hold) and align both files.

These belong in `I7c` because the contract this tentacle exposes is only as strong as the sibling tentacles' adherence. Audit annually as part of the limit-config review.

### I7d. Per-symbol `limits.toml` overrides — concrete schema (cycle 3 deferred → ratified)

Epic 4 says "per-symbol and account-wide limits" but never specified the file layout. Without a schema, callers will invent their own and the hot-reload tightening-only invariant becomes unenforceable. Ratify it now so Epic 4 implements once, not twice.

**Schema (TOML, hot-reload safe):**

```toml
# risk/limits.toml — single file, versioned in git, hashed at session start
schema_version = 2          # bump on breaking change; loader refuses unknown versions
config_id      = "2026-05-17.a"  # opaque human-meaningful tag, recorded in every breach

[account]
# Account-wide ceilings. These are HARD; per-symbol overrides may only tighten, never loosen.
max_single_name_pct      = 0.15   # check #5; 15% of NetLiq
max_gross_exposure_pct   = 1.00   # check #6; no leverage in v1
max_daily_turnover_pct   = 0.50   # check #8
kelly_fraction           = 0.25   # check #7 default
fatfinger_band_pct       = 0.15   # check #4; ±15% of prev_close
min_intent_notional_usd  = 50.00  # check #3
dd_daily_pct             = -0.02  # breaker
dd_weekly_pct            = -0.05
dd_ptt_pct               = -0.10

[defaults.symbol]
# Default per-symbol envelope (applied when a symbol is in the universe but not listed below).
max_notional_pct         = 0.15   # cannot exceed account.max_single_name_pct
kelly_fraction           = 0.25
fatfinger_band_pct       = 0.15

# Per-symbol overrides. ONLY tighter values are accepted; loosening fails validation.
# Schema: each table is [symbols."<TICKER>"] with any subset of the defaults.symbol keys.

[symbols.TSLA]
# Single-name vol much higher than mean; cap notional + tighten fat-finger band.
max_notional_pct   = 0.05    # 5% vs 15% default — survivable on a 50% gap
kelly_fraction     = 0.15    # cut Kelly fraction; the edge is noisier here
fatfinger_band_pct = 0.25    # WIDER band is a LOOSEN — refused by loader

[symbols.AAPL]
max_notional_pct   = 0.10    # tighter than default, looser than TSLA
# kelly_fraction inherits 0.25, fatfinger inherits 0.15

[symbols.SPY]
# Index ETF: cap higher would be a loosen; instead document parity by repeating default.
max_notional_pct   = 0.15
kelly_fraction     = 0.25
```

**Loader invariants (enforced at load time, before any chain run uses the config):**
1. `schema_version` must equal a value the loader knows; unknown → refuse to load (HALT, `rule_id=LIMITS_SCHEMA_UNKNOWN`).
2. Every override key must exist in `defaults.symbol`; unknown keys are typos and refuse the load.
3. For each `[symbols.X]` value `v` and matching `defaults.symbol.d`: `v <= d` for "ceiling" keys (`max_notional_pct`, `kelly_fraction`, `fatfinger_band_pct`). A `>` is a loosen and requires `--ack-loosen` per existing rule.
4. Account ceilings are the absolute outer bound: `symbols.X.max_notional_pct <= account.max_single_name_pct` always.
5. Hot reload re-runs all 4 checks against the new file; failure leaves the previous config active and emits `LIMITS_RELOAD_REJECTED` to `breaches.jsonl`.
6. `config_id` is appended to every `BreachEvent` and `OrderRefused` record so post-mortems can correlate behavior with the active limit set.

Universe-mismatch rule: if `[symbols.X]` exists but `X` is not in the day's universe, that's a soft warning (logged once at load, not a HALT) — common during ticker delisting. If `X` IS in the universe but missing from `[symbols.*]`, `defaults.symbol` applies (no warning).

Action: ratify in Epic 4; close the cycle-3 deferred item.

### I7e. Risk metrics emission to observability — concrete metric schema

Epic 8 covers the **digest** (human-readable end-of-day email). It doesn't cover **live metric emission** — utilization-over-time, breach rates, cooldown remaining — which observability needs to dashboard and alert on. Defining the schema here lets Epic 8 emit and observability consume against a stable contract.

Follows `observability/CONTEXT.md` line 29 naming: `bot_<dept>_<noun>_<unit>`. All metrics carry the labels `mode` (live|paper|whatif), `config_id` (from I7d), and where applicable `rule_id`, `symbol`.

| Metric | Type | Labels | When emitted |
|---|---|---|---|
| `bot_risk_chain_runs_total` | counter | `mode`, `outcome=approved|rejected` | every `check_chain.run` invocation |
| `bot_risk_check_rejections_total` | counter | `mode`, `rule_id`, `symbol` | every per-check rejection (one chain run can emit zero or one — fail-fast) |
| `bot_risk_intent_notional_usd` | histogram | `mode`, `symbol`, `side` | every intent that reaches the chain, regardless of outcome |
| `bot_risk_limit_utilization_ratio` | gauge | `rule_id`, `scope=account|symbol`, `symbol` | refreshed each chain run; value in [0,1+], values >0.8 already alertable per Epic 8 |
| `bot_risk_halt_active` | gauge | `mode` | 1 if `HALT` (or `PAPER_HALT`) exists, 0 otherwise; sampled by heartbeat |
| `bot_risk_halt_total` | counter | `rule_id` | incremented when a HALT is created (not when re-sampled) |
| `bot_risk_unhalt_cooldown_remaining_seconds` | gauge | `rule_id` | nonzero only when a HALT is active; drives "can the operator resume yet" alerts |
| `bot_risk_heartbeat_age_seconds` | gauge | (none) | written by heartbeat process itself; SLO breach at >120s |
| `bot_risk_drawdown_pct` | gauge | `window=daily|weekly|ptt` | refreshed at session start from `equity_curve.parquet` |
| `bot_risk_snapshot_age_seconds` | gauge | (none) | now - snapshot.ts; alert if >600s during a chain run |
| `bot_risk_config_reload_total` | counter | `outcome=accepted|rejected_loosen|rejected_invalid` | every `limits.toml` reload attempt |

Wire format: emit via the `observability` metrics library (no direct Prometheus dependency in this tentacle); fallback when observability is unavailable is to append a `MetricSample` JSON line to `risk/metrics.jsonl` so post-hoc replay is possible. The gauge sampler runs from `risk_heartbeat.py` (Epic 1) so adding metrics doesn't add a new process.

Alerts (proposed, owned by observability but specified here):
- `bot_risk_halt_active{mode="live"} == 1` for >5m during 09:30-16:00 ET → page operator.
- `bot_risk_limit_utilization_ratio > 0.95` for any `rule_id` → ticket-level alert.
- `bot_risk_heartbeat_age_seconds > 300` → page (same threshold execution uses to refuse orders).
- `rate(bot_risk_check_rejections_total{rule_id="PRICE_BAND_FATFINGER"}[1h]) > 0` → page (a fat-finger reject means we almost traded on bad data).

Action: list under Epic 8 deliverables; add a sub-bullet to Epic 16 ensuring the `mode` label is set from `ChainMode` at the emission site.

### I7f. Replay harness embargo — time-travelling the limits.toml correctly

Cycle 1 Epic 9 says "replay the check chain" but is silent on **which `limits.toml` the replay uses**. The implicit default (read the current working file) is dangerously wrong: a backtest of "what would yesterday's orders have done?" must use yesterday's limits, not today's tightened post-mortem version. Without this, paper-mode soak runs (Epic 10) produce results that disagree with prod for reasons that look like bugs.

**Embargo rule:** every chain run in `PAPER` or `WHATIF` mode against a historical snapshot MUST resolve `limits.toml` to the version that was in force at the snapshot's timestamp. Three mechanisms:

1. **Per-snapshot config pinning (already half-built).** Epic 4 says "every snapshot records the limit-config hash." Strengthen: snapshots also record `config_id` (from I7d) AND a copy of the resolved limit values inline under `snapshot.limits_resolved`. The inline copy is the source of truth for replay — git history is the audit trail, the snapshot is the runtime input. Survives `limits.toml` git history rewrites, file deletions, schema migrations.

2. **Explicit `--as-of` semantics in the replay CLI.** `risk-mgmt replay --snapshot 2026-03-12 --as-of snapshot` (default) reads `snapshot.limits_resolved`. `--as-of head` uses today's `limits.toml` (only useful for "would today's stricter limits have caught yesterday's bad fill?" analyses; loud warning printed). `--as-of <git-sha>` reads `limits.toml` at that commit. No silent fallback — missing or ambiguous `--as-of` is a CLI error.

3. **Time-travel guards on derived state.** Drawdown thresholds and Kelly fractions are config-derived. Replay must use the *snapshot's* `equity_curve.parquet` slice (rows up to and including the snapshot date), not today's full curve. Equivalent rule for the 60-day HWM: compute from `equity_curve[date <= snapshot.date].tail(60)`, never from `equity_curve.tail(60)` at replay time.

4. **Paper soak (Epic 10) specifically:** the 30-day paper run must replay each day with `--as-of snapshot` and ALSO record a `--as-of head` shadow run; if the two disagree on any decision, surface in the daily digest as `LimitDriftDetected`. That's the signal that a limit change in the middle of the soak invalidated earlier soak days — restart the clock or accept the regression.

Concretely: `snapshot.json` schema gains:
```
{
  ...,
  "config_id": "2026-05-17.a",
  "limits_hash": "sha256:...",
  "limits_resolved": { /* full materialized limit set, no inheritance */ }
}
```

The materialized form (with per-symbol overrides already resolved against defaults) means the replay loader needs zero knowledge of the override-inheritance rules — they were resolved at snapshot time. Snapshots become trivially diff-able when limits change.

Action: extend Epic 2 (snapshot loader) to write `limits_resolved`; extend Epic 9 (replay) with the `--as-of` flag and embargo enforcement; extend Epic 10 acceptance with the shadow-run divergence check.

### I9. Kill-switch propagation latency budget (cycle 5)

The I5 contract requires `execution` to `stat HALT` and parse `heartbeat.json` **per order**. With ~10 contracts/session and 5 min between snapshot and first fill, propagation latency isn't a hot path — but it IS a correctness budget. Two numbers to ratify so the contract is testable, not vibes:

- **Per-order check overhead:** `stat(HALT) + read(heartbeat.json) + json.loads` must complete in **<5 ms p99** on the Mac mini SSD. Measured floor: `stat` ~50 us, 80-byte JSON read+parse ~300 us. Budget gives 10x headroom for FS cache misses and Python GIL hiccups. Test: `bench_killswitch.py` runs 10k iterations cold-cache, asserts p99 < 5 ms. Belongs in Epic 14 acceptance.
- **End-to-end HALT propagation:** time from `touch HALT` (operator or breaker) to the first order `execution` refuses must be **<= max(per-order-check-cadence, 1s)**. Since `execution` checks per order and orders are submitted sequentially, the worst case is "HALT touched after order N is in flight to IBKR" — order N may execute, N+1 will not. This is the **one-order-leak** budget we explicitly accept: a HALT mid-batch loses at most one in-flight order. Document this loud: it's not a bug, it's the v1 spec. Tighter (cancel-in-flight) requires `placeOrder` cancellation handling, deferred to v2.
- **Heartbeat-stale latency:** `risk_heartbeat.py` writes every 60s; `execution` refuses at age > 300s. So a heartbeat-process crash takes 240-300s to halt trading (300s ceiling minus up-to-60s since last successful write). Acceptable because the bot only fires at 09:35 ET — a heartbeat crash at 09:30 still halts well before any order. Test: kill the heartbeat process at T=0, assert order placement at T=301s raises `OrderRefused(reason="HEARTBEAT_STALE")`.

Action: add the three numeric assertions to Epic 14 (`risk-mgmt verify-execution-contract` CLI). Reject any execution impl that fails them.

### I10. Breach severity tiers (cycle 5)

Today every `BreachEvent` is flat — the only distinction between "rejected one intent for being under $50" and "drawdown breaker tripped" is the `rule_id`. That couples downstream consumers (digest, metrics, cooldown lookup, paging) to a hard-coded `rule_id → severity` map maintained in N places. Ratify a single `severity` field on `BreachEvent` so the routing is one lookup at emission, not N at consumption.

**Tiers:**

| Tier | Meaning | Filesystem side-effect | Digest section | Pager |
|---|---|---|---|---|
| `INFO` | Silent reject, expected operationally | breaches.jsonl line | "rejects" rollup count only | no |
| `WARN` | Reject worth a glance, but no HALT | breaches.jsonl line | "rejects" itemized | no |
| `HALT` | Reject + HALT flag flipped | breaches.jsonl + HALT + halt_audit.log | "halts" section, top of digest | no (operator sees at 09:35 ET) |
| `CRITICAL` | HALT + immediate page | all of HALT + pager-out-of-band | "halts" section + RED banner | YES |

**Rule → tier mapping (initial; lives in `risk/severity_map.toml` per-rule, hot-reloadable):**

| `rule_id` | Tier | Why |
|---|---|---|
| `DUPLICATE_INTENT` | INFO | Strategy bug to fix tomorrow, no risk today |
| `INTENT_NONZERO` | INFO | Rounding-noise rebalance, expected |
| `KELLY_CAP` | WARN | Sized down, day continues |
| `MAX_SINGLE_NAME_NOTIONAL` | WARN | Sized down or rejected; day may continue per intent |
| `DAILY_TURNOVER_CAP` | WARN | Batch capped; day continues |
| `MAX_GROSS_EXPOSURE` | HALT | Implies a sizing bug — investigate before another batch |
| `SNAPSHOT_FRESH` | HALT | Stale equity → wrong sizing |
| `POSITION_RECONCILES` | HALT | Book wrong; re-snapshot required |
| `FEATURE_SANITY` | HALT (multi-symbol) / WARN (single-symbol) | NaN/inf in pipeline |
| `PRICE_BAND_FATFINGER` | CRITICAL | Almost traded on a 10x-wrong price; operator wakes for this |
| `BROKER_AVAILABILITY` | HALT | IBKR side; skip the day |
| `STRATEGY_INTENTS_TIMEOUT` | HALT | strategy-engine didn't deliver |
| `LIMITS_RELOAD_REJECTED` | WARN | Prior config still active; loud but not blocking |
| `LIMITS_SCHEMA_UNKNOWN` | CRITICAL | Loader broken; can't enforce anything |
| Drawdown daily -2% | HALT | Standard breaker |
| Drawdown weekly -5% | HALT | Standard breaker |
| Drawdown PTT -10% | CRITICAL | Post-mortem mandated; operator must engage immediately |
| `UNHALT_FORCE_OVERRIDE` | WARN | Visibility, not prevention (per I6) |
| `LimitDriftDetected` | INFO | Soak-mode signal only |

**Implementation:**
- `BreachEvent` gains a required `severity: str` field ("info" | "warn" | "halt" | "critical").
- `make_breach()` looks up the tier from `risk/severity_map.toml` at construction; unknown `rule_id` defaults to `WARN` and emits a meta-breach `UNMAPPED_SEVERITY` (so adding a new rule without a tier entry is loud).
- The cooldown table in I6 keys off `severity` AND `rule_id` — `CRITICAL` rules get the longer of the two cooldowns; `INFO`/`WARN` skip cooldown entirely (no HALT, nothing to unhalt).
- Metric `bot_risk_check_rejections_total` gains a `severity` label so the I7e alert "fat-finger reject -> page" generalizes to `severity="critical" -> page`, removing one hard-coded rule_id from the alert config.
- Pager integration is out-of-scope for this tentacle: `severity=critical` events MUST be tagged in `breaches.jsonl` so `observability` can subscribe; no direct pager call from risk.

Action: extend Epic 8 with `severity_map.toml` loader and tier field; extend Epic 21 metrics with the `severity` label.

### I11. Module layout drift (cycle 5)

`__init__.py` imports `Approved, ChainMode, Rejected, run_check_chain, diff_targets_to_intents` from `risk_management.chain`, but the codebase only has `risk_management.diff` (with `diff_targets_to_intents` and dataclasses). `chain.py` is missing — package import fails today. Ratify the module split so implementation lands in the right place:

- `risk_management.chain` — public entry points: `run_check_chain`, `diff_targets_to_intents`, the `ChainMode` enum, `Approved` / `Rejected` result types. This is the single import surface for `execution` and `strategy-engine`.
- `risk_management.diff` — internal implementation detail. Keep `OrderIntent`, `TargetPosition`, `TargetPositionBatch`, `Side`, `MarkMissing`, `BatchChecksumError`, `DiffResult` here, but re-export `diff_targets_to_intents` through `chain` so callers never import `risk_management.diff` directly. Treat `diff.py` as a `_diff` private module in spirit (don't rename mid-flight; instead document the import boundary).
- Drift fix: also reconcile `DiffResult` vs. the current `diff_targets_to_intents` return type. The signature says `-> list[OrderIntent]`, but I4's atomic-batch contract requires a way to signal "batch rejected, all-or-nothing." Either return `DiffResult` (carries `halt_reason`/`halt_rule_id`) OR keep raising `MarkMissing` and document raise-on-batch-reject as the contract. Pick ONE before Epic 13 lands; the current half-and-half (DiffResult defined but unused) is the worst of both.

Action: when Epic 3/13 implementation lands `chain.py`, the PR description must include (a) which return contract was picked and (b) a one-line update to this section deleting the "pick one" choice.

### I12. Local domain types drift from `shared_core.contracts` (cycle 6)

`risk_management.diff` defines local `OrderIntent`, `TargetPosition`, `TargetPositionBatch`, `Side` AND `risk_management.chain` defines a local `ChainMode` + `risk_management.breaches` defines a local `BreachEvent` — but `shared_core.contracts` already exports canonical versions of every one of these (T3, T7, T8, T9, T11, T12) with stricter invariants: `Symbol`-typed (not `str`) symbol fields, `slots=True` for memory, ratified KD1 `_decimal_guard`, and a different `OrderIntent` field set (`intent_id, symbol: Symbol, side, delta_qty, mark, batch_id, decision_id` vs. local's `symbol: str, side, qty, target_qty, current_qty, ...`). The local types pre-date `shared_core` and are now schema-divergent — `execution` and `strategy-engine` cannot consume `risk_management.diff.OrderIntent` and `shared_core.contracts.OrderIntent` interchangeably. This is a cross-tentacle break, not a local cleanup.

**Action:** fold into Epic 18 (CONTEXT-reconciliation PR). The reconciliation must (a) pick ONE schema for `OrderIntent` — the `shared_core` version is the published contract, so this tentacle should adopt it and add the `current_qty`/`target_qty` context as either chain-context fields or a sibling `IntentContext` dataclass, (b) re-export `Side`/`ChainMode`/`BreachEvent`/`TargetPosition*` from `shared_core.contracts` through `risk_management` so existing call sites keep working during migration, (c) delete `risk_management.diff.{OrderIntent, TargetPosition, TargetPositionBatch, Side}` after callers cut over. Cycle 5's `chain.py` landing (Epic 25 partial) is verified — package imports clean and 52 tests pass; this is the next blocking integration item, NOT a same-cycle fix.

### I13. Cycle-8 landings — calibration, divergence gate, metrics emitter

Three modules landed in cycle 8 to convert previously hand-waved priors into
data-driven inputs and to pre-wire observability:

1. **`risk_management.calibration`** (`calibrate_drawdown_thresholds`) — Closes
   the I2 calibration loop. Takes a daily-return distribution (the cycle-7
   Sharpe +2.03 baseline's concatenated fold returns) and emits
   `(dd_daily_pct, dd_weekly_pct, dd_ptt_pct)` quantized to a 1 bp grid, with
   a `rationale` string the operator pastes into the limits.toml commit
   message. Default target catches the worst ~1.5% of historical days
   (CONTEXT I2's "1-2% not 10%" guidance). Skew adjustment per Sortino & van
   der Meer 1991 — a negative left tail tightens, never loosens. Floor /
   ceiling clamps keep low-variance strategies from hair-trigger HALTs and
   stop a backtest with an anomalous tail from loosening past the operator's
   risk appetite. Pure stdlib — no numpy dep in the risk-management package.
   Recommended wiring: invoked once per backtest promotion, output reviewed
   against existing `limits.toml`, change goes through the normal
   tighten-only hot-reload (or `--ack-loosen` if calibration suggests
   loosening — which is the right place to make that decision deliberately).

2. **`risk_management.live_divergence`** (`evaluate_live_divergence`) — The
   "suspiciously high backtest" pre-trade gate. Computes a z-score of trailing
   live returns vs. the backtest distribution and emits a verdict with
   `breach=True` when one-sided downside z exceeds `sigma_threshold` (default
   3.0). One-sided by default because a lucky day shouldn't train the operator
   to ack every divergence; over-performance is a `backtesting` concern, not a
   chain concern. Returns a `LiveDivergenceVerdict` with `as_breach_kwargs()`
   that plugs directly into `risk_management.breaches.make_breach`. New
   `rule_id="LIVE_BACKTEST_DIVERGENCE"` (HALT severity by default per I10
   tiers). Insufficient-data path emits a distinct rule_id rather than silently
   passing — observability sees the soak window is still warming up. Wiring
   target: Epic 11 (09:34 ET broker-availability preflight) becomes the
   natural home — same pre-chain runtime slot.

3. **`risk_management.metrics_emitter`** (`MetricsSampler`, `new_sampler`,
   `SCHEMA`) — Closes I7e by frozen-schema'ing all 11 metric names + types +
   required-label sets and providing the JSONL fallback specified there.
   Sampler instances are pinned to `(mode, config_id)` so every emitted
   sample auto-carries the labels required by every metric — no per-call
   bookkeeping. Mode dispatch matches I7a: LIVE -> `risk/metrics.jsonl`,
   PAPER -> `risk/paper/metrics.jsonl`, WHATIF -> in-memory only. The
   schema is the contract surface observability writes against; samples in
   the JSONL are guaranteed shape `{name, value, labels, ts_unix, type}`.
   Unknown metric names raise loudly (typo guard); missing required labels
   raise (call-site bug). Unknown extra labels pass through (ad-hoc
   dashboards). When observability's metrics library lands, it drains
   `risk/metrics.jsonl` or replaces the sink in `_sink_path_for` — the
   schema itself doesn't change.

Test coverage: 40 new tests (10 calibration + 13 divergence + 17 metrics),
all passing. Total risk-management suite is now 92 tests; the 8 pre-existing
test_diff.py failures are unrelated cycle-7 drift from
`shared_core.TargetPositionBatch` gaining `decision_id`/`mode`/`solver_status`
fields — out of scope here, file an Epic 26 follow-up to refresh those
fixtures.

Action: thread `new_sampler(mode, limits.config_id)` through `run_check_chain`
in a future cycle so the existing chain start/finish path automatically emits
`bot_risk_chain_runs_total{outcome=...}` and `bot_risk_check_rejections_total`.
Today the sampler is plumbed but the chain itself doesn't yet call it — the
schema/sink is what was load-bearing for unblocking observability work.

### I14. VIX regime gate — soft vs. hard split (cycles 10 + 11)

Cycle 10 landed two complementary mechanisms for the cycle-9 finding
that Ridge bleeds Sharpe -0.252 in the top-quintile (``vix_high``)
VIX-proxy bucket. They look like they overlap at the 80th-percentile
boundary; they don't. Document the split so future cycles don't try to
"deduplicate" them into a single switch:

  +--------------+--------------------------+----------------------------+
  | Layer        | Strategy-engine          | Risk-management            |
  | Module       | ``sizing.py``            | ``regime_gate.py``         |
  | Mechanism    | Soft dampening of target | Hard ``VIX_HIGH_REGIME_*`` |
  |              | vol; weights -> 0 at 80+ | breach + LIVE-only HALT    |
  | Trigger band | 60-80 pct: half-vol      | 80+ pct: triggered=True    |
  |              | 80+ pct: zero-vol        | LIVE -> halt=True          |
  | Failure mode | Sized-down or empty book | HALT (LIVE) / audit (PAP)  |
  +--------------+--------------------------+----------------------------+

**Why both:** soft dampening leans the book off risk one tier early
(60-80 band) so we don't whipsaw at the 80th-pct boundary. At 80+ the
soft layer emits zero weights, which makes the diff step naturally
produce zero ``OrderIntent``s — so the chain doesn't run any per-intent
checks, and there's nothing for ``execution`` to submit. The hard
``VixHighGate`` runs in parallel and writes a ``BreachEvent`` to
``breaches.jsonl`` with ``rule_id=VIX_HIGH_REGIME_HALT`` regardless.

**Why this is NOT double-blocking:** the two layers operate on
different objects. Soft dampening zeroes out **weights** (= no orders
are ever proposed). The hard gate is a **breach** + an explicit
``halt=True`` signal the scheduler reads. When weights are zero AND
the gate triggers, the scheduler sees zero orders to submit AND
``decision.halt is True``. Both observations agree; neither suppresses
the other. The audit trail is preserved in ``breaches.jsonl`` even
when no order was at risk of being submitted — that's the point:
post-mortem analysis ("did we trade on a vix_high day in cycle 11?")
reads the breach stream, not the absence-of-orders.

**Paper-mode metric emission (cycle 11 add):** ``VixHighGate.evaluate``
now accepts an optional ``sampler: MetricsSampler``. PAPER-mode
callers thread the sampler in so every triggered evaluation emits
``bot_risk_halt_total{rule_id="VIX_HIGH_REGIME_HALT"}`` to
``risk/paper/metrics.jsonl``. This preserves the OOS gate-firing
signal alongside the breach record. Without metric emission in PAPER,
the I7e dashboards would show zero VIX-halt activity during a paper
soak (since paper never flips the live ``HALT`` flag) and an analyst
validating the cycle-9 finding would have to cross-join breaches
against equity-curve days. The metric makes the data first-class.

**Action items already closed in cycle 11:**
- ``regime_gate.evaluate(..., sampler=...)`` plumbed; emission failures
  swallowed (matches ``chain._emit_chain_metrics`` semantics).
- ``tests/risk_management/test_regime_gate.py`` covers the zero-weight
  + audit-trail case (breach recorded; no order submission path
  exercised) AND the PAPER-mode metric flush.

**When/whether to drop one layer:** if a future model handles
``vix_high`` cleanly (cycle-9 follow-up retraining), the right
sequence is (a) raise the hard-gate percentile threshold first
(95+ becomes the "almost never" floor), (b) keep soft dampening as a
permanent vol-target governor — it's strategy-side risk control, not
regime gating. The hard gate is the safety net; the soft dampener is
the steering wheel. Removing the soft layer would re-introduce the
whipsaw at the boundary the soft layer was designed to absorb.

### I8. Operational: where the audit log goes

Baseline says "append-only `halt_audit.log`" but doesn't specify rotation, retention, or backup. On a Mac mini with 245GB this matters less than on a server, but:
- **Rotation:** weekly logrotate (or a launchd job) → `halt_audit.YYYY-WW.log.gz`, kept indefinitely (text gzips to nothing).
- **Backup:** `risk/` and `state/` directories rsync nightly to an external drive or iCloud Drive folder. Snapshots, breaches.jsonl, equity_curve.parquet, halt_audit.log all belong in the backup set. A laptop-drive failure should not lose the trade-decision audit trail.
- **Out of scope:** offsite/cloud backup, encryption-at-rest — solo operator, non-PII data, the threat model is "Mac mini SSD dies", not "nation-state adversary".
