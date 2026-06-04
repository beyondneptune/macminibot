# strategy-engine

Daily portfolio construction: convert ML signals into target positions via vol-targeted mean-variance optimization with turnover control. **Not a live signal generator. Not a tick consumer.**

## Status
Greenfield — no source code yet. **Architecture ratified 2026-05-17 after substrate review.** The earlier draft assumed live tick streaming and intraday inference; that's been rejected. The IBKR market-data account does not stream the breadth we need cheaply, the ML edges from `ml-research` are daily-horizon, and this is a solo-operator bot on a 16 GB Mac mini. Once-per-day rebalance against T-1 dollar bars is the agreed cadence. Update this file as code lands.

## Performance benchmark (target to beat)
- **Ridge baseline, cycle-7 backtest: Sharpe +2.03 net of costs** with dollar-neutral weights normalized so `|w|` sums to 1.0. This is the bar every future optimizer change must clear out-of-sample on the same harness before promotion. Any new variant (richer signal model, alternative penalty, alternative covariance estimator) that lands sub-+2.03 net Sharpe on the canonical scenario set is rejected at review and either tuned or shelved. The +2.03 figure is the *net* number — gross Sharpe alone is not a substitute. Re-baseline whenever the underlying scenario set is changed; record the new number here and the cycle that produced it.

## Scope
- **Daily rebalance loop at 09:35 ET** — gated by `pandas_market_calendars.NYSE`, fired by the scheduler tentacle after overnight gap is observed
- Signal ingestion contract from `ml-research` (validate, winsorize, reject malformed)
- Covariance estimation with Ledoit-Wolf shrinkage; HRP fallback above n=500
- Mean-variance optimizer with risk-aversion, gross-exposure cap, per-name cap, and turnover penalty
- Vol-targeting scaler: rescale optimal weights so ex-ante portfolio vol matches σ_target = 12% annualized
- Cross-asset normalization: optimize in risk-units, convert to contracts at the boundary via a contract-metadata registry
- Turnover gate: skip rebalances that don't move the needle (no-trade band)
- Edge-case rules: holidays, halts/LULD, new listings, signal blowups, ill-conditioned Σ, stale bars, NAV jumps, zero-signal days, **corporate actions** (added in improvements)
- Target-position emitter to `execution` (one `TargetPosition` per name, with full reasoning trace)
- Reconciliation loop: previous-day target vs. realized fill vs. current book — feeds tomorrow's `w_prev`
- Backtest harness wrapper (calls into `backtesting` tentacle but owns the strategy-engine-specific scenarios)
- Kill switch + dry-run mode (read all inputs, emit zero `TargetPosition`s, log full trace)

## Boundaries
- **Owns:** the daily-rebalance loop, the optimizer, the vol-target scaler, the turnover gate, the contract-metadata registry, the edge-case rule book, the reconciliation ledger, the dry-run harness
- **Consumes from:**
  - `market-data` — T-1 close dollar bars (read API over ArcticDB), used to build the returns matrix for Σ estimation
  - `ml-research` — `Signal{symbol, r_hat_daily, confidence, model_version, as_of}`, one row per name per day; payloads outside the contract are **rejected, not coerced**
  - `shared-core` — `Symbol`, `Side`, `ContractSpec`, `Money`, `TargetPosition`, `NavSnapshot` types
  - `risk-management` — `NavSnapshot` from the 09:30 ET snapshot; strategy-engine never calls IBKR for NetLiq directly (see Key Decision above)
  - `compliance` — corporate-action feed (ex-dates, splits, mergers, spinoffs, index reconstitutions); see Improvement #2
- **Provides to:**
  - `risk-management` — `TargetPositionBatch` (declarative end-state targets); risk-management diffs vs current book, vets, and lowers to `OrderIntent`s. risk has veto and we honor rejections by re-emitting a corrected batch on the next session.
  - `execution` — does **not** receive directly from strategy-engine. execution receives `OrderIntent` batches from risk-management, which in turn derives them from our `TargetPositionBatch`.
  - `observability` — optimizer status (solver iterations, objective, dual residuals), Σ condition number, turnover ratio, signal-rejection counts, edge-case-rule firings, vol-target tracking error
  - `control-plane` — current dry-run flag, current kill-switch state, last-rebalance summary
- **Does NOT own:** the ML model itself (`ml-research`), order types or routing (`execution`), pre-trade risk checks (`risk-management`), the dollar bars themselves (`market-data`). We're the **portfolio constructor**, nothing more.

## Key Decisions (ratified 2026-05-17)

### Canonical naming: `TargetPosition` vs `OrderIntent` (ratified cross-tentacle, cycle 2)
- **strategy-engine emits `TargetPosition`** — a declarative end-state ("name X should be Y contracts at end-of-session"). One per name per rebalance.
- **risk-management diffs `TargetPosition` against current book** to produce `OrderIntent` records (imperative "buy/sell N contracts"). risk-management owns the `TargetPosition → OrderIntent` lowering; strategy-engine does not emit `OrderIntent`s.
- **execution consumes `OrderIntent`s** (post-risk), never `TargetPosition`s directly.
- This eliminates the cycle-1 ambiguity where risk-management spoke of `OrderIntent` and strategy-engine of `TargetPosition`. Both terms now have distinct, non-overlapping meanings.

### Daily-batch contract with `execution` (ratified cross-tentacle, cycle 2)
- execution receives a **single batch of `OrderIntent`s once per session** (post-risk), not a stream of intents with per-message `urgency`. The batch is the entire session's order list, handed over at ≈09:35 ET.
- execution's scheduler has the **entire trading session** (≈09:35 → 15:55 ET, respecting half-day early close) to fill the batch via its own TWAP/VWAP slicing. No microsecond-budget intents from this tentacle.
- Batch schema (handed from risk-management to execution; strategy-engine produces the upstream `TargetPosition` batch with identical envelope):
  ```
  TargetPositionBatch {
      decision_id: str               # hash covering as_of, model_version, signal_payload, w_prev, contract_spec_version, solver_status
      as_of: int                     # int64 ns UTC, the 09:35 ET wake timestamp
      session_close: int             # int64 ns UTC, NYSE close for the day (respects half-days)
      valid_until: int               # int64 ns UTC, == session_close - 5min buffer
      nav_snapshot: Money            # NetLiq read from risk-management's 09:30 snapshot (see below)
      nav_snapshot_source: str       # always "risk-management.snapshot@09:30ET" — never a direct IBKR call
      targets: list[TargetPosition]  # one per name in the optimized universe (UNION with held names — see Improvement #6)
      targets_count: int             # == len(targets); checksum for atomic-rename loaders, see risk-management I7b
      mode: ChainMode                # LIVE | PAPER | WHATIF; read from control-plane at wake, frozen for session (Improvement #7)
      supersedes: str | None         # prior decision_id on mid-session re-emission, NULL on the morning batch (Improvement #4)
  }
  ```
- Only one batch per session. If the optimizer re-runs (e.g. operator manual trigger), the new batch supersedes the prior `decision_id` and execution must cancel-and-replace unfilled work from the prior batch.

### NAV source: read from `risk-management`'s 09:30 snapshot (ratified cross-tentacle, cycle 2)
- strategy-engine **MUST NOT** call IBKR (`reqAccountSummary`, `reqAccountUpdates`) directly to obtain NetLiq for sizing. All NAV reads come from `risk-management`'s 09:30 ET snapshot.
- Rationale: a direct IBKR call from this tentacle races with cash movements (deposits, dividends settling, FX sweeps, journal entries) and would produce a `nav_snapshot` inconsistent with the one risk-management uses for its own limit checks. Reading the same snapshot guarantees both tentacles agree on the denominator.
- Contract: risk-management publishes `NavSnapshot{nav: Money, as_of_ns: int, account_id: str, snapshot_version: str}` at 09:30 ET. strategy-engine reads it synchronously at 09:35 wake. Freshness check is the **intersection** of risk's window and our own slack bound (see Improvement #8): valid iff `snapshot.as_of_ns ∈ [09:30:00 ET, 09:35:00 ET]` AND `now - snapshot.as_of_ns < 10 min`. Either clause failing → halt rebalance, alert, no `TargetPosition` emission.
- `nav_snapshot.snapshot_version` is included in the `decision_id` hash so any snapshot replacement is fully auditable.

### Cadence and trigger
- **Once per day, 09:35 ET.** Five minutes after open lets the auction print, overnight news reprice, and the opening cross settle. Avoids the first-minute LULD churn. Gated by `pandas_market_calendars.NYSE.schedule()` — US holidays skip the entire wake; half-days fire at 09:35 ET regardless and respect the early close in the `valid_until` field of every `TargetPosition`.
- No intraday rebalancing. No reactive position changes on news. Intraday signal-driven trading is explicitly out of scope and requires a separate tentacle if ever wanted.

### Sizing philosophy
- **Vol-targeted risk-parity with mean-variance tilts**, not pure Kelly. Kelly assumes you know `r_hat` exactly; we have noisy ML edges with `IC ≈ 0.03–0.08` on dollar bars, and full-Kelly on a noisy edge ruins compounding. AQR (Asness/Frazzini/Pedersen, *Leverage Aversion and Risk Parity*, 2012) and Bridgewater's All Weather demonstrate vol-targeting + risk parity is the right primitive for noisy multi-asset signals at daily cadence. We start there, then tilt by `r_hat` through a quadratic objective with bounded risk aversion.

### Optimizer
```
Σ_shrunk = (1 - δ) · Σ_sample + δ · F                # Ledoit-Wolf shrinkage; F = constant-correlation target
w* = argmax_w [ w'·r_hat - (λ/2)·w'·Σ_shrunk·w
                - γ_L1·||w - w_prev||₁
                - γ_L2·||w - w_prev||₂²              # ridge term, see Improvement #1
              ]
     s.t.  ||w||₁ ≤ L_gross                          # gross-exposure cap, default 2.0
           |w_i| ≤ c_max                             # per-name cap, default 0.05 (5% NAV)
           sum(w_i) ∈ [-net_cap, +net_cap]           # net-exposure cap, default 1.0

σ_p = sqrt(w*' · Σ_shrunk · w*) · sqrt(252)
w_final = w* · (σ_target / σ_p)                      # σ_target = 12% annualized
```
- λ (risk aversion) is calibrated such that with `r_hat = 0` the unconstrained solution recovers minimum-variance — keeps the tilt component bounded
- Σ window: 252 trading days of T-1 dollar-bar close-to-close returns, EWMA-weighted (half-life ≈ 60 days) so recent regimes dominate without throwing out the long tail

### Solver selection (refined — see Improvement #4)
| n (names) | Solver | Why |
|---|---|---|
| n ≤ 50 | cvxpy + ECOS | Interior-point, cleanest duality gap on small dense problems |
| 50 < n ≤ 500 | cvxpy + OSQP, warm-started from yesterday's w* | OSQP is ADMM-based and admits warm-start; daily Σ shifts are small so warm-start saves 5–20× iterations |
| n > 500 | HRP (López de Prado 2016, *Building Diversified Portfolios that Outperform Out-of-Sample*) | Σ inversion gets numerically unstable; HRP is hierarchical and doesn't invert |
- All solver paths emit `solver_status`, `iterations`, `objective`, and `primal_residual` to `observability`. Non-optimal status → fallback to HRP for the day and alert.

### Cross-asset normalization
- Optimizer thinks in **risk-units** (dimensionless weight × per-contract notional). Translation to broker-side **contracts** happens at the emitter boundary using `ContractSpec.multiplier`:
  - ES futures: 50 × index level
  - EURUSD spot: 100,000 base currency notional
  - US equities: 1 share = 1 unit notional × price
  - GLD ETF: 1 share = price (no special multiplier)
- The `ContractSpec` registry is loaded at startup from `shared-core`, refreshed on weekly universe rebuilds. Stale `ContractSpec` (multiplier change unannounced) is the single biggest source of silent position-sizing bugs — registry version is hashed into every `decision_id`.

### Signal contract (the inbound boundary)
```
Signal {                         # mirrors shared_core.contracts.signal (ratified with ml-research, cycle 2)
    symbol: Symbol               # canonical, matches market-data universe registry
    r_hat_daily: Decimal         # EXPECTED DAILY DECIMAL RETURN, e.g. Decimal("0.0023") for 23 bps
    confidence: Decimal          # in [0, 1], used as a multiplicative tilt scaler
    model_version: str           # ml-research model registry hash
    as_of_ns: int                # int64 ns since epoch, UTC, must be ≥ T-1 close
}
```
- `Decimal` (not `float`) is load-bearing: the optimizer scales `r_hat` against `nav_snapshot` (already `Money`/`Decimal`) and rounds to contract lots; a float round-trip can flip the last lot of a near-boundary trade. Convert to `float64` only at the cvxpy boundary, never persist a float in the trace.
- Field name is `as_of_ns` (not `as_of`) to match ml-research's locked spec; mixing both names across tentacles is how a 1000× timestamp-unit bug ships.
- **z-scores, ranks, ICs, and unsized "signal strength" floats are rejected.** Strategy-engine refuses to guess the scale. If `|r_hat_daily| > 0.05` (5% daily expected return), winsorize to 3σ of the cross-sectional distribution and log to observability — almost certainly a unit error upstream.
- Missing fields, NaN, or `as_of` older than 36h → entire payload rejected, alert raised, yesterday's `w_prev` carries forward (degenerate "do nothing" rebalance).

### Turnover control
- L1 penalty term `γ_L1 · ||w - w_prev||₁` with `γ_L1 ≈ 2 × round_trip_cost_bps` per Gârleanu-Pedersen (2013, *Dynamic Trading with Predictable Returns and Transaction Costs*). L1 produces a literal no-trade region — small changes in `r_hat` don't trigger trades.
- Plus L2 ridge `γ_L2 · ||w - w_prev||₂²` for numerical stability and OSQP warm-start friendliness (see Improvement #1). `γ_L2` is small (≈ 0.1 × γ_L1) — its job is conditioning, not cost.
- **No-trade band post-optimizer:** if `|Δw_i| < 0.5% NAV` for a name, freeze `w_final_i := w_prev_i`. Stops paying commissions on 0.1% drifts.
- **Turnover ceiling:** if total `||w - w_prev||₁ > 1.0` (100% gross turnover), scale the trade down — almost always a signal blowup, not a real regime change.

### Edge-case rule book (each is a codified, unit-tested rule)
| Trigger | Action |
|---|---|
| US market holiday (`NYSE.schedule` empty) | Skip wake entirely, no rebalance |
| Halt / LULD on name `i` | Freeze `w_prev_i`; if halt > 1 session, emit flatten-via-TWAP target to `execution` |
| New listing (<60 days of dollar bars) | Exclude from universe |
| New listing (60–252 days) | Include with `c_max := 0.25 × c_max` |
| Signal blowup: `|r_hat_i| > 5σ` cross-sectional | Winsorize to 3σ, log to observability |
| Σ ill-conditioned (`cond(Σ_shrunk) > 1e6`) | Fall back to diagonal Σ, emit alert |
| Stale dollar bar (>48h since last bar), 1 name | Drop name, redistribute its `w_prev` weight pro-rata over remaining universe |
| Stale bars on ≤10% of universe | Drop affected names as above; log to observability; proceed |
| Stale bars on 10–30% of universe | Drop affected names AND scale `σ_target := 0.5 × σ_target` for the session (less universe → less diversification → lower risk budget) |
| Stale bars on >30% of universe | Halt rebalance entirely (carry `w_prev`); alert; the data substrate is the suspect, not the strategy. Distinct from "Σ ill-conditioned" — that's a numerical failure on a complete dataset; this is a dataset-completeness failure |
| NAV jump > 10% intraday vs T-1 close | Halt, dry-run mode, require operator ack |
| All signals near zero (`max_i \|r_hat_i\| < 5 bps`) | Decay positions to cash at 0.2 / day (not all-flat in one shot — avoids paying full turnover on a quiet day) |
| **Corporate action ex-date** (see Improvement #2) | Adjust `w_prev` before optimization; never treat split-induced share-count change as a trade signal |
| **Index reconstitution day** | Defer rebalance by 1 session for affected names — let the herd trade first |

### Reconciliation
- End-of-day reconciliation reads `execution`'s actual fills and `risk-management`'s rejection events. Produces tomorrow's `w_prev` from realized positions, not from yesterday's targets. Drift between target and realized is logged per-name and surfaces as a daily report.

## Conventions
- All decisions carry a `decision_id` = `hash(as_of || model_version || signal_payload || w_prev || contract_spec_version || solver_status)`. Fully reproducible — feed the same inputs, get the same `decision_id`, get the same targets.
- Reasoning trace on every `TargetPosition` includes: `r_hat`, `confidence`, `w_optimal_pre_vol_target`, `w_optimal_post_vol_target`, `w_after_no_trade_band`, `sigma_p`, `solver_status`, `edge_case_rules_fired[]`. Lets `compliance` and operator post-mortems reconstruct the day.
- **The optimizer never writes to disk.** Inputs read-only, outputs emitted as messages. State (w_prev, contract_spec version, last-fired-edge-rules) lives in a single small append-only ledger owned by this tentacle.
- **Dry-run mode is identical to live except no `TargetPosition` is emitted.** Every test fixture runs in dry-run; promotion to live is a single config flag flip.
- Every optimizer call has a wall-clock budget of 30 seconds. Overrun → fallback to HRP and alert.
- No NumPy random seeds without explicit seed parameter. Reproducibility is non-negotiable.
- All timestamps are int64 ns UTC, matching `market-data`'s convention.

## Improvements over the baseline (this revision)

### Improvement #1 — Hybrid L1 + L2 turnover penalty (not pure L1)
**Baseline says:** L1 turnover penalty per Gârleanu-Pedersen 2013.
**Refinement:** L1 alone matches per-share commission + half-spread (linear in `|Δw|`), but is non-smooth at zero, which hurts OSQP warm-start convergence and over-regularizes small trades. DeMiguel, Martín-Utrera, Nogales (2014, *Size Matters: Optimal Calibration of Shrinkage Estimators for Portfolio Selection*, JBF) and Olivares-Nadal & DeMiguel (2018, *A Robust Perspective on Transaction Costs in Portfolio Optimization*, OR) both find a **hybrid L1 + small L2** dominates pure L1 on out-of-sample Sharpe when the cost function has a quadratic impact component (which TWAP execution at our size profile does). L2 ridge term also gives OSQP a strictly convex objective → guaranteed unique optimum, 2–5× faster warm-start. Decision: keep L1 as the dominant cost term (`γ_L1 ≈ 2 × round_trip_cost_bps`), add small L2 (`γ_L2 ≈ 0.1 × γ_L1`) for conditioning and warm-start speed.

### Improvement #2 — Corporate-action handling as a first-class edge case
**Baseline's edge-case list omitted** dividend ex-dates, splits, mergers, spinoffs, index reconstitutions, and quadruple-witching Fridays. These create **artificial drift** in `w_prev` between yesterday's close and today's open that the optimizer would otherwise interpret as a trade signal. A 2:1 split halves yesterday's `w_prev_i × price`; without adjustment, the optimizer "sees" a 50% NAV drop on that name and tries to buy back, paying real commission on a phantom signal. Required: a corporate-action feed (subscribed via `compliance` or directly from IBKR's `reqContractDetails` corporate-action tags), an `apply_corporate_actions(w_prev, ca_events, as_of) -> w_prev_adj` adjustment step run **before** the optimizer, and a unit test per action type (split, reverse-split, cash div, stock div, spinoff, merger, ticker change). Quadruple-witching Fridays (3rd Friday of Mar/Jun/Sep/Dec) get an additional `c_max := 0.5 × c_max` haircut on futures positions for the session — historical realized vol spikes 30–60% intraday.

### Improvement #3 — Solver tier refinement and warm-starting
**Baseline says:** ECOS for n ≤ 500, HRP above. **Refinement:** ECOS is interior-point and re-solves from scratch every call — no warm-start. For a daily rebalance where Σ changes slowly, OSQP's warm-start is a 5–20× iteration saver from 50 names upward (Stellato et al. 2020, *OSQP: An Operator Splitting Solver for Quadratic Programs*). ECOS still wins at n < 50 because its duality gap is cleaner on small dense problems and warm-start savings don't matter. HRP above 500 is correct as-is — Σ inversion is the failure mode. Tiered selection codified in the table above. Warm-start cache = previous day's primal+dual solution, invalidated when the universe changes by >5% or when `cond(Σ)` jumps by >2× day-over-day.

### Improvement #4 — Cancel-and-replace semantics for mid-session re-emission (cycle 3)
**Baseline gap:** the "Daily-batch contract" key decision says "If the optimizer re-runs (e.g. operator manual trigger), the new batch supersedes the prior `decision_id` and execution must cancel-and-replace unfilled work from the prior batch." That sentence is the whole spec — it doesn't say *how* the supersede happens, *who* cancels, what happens to **filled** legs, or how the new optimization knows about partial fills. Cycle 2 surfaced this and left it open.
**Resolution (this cycle):**
1. **Re-emission preconditions** — `control-plane` exposes `strategy-engine reemit --reason "<text>"`. The command is **rejected** unless (a) a prior `TargetPositionBatch` was emitted today, (b) `now < session_close - 30min` (no re-emissions in the last half hour — the new batch needs runway to fill), (c) the kill switch is not active, (d) the operator has supplied a `--reason` string that will be persisted in the new batch's manifest. No silent re-emissions; the operator is on the hook in the audit log.
2. **Snapshot semantics** — re-emission re-reads `risk-management`'s `NavSnapshot` (which is still the 09:30 ET snapshot — risk does not re-snapshot intraday in v1) AND re-reads `risk-management.fills_so_far(prior_decision_id)` and `execution.unfilled_legs(prior_decision_id)`. The new optimization is seeded with `w_prev := realized_position_now` (computed from snapshot + filled legs), **not** the morning's `w_prev`. This is the only correct denominator: re-optimizing against the morning book would re-order legs that already filled and double-trade.
3. **Batch envelope** — the new `TargetPositionBatch` carries `supersedes: str = prior_decision_id` (NULL on the morning batch). Identical envelope schema otherwise. The new `decision_id` hash includes `supersedes` so it is provably distinct.
4. **Lowering contract (risk-management's job, documented here for symmetry)** — when risk-management receives a batch with non-NULL `supersedes`, it MUST: (a) issue cancel requests to `execution` for every `OrderIntent` from `supersedes` that has `status ∈ {WORKING, PARTIALLY_FILLED.remaining > 0}`; (b) **wait for cancel acks** (or a 5s timeout → HALT) before computing the new `TargetPosition → OrderIntent` diff; (c) compute the diff using post-cancel realized positions, not theoretical morning positions; (d) emit the new `OrderIntent` batch with `replaces: prior_intent_id` populated wherever a 1:1 replacement exists. Partial fills are honored: if morning emitted BUY 100 and 40 filled before cancel, the new diff sees +40 in the book and sizes accordingly.
5. **What strategy-engine does NOT do** — strategy-engine never issues cancels. It never touches `OrderIntent`. It re-emits a fresh `TargetPositionBatch`; risk-management is the sole canceller. This preserves the cycle-2 invariant that strategy-engine is declarative-end-state only.
6. **Audit** — every re-emission writes a row to the strategy-engine append-only ledger: `(prior_decision_id, new_decision_id, reason, ts_reemit, w_prev_morning, w_prev_now)`. The end-of-day reconciliation loop reads both batches when rebuilding tomorrow's `w_prev`.
7. **Observability** — re-emissions are a tracked counter; >1 re-emission per session for 3 consecutive sessions is an alert (operator is probably fighting the optimizer; investigate).

### Improvement #5 — Corporate-action feed: consumer-side schema and producer-side ownership (cycle 3)
**Baseline gap:** Improvement #2 introduced the corporate-action edge case but did not specify the wire format, did not assign producer-side ownership, and `compliance/CONTEXT.md` does not currently mention the feed. Cycle 2 flagged it.
**Consumer-side schema (this is what strategy-engine demands; producer must conform):**
```python
# shared_core/contracts/corporate_action.py — to be mirrored from this spec
from dataclasses import dataclass
from decimal import Decimal
from shared_core.contracts.symbol import Symbol

@dataclass(frozen=True, slots=True)
class CorporateAction:
    symbol: Symbol                       # canonical, matches market-data universe registry
    action_type: str                     # enum: "split" | "reverse_split" | "cash_dividend" | "stock_dividend" |
                                         #       "spinoff" | "merger_cash" | "merger_stock" | "ticker_change" |
                                         #       "delisting" | "rights_offering"
    ex_date: int                         # int64 ns UTC, 09:30 ET of the ex-date session
    ratio_numerator: int                 # split 2-for-1 → num=2, den=1; reverse 1-for-10 → num=1, den=10
    ratio_denominator: int               # both = 1 for non-ratio actions (cash div, ticker change)
    cash_amount: Decimal                 # per pre-action share; 0 for non-cash actions
    new_symbol: Symbol | None            # populated for ticker_change, spinoff (parent's child), merger_stock
    payable_date: int | None             # int64 ns UTC, when cash actually hits the account (None for non-cash)
    source: str                          # "ibkr.reqContractDetails" | "vendor.<name>" | "manual.<operator_id>"
    as_of: int                           # int64 ns UTC, when the producer published this row
```
**Producer-side ownership recommendation:** the corporate-action feed should be owned by `compliance`, NOT `market-data`. Rationale:
- `market-data` owns the **price/volume substrate**; corporate actions are a categorically different feed (events, not bars; legal/registrar source-of-truth, not exchange tape). Conflating them dilutes the `market-data` boundary, which is already "bars + ticks via ArcticDB".
- `compliance` already owns the IBKR account-status surface and is the natural reader of `reqContractDetails` corporate-action tags. The feed lives next to the audit log it serves.
- `risk-management` is a consumer too (a split changes `MAX_SINGLE_NAME_NOTIONAL` denominator overnight); a single producer keeps risk and strategy-engine reading from the same schema, eliminating the silent-drift class of bug.
- Filing escalated: `compliance/CONTEXT.md` needs a "Corporate-action feed" section; `shared-core` needs the dataclass above; `risk-management` needs a consumer hook in its `BROKER_AVAILABILITY` / `POSITION_RECONCILES` checks (a missing split adjustment will trip reconciliation, not just sizing).
**strategy-engine consumer behavior:**
- At 09:33 ET (2 min before wake) the rebalance loop reads `compliance.corporate_actions(ex_date=today)` synchronously. Empty list is the common case and a valid response. Missing feed (>5 min stale, or producer unreachable) → halt rebalance, alert, no batch — same failure mode as a stale `NavSnapshot`.
- `apply_corporate_actions(w_prev, ca_events, as_of) -> w_prev_adj` runs **before** the optimizer, per Improvement #2. The unit-test fixture per action type lives here; the producer's contract test asserting schema parity lives in `compliance`.
- `decision_id` hashes the sorted tuple of `(symbol, action_type, ex_date)` for the day's actions, so a missing or duplicated action makes a downstream replay fail loudly.

### Improvement #6 — Halt/exclusion semantics vs risk-management's flatten-on-missing-target rule (cycle 3, cross-tentacle bug fix)
**The conflict (discovered reading `risk-management/CONTEXT.md` I4):** risk-management's `diff_targets_to_intents` rule says *"For every symbol in `snapshot.positions` NOT in `targets`: emit a flatten intent (`SELL` to zero). Missing-from-targets means 'exit', never 'hold'."* But strategy-engine's edge-case rule book says for halted/LULD names: *"Freeze `w_prev_i`"* — which the natural implementation reads as "omit this name from the optimizer and from the target list." Under risk-management's rule, omission → flatten → strategy-engine accidentally instructs risk to liquidate every halted, stale-bar, or new-listing-excluded name. The opposite of intent.
**Fix (this cycle): strategy-engine emits an EXPLICIT target for every name with non-zero `w_prev`, even when the rule book says "freeze" or "exclude".** Specifically:
- The `TargetPositionBatch.targets` list is the **full union** of (a) names in the optimized universe and (b) names with non-zero `w_prev` carried forward by an edge-case rule.
- A frozen/held name gets a `TargetPosition` with `contracts = w_prev_i_in_contracts` (i.e. the realized current holding rounded to lot) AND `hold_reason: str` populated with the edge-case rule id that fired (`"halt.luld"`, `"stale_bar.>48h"`, `"new_listing.<60d"`, `"corporate_action.spinoff_pending"`, etc.). `hold_reason` is NULL for normally-optimized names.
- risk-management's diff sees `target_qty == current_qty` for these names → `INTENT_NONZERO` check trips → no `OrderIntent` is emitted. The held position is preserved without any cross-tentacle gymnastics.
- Names truly intended to flatten (zero-signal decay, optimizer set `w_i = 0`, post-no-trade-band freeze that resolved a small position to zero) get `contracts = 0` with `hold_reason = NULL`. risk-management's flatten path fires correctly on these and only these.
- New `TargetPosition` schema field: `hold_reason: str | None`. Add to the dataclass in `shared-core` alongside the other ratified fields. The reasoning trace already captures `edge_case_rules_fired[]`; `hold_reason` is the single-rule-id summary that risk-management reads without parsing the trace.

This is a real bug, not a stylistic one: without the fix, a single halted name on a quiet session triggers a forced liquidation through the entire risk → execution path. The fix is one extra field and one rule about how the targets list is assembled. Cheap, eliminates the largest cross-tentacle silent failure mode surfaced so far.

### Improvement #7 — `ChainMode` propagation: `control-plane → strategy-engine → risk-management` (cycle 4)
**The gap (cycle 3 left open):** `risk-management/CONTEXT.md` I7a ratified `ChainMode ∈ {LIVE, PAPER, WHATIF}` with **no default value** — every callsite must pass it explicitly. But the upstream flow into risk is `strategy-engine`'s `TargetPositionBatch`, whose schema (key decision "Daily-batch contract", cycle 2) carries only a `dry_run: bool` flag. A bool cannot express the three-way distinction risk's chain requires; in particular, `dry_run=True` could mean either PAPER (write paper sidecars) or WHATIF (zero side-effects), and risk has no way to know which. Backtest runs would currently get classified as PAPER and pollute `risk/paper/` with replay noise.

**Resolution:**
1. **Replace `dry_run: bool` with `mode: ChainMode`** on `TargetPositionBatch`. The dataclass field is `mode: ChainMode` (imported from `shared_core.contracts.chain_mode`, mirrored from risk's enum). Old `dry_run` field is removed, not deprecated — the batch schema is greenfield and there are no live consumers yet.
2. **Source of truth: `control-plane`.** `control-plane` owns a process-wide `current_mode: ChainMode`, settable via `bot mode set <live|paper|whatif>` and readable via `bot mode get`. Strategy-engine reads it synchronously at 09:35 wake (same pattern as the kill switch read). No env-var fallback — explicit operator action only. Mode change requires `--reason "<text>"` and writes an audit row.
3. **Wake-time consistency:** the mode read at 09:35 ET is **frozen for the entire session**. A mid-session `bot mode set` takes effect only on the next session's wake (or on a `strategy-engine reemit`, where it is re-read). This prevents the operator from accidentally flipping LIVE → PAPER between batch emission and risk's chain run, which would leak a live batch into paper sinks.
4. **Backtest harness uses `WHATIF`, not `PAPER`.** The CI backtest scenarios (todo Epic 14) pass `mode=ChainMode.WHATIF` explicitly. PAPER is reserved for the 30-day soak gate documented in `risk-management/CONTEXT.md` Epic 10; it writes to real (parallel) paths and must not be triggered by unit tests.
5. **Kill-switch interaction:** kill switch and `mode` are orthogonal. Kill switch flips emit/no-emit; mode flips which downstream sinks risk uses. A LIVE-mode kill-switched session still emits zero `TargetPosition`s; a PAPER-mode session with no kill switch still emits a full batch into paper sinks.
6. **Schema mirror:** add `ChainMode` to `shared-core/contracts/chain_mode.py` so risk-management and strategy-engine import the same enum. Cross-tentacle PR — file alongside the `Signal` rename diff so both ratifications land together.

### Improvement #8 — Snapshot freshness window: ratify the tighter risk-side bound (cycle 4, cross-tentacle fix)
**The conflict (risk-management/CONTEXT.md I7c #2 surfaced this against us):** our "NAV source" key decision says missing/stale snapshot is *">10 min old → halt rebalance"*. Risk-management's chain check `SNAPSHOT_FRESH` requires `snapshot.ts ∈ [09:30:00 ET, 09:35:00 ET]` — a 5-minute hard window. If the snapshot lands at 09:30:00 ET and our chain runs at 09:39:00 ET, we'd emit a batch (still inside our 10-min bound) but risk would reject it with `SNAPSHOT_FRESH` failure — guaranteed HALT every time the wake slips even slightly.

**Resolution (ratify here, propagate to risk):** strategy-engine adopts the **intersection** of both bounds: snapshot is valid iff `snapshot.ts ∈ [09:30:00 ET, 09:35:00 ET]` AND `now - snapshot.ts < 10 min`. The first clause is risk's bound (the morning snapshot's identity); the second is our own (operator-friendly catch for clock skew or a delayed wake). At 09:35 ET wake both clauses hold trivially; at 09:39:30 ET (a 4.5-min slip) the second clause still holds but the first does too because we only require `snapshot.ts` itself to be in the window, not `now`. The combined check fails iff the snapshot is from yesterday, from a re-snapshot attempted outside the morning window, or our wake slipped past 09:40 ET — all of which should halt. Update "NAV source" key decision wording to match; risk-management's I7c flagged the same need from the other side, so this is a one-PR alignment.

## Cross-tentacle gaps surfaced but not fixed here

### Resolved in cycle 2
- **`TargetPosition` vs `OrderIntent` naming** — resolved. strategy-engine emits `TargetPosition` (declarative); risk-management lowers to `OrderIntent` (imperative). See "Canonical naming" key decision.
- **Daily-batch contract with `execution`** — resolved. execution receives a single `OrderIntent` batch per session (post-risk), session-long fill window. See "Daily-batch contract" key decision for the schema.
- **NAV double-read with `risk-management`** — resolved. strategy-engine reads NetLiq from risk-management's 09:30 ET `NavSnapshot`; no direct IBKR call. See "NAV source" key decision.

### Resolved in cycle 3
- **Cancel-and-replace semantics on mid-session re-emission** — resolved. See Improvement #4: preconditions, snapshot re-read, `supersedes` envelope field, risk-management cancel ownership, post-cancel diff.
- **Corporate-action feed: consumer schema + producer owner** — resolved consumer-side. See Improvement #5: `CorporateAction` dataclass, `compliance` recommended as producer owner, 09:33 ET synchronous read, failure mode mirrors `NavSnapshot` staleness.
- **Halt/exclude vs risk-management flatten conflict** — resolved. See Improvement #6: targets list is the full union of optimized + held names, new `hold_reason` field on `TargetPosition`, risk's `INTENT_NONZERO` check naturally absorbs frozen positions without a special case.

### Resolved in cycle 4
- **`Signal` field-name and type drift vs ml-research** — resolved in-place: `Signal` schema now reads `r_hat_daily: Decimal`, `confidence: Decimal`, `as_of_ns: int` matching ml-research's locked spec. `Symbol` typed (not bare `str`). Cross-tentacle `shared-core` dataclass PR still needed; both tentacles will import from there once landed.
- **`ChainMode` propagation from `control-plane`** — resolved. See Improvement #7: `dry_run: bool` removed from `TargetPositionBatch` and replaced with `mode: ChainMode`, sourced from `control-plane`, frozen at wake, mirrored in `shared-core`. Aligns with risk-management I7a's no-default `mode=` requirement.
- **Snapshot freshness window mismatch** — resolved. See Improvement #8: strategy-engine adopts the intersection of risk's `[09:30:00, 09:35:00 ET]` window and our `< 10 min` slack bound. Updated "NAV source" key decision wording.
- **Progressive Σ / stale-bar degradation** — resolved. Edge-case rule book now has three tiers (≤10% / 10–30% / >30% of universe stale) with proportional responses: drop-and-redistribute, drop + halve σ_target, and full halt. Cycle 1's single rule was binary; this captures the dataset-completeness failure mode separately from the numerical-conditioning one.

## Improvements (cycle 6 — incremental on the landed implementation)

### Improvement #12 — `TargetPosition.contracts: Decimal | None` field for Alpaca lot-rounded lowering
**The drift:** cycle-5 Improvement #11 demands the emitter lower `target_weight` (risk-units) to a venue-side share count via `contracts = round(w_final * NAV / price, fractional_precision)`. The local `TargetPosition` stub only carried `target_weight: float`, so the emitter epic (todo last bullet) would have had to retro-fit the field and break every existing consumer. Added `contracts: Decimal | None` now with `None` default — non-disruptive for the 32 pipeline tests, contractually required to be non-None for any LIVE/PAPER batch handed to risk-management. Decimal (not float) matches the cycle-2 r_hat/NAV typing rationale: a float round-trip can flip the last lot on a near-boundary trade. The shared-core mirror has `target_qty: Decimal` as the canonical type; the local stub's split into `target_weight` (optimizer-native) plus `contracts` (venue-lowered) preserves both for the trace without losing precision at the lowering step.

### Improvement #13 — Plumb `gamma_turnover_l2` through `RebalanceInputs` (Ridge-baseline sizing hook)
**The drift:** the optimizer accepts `gamma_turnover_l2` with a default of `0.1 * gamma_turnover` (Improvement #1's hybrid penalty), but `RebalanceInputs` did not expose it — so the only way to dial L2 from the scheduler was to monkey-patch the optimizer call. Two reasons this matters now: (a) the ml-research Ridge-as-baseline ratification means `r_hat` already carries the linear-model shrinkage; piling more L2 on top in the sizing step is double-shrinkage and silently flattens the tilt for the cleanest, best-conditioned signal class; (b) a future per-model-family hook (read `model_version` → look up `gamma_turnover_l2` in a config registry) needs the kwarg already wired end-to-end or it's a multi-file diff at the wrong moment. Setting `gamma_turnover_l2=0.0` for linear-signal sessions keeps L1 (which is genuine cost-of-trading) and removes the conditioning ridge that exists only to speed OSQP warm-starts and prevent degeneracy — both moot when `r_hat` is already linear-shrunk and the universe is small enough for ECOS (no warm-start anyway). Default kept as `None` so the optimizer's hybrid stays the steady-state; the kwarg is an escape hatch, not a behavior change.

## Improvements (cycle 8 — Ridge-baseline alignment)

### Improvement #14 — Codify when `gamma_turnover_l2` is zero vs default (per-signal-family policy)
**Why now:** Improvement #13 wired the kwarg end-to-end but left the *policy* implicit. Cycle 7's Ridge backtest hit Sharpe +2.03 net using a signal that is *already* the closed-form L2-shrunk linear projection (`(X'X + αI)^-1 X'y`). The MV optimizer's `γ_L2 · ||w - w_prev||₂²` term then shrinks the sized weights toward `w_prev` a second time — and because the Ridge `r_hat` is already a low-norm, well-conditioned target, the L2 penalty has no degeneracy to fight and exists only as cost-free drag on the tilt. The double-shrinkage flattens exactly the signal class the bot is best at trading. Codify the policy so future cycles do not silently re-introduce it.

**Policy (read `model_version` prefix at wake; mirror in the scheduler):**

| `model_version` family | `gamma_turnover_l2` | Rationale |
|---|---|---|
| `ridge.*`, `lasso.*`, `elasticnet.*`, `ols.*` | **`0.0`** | Linear `r_hat` is already L2/L1 shrunk; second ridge is double-shrinkage. Universe ≤ 50 → ECOS path → no warm-start to protect → safe to drop. |
| `gbm.*`, `xgb.*`, `lgbm.*`, `rf.*` | **`0.1 × γ_L1`** (default) | Tree-based `r_hat` is unshrunk and can be locally pathological at signal cliff edges; the ridge term is genuine numerical insurance. |
| `nn.*`, `mlp.*`, `transformer.*` | **`0.1 × γ_L1`** (default) | Same as trees: unbounded output scale, real risk of single-name spikes. |
| `ensemble.*` | **`0.05 × γ_L1`** (half-default) | Averaging dampens spikes; halve the conditioning ridge but do not zero it. |
| `unknown` / unmapped prefix | **`0.1 × γ_L1`** (default) | Conservative — unknown signal class gets the conditioning ridge until classified. |

The policy lives in `strategy_engine.config.l2_policy.py` as a single `Dict[str, float | None]` keyed by family prefix; the scheduler resolves `model_version.split(".", 1)[0]` against it at wake and writes the resolved value into `RebalanceInputs.gamma_turnover_l2`. **The L1 turnover penalty is never touched by this policy** — L1 encodes literal round-trip cost and is invariant across signal families. Resolved `gamma_turnover_l2` is recorded in the reasoning trace alongside `model_version` so a post-mortem can confirm the policy fired correctly. Operator override path: `control-plane` can pin a session-wide `gamma_turnover_l2` value that bypasses the table (audit row required, same `--reason` discipline as `bot mode set`).

### Improvement #15 — Optimizer normalization parity with backtest (`|w|`-sum vs gross-cap)
**The drift:** the cycle-7 Ridge backtest evaluates dollar-neutral weights normalized so `sum(|w_i|) == 1.0` (a unit-L1, dollar-neutral budget). The MV optimizer's constraint set, by contrast, expresses the gross-exposure cap as `||w||₁ ≤ L_gross` with `L_gross = 2.0` default and adds a *separate* `sum(w_i) ∈ [-net_cap, +net_cap]` net cap, then post-multiplies by `(σ_target / σ_p)` from the vol-target scaler. The vol-target step is the only thing that pins the absolute scale of `w` — without it, the MV solution is degenerate-by-scale on a pure linear objective (no risk term forces a unique norm). Two concrete problems this creates against the backtest's `|w|=1.0` convention:

1. **Scale mismatch at the backtest boundary.** When the backtest harness (todo Epic 14) hands `r_hat` to the optimizer and reads `w_final` back, it currently re-normalizes to `|w|=1.0` *after* the vol-target scaler ran — which silently undoes the vol-target work and replaces it with the backtest's own L1 budget. The +2.03 Sharpe was measured under this re-normalization; the live optimizer's vol-targeted weights are a *different* portfolio at the same `r_hat`. Cycle 7's measurement and cycle 8's live path are not the same strategy.
2. **Dollar-neutrality is a soft constraint, not enforced.** `net_cap = 1.0` allows a 100%-net portfolio; the backtest enforces `sum(w_i) == 0` exactly. With a long-biased `r_hat` cross-section (common in trending regimes) the live optimizer will silently lean long while the backtest stays neutral — even before scale effects.

**Resolution (two-part):**
- **(a) Add a `normalization_mode: NormalizationMode` field to `RebalanceInputs`** with enum values `{vol_target, l1_unit_dollar_neutral, gross_cap}`. Default `vol_target` (today's behavior, unchanged). The backtest harness sets `l1_unit_dollar_neutral` explicitly. In `l1_unit_dollar_neutral` mode the optimizer adds the hard equality `sum(w_i) == 0` to the constraint set, replaces `||w||₁ ≤ L_gross` with `||w||₁ == 1.0` (equality), and **skips the vol-target post-multiply entirely** (`w_final := w*`). This makes the live optimizer reproduce the backtest's portfolio bit-for-bit when run on the same `(r_hat, Σ)` pair — a prerequisite for trusting the +2.03 Sharpe number when promoting.
- **(b) Add a parity-regression test** that feeds a deterministic `(r_hat, Σ)` snapshot through both the backtest pipeline and the live optimizer with `normalization_mode=l1_unit_dollar_neutral`; assert `np.allclose(w_live, w_backtest, atol=1e-8)`. This becomes a guard against silent divergence as the optimizer evolves. **Whenever the live path changes (new constraint, new penalty term, new solver), the parity test must still pass under `l1_unit_dollar_neutral` or the change is rejected.**

The default-mode behavior (`vol_target`) stays the steady-state for the live bot — vol-targeting is the right primitive for a real account because it bounds session-level risk to a known number, which an L1-unit constraint does not. The `l1_unit_dollar_neutral` mode exists strictly so the backtest and the live optimizer agree on portfolio identity, not so the live bot ships under that constraint. The third mode (`gross_cap`) is reserved for future stress scenarios that want the inequality form without vol-targeting; not wired in cycle 8.

### Shared-core integration (cycle 7 — landed)
The cycle-6 "deferred" gap is closed. `Signal`, `NavSnapshot`, `ChainMode`, and `Symbol` are now imported directly from `shared_core.contracts` and re-exported through `strategy_engine.contracts`. The canonical field-name and type changes propagated cleanly:
- `Signal.symbol` is `Symbol` (not bare `str`); `Signal.confidence` is `float` (not `Decimal`, per shared_core T6 — confidence is a unitless ratio, not money).
- `NavSnapshot` fields are `net_liq`/`snapshot_ns`/`buying_power`/`positions`/`snapshot_version` (sha256 hex) per shared_core T10. The scheduler reads `inputs.nav_snapshot.net_liq` (was `.nav`) into the batch.
- `ChainMode` enum values are lowercase strings (`"live"`/`"paper"`/`"whatif"`) in shared_core; name-only access (`ChainMode.WHATIF`) is unchanged.
- The scheduler pipeline now keys `prev_positions: Dict[Symbol, float]` and `returns.columns: Symbol-typed` (pandas accepts `Symbol` as a column label because the frozen dataclass is hashable).

**Wrapper kept for `TargetPosition` and `TargetPositionBatch`** (option (b) from the cycle-6 deferred note). The shared_core canonical forms drop the reasoning-trace fields (`target_weight`, `r_hat`, `confidence`, `w_prev`, `sigma_contribution`, `decision_id`, `mode`, `solver_status`, `sigma_portfolio_annualized`, `edge_case_rules_fired`) that CONTEXT declares load-bearing for compliance and operator post-mortems. The local envelope carries them; `to_shared_core()` lowers to the canonical form by collapsing the per-name trace into `TargetPosition.reason_trace: dict[str, Any]` and aliasing `batch_id := decision_id`. A formal extension proposal is filed in `shared-core/todo.md` (Epic 16) for the shared-core owner to ratify: either extend shared_core with the trace fields (collapse strategy-engine `contracts.py` to a re-export) or bless the sidecar pattern (no further changes). Strategy-engine is unblocked either way; 33 tests pass (32 prior + 1 new lowering parity test).

## Improvements (cycle 5 — landed alongside initial `src/strategy_engine/` implementation)

### Improvement #9 — Explicit solver fallback chain with transition criteria
**The gap (drift between CONTEXT and `src/strategy_engine/optimizer.py`):** CONTEXT's "Solver selection" table promises a tiered ECOS / OSQP / HRP pick by universe size, and the rule "Non-optimal status → fallback to HRP for the day and alert". The current optimizer goes straight from a non-`optimal` cvxpy status to HRP — no intermediate retry on the same QP family. Two known failure modes are silently downgrading: (a) OSQP returning `solved_inaccurate` because the warm-start was poisoned by a universe-changed prior solution (fixable by cold-start retry), and (b) ECOS hitting `feastol` on a near-binding gross-exposure cap (fixable by SCS retry with looser tolerances). HRP is the *correct* final fallback but is too lossy as the first response — it discards the entire MV objective.

**Resolution: a three-stage chain, codified.**
1. **Stage 1 — chosen tier (ECOS or OSQP), default tolerances.** Standard path.
2. **Stage 2 — same tier, cold-start + looser tolerances.** If Stage 1 returns `infeasible_inaccurate`, `unbounded_inaccurate`, or `solver_error`: re-solve from cold (`warm_start=False` even if a cache exists), one-decade looser primal/dual tolerances (`eps_abs=1e-6, eps_rel=1e-6` for OSQP; `abstol=1e-7, reltol=1e-7` for ECOS), and a higher `max_iter` (200k for OSQP). Increment `solver.stage2_retry` counter to observability. If Stage 2 returns `optimal` or `optimal_inaccurate`, accept and log the retry.
3. **Stage 3 — cross-family retry with SCS, before HRP.** If Stage 2 still fails: try SCS (cvxpy's `cp.SCS`) with default tolerances. SCS is splitting-cone, doesn't share OSQP's quadratic-form sensitivity, and frequently lands when OSQP stalls on a poorly scaled `Σ`. SCS is a build-time hard dep already pulled in by cvxpy, so no new package. Stage 3 accept criteria are identical to Stage 1.
4. **Stage 4 — HRP fallback.** If Stage 3 also fails: invoke `_hrp_allocate(Sigma_sym)`. `solver_status` is set to `fallback_hrp_after_3_retries`, an alert fires, and `decision_id` records the final status as required.

The chain is **bounded at 30s wall-clock total** (the per-call budget from CONTEXT Conventions); Stage 2/3 each carry a 5s timeout so the 30s budget is preserved with room to spare. The HRP stage is essentially instantaneous. Observability gets a 4-bucket histogram of which stage produced the day's solution; healthy steady-state should be 99%+ Stage 1, sub-1% Stage 2, ~0 Stage 3/4. A Stage-3 hit on 3 consecutive sessions or any Stage-4 hit is an operator alert — the substrate is the suspect (Σ ill-conditioning that the diagonal-fallback rule missed, or a Signal payload that survived validation but is numerically pathological).

### Improvement #10 — Warm-start state caching schema (concrete, on-disk)
**The gap:** CONTEXT Improvement #3 promises "Warm-start cache = previous day's primal+dual solution, invalidated when the universe changes by >5% or when `cond(Σ)` jumps by >2× day-over-day" but does not specify the storage format, the invalidation key, or how the cache survives a process restart. Without a concrete spec, the OSQP warm-start savings claimed in CONTEXT (5–20× iterations) won't be realized on the first call of any new process.

**Resolution: a single tentacle-owned ledger file under the data root, append-only.**
- **Location:** `data/strategy_engine/warmstart_cache/{date_iso}.parquet` (one file per session; rotates with the rebalance ledger).
- **Schema:** `(universe_hash: str, primal: list[float], dual_eq: list[float], dual_ineq: list[float], sigma_cond: float, n_universe: int, written_at_ns: int, decision_id: str)`. Parquet for typed-array storage and cheap re-read.
- **Universe-hash key:** `sha256(sorted_join(symbols, "|"))`. Cache hit requires **exact** universe match; the >5% drift rule applies to the *cond-jump invalidation* below, not the universe key itself (mismatched universes have differently dimensioned primal/dual vectors and can't be warm-started at all).
- **Invalidation gates (all must pass for a hit):**
  1. `universe_hash == cached.universe_hash` (exact dimension match).
  2. `cond(Σ_today) / cached.sigma_cond ∈ [0.5, 2.0]` (the day-over-day cond-jump gate from CONTEXT Improvement #3).
  3. `cached.written_at_ns` is from yesterday's session (no skipping a day — staleness > 96h trading-hour-aware → cold start).
  4. ChainMode of the new call matches the mode the cache was written under (no LIVE↔PAPER warm-start leakage; cached state is mode-segregated).
- **Write path:** at the *end* of `run_rebalance`, after the optimizer returns `optimal`, persist `(primal=w_var.value, dual_eq=constraints[*].dual_value, dual_ineq=...)` keyed by today's `decision_id`. A failed solve or any non-`optimal` status writes **nothing** — poisoned warm-starts are how OSQP stalls cascade across sessions.
- **Read path:** at the *start* of `mean_variance_optimize`, the scheduler attempts `load_warmstart(date=today-1, universe_hash=...)`. Miss is cheap and silent (cold start, no alert). Hit is logged to observability with the cached `decision_id` so a downstream investigation can chain back through the cache provenance.
- **Atomic write:** standard temp-file + rename (same pattern as risk-management I7b). Half-written parquet is the source of mysterious warm-start "hits" that aren't.
- **Hard cap on cache size:** retain only the last 21 sessions; older files are pruned at write time. The cache is a performance optimization, not a system-of-record — the append-only rebalance ledger remains the durable history.

### Improvement #11 — Alpaca-specific `ContractSpec` and execution-pivot adjustments
**The gap (cycle 5 pivot):** the project is pivoting from IBKR to Alpaca paper for v1. CONTEXT's "Cross-asset normalization" table only lists IBKR-relevant examples (ES futures, EURUSD spot, US equities, GLD ETF) and references `ContractSpec.multiplier` semantics built around futures and FX. Alpaca's product surface is narrower and has different lot/precision semantics; if we ship the IBKR-shaped registry into an Alpaca-driven path, the `contracts = round_to_lot(...)` lowering in the emitter (todo Epic 10) silently mis-sizes every order. Three concrete deltas:

**1. Asset-class subsetting.** Alpaca (paper, equities-only at the v1 tier we're using) supports `us_equity` and `crypto`. It does **not** offer futures, options, or spot FX. The `ContractSpec.asset_class` enum on this side narrows to `{us_equity, crypto}` — the futures/FX rows in CONTEXT's normalization table become out-of-scope for v1 and gain a note: "IBKR-roadmap; not active in the Alpaca-paper pivot". The optimizer logic is asset-class-agnostic so no math changes; only the registry contents and the emitter's `round_to_lot` behavior do.

**2. Fractional shares change `round_to_lot`.** Alpaca permits fractional notional and fractional-share orders for most US equities (down to 6-decimal precision) but **only on market orders during regular hours** and **only on names flagged `fractionable=True`** in their `/v2/assets` endpoint. `ContractSpec` gains:
- `fractionable: bool` (read from Alpaca `/v2/assets.fractionable`).
- `fractional_precision: int` (always 6 for Alpaca per current API; kept as a field so we can lower it to 4 or 2 per-name without a code change).
- `min_order_qty: Decimal` (Alpaca floors fractional notional at $1 of dollar-value or 0.000001 share — we encode both as `min_order_notional: Decimal` and `min_order_qty: Decimal` and the emitter takes the binding constraint).

The emitter's lowering becomes: if `fractionable` → `round_to_lot = round(w_final * NAV / price, fractional_precision)`; else → `round_to_lot = floor(w_final * NAV / price)` (integer shares, no fractional). Result: small NAVs ($10k–$50k paper accounts that are realistic for personal trading) can still hold positions in $300+ names like SPY without the entire weight rounding to zero.

**3. Symbol mapping and corporate-action source switch.** Alpaca uses bare ticker strings (`"AAPL"`, `"BRK.B"` with a literal dot) and does NOT expose IBKR's `reqContractDetails` corporate-action tags. Improvement #5's `CorporateAction.source` field gains a new enum value `"alpaca.corporate_actions"` (Alpaca exposes `/v2/corporate_actions` with daily-snapshot ex-date payloads). The `compliance` producer Epic gains a sibling subtask: implement an Alpaca adapter alongside (eventually) the IBKR one. The `symbol` canonicalization in `shared-core` uses Alpaca's wire format as the canonical string for v1 — IBKR-style `Contract` IDs become a translation layer if/when we re-add IBKR later. This avoids re-keying the rebalance ledger on a venue switch.

**Three small but load-bearing notes for the implementation:**
- `ContractSpec.multiplier` is **1** for every Alpaca product (equities and crypto), so the `mult=50` (ES) etc. rows in CONTEXT's table are stubs not exercised by v1. The field stays so the IBKR re-add is a config change, not a schema change.
- `valid_until` in `TargetPositionBatch` must respect Alpaca's regular-hours window for fractional orders. Cap `valid_until_ns = min(session_close_ns, regular_hours_close_ns) - 5min_buffer`. Fractional orders submitted after 15:55 ET are rejected by Alpaca and would silently leave a target unfilled.
- Alpaca's paper account NAV (`/v2/account.equity`) replaces IBKR's `reqAccountSummary` in `risk-management`'s 09:30 snapshot. The "NAV source" key decision (read from risk-management's snapshot, not directly) is **unchanged** — only the implementation behind risk-management's snapshot swaps. Strategy-engine is venue-agnostic by design and stays so.
