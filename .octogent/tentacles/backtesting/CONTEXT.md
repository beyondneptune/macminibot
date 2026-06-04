# backtesting

Walk-forward replay of the once-per-day rebalance strategy with IBKR-conservative fee/slippage models, purged & embargoed CV, and DSR/PBO go/no-go gates.

## Status
Greenfield — no source code yet. Architecture is **ratified** (this document). The 10 epics in `todo.md` are sized for an agent each.

## Scope
- Walk-forward **orchestrator** that IMPORTS `ml_research.cv.WalkForwardHarness` + `ml_research.cv.CPCVSplitter` (ml-research is the OWNER of split logic — no parallel splitter here) and wraps them with the execution sim, fee/slippage models, and metrics
- IBKR Pro Tiered fee model (equities, ES futures, IDEALPRO FX)
- Almgren-Chriss square-root slippage with spread-half floor; MOO orders default
- Execution simulator that writes the **same trade-journal schema** as live (`execution` tentacle owns the schema; see "Trade-journal schema" below for the ratified columns/dtypes)
- No-lookahead wrapper + fuzz test on dollar-bar close-times
- Metrics: per-fold Sharpe/Sortino/Calmar, **PSR**, **DSR** (gate ≥ 0.95), **PBO**, turnover, fee drag, hit rate, live-vs-backtest fill parity
- **Bootstrap Sharpe CIs + false-discovery floor** via stationary block-bootstrap (Politis-Romano 1994) and randomized-feature monte carlo (alpha must clear the 95th-pct of shuffled-feature DSR)
- HTML report renderer (one self-contained file per run; reproducible from manifest)

## Out of scope
- Event-driven tick replay (overkill for once-per-day rebalance)
- ML model design (owned by `ml-research`)
- Position sizing (owned by `strategy-engine`)
- Cloud backtesting platforms — runs locally on the Mac mini M4

## Boundaries

| Direction | Tentacle | Contract |
|---|---|---|
| Consume | `ml-research` | **`ml_research.cv.WalkForwardHarness` + `CPCVSplitter`** (split logic OWNER — backtesting imports, never reimplements) AND signal generator artifact (predict overnight on T-1 dollar bars) |
| Consume | `strategy-engine` | Position-sizer + portfolio constructor, REPLAYED through the production code path — no parallel implementation |
| Consume | `market-data` | T-1 dollar bars (or proxy daily bars pre-2010) for the 10-name bootstrap universe |
| Consume | `execution` | Trade-journal schema (single source of truth for fills, live + backtest) — `execution` is OWNER, writes both live trades and backtest fills in the same parquet format |
| Produce | `ml-research` | Per-fold metrics JSON, DSR/PBO numbers for promotion gates |
| Produce | `observability` | HTML report, live-vs-backtest parity table |
| Produce | `risk-management` | Drawdown/turnover envelopes from OOS folds |

## Key Decisions (ratified)

### Strategy class
Once-per-day rebalance, ML-driven multi-asset systematic. Predict overnight on T-1 dollar-bar features, execute at next session **open** via MOO. Hold for the session, flatten or roll at next open. Not event-driven.

### Universe (v0 → v1)
- **v0 — 10-name bootstrap**: AAPL, MSFT, JPM, JNJ, XOM, KO, PG, GLD, SPY, TLT — all continuously liquid since the 1990s. Survivor-bias-acceptable for v0 only.
- **v1 — MANDATORY before scaling to 5,000 names**: PIT universe service ingesting historical SP500 constituents (Wikipedia revision history) + delisted tickers (SEC EDGAR Form 25). Delisted name = terminal return = last close → delisting value (zero or cash-out per Form 25 notes).

### Walk-forward (expanding window)
| Param | Value | Source |
|---|---|---|
| Initial train | 2000-01 → 2007-12 (8 yr, spans dot-com + pre-GFC) | — |
| Embargo | 5 trading days | López de Prado AFML §7.4 |
| Purge | 1 day around fold boundary | López de Prado AFML §7.2 |
| Test fold | 1 year (≈252 obs × 10 names ≈ 2,520 trades — stable Sharpe SE) | — |
| Slide | 1 year → ~18 OOS folds 2008 → 2026 | — |
| Refit cadence | Every slide (annual) | — |
| Hyperparam selection | **CPCV inside each train fold**, N=6, k=2 → 15 combos | López de Prado AFML Ch. 12 |

**Implementation**: split logic is imported from `ml_research.cv.walk_forward` — the **canonical signature ratified in `ml-research/CONTEXT.md`** is:

```python
WalkForwardHarness(
    train_min_years=5, train_max_years=15,
    val_days=252,
    embargo_days=5, purge_days=1,
    test_days=21, refit_cadence="monthly",
).split(index) -> Iterator[FoldSpec]
# FoldSpec = (train_idx, val_idx, test_idx, fold_id, fold_meta)
```

`CPCVSplitter(N=6, k=2)` is fed into the inner hyperparam search. Backtesting NEVER owns splitter code.

**Cycle-3 reconciliation (was a real bug)**: prior versions of this CONTEXT documented a different signature (`train_init`, `test_years=1`, returning `(train_idx, test_idx)`) — that was drift from cycle 1 vs the ml-research-ratified contract. The CI test below now pins the ml-research-canonical kwargs. Annual-cadence walk-forward (the "1-year test fold" intent originally documented here) is expressed by configuring `test_days=252, refit_cadence="annual"` rather than a parallel splitter — i.e. the harness is reconfigured, not reimplemented.

CI contract test (`tests/contracts/test_ml_research_imports.py`, runs in both tentacles):
```python
def test_walk_forward_signature_stable():
    from ml_research.cv.walk_forward import WalkForwardHarness
    from ml_research.cv import CPCVSplitter
    sig = inspect.signature(WalkForwardHarness.__init__)
    expected = {"train_min_years", "train_max_years", "val_days",
                "embargo_days", "purge_days", "test_days", "refit_cadence"}
    assert expected <= set(sig.parameters)
    # FoldSpec is a 5-tuple — explicit unpack guard
    h = WalkForwardHarness()
    first = next(iter(h.split(_synthetic_index())))
    assert len(first) == 5  # (train_idx, val_idx, test_idx, fold_id, fold_meta)
    assert {"N", "k"} <= set(inspect.signature(CPCVSplitter.__init__).parameters)
```

Nested CV: outer walk-forward is **never** used for selection — `model_spec` is frozen before the OOS year is touched.

### Fee model — IBKR Pro Tiered (retail, <300k shares/mo) + Alpaca branch
Source: IBKR Pricing → Commissions → "US Tiered" page (verified Q1 2026; numbers checked against `https://www.interactivebrokers.com/en/pricing/commissions-home.php`).

| Asset | Formula | Notes |
|---|---|---|
| US equity (AAPL, GLD, SPY, TLT) | `max($0.35, $0.0035 × shares)` IBKR commission + **~$0.003/sh pass-through** (NYSE/NSDQ exch fees + SEC §31 + TAF + FINRA + NSCC clearing) → **all-in ~$0.0065/share, $0.35 floor, 1% trade-value cap** | Conservative; pass-through varies ±20% by venue/order-type |
| ES future | `$0.85 IBKR + $1.38 CME exchange + $0.02 reg = $2.25/contract/side` | E-mini S&P 500 |
| EURUSD IDEALPRO | `max($2.00, 0.20 bps × notional)` per side | Spot FX |

**SEC §31 fee** is sell-side only and rate changes mid-year — model with annual schedule lookup, not a hard-coded constant.

**Cycle-5 add — Alpaca broker branch (`alpaca_commission_free_2026q1`)**: Alpaca offers $0 commission on US equities + ETFs; the only mandatory pass-throughs are regulatory (SEC §31 sell-side + FINRA TAF). When the run manifest's `broker` key is `"alpaca"`, the fee model substitutes:
- `commission = 0.0`
- `exchange_fee = 0.0` (Alpaca absorbs venue routing fees on the retail tier)
- `reg_fee = SEC_31_RATE[year] * notional_usd` if `side == "SELL"` else `0`, plus `FINRA_TAF_PER_SHARE[year] * shares` (sell-side, currently $0.000166/share capped $8.30/trade)
- `clearing_fee = 0.0`

Backtests are run **twice** for any strategy targeted at production-on-Alpaca: once with `ibkr_pro_tiered_2026q1` (canonical conservative gate) and once with `alpaca_commission_free_2026q1` (realistic expected fee drag for actual deployment). The aggregate metrics file records both Sharpe numbers; promotion gates apply to the IBKR-conservative number so a fee-regime regression doesn't silently inflate DSR. The fee-drag delta between the two is the **broker subsidy** line in the HTML report and is the load-bearing input for the "should we switch brokers" decision. Selector: `FeeModel.for_broker(broker: Literal["ibkr_pro", "alpaca"]) -> Callable`.

### Slippage model
- Almgren-Chriss square-root: `slip_bps = σ_daily_bps × k × sqrt(Q / ADV)`, k ≈ 0.1 (Almgren, Thum, Hauptmann, Li 2005 — *Direct Estimation of Equity Market Impact*, Risk).
- **Spread half-cost floor** per asset class: equity ≥ 1 bp, ES ≥ 0.25 tick, EURUSD ≥ 0.1 pip.
- For retail size (<1% ADV) market impact is negligible; spread dominates → floor binds in practice.
- MOO orders default (deepest book at the open auction); LMT-on-open is the alternative for the report comparison.

### Pitfall guards (3 named + 1 added)
| Pitfall | Guard | Enforcement |
|---|---|---|
| Label leakage via overlapping windows | Purged k-fold + embargo | Unit test in `cv_splitter_test.py` asserting no train-set timestamp lies within `[test_start - purge, test_end + embargo]` |
| Look-ahead via dollar-bar close-time | `no_lookahead_wrapper.py` proxies all feature reads; raises if `ts > current_t` | Property-based fuzz test (Hypothesis) generating random access patterns |
| Refit on test set (hyperparam peeking) | Nested CV; `model_spec` is hashed + frozen before OOS fold opens | CI check: hash recorded in manifest, compared post-run |
| **Trade-on-close vs trade-on-open mismatch** (added) | Signal computed on `close[T-1]`, fill priced at `open[T]` with overnight gap captured in PnL attribution | Manifest records both prices per trade; report flags gap-PnL separately so it isn't credited to alpha |

### Solver / library choice
**Roll-your-own walk-forward driver** (Python + pandas + numpy + numba for hot loops), NOT vectorbt/backtrader/lean.

Reasoning:
- The strategy is **once-per-day** — there is no event-loop hot path to vectorize. vectorbt's main value (vectorized signal evaluation across millions of bars) is wasted.
- backtrader and lean are event-driven OOP frameworks built for tick/bar streaming; their walk-forward and purged-CV support is bolt-on at best.
- Live/backtest parity demands we **call the production `strategy-engine` code path directly** — wrapping a third-party framework adds a translation layer that's exactly the kind of skew we're trying to eliminate.
- López de Prado's CPCV and DSR are not first-class in any mainstream backtester; we'd implement them ourselves anyway.
- Dependency budget: pandas, numpy, numba, pyarrow, scipy, scikit-learn (for splitter base classes). All on the Mac mini's 16 GB easily.

### Upstream data requirements (binding on `market-data`)

The slippage model (Almgren-Chriss sqrt with spread-half floor) consumes `Q/ADV`. Backtesting requires `market-data` to expose, alongside every dollar/daily bar read:

| Field | Type | Definition | Notes |
|---|---|---|---|
| `adv_21d_shares` | `int64` | Rolling 21-trading-day mean of daily `volume` (shares for equity/ETF; contracts for futures; notional units for FX) | Lagged — uses bars strictly < `as_of_ts`; never includes the current bar (look-ahead guard) |
| `adv_21d_notional_usd` | `float64` | Rolling 21-day mean of `volume × vwap` in USD; cross-asset comparable | FX cross-rated to USD via `market-data`'s reference snapshot |
| `spread_bps_median_21d` | `float64` | Median quoted spread over the trailing 21 days, bps | NULL until L2/BID_ASK storage unlocks; backtesting falls back to the asset-class spread floor when NULL |

Justification: the slippage model's `sqrt(Q / ADV)` term is meaningless without an ADV column on the read side. Pulling raw volume and computing the rolling mean inside `backtesting` would (a) duplicate logic across `ml-research` (which already needs identical rolling stats for `realized_vol_21d`) and (b) miss the look-ahead guard that `market-data`'s no-lookahead read API enforces centrally. **Action: filed against `market-data/todo.md` as part of bar-read API v1.**

### HTML report contract (binding on `observability`)

The `report.html` artifact written under `runs/<run_id>/report.html` is:
- **Generated by `backtesting`** (this tentacle owns the renderer — Epic 9), self-contained single file, deterministic from `manifest.json` + per-fold parquet trades.
- **Served by `observability`**: `observability` runs a static file handler rooted at `runs/` and exposes `https://<host>/runs/<run_id>/report.html`. No dynamic generation server-side. No database round-trip on serve.
- **Index page**: `observability` additionally renders `runs/index.html` listing all runs (sortable by `run_id`, `strategy_sha`, `DSR`, `git_sha`) — also generated from the manifests, not from a DB.
- **Retention**: `observability` owns the policy (proposed: keep last 50 runs hot on disk, archive older to R2 via the `market-data` cold-tier pattern).

This split keeps `backtesting` free of HTTP concerns and keeps `observability` free of pandas/matplotlib deps. **Action: documented in `observability/CONTEXT.md` (not yet ratified there — pending cross-tentacle PR).**

### `model_spec_hash` canonicalizer (provenance contract with `ml-research`)

`manifest.json["model_spec_hash"]` is computed by **`ml-research`**, not by `backtesting`. Specifically:

- **Owner**: `ml_research.registry.spec.canonicalize_and_hash(model_spec: dict) -> str`
- **Canonicalizer**: `json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)` of a dict with keys `{git_sha, feature_spec_hash, label_spec_hash, hyperparams, train_window, refit_cadence, lgbm_objective}` — the same canonicalization used by `ml-research`'s registry manifest (see `ml-research/CONTEXT.md` Conventions, "Every training run produces a manifest"). Hash function: `hashlib.sha256` hex digest.
- **Frozen-before-OOS guarantee**: `ml-research` writes the hash at the moment `model_spec` is finalized for a given outer fold; `backtesting` reads it via the registry handle and copies it verbatim into the run manifest. If `backtesting` ever has to compute it locally, that's a contract violation (parallel hashers will drift).
- **CI check**: `tests/contracts/test_model_spec_hash.py` (lives in `ml-research`, runs in both CIs) round-trips a fixture spec through the canonicalizer and asserts the hash matches a golden hex string.

**Open coordination item**: `ml-research/CONTEXT.md` does not yet name `canonicalize_and_hash` or pin the field set. Flagged for cycle-3 ml-research follow-up.

### Reproducibility — RNG seed contract (cycle 4)

Every stochastic step in `backtesting` derives from a **single root seed** recorded in `manifest.json["root_seed"]` (uint64, default `0xB10CC0DE`, overridable per-run). Sub-seeds are derived deterministically via `numpy.random.SeedSequence(root_seed).spawn(n)` so two parallel resamplers cannot share state and bytewise re-running a manifest reproduces every CI/MC number to the last digit.

| Consumer | Sub-seed slot | Notes |
|---|---|---|
| Stationary block-bootstrap Sharpe CI (Epic 8) | `spawn[0]` per fold; per-fold seed = `spawn[0].spawn(fold_id)` | Politis-Romano resampler; B=10,000 |
| Politis-White auto block-length | `spawn[1]` | Itself uses bootstrap internally |
| Randomized-feature monte carlo / FDF (Epic 11) | `spawn[2].spawn(perm_id)` per of 100 column permutations | Per-column-independent shuffle |
| Romano-Wolf stepdown bootstrap (Epic 12) | `spawn[3]` | Family-wise null distribution |
| LightGBM `feature_fraction` / `bagging` (delegated to ml-research) | passed via `model_spec.hyperparams["seed"]` — NOT spawned here | `ml-research` owns model RNG; we only forward |

Contract: `numpy.random.default_rng(sub_seed)` is the only acceptable RNG constructor; bare `numpy.random.*` or `random.*` calls are CI-banned (AST scan in `tests/contracts/test_no_global_rng.py`). Re-running with the same `root_seed` + same input parquet → byte-identical `metrics/aggregate.json`; the report renderer (Epic 9) asserts this on every run via a self-replay check.

### Run cache strategy (cycle 4)

Walk-forward replay over 18 folds × refit + 10,000-resample bootstrap × 100-shuffle FDF dominates wall-clock; re-running an *identical* config (same manifest inputs modulo `run_id` + `git_sha-of-renderer-only` diffs) must not refetch bars or refit models. Three-layer cache, keyed by content hash, evicted LRU:

| Layer | Key | Location | Hit semantics |
|---|---|---|---|
| **Bar reads** | `sha256(symbol, bar_kind, start, end, market_data_schema_version)` | `cache/bars/<key>.parquet` | Owned by `market-data`'s read API (per its CONTEXT) — backtesting just consumes; we never write here |
| **Per-fold model artifact** | `model_spec_hash` (ml-research-canonical, see above) + `train_idx_hash` | `cache/models/<model_spec_hash>/<train_hash>.lgb` | Hit ⇒ skip refit, load booster directly; controlled by `ml_research.registry.load_or_fit(...)` — `backtesting` calls, never bypasses |
| **Per-fold metrics + journal** | `sha256(model_spec_hash, test_idx_hash, fee_model_version, slippage_model_version, root_seed_spawn[0..3])` | `cache/folds/<key>/{trades.parquet,metrics.json}` | Hit ⇒ skip the entire fold replay; manifest records `cache_hit=true` per fold so the report flags how much was actually computed |

Cache **invalidation triggers**: any change to `fee_model.version`, `slippage_model.version`, `market_data_schema_version`, `model_spec_hash`, or `root_seed` busts the relevant layer. `--no-cache` CLI flag forces full recompute (used for promotion runs — promotion DSR must be from cold-cache numbers, gated in CI). Size cap: 20 GB on the M4's 245 GB disk (LRU evict); enforced by a `cache_gc` cron hook in the run wrapper, not at read time.

### Compute budget (Mac mini M4, 16 GB / 245 GB)

A full v1 walk-forward run (18 folds × annual refit × 10-name universe × 26 years of dollar bars + Epic 8 bootstrap + Epic 11 FDF) targets the following budget. Exceeding any line is a CI failure (recorded in `manifest.json["compute"]`):

| Resource | Soft cap | Hard cap | Enforcement |
|---|---|---|---|
| Wall-clock per run (warm cache) | 5 min | 15 min | timeout in `run_backtest` entrypoint |
| Wall-clock per run (cold cache, full refit + bootstrap) | 45 min | 90 min | timeout; FDF (Epic 11, ~100× refit) is **excluded** — runs separately at promotion only |
| Peak RSS | 6 GB | 10 GB | `resource.setrlimit(RLIMIT_AS, hard)`; leaves headroom for macOS + IBKR Gateway co-resident |
| CPU cores | 6 of 10 (M4 has 4P+6E) | 8 of 10 | `numba.set_num_threads`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `LGBM num_threads` — all set from a single `BACKTEST_NUM_THREADS` env var so we don't fight ourselves with thread oversubscription |
| Per-fold parquet write | 50 MB | 200 MB | post-write size check; >hard cap flags a schema regression (e.g. accidentally writing per-tick instead of per-fill) |
| Cache footprint | 15 GB | 20 GB | `cache_gc` LRU evict |

Justified by 10-name v0; v1 (5,000-name PIT universe) will require re-budgeting and likely a streaming/chunked refit — flagged in the v1 universe expansion epic, not addressed here.

## Conventions

### Run manifest (reproducibility bar)
Every run writes `manifest.json`:
```
{
  "run_id": "<ulid>",
  "git_sha": "...",
  "data_window": {"start": "2000-01-03", "end": "2026-04-30"},
  "universe_snapshot_id": "...",        // hash of PIT universe at run start
  "strategy_id": "...", "strategy_sha": "...",
  "model_spec_hash": "...",             // frozen before OOS open
  "fee_model": {"name": "ibkr_pro_tiered_2026q1", "version": "..."},
  "slippage_model": {"name": "almgren_sqrt_k0.1", "version": "..."},
  "cv": {"embargo_days": 5, "purge_days": 1, "cpcv": {"N": 6, "k": 2}},
  "metrics": {...}
}
```

### Trade-journal schema (cross-tentacle contract — OWNED by `execution`)
`execution` is the OWNER and writes both live trades and backtest fills in the same parquet format. Backtesting's `ExecutionSimulator` calls `execution.journal.write_fill(...)` — no second writer, no second schema. This makes the live-vs-backtest parity report a straight `JOIN ... USING (client_order_id)`.

**Concrete parquet schema (v1, ratified):**

| Column | dtype | Nullable | Notes |
|---|---|---|---|
| `client_order_id` | `string` (ULID) | NO | **PK component**; deterministic in backtest = `hash(run_id, decision_id)` |
| `backtest_run_id` | `string` | YES | NULL in live; set in backtest. **PK component** |
| `decision_id` | `string` (ULID) | NO | links to upstream signal row |
| `ts_decision` | `timestamp[ns, UTC]` | NO | when strategy emitted intent |
| `ts_submit` | `timestamp[ns, UTC]` | NO | when OMS sent to broker |
| `ts_ack` | `timestamp[ns, UTC]` | YES | broker ack (NULL in backtest) |
| `ts_fill` | `timestamp[ns, UTC]` | NO | fill timestamp |
| `symbol` | `string` | NO | canonical (e.g. `AAPL`, `ESZ6`, `EUR.USD`) |
| `asset_class` | `dictionary<string>` | NO | `equity` / `future` / `fx` |
| `side` | `dictionary<string>` | NO | `BUY` / `SELL` |
| `qty` | `int64` | NO | signed by convention? NO — unsigned; sign comes from `side` |
| `order_type` | `dictionary<string>` | NO | `MOO` / `LMT` / `MKT` |
| `tif` | `dictionary<string>` | NO | `OPG` / `DAY` / `IOC` |
| `limit_px` | `float64` | YES | NULL for MOO/MKT |
| `fill_px` | `float64` | NO | actual or simulated |
| `fill_qty` | `int64` | NO | partials allowed → multiple rows same `client_order_id`+`fill_seq` |
| `fill_seq` | `int32` | NO | **PK component**; `0` if single fill |
| `ref_px_close_tm1` | `float64` | NO | T-1 close used by signal (pitfall #4 attribution) |
| `ref_px_open_t` | `float64` | NO | T open used for fill |
| `commission` | `float64` | NO | IBKR side only |
| `exchange_fee` | `float64` | NO | venue pass-through |
| `reg_fee` | `float64` | NO | SEC §31 + TAF + FINRA |
| `clearing_fee` | `float64` | NO | NSCC etc. |
| `slippage_bps_modeled` | `float64` | YES | NULL in live |
| `slippage_bps_realized` | `float64` | NO | `(fill_px - ref_px_open_t) / ref_px_open_t * 1e4 * side_sign` |
| `gap_pnl` | `float64` | NO | `(open_t - close_tm1) * qty * side_sign` |
| `session_pnl` | `float64` | NO | intra-session PnL only |
| `venue` | `string` | YES | `NYSE` / `NSDQ` / `CME` / `IDEALPRO` |
| `route` | `string` | YES | IBKR SMART / DIRECT |
| `account` | `string` | NO | IBKR account id; `BACKTEST` literal in sim |
| `strategy_sha` | `string` | NO | for parity join across live/backtest |

**Primary key**: `(client_order_id, fill_seq, backtest_run_id)` — `backtest_run_id` is part of PK so the same `client_order_id` can appear in both a live row (PK = `(coid, 0, NULL)`) and N backtest rows (PK = `(coid, 0, run_id_X)`).

**Partitioning**: `runs/<run_id>/trades/fold=YYYY/part-*.parquet` in backtest; `live/trades/dt=YYYY-MM-DD/part-*.parquet` in live. Same schema → `pyarrow.dataset.dataset([live_path, bt_path]).to_table()` works without translation.

**Contract test** (`tests/contracts/test_journal_schema.py`): import `execution.journal.SCHEMA` (a `pyarrow.Schema`), assert it equals the pinned schema above; both tentacles' CI run it.

**Note on bps dtypes** (cycle 4): `slippage_bps_modeled` / `slippage_bps_realized` are `float64` and **NOT** subject to `shared-core`'s pending `Decimal`-vs-fixed-`int64` decision. That decision binds **prices and quantities** (where rounding error compounds into PnL and reconciles to the penny against IBKR statements); bps measurements are derived ratios already lossy to ~1 ULP of float64 at any realistic scale, and the report's bucketed parity histograms (5 bps bins) swamp the dtype-precision difference. Decision is recorded here so when `shared-core` ratifies its Decimal type, the journal schema is not retroactively rewritten for these two columns. `fill_px`, `limit_px`, `ref_px_*` **will** follow whatever `shared-core` decides — flagged for re-ratification in that PR.

### Walk-forward output layout
```
runs/<run_id>/
  manifest.json
  trades/fold=YYYY/part-0.parquet         # journal, same schema as live
  metrics/fold=YYYY.json                  # per-fold metrics
  metrics/aggregate.json                  # DSR, PBO, mean/std Sharpe
  report.html                             # self-contained
```

### Metrics & gates
| Metric | Definition | Gate |
|---|---|---|
| Sharpe (per fold) | annualized, geometric | report mean ± std |
| Sortino | downside-vol Sharpe | report |
| Calmar | CAGR / max-DD | report |
| **PSR** | Probabilistic Sharpe Ratio, skew/kurt-corrected (Bailey & López de Prado 2012) | report |
| **DSR** | Deflated Sharpe Ratio (Bailey & López de Prado 2014) — corrects for N trials | **≥ 0.95 to promote** |
| **PBO** | Probability of Backtest Overfitting (Bailey et al. 2016, combinatorial symmetric CV) | **≤ 0.5 to promote** |
| Turnover | annualized 2-way notional / NAV | report (informs fee drag) |
| Fee drag | gross Sharpe − net Sharpe | report |
| Hit rate, avg win / avg loss | per fold | report |
| Live-vs-backtest fill parity | % of live fills within ±X bps of modeled | gate when live data exists |
| **Sharpe 95% CI (bootstrap)** | Stationary block-bootstrap (Politis-Romano 1994), block-size via Politis-White, B=10,000 resamples on daily returns | report; **CI lower bound > 0** required to promote |
| **False-discovery floor (FDF)** | Shuffle features per-column 100×, retrain, record DSR; promotion DSR must exceed 95th-pct of shuffled-feature DSR | gate; protects against the "DSR > 0.95 but on noise" failure mode |

## Improvements over baseline (delta vs. generic bootstrap)

1. **Trade-on-open/close pitfall + gap-PnL attribution**: added 4th pitfall; signal at `close[T-1]`, fill at `open[T]`, overnight gap broken out so it isn't credited as alpha. Cheap to implement, catches the most common "my Sharpe looks great in backtest" bug for daily-rebalance strategies.

2. **Library choice ratified to roll-your-own with reasoning** (not vectorbt/backtrader/lean): because once-per-day cadence wastes vectorization, and live/backtest parity requires calling the production `strategy-engine` directly with no wrapper translation layer.

3. **Trade-journal schema is a cross-tentacle contract owned by `execution`**: identified a gap — `execution/CONTEXT.md` does not yet pin the schema. Action item escalated as Epic 0 dependency; backtesting drafts, `execution` ratifies, both write to the same parquet definition so live-vs-backtest parity reports are a straight diff rather than a translation. **Cycle 2: concrete schema (columns, dtypes, PK) now pinned above; contract test added.**

4. **(Cycle 2) Import ml-research's split logic, never reimplement**: `WalkForwardHarness` + `CPCVSplitter` live in `ml_research.cv`; backtesting imports and wraps them. CI signature contract test runs in both tentacles so drift breaks the build immediately rather than silently producing two parallel splitters with subtly different embargo semantics.

5. **(Cycle 2) Bootstrap CIs + false-discovery floor**: stationary block-bootstrap (Politis-Romano 1994; block-length via Politis-White automatic selection) yields Sharpe 95% CI — **lower bound > 0 required to promote**. Independently, randomized-feature monte carlo (shuffle each feature column, retrain, repeat 100×) sets a **DSR false-discovery floor** at the 95th-pct of the null distribution. DSR > 0.95 alone catches multiple-trial overfitting but not "signal happens to look like noise"; the FDF gate closes that hole. Cost: ~100× the inner-loop refit cost, run once at promotion time only — not every backtest.

6. **(Cycle 3) Splitter signature reconciled with `ml-research`** — sibling-CONTEXT diff caught a real drift: backtesting documented `WalkForwardHarness(train_init=..., test_years=1)` returning a 2-tuple; ml-research's ratified contract is `(train_min_years, train_max_years, val_days, embargo_days, purge_days, test_days, refit_cadence)` returning a 5-tuple `FoldSpec`. Updated the import contract and the CI test to the upstream-canonical form. Annual-cadence walk-forward is now expressed by configuring `test_days=252, refit_cadence="annual"` rather than a forked splitter — preserving the "wrap, don't replace" principle that anchors the leakage-safety guarantees.

7. **(Cycle 3) Closed three cross-tentacle gaps that cycle 2 only flagged**: (a) **21-day rolling ADV** is now a named field on `market-data`'s bar-read API with a documented look-ahead guard (lagged, never current bar); (b) **`report.html`** ownership split — `backtesting` generates the self-contained HTML, `observability` serves it as a static file with an auto-rendered run index; (c) **`model_spec_hash` provenance** assigned to `ml_research.registry.spec.canonicalize_and_hash` with a pinned canonicalizer (`json.dumps(sort_keys=True, separators=(",",":"),default=str)` + `sha256`) and a round-trip golden-hash CI test. Each gap is now a concrete action item against the owning tentacle rather than a "TBD" in our manifest.

8. **(Cycle 4) Reproducibility, run cache, and compute budget pinned**: (a) single `root_seed` in manifest with `SeedSequence.spawn` deriving sub-seeds for each stochastic consumer (bootstrap CI, Politis-White block-length, FDF, Romano-Wolf) — bare `np.random.*` is AST-banned in CI; bytewise replay guaranteed. (b) three-layer content-hash cache (bars / models / per-fold trades+metrics) keyed on the same hashes used for promotion gates — promotion runs gated to cold-cache via `--no-cache`. (c) Mac-mini-M4 budget table (5 min warm / 45 min cold wall-clock, 6 GB RSS, 6 cores) recorded in `manifest.json["compute"]` and CI-enforced. Together these close the "you said it reproduced but I can't tell if it actually re-ran" failure mode and stop the M4 from getting OOM-killed by parallel-fold refits.

9. **(Cycle 4) `slippage_bps` dtype decision recorded ahead of `shared-core`**: `float64` for the two bps columns is the ratified choice and explicitly *not* gated on shared-core's Decimal-vs-int64 decision (which binds prices/quantities only). Prevents a schema rewrite cascade when shared-core ratifies.

10. **(Cycle 4) Splitter-signature drift in `todo.md` Epic 4** caught and fixed: CONTEXT.md was reconciled in cycle 3 to the ml-research-canonical `{train_min_years, train_max_years, val_days, embargo_days, purge_days, test_days, refit_cadence}` 5-tuple-FoldSpec API, but `todo.md` Epic 4 still listed the legacy `{train_init, test_years}` 2-tuple. Two places, same contract — drift between them is exactly the class of bug we're trying to prevent. Now consistent.

11. **(Cycle 3) Multiple-testing across strategies — Romano-Wolf stepdown as a v2 epic**: DSR's deflation only corrects for N trials of the **same** strategy family; once we're comparing >1 strategy class (e.g. momentum vs mean-reversion vs vol-carry) the family-wise error rate balloons again. White's Reality Check (Econometrica 2000) and Hansen's SPA (J. Bus. Econ. Stat. 2005) handle the joint null; Romano-Wolf stepdown (Econometrica 2005) is strictly more powerful (controls FWER without sacrificing as much power as Bonferroni) and is the modern default. Not on the v1 critical path (we only have one strategy class), but added as Epic 12 so the gate exists before strategy proliferation makes p-hacking trivial.

12. **(Cycle 5) Alpaca branch in the fee model**: Alpaca's $0-commission retail offer is a real broker option for v0 (10-name equity universe); modelling only the IBKR-conservative path would silently overstate true production fee drag whenever the strategy is actually deployed against Alpaca. The fee model now carries two pinned versions (`ibkr_pro_tiered_2026q1` and `alpaca_commission_free_2026q1`); every promotion run produces *both* aggregate metric sets, the IBKR number governs the DSR ≥ 0.95 gate (conservative-by-construction), and the delta becomes a first-class "broker subsidy" line in the report so the broker-choice decision is informed by per-strategy turnover characteristics rather than priors.

13. **(Cycle 5) Live-paper parity report against Alpaca paper-trading**: live-vs-backtest parity (Epic 10) currently presumes IBKR live fills. Alpaca's paper-trading API is free, instantly available, and returns realistic fills against the Alpaca real-time book — making it the right *daily* canary for live/backtest skew detection long before the strategy touches a funded account. New Epic 16 wires the Alpaca paper-trading account as a continuous shadow runner; the parity diff (same `strategy_sha`, same decision dates, same Epic 0 journal schema) flags drift in fill-price, slippage-modeled-vs-realized, and fill-rate within one trading day rather than waiting for funded-account history to accumulate. Cost is ~zero (Alpaca paper is free and rate-limit-friendly) and the early-warning value is large.

14. **(Cycle 5) Regime-stratified evaluation breakdown**: a single all-fold DSR can pass the 0.95 gate while the strategy is structurally broken in (e.g.) high-vol regimes — the OOS folds happen to be vol-low-weighted. New Epic 17 stratifies every metric by an ex-ante regime label (VIX terciles + yield-curve slope sign + SPY 200-DMA above/below; computed from `market-data`, lagged so the labels themselves are no-lookahead). The aggregate report shows DSR / Sharpe / PBO / fee-drag *per regime bucket* and the report-renderer flags any bucket where DSR collapses by >0.3 vs all-fold even if the aggregate gate passes. Catches "works on average, blows up in 2008/2020-style regimes" before live deployment rather than during it.

15. **(Cycle 6) `WalkForwardDriver` now broker-aware** — Cycle 5 landed the Alpaca fee model (`equity_alpaca`) but the driver's `_run_fold` only ever dispatched to `equity_tiered`, leaving the `alpaca_commission_free_2026q1` branch as dead code from the driver's POV and the "run twice" architecture non-functional. Driver gains `broker: Literal["ibkr_pro","alpaca"]` (defaults to IBKR — conservative), validates `broker="alpaca"` requires `asset_class="equity"` (Alpaca's commission-free retail tier is US-equities-only; the IEX-only data caveat is upstream of this tentacle), and derives `side` from delta-shares sign so sell-side regs (SEC §31 + FINRA TAF) attract correctly on rebalances flipping long-to-flat. Promotion runs now actually produce both fee totals; the broker-subsidy delta (IBKR − Alpaca) is a real number on the same fills, not a back-of-envelope. Test `test_driver_alpaca_has_lower_fee_drag_than_ibkr` pins the inequality on synthetic fills so a future refactor cannot silently re-route Alpaca traffic through the IBKR path.

16. **(Cycle 6) DSR threshold calibration helper (`dsr_expected_max_sr`)** — CONTEXT pinned a generic `DSR ≥ 0.95` gate that implicitly assumes `sigma_SR = 1.0` across trials (the standard textbook deflation). With Epic 6's WalkForwardDriver now producing per-fold Sharpes, we have *observed* cross-fold variance and can calibrate the deflation benchmark to actual strategy stability rather than the worst-case prior: `dsr_expected_max_sr(n_trials, sr_trials_std, periods)` exposes Bailey & López de Prado (2014) eq. (5) directly and feeds into `psr(returns, benchmark_sr=...)`. Stable strategies (low cross-fold Sharpe std) face a *looser* effective DSR hurdle — the generic 0.95 was paying for variance we don't actually have, leaving real alpha gated out. The existing `dsr()` is refactored to call the helper with `sigma=1.0` so the generic gate is preserved by default and the calibrated path is opt-in (caller computes their own benchmark from `result.fold_metrics`).

17. **(Cycle 8) `bootstrap_sharpe_ci` — Politis-Romano stationary bootstrap CI on OOS Sharpe** — closes the CONTEXT-pinned `Sharpe 95% CI lower bound > 0` promotion gate. Cycle 7 produced +2.03 net Sharpe on the Alpaca-broker Ridge baseline and honestly flagged it as "suspiciously high for a tech-mega-cap 2024-2026 bull window"; the new bootstrap CI (B=10,000, mean-block-length via Politis-White automatic selection with the Patton-Politis-White 2009 correction) answers that question quantitatively. New API: `bootstrap_sharpe_ci(returns, B=10_000, block_length="auto", alpha=0.05, periods=252, seed|rng) -> BootstrapSharpeResult` with fields `(point_estimate, ci_lower, ci_upper, p_positive, block_length, n_resamples, alpha, bootstrap_distribution)`. The dataclass exposes the full resampled distribution so the report renderer (Epic 9) can plot the density without re-running the bootstrap. Deterministic on `seed` per the cycle-4 RNG contract.

   **Empirical result on the cycle-7 backtest** (`scripts/12_backtest_ridge_e2e.py`, 27 folds, OOS 2024-01 → 2026-05 on AAPL/MSFT/GOOGL/AMZN/META/TSLA/NVDA/GLD): bootstrap-on-log-equity-diff point estimate +1.943, 95% CI [+0.657, +3.220], `p(true Sharpe > 0) = 0.9983`. The lower bound clears zero by a wide margin, the upper bound rules out +3.5+ as the population value. Politis-White auto-block-length returned 1.0 (degenerate IID), which is the *expected* result for a daily-rebalanced cross-sectional long-short — the strategy explicitly orthogonalises across days. The verdict: +2.03 is *real* but the true value with 95% confidence lies somewhere in [+0.66, +3.22]; calling +2.03 "the answer" is the upper-middle of that band, not a point fact. Paper-trading sizing should be calibrated against the lower bound, not the point estimate.

19. **(Cycle 9) Regime-stratified DSR breakdown implemented + applied to cycle-7 Ridge — exposes that +2.03 net Sharpe is concentrated in low-vol regimes**. New module `src/backtesting/regime.py` provides `RegimeTagger.from_series(vix_proxy)` (rolling 252-day quintile cutoffs, bottom 20% = `vix_low`, top 20% = `vix_high`, middle 60% = `vix_mid`, all causal via `shift(1)` like `ml_research.features._vix_tercile_features`) and a top-level `tag_regime(date, vix_proxy_value, *, tagger) -> {"vix_low","vix_mid","vix_high"}`. `WalkForwardDriver` gains optional `regime_tagger` + `vix_proxy` kwargs (both-or-neither contract); when provided the driver attaches a `regime_metrics: list[RegimeMetrics]` to `WalkForwardResult` reporting per-bucket (n_obs, Sharpe_net, DSR, fee_drag_sharpe). Per-bucket DSR uses `n_trials=1` because regime sub-sampling is *replication* of one fitted model, not independent search (cycle-8 fix #18 applies). **Empirical result on the cycle-7 Ridge backtest** (`scripts/14_regime_stratified_ridge.py`, 27 folds, OOS 2024-01 → 2026-05): aggregate +2.027 net Sharpe decomposes to `vix_low: +2.937 (n=106, DSR=0.963)`, `vix_mid: +2.081 (n=408, DSR=0.997)`, `vix_high: −0.252 (n=53, DSR=0.455)`. The +2.03 headline does NOT hold in high-vol regimes — it is materially concentrated in low-vol bull windows (~96% of OOS observations were vix_low/mid in the 2024-2026 sample, reflecting the actual realized regime mix; the 53 vix_high days lose money on a Sharpe basis). The cycle-5 flag rule ("DSR collapse > 0.3 vs aggregate") didn't trigger here only because the *broken* `n_trials=27` aggregate DSR collapsed to 0.0 already (cycle-8 issue #18); against the cycle-8-fixed aggregate DSR (≈ 0.998 with `n_trials=1`), the vix_high bucket's 0.455 represents a collapse of ~0.54 — well over the 0.3 threshold. **Action item against Epic 9 (report renderer)**: the collapse-threshold comparison MUST use the cycle-8-fix-honest aggregate DSR, not the legacy fold-count one, or the flag is mute when it matters most. Paper-trading sizing on this Ridge baseline should be reduced when VIX-proxy enters the top-quintile bucket (or the strategy should flatten in that regime altogether) — running the cycle-7 sizing through vix_high regimes is a known negative-edge proposition.

18. **(Cycle 8) DSR gate fix: `n_trials = n_alphas × n_seeds × n_feature_specs`, NOT the walk-forward fold count** — Cycle 7 set `n_trials = len(result.fold_metrics)` (e.g. 27 for the Ridge backtest), which conflates *sequential walk-forward evaluations of one fitted model* with *independent hyperparameter trials of a strategy family*. Bailey & López de Prado (2014) deflation is designed for the latter only — folds are the *replication* mechanism, not the *search* mechanism, and deflating Sharpe for "number of times we ran the fold loop" double-penalises a strategy that has been honestly cross-validated. With `n_trials=27` and the cycle-7 OOS series, DSR collapsed to 0.0000 even though both the bootstrap CI lower bound and the per-fold PSR distribution clearly indicated a real signal. The cycle-8 fix: callers MUST pass the cardinality of the hyperparameter search performed *outside* the walk-forward harness — `n_alphas × n_seeds × n_feature_specs × n_label_specs`. For the cycle-7 Ridge baseline (`alpha=1.0` only, Ridge is deterministic, one feature spec, one label spec) this is `1 × 1 × 1 × 1 = 1` and DSR correctly collapses to PSR vs zero (= 0.9986), which is the truthful answer for "we did not actually search". Once Epic 5's LightGBM grid sweep lands (`num_leaves ∈ {15,31,63}`, `learning_rate ∈ {0.03,0.05,0.1}`, 3 seeds, 2 feature specs → 54 trials), the DSR honest gate will tighten accordingly. This decouples *replication count* from *search count* — the cycle-7 conflation is exactly the failure mode the DSR paper warns against. **Action: add a `RunManifest.search_cardinality` field at Epic 9 time so the gate can be CI-enforced from the manifest rather than hand-counted in each script.**
