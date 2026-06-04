# market-data

Real-time and historical exchange data ingestion, normalization, order-book reconstruction, tick archive.

## Status
Greenfield — no source code exists yet. **Architecture ratified 2026-05-17 after two research synthesis passes** (substrate + storage + universe). Treat the notes below as the agreed design, not a starting proposal.

## Architecture (one-paragraph summary)
Single-source **IBKR Gateway** / TWS API. Live ticks stream in → derived **dollar bars** (event-driven, ~50–100/day per contract — the primary ML substrate) and **daily bars** (the slow-factor layer) are written to **ArcticDB with local LMDB backend** (~10× compression vs vanilla Parquet+ZSTD, built by Man AHL for this exact workload). **Raw ticks are NOT persisted on disk** — IBKR re-serves any 6-month window per symbol in seconds at 60 req/10min pacing, so feature re-engineering pulls on demand. **Cloudflare R2 free tier** (10 GB, zero egress) is the cold cache for occasionally-needed archived bar partitions. Universe: **top ~5,000 contracts** at dollar-bar resolution + **full IBKR universe (~50,000)** at daily-bar resolution.

## Scope
- IBKR Gateway / TWS API client (data side only — trading client lives in `execution` per existing fault-isolation convention)
- Universe registry: per-asset-class ranking by 30-day ADV; weekly recompute; emit top-5k dollar-bar universe and full-universe daily-bar universe
- Head-timestamp pre-pass: `reqHeadTimeStamp` for every (contract, whatToShow) before any backfill — never spend a request on non-existent data
- Live tick streaming via `reqTickByTickData` for the top-5k universe
- Real-time dollar-bar derivation: rolling notional accumulator per contract, emit bar when threshold crossed
- One-shot historical tick backfill (~6 mo IBKR retention): `reqHistoricalTicks` walking `endDateTime` backwards, deriving bars in the same pass
- Daily-bar backfill across the full IBKR universe (~30 yr depth where available)
- ArcticDB storage layer: read/write/append wrapper; library naming `<asset_class>.<bar_type>` / `<exchange>.<symbol>`
- R2 cold-tier archive: aged bar partitions pushed to R2 free tier with transparent read fallthrough
- Bar-read API for downstream tentacles (`ml-research`, `strategy-engine`, `backtesting`)

## Universe scope (locked)
**~5,000 contracts at dollar bars, ~50,000 at daily bars.** Selection criteria:
- US equities: S&P 1500 ranked by 30-day ADV (top 1,500)
- ETFs: top 500 US-listed by ADV (includes all major commodity ETFs: GLD, SLV, USO, UNG, DBC, GDX, etc.)
- Futures: ~250 active product codes × ~10 active expiries (~500 active contracts; covers all commodity, equity-index, rates, FX, crypto)
- Spot FX: ~30 major + minor IDEALPRO pairs
- Crypto: all available IBKR/Paxos spot (~10)
- Cash indices: ~30 major (S&P, Nasdaq, Dow, FTSE 100, DAX, Nikkei 225, HSI, etc.)
- Non-US equities: top ~2,000 by ADV across LSE, Xetra, Euronext, TSE, HKEX, ASX (requires paid market-data subs per exchange)
- **Excluded**: options (sparse historical depth on IBKR, separate strategy class); cash bonds (limited bar API — use rate futures as proxy); **BID_ASK quotes** (storage prohibitive on this hardware — flagged as highest-ROI future-upgrade target)

## Storage allocation (Mac mini M4, ~140 GB usable on internal SSD; no external storage budget)

| Layer | Size | Notes |
|---|---|---|
| Hot tick buffer | ~2 GB | ~1h streaming, in-memory + minimal spill |
| Dollar bars (ArcticDB/LMDB) | ~12 GB | Top 5,000 × 10 yr |
| Daily bars (ArcticDB/LMDB) | ~10 GB | Full IBKR universe × 30 yr |
| Pre-computed features | ~30 GB | Owned by `ml-research`, stored here for read locality |
| Compliance journal | ~10 GB | Owned by `compliance` |
| Trained models | ~5 GB | Owned by `ml-research` |
| Scratch / logs / cache | ~10 GB | |
| **R2 cold tier (cloud)** | **10 GB free** | Aged bar partitions, on-demand pull |
| **Local total** | **~79 GB** | |
| **Headroom** | **~61 GB** | for growth + accumulation of compliance journal |

## Boundaries
- **Owns**: IBKR data client, universe registry, head-timestamp store, tick-stream → dollar-bar pipeline, ArcticDB storage layer, R2 archive, backfill orchestrator, pacing guard, bar-read API
- **Consumes from**: `shared-core` (Symbol, Tick, Bar, Venue types); `infrastructure` (IBKR Gateway supervision, secrets for IBKR account credentials)
- **Provides to**: `ml-research` (read API over dollar bars + daily bars); `strategy-engine` (live dollar-bar feed for inference); `execution` (real-time prices for risk pricing); `backtesting` (historical bars); `treasury` (FX cross-rates and venue marks for reconciliation); `risk-management` (mark prices for unrealized P&L)
- **Does NOT own**: trading clients (that's `execution` — strict separation per convention), feature engineering itself (that's `ml-research`), L2 order-book quotes (deferred — no BID_ASK storage on this hardware)

## Key Decisions (ratified 2026-05-17)
- **Dollar bars over time bars** — López de Prado, *Advances in Financial ML* Ch. 2: event-driven bars yield returns closer to Gaussian with lower serial autocorrelation, respecting ML estimator IID assumptions. Replicated by Hudson & Thames and a 2025 *Financial Innovation* crypto study.
- **ArcticDB (LMDB local backend) over vanilla Parquet+ZSTD** — Man AHL benchmark: ~10× compression, dropped to ~40% of prior tick-store footprint. Open-source via MongoDB. LMDB backend means no server, runs entirely local.
- **5,000 contracts at dollar bars** — practical breadth ceiling on the internal SSD. Wolff (2024) suggests 1,500–2,000 well-chosen may give ≥95% of Sharpe for cross-sectional strategies (covariance becomes ill-conditioned beyond ~1k); trend/vol/cross-asset strategies that don't rely on full-universe covariance benefit from more, so 5k is the default unless model class says otherwise.
- **No on-device raw tick retention** — IBKR re-serves any 6-month window per symbol in seconds at 60 req/10min pacing; pre-staging is wasted storage. The original "14-day raw tick sandbox" plan was rejected during the second-pass architecture review.
- **R2 free tier for cold archive** — 10 GB, zero egress, no ongoing cost.
- **TRADES only, no BID_ASK** — L2 microstructure features deferred until paid storage is available. Flagged as the highest-ROI future upgrade per substrate research (Kolm/Turiel/Westray *Deep Order Flow Imbalance*, SSRN 3900141).
- **Now-or-never tick backfill** — IBKR retains only ~6 months of ticks; backfill must run as soon as the pipeline is live to capture the deepest available history before it rolls off forever.
- **History depth target: ~10 years** — Robot Wealth window-length study + arXiv 2502.12537 + Gu/Kelly/Xiu (RFS 2020) show a U-curve. Pre-2008 hurts due to microstructure regime shifts (decimalization, HFT rise, post-MIFID2 fragmentation).

## Conventions
- All bars are immutable once written; corrections appended as a new version referencing the prior, never in-place
- Dollar-bar threshold is per-contract (calibrated by 30-day ADV) — recorded in bar metadata so old bars remain interpretable across recalibrations
- Live derivation is event-driven, not polled — a bar emits the moment notional crosses threshold
- ArcticDB library/symbol naming: `<asset_class>.<bar_type>` library, `<exchange>.<symbol>` symbol (e.g. library `equity.dollar_bars`, symbol `NASDAQ.AAPL`; library `future.daily_bars`, symbol `CME.ES`)
- All timestamps stored as int64 nanoseconds since Unix epoch, UTC — never floats, never milliseconds, never local time
- Live tick stream and historical pull use **separate clientIds** to keep pacing budgets independent
- Backfill workers acquire from a global token-bucket pacing guard: 60 req / 10 min per clientId, 6 req / 2 sec per (contract, exchange, tickType); BID_ASK counts as 2 (irrelevant now but enforced for future)
- `reqHeadTimeStamp` is consulted before every backfill range — non-existent data is never requested
- Universe membership is recomputed weekly from ADV; new members trigger backfill, removed members archive (never delete)
- Trading client (in `execution`) and this data client are **separate processes/connections** — fault isolation matters more than connection efficiency, per existing `execution` convention
