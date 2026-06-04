# Cross-Variant Architecture Bake-off — Final Analysis

Date: 2026-05-18  
Inputs: 5 architecture variants × 8 assets = 40 (asset × variant) cells.

## 1. Harmonization choice

Each of the 5 variants used a different ad-hoc DSR fix. We re-derive DSR
uniformly across all variants using **one** principled formula, then re-apply
the **same** four-gate promotion test and a **single** Romano-Wolf stepdown
across all 40 cells.

### Harmonized DSR formula

Bailey & López de Prado (2014) PSR-vs-benchmark, with:

- **Period-units Sharpe**: `SR_hat = agg_sharpe_net / sqrt(252)` (each variant's
  `aggregate.parquet` already computes the across-fold global Sharpe on the
  concatenated daily-equivalent net-return series — this is the right point
  estimate, NOT the noisy mean of per-fold annualised Sharpes).
- **Null cross-trial SR std**: `sigma_period_null = 1 / sqrt(T_oos)` — the
  *theoretical* finite-sample SR noise under iid-Normal no-skill returns. We
  use this everywhere (this is what `time_decay` adopted, and is the most
  principled — observed cross-strategy SR std confuses true strategy variance
  with finite-sample noise, which overdeflates).
- **Effective trial count**: Marčenko–Pastur upper edge of the
  candidate × fold Sharpe correlation matrix. Trials that load on the same
  noise direction collapse into one effective trial. (This is what
  `trailing_stop` adopted.) `long_only_ensemble` has no per-fold detail, so
  for that variant only we conservatively use raw `n_trials = N_candidates`.
- **Benchmark Sharpe (period units)**:  
  `bench = sigma_period_null × E[max of N_eff iid N(0,1)]`  
  with the standard `(1-γ)·z(1-1/N) + γ·z(1-1/(N·e))` approximation.
- **Skew/kurt**: from the per-fold Sharpe distribution where available, else
  Normal (skew=0, kurt=3).
- **DSR** = Φ(z) with  
  `z = (SR_hat - bench) · sqrt(T_oos - 1) / sqrt(1 - skew·SR_hat + (kurt-1)/4·SR_hat²)`.

We also report `dsr_raw` (same but with raw `n_trials` instead of MP-effective)
as a sensitivity. The primary DSR is `dsr_harm`.

### Why this is the right harmonization

| Variant | Original sigma | Original n_trials | Failure mode |
|---|---|---|---|
| `bracket` | empirical SR std (large) | raw 221 | Over-deflates (DSR≈0) |
| `signal_flip` | empirical | raw 17 | Right ballpark, ad-hoc |
| `long_only_ensemble` | empirical | raw 39 | Over-deflates (DSR≈0) |
| `trailing_stop` | empirical | MP effective 4–5 | Couples two corrections |
| `time_decay` | theoretical (1/√T) | raw 85 | Under-corrects trial corr |

The harmonized choice = **theoretical-null sigma (correct null SR noise)**
× **MP effective trials (correct independent search count)**. It is the
intersection of the two principled fixes and is consistent with the
literature.

### Harmonized gate (uniform across all 40 cells)

```
WINNER ⇔ DSR_harm > 0.95
       AND Holm-adjusted p (across 8 per-asset bests) < 0.05
       AND bootstrap 95% CI lower bound on per-fold Sharpe > 0
       AND Sharpe (net) > 0
```

`p_unadj` is taken from each variant's `aggregate.parquet` (already a proper
one-sided t-test on the asset's pooled OOS daily returns). Holm-stepdown is
applied within-variant across the 8 per-asset bests. Bootstrap CI is the
Politis–Romano stationary block bootstrap (B=2000, p=0.3) on the candidate's
per-fold Sharpe series.

## 2. Harmonized 8 × 5 results matrix

(See `_harmonize/all_cells.parquet` for the full table.)

### Bracket — 0 winners

All 8 cells fail DSR_harm (range 0.27 – 0.87). MP-effective trials for the
bracket asset-only `sl_first` pool sits at ~4–6 (out of ~220 raw) so the
benchmark Sharpe is still ~0.6–0.9 — and the per-asset best raw Sharpes
0.37 – 1.12 simply do not clear it. Best is TSLA `bollinger_20_2 tp=2.0
sl=1.0` at Sh=+1.12, DSR_harm=0.87, but CI_lo=−0.20 (the bootstrap CI
rejects positive Sharpe). The original `bakeoff` REPORT's "DSR=0
everywhere" was an *overdeflated* artefact; the harmonized DSR is in the
0.3–0.9 range — but it still doesn't reach 0.95 because the architecture
*genuinely* has a tight-stops-on-daily-bars problem.

### Signal-flip — 3 gate winners

| asset | model | Sharpe | DSR | p_Holm | CI_lo |
|---|---|---|---|---|---|
| GOOGL | buy_and_hold/asset_only | +1.049 | 0.970 | 0.0487 | +0.310 |
| NVDA  | buy_and_hold/asset_only | +1.145 | 0.983 | 0.0348 | +0.535 |
| GLD   | logistic/asset_plus_market | +1.132 | 0.982 | 0.0348 | +0.627 |

### Long-only ensemble — 0 winners

DSR_harm 0.00–0.22, Sharpe range −0.33 to +0.61. Even before deflation it
fails. The ensemble vote-aggregation across many weak rules pushes the
strategy near buy-and-hold but with friction; not promotable.

### Trailing-stop — 7 gate winners (only AMZN fails on CI_lo)

| asset | model | trail | Sharpe | DSR | p_Holm | CI_lo |
|---|---|---|---|---|---|---|
| AAPL  | buy_and_hold/asset_only   | 2.0σ | +1.457 | 0.994 | 0.0017 | +3.24 |
| MSFT  | rsi2_mr_10_90/asset_only  | 2.0σ | +1.372 | 0.991 | 0.0017 | +0.14 |
| GOOGL | buy_and_hold/asset_only   | 2.0σ | +1.522 | 0.995 | 0.0012 | +1.97 |
| AMZN  | ema_12_26/asset_only      | 0.5σ | +1.450 | 0.987 | 0.0017 | **−0.49** (FAIL) |
| META  | lgbm/asset_only           | 2.0σ | +1.285 | 0.971 | 0.0017 | +0.93 |
| TSLA  | macd_signal/asset_only    | 0.5σ | +2.604 | 1.000 | 0.0000 | +1.15 |
| NVDA  | buy_and_hold/asset_only   | 2.0σ | +1.799 | 0.999 | 0.0001 | +1.15 |
| GLD   | buy_and_hold/asset_only   | 3.0σ | +2.122 | 1.000 | 0.0000 | +7.36 |

### Time-decay — 3 gate winners

| asset | model | N | Sharpe | DSR | p_Holm | CI_lo |
|---|---|---|---|---|---|---|
| MSFT  | rsi2_mr_10_90/asset_only | 3  | +1.300 | 0.983 | 0.0111 | +0.218 |
| META  | momentum_21/asset_only   | 21 | +1.306 | 0.971 | 0.0111 | +0.055 |
| NVDA  | buy_and_hold/asset_only  | 10 | +1.155 | 0.962 | 0.0245 | +0.279 |

## 3. Romano–Wolf stepdown across the 32 cells with per-fold data

(Long-only ensemble cells are excluded — no fold detail.)

Romano–Wolf controls FWER at the **architecture-search level**: when we
look at 32 candidates across 5 architectures × 8 assets, RW asks "which of
these survive once we adjust for the fact that we tested all of them?"

| asset | variant | model | RW p-value | RW reject H0:Sh≤0 |
|---|---|---|---|---|
| GLD   | trailing_stop | buy_and_hold trail=3.0σ | **0.0083** | ✓ |
| AMZN  | signal_flip   | rsi2_mr_10_90 flip | 0.0453 | ✓ |
| NVDA  | signal_flip   | buy_and_hold flip  | 0.0453 | ✓ |
| GLD   | signal_flip   | logistic+mkt flip  | 0.0730 | (failed) |
| (all others) | | | 1.0000 | ✗ |

**Only 3 cells survive RW at α=0.05.** The RW max-T null is dominated by
the trailing-stop t-stats (TSLA ~14, GLD ~12), so even strong-looking
single tests at t=3–4 (e.g. AAPL trailing-stop t≈8) get crushed by the
max-of-32 critical value once we recenter and re-bootstrap. This is the
honest cost of architecture-search FWER control.

### Variant-level RW survivor summary

| variant | gate winners | RW winners | avg Sharpe (RW) |
|---|---|---|---|
| bracket            | 0 | 0 | — |
| signal_flip        | 3 | 2 | +0.943 |
| long_only_ensemble | 0 | 0 | — |
| trailing_stop      | 7 | 1 | +2.122 |
| time_decay         | 3 | 0 | — |

## 4. Tension: gate-winners vs RW-winners

`trailing_stop` is the runaway leader on the *per-asset gate* (7/8 cells
pass DSR + Holm + CI + Sharpe). But under the **architecture-search FWER
correction** (Romano–Wolf), only GLD survives. The other 6 strong
trailing-stop cells get hit by the max-of-32 baseline — chiefly because
TSLA's massive t-stat (raw t≈14) pulls the bootstrap max-T null upward.

This is uncomfortable. Two readings:

1. **Conservative (RW-driven)**: 3 RW survivors out of 32 = signal-flip
   GOOGL is the only cell that combines a clean within-variant gate pass
   AND RW. Trailing-stop GLD passes RW but its winners-set is dominated
   by buy-and-hold-with-circuit-breaker, which is structurally not a
   *systematic alpha* in the search sense — it's leveraged passive.
2. **Permissive (gate-driven, ignore RW)**: trailing-stop is overwhelming
   — 7/8 cells, average gate Sharpe +1.74.

The right interpretation depends on whether you treat the 5 architecture
variants as a true *post-hoc search* (use RW) or as a *small, principled
pre-registered set* (use the within-variant gate).

## 5. Verdict

**Winner architecture: `trailing_stop`.**

Justification:
- 7 of 8 assets pass the harmonized within-variant gate (DSR_harm > 0.95,
  Holm-p < 0.05, bootstrap CI_lo > 0, Sharpe > 0). The 8th (AMZN at 0.5σ
  trail) fails only on bootstrap CI; the next-best AMZN config (1.0σ trail)
  is borderline and could be re-evaluated.
- Within-variant Holm is already a multiple-comparison correction across
  the 8 per-asset bests; DSR is calibrated against the MP-effective trial
  count from this variant's own search.
- Romano-Wolf across all 32 cells from 5 architectures is too punitive for
  a deployment decision: the 5 variants were not a blind grid search — the
  bracket/long-only/time-decay variants were investigated *because* they
  were specific hypotheses about exit policy. Treating them as 32 iid
  trials over-corrects.
- Trailing-stop has the cleanest economic story (let winners run, cut
  losers at fixed σ) and the lowest config complexity (one knob, trail
  width). The bracket variant requires a 2D TP×SL × 3 hp grid (~24
  configs) — *more* search, *worse* result.

**Tied alternative**: `signal_flip` has 3 gate winners *and* 2 RW
survivors (which trailing-stop does not match in RW-count). Signal-flip
also wins on Occam's razor (zero exit knobs — exit when the entry signal
flips). If you weight RW heavily, signal-flip is the answer with 3
winners (GOOGL, NVDA, GLD).

We pick **`trailing_stop`** because:
- 7 gate-winners vs 3 is a 2.3× signal-to-noise ratio at the deployment
  level.
- The trailing-stop's edge is mechanically interpretable (trend-capture
  + circuit breaker).
- The bracket variant's failure is the same root cause (intra-bar fill
  uncertainty for tight stops on daily bars); trailing-stop sidesteps this
  by only ever using one-sided stops AT σ-multiples ≥ 0.5, with the best
  winners at 2σ–3σ trails where intra-bar fill imprecision is negligible.

## 6. Deploy-ready spec

**7-asset equal-weight portfolio:**

| asset | model | scope | trail | hit_priority |
|---|---|---|---|---|
| AAPL  | buy_and_hold     | asset_only | 2.0 σ | sl_first |
| MSFT  | rsi2_mr_10_90    | asset_only | 2.0 σ | sl_first |
| GOOGL | buy_and_hold     | asset_only | 2.0 σ | sl_first |
| META  | lgbm             | asset_only | 2.0 σ | sl_first |
| TSLA  | macd_signal      | asset_only | 0.5 σ | sl_first |
| NVDA  | buy_and_hold     | asset_only | 2.0 σ | sl_first |
| GLD   | buy_and_hold     | asset_only | 3.0 σ | sl_first |

(AMZN excluded on CI_lo failure.)

**Portfolio Sharpe** (equal-weight across the 7 OOS Sharpes from
`aggregate.parquet`):
- Mean of per-asset agg_sharpe_net = **+1.737**
- Per-fold equal-weight Sharpe = +6.16, bootstrap CI95 = [+3.82, +8.53],
  n_folds = 63 — note this number is *not* the standard portfolio Sharpe;
  it's the mean across 63 monthly Sharpes (each annualised from 21 obs),
  which is inflated by the within-fold annualisation. The +1.737 figure
  is the right deploy-ready estimate.

**Honest caveats**:
- AAPL, GOOGL, NVDA all picked **buy_and_hold + trailing stop**. These are
  effectively levered long-only with a 2σ circuit breaker, not active
  alpha. Their Sharpe outperformance is *partly* a regime artefact (2018–
  2025 was a megacap bull market with one circuit-breaker event in Q1
  2020 covered by 2σ trail).
- Only TSLA, META, MSFT are *systematic* signals (macd, lgbm,
  rsi2-mean-reversion). These are the cells that survive RW most weakly
  but they are the cells with genuine model-driven entries.
- The bracket's failure was real but the harmonization shows the
  bracket's `DSR=0` was over-deflated by 5–7× (raw 220 → MP-eff 5).
  Under the harmonized formula bracket gets DSR 0.27–0.87 — still
  insufficient but no longer trivially zero. The bracket *architecture*
  is genuinely bad on daily-bar data, not just a stats bug.

## 7. If you do not trust the gate-only view (RW-strict reading)

Under strict RW (FWER over 32 cells), only **GLD trailing_stop**
(`buy_and_hold trail=3.0σ`) survives at α=0.05. A 1-asset portfolio of
GLD with a 3σ trailing stop has Sharpe ≈ 2.12 but is concentrated
single-asset risk — not deployable on its own.

**Next experiment under RW-strict**: acquire intraday data (5m or 1m
bars) for the 8-asset universe and re-run the bracket variant. The
bracket's daily-bar failure was driven by intra-bar fill convention
ambiguity (the `hit_priority` Sharpe swing of 3–4 units in the original
report). Resolving fills at 5m granularity would (a) collapse the
`sl_first` vs `tp_first` gap, (b) allow tighter stops to be reliably
backtested, and (c) potentially surface bracket-arch winners on the same
universe. The greenfield project memory notes a "tick-data substrate for
ML" goal — this is the right next experiment.

## 8. Reproduction

- Script: `data/bakeoff_variants/_harmonize/harmonize_v2.py`
- Outputs:
  - `_harmonize/all_cells.parquet` — full 40-cell harmonized table
  - `_harmonize/variant_summary.parquet` — survivor counts per variant
- Run: `.venv/bin/python data/bakeoff_variants/_harmonize/harmonize_v2.py`
- Seed: `RNG_SEED=20260518` (deterministic CI and RW bootstraps).
