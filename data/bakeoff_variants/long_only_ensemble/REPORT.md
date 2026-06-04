# Long-Only Ensemble Bake-off Report

Architecture variant of `scripts/50_per_asset_bakeoff.py` —
each candidate from the per-asset bake-off is treated as a *voter*.
The ensemble takes a long position when at least **K** candidates
emit a long signal. Short signals are discarded (never traded).
Exit: one-bar hysteresis (drop position when votes < K-1) and/or
intra-bar bracket TP/SL. Hit priority: `sl_first` only.

**K grid**: [3, 4, 5].  
**Candidates**: rule-based + ML (ridge, logistic, lgbm) × (asset_only, asset_plus_market).  
**Walk-forward**: 8y train min, 1y inner-val, 5d embargo, 1d purge, 21d test, monthly slide.

## Promotion gate (same as main bake-off)
`p_Holm_acrossAssets < 0.05` AND `DSR > 0.95` AND `bootstrap-CI lower > 0` AND `Sharpe > 0`.
Variance-calibrated DSR with `n_trials = K_grid × TP/SL grid` per asset.

## Winners (one per asset; or NO WINNER)

| asset | verdict | K | tp_σ | sl_σ | Sharpe (net) | DSR | p_Holm_8 | p_Holm_strict | CI_lo | CI_hi | n_entries |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | NO WINNER | — | — | — | — | — | — | — | — | — | — |
| MSFT | NO WINNER | — | — | — | — | — | — | — | — | — | — |
| GOOGL | NO WINNER | — | — | — | — | — | — | — | — | — | — |
| AMZN | NO WINNER | — | — | — | — | — | — | — | — | — | — |
| META | NO WINNER | — | — | — | — | — | — | — | — | — | — |
| TSLA | NO WINNER | — | — | — | — | — | — | — | — | — | — |
| NVDA | NO WINNER | — | — | — | — | — | — | — | — | — | — |
| GLD | NO WINNER | — | — | — | — | — | — | — | — | — | — |

## Aggregate portfolio Sharpe
Equal-weight, daily-rebalanced portfolio of the 0 winners:
**Sharpe (net) = +nan**

## Best K per asset (ranked by DSR, then p_unadj)

| asset | n_candidates | best K | best Sharpe | best DSR | CI_lo (boot) |
|---|---|---|---|---|---|
| AAPL | 17 | 4 | +0.444 | 0.000 | (not bootstrapped) |
| MSFT | 17 | 3 | +0.093 | 0.000 | (not bootstrapped) |
| GOOGL | 17 | 5 | +0.418 | 0.001 | (not bootstrapped) |
| AMZN | 17 | 5 | -0.175 | 0.012 | (not bootstrapped) |
| META | 17 | 5 | +0.159 | 0.156 | (not bootstrapped) |
| TSLA | 17 | 5 | -0.333 | 0.003 | (not bootstrapped) |
| NVDA | 17 | 4 | +0.561 | 0.057 | (not bootstrapped) |
| GLD | 17 | 5 | +0.608 | 0.000 | (not bootstrapped) |

## Per-asset Sharpe by K (best TP/SL per K)

| asset | K=3 Sharpe | K=4 Sharpe | K=5 Sharpe | K=3 n_entries | K=4 n_entries | K=5 n_entries |
|---|---|---|---|---|---|---|
| AAPL | +0.345 | +0.444 | +0.346 | 1313 | 1254 | 1182 |
| MSFT | +0.093 | +0.045 | +0.051 | 1322 | 1317 | 1288 |
| GOOGL | +0.354 | +0.404 | +0.418 | 1320 | 1305 | 1270 |
| AMZN | -0.255 | -0.250 | -0.175 | 1321 | 1286 | 1231 |
| META | +0.128 | +0.090 | +0.159 | 1321 | 1297 | 1257 |
| TSLA | -0.336 | -0.374 | -0.333 | 1294 | 1228 | 1115 |
| NVDA | +0.524 | +0.561 | +0.442 | 1315 | 1291 | 1224 |
| GLD | +0.498 | +0.518 | +0.608 | 1315 | 1294 | 1224 |

## Comparison vs prior single-model architecture

Main bake-off (`data/bakeoff/REPORT.md`) winners under the same gate: **0/8**.
Best raw OOS Sharpe in the single-model run was TSLA bollinger_20_2 (+1.118) — passed Holm-8 but failed DSR. The ensemble reduces the trial count from ~340 → 39 per asset, lowering the DSR deflation hit.
Long-only ensemble winners under the same gate: **0/8**.

## Methodology
- **Voter pool**: every candidate from the main bake-off (rule + ML × scope).
- **Long vote**: candidate signal == +1 → vote 1, else vote 0 (shorts discarded).
- **Entry**: ensemble long when sum(votes) >= K.
- **Exit**: one-bar hysteresis (votes < K-1) and/or intra-bar bracket TP/SL.
- **TP/SL grid**: same as main bake-off — TP ∈ {0.5, 1.0, 1.5, 2.0}σ × SL ∈ {0.5, 1.0, 1.5}σ + no-bracket EOD-close.
- **Hit priority**: `sl_first` only (the conservative convention).
- **Stage-1 correction**: variance-calibrated DSR with `n_trials = #ensemble_configs_per_asset`.
- **Stage-2 correction**: Bonferroni-Holm across the 8 best-of-asset p-values.
- **Fees / slippage**: identical to main bake-off (`backtesting.fees.equity_alpaca`, `backtesting.slippage.slippage_bps_with_floor`, k=0.1).

## Limitations
- Ensemble vote tallies are computed on raw candidate signals; we do not weight by candidate quality (a known win for stacked ensembles).
- Long-only by design — any short-side alpha in candidates is discarded.
- Hysteresis (K → K-1) is fixed; a wider deadband (K → K-2) may further reduce churn.
- DSR ranking uses the candidate's own OOS distribution (PSR with skew/kurt correction).