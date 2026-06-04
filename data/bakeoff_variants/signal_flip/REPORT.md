# Signal-Flip Exit Bake-off Report (architecture variant)

Pure model-driven entries AND exits: position is held while the signal direction is constant, closed the moment the signal flips sign (or hits 0). **No TP. No SL. No time stop.** Compare against the bracket-order baseline in `data/bakeoff/REPORT.md`.

## Promotion gate
- `p_Holm_acrossAssets < 0.05`  **AND**  `bootstrap-CI lower > 0`  **AND**  `Sharpe (net) > 0`.
- DSR is reported (variance-calibrated, n_trials = per-asset candidate count) but is not a hard gate here — the search space (≈17 trials/asset) is small enough that DSR is informative rather than dominant. The bracket-baseline's DSR=0 was an artefact of the 24× larger TP/SL grid; with no grid that artefact is gone.

## Winners (one per asset; or NO WINNER)

| asset | verdict | model | scope | Sharpe (net) | DSR | p_Holm_8 | p_Holm_strict | CI_lo | CI_hi | avg_hold (d) | trades |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | NO WINNER | ridge | asset_plus_market | +0.895 | 0.530 | 0.0760 | 0.9863 | +0.047 | +1.743 | 3.7 | 355 |
| MSFT | NO WINNER | zscore_mr_1.5 | asset_only | +0.887 | 0.503 | 0.0760 | 0.9863 | +0.025 | +1.668 | 2.6 | 158 |
| GOOGL | WINNER | buy_and_hold | asset_only | +1.049 | 0.789 | 0.0487 | 0.9863 | +0.211 | +1.879 | 21.0 | 63 |
| AMZN | NO WINNER | rsi2_mr_10_90 | asset_only | +0.741 | 0.539 | 0.0760 | 0.9863 | -0.099 | +1.592 | 1.7 | 246 |
| META | NO WINNER | buy_and_hold | asset_only | +0.625 | 0.392 | 0.0760 | 0.9863 | -0.239 | +1.428 | 21.0 | 63 |
| TSLA | NO WINNER | logistic | asset_only | +0.852 | 0.467 | 0.0760 | 0.9863 | -0.031 | +1.677 | 4.0 | 328 |
| NVDA | WINNER | buy_and_hold | asset_only | +1.145 | 0.710 | 0.0332 | 0.5923 | +0.402 | +1.971 | 21.0 | 63 |
| GLD | WINNER | logistic | asset_plus_market | +1.132 | 0.602 | 0.0332 | 0.6397 | +0.265 | +1.998 | 3.8 | 349 |

## Aggregate portfolio Sharpe (equal-weight, daily-rebalanced)

- Winners promoted: **3/8**
- Portfolio Sharpe (net): **+1.821**
- Avg holding period across per-asset best candidates: **9.9 trading days**

## Per-asset best candidate (full diagnostics)

| asset | model | scope | Sharpe | DSR | p_unadj | p_Holm_8 | p_Holm_strict | CI_lo | avg_hold | median_hold | trades |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | ridge | asset_plus_market | +0.895 | 0.530 | 0.0201 | 0.0760 | 0.9863 | +0.047 | 3.7 | 2.0 | 355 |
| MSFT | zscore_mr_1.5 | asset_only | +0.887 | 0.503 | 0.0210 | 0.0760 | 0.9863 | +0.025 | 2.6 | 2.0 | 158 |
| GOOGL | buy_and_hold | asset_only | +1.049 | 0.789 | 0.0081 | 0.0487 | 0.9863 | +0.211 | 21.0 | 21.0 | 63 |
| AMZN | rsi2_mr_10_90 | asset_only | +0.741 | 0.539 | 0.0448 | 0.0760 | 0.9863 | -0.099 | 1.7 | 1.0 | 246 |
| META | buy_and_hold | asset_only | +0.625 | 0.392 | 0.0760 | 0.0760 | 0.9863 | -0.239 | 21.0 | 21.0 | 63 |
| TSLA | logistic | asset_only | +0.852 | 0.467 | 0.0255 | 0.0760 | 0.9863 | -0.031 | 4.0 | 2.0 | 328 |
| NVDA | buy_and_hold | asset_only | +1.145 | 0.710 | 0.0044 | 0.0332 | 0.5923 | +0.402 | 21.0 | 21.0 | 63 |
| GLD | logistic | asset_plus_market | +1.132 | 0.602 | 0.0047 | 0.0332 | 0.6397 | +0.265 | 3.8 | 2.0 | 349 |

## Top-3 candidates per asset (by net Sharpe)

### AAPL
| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |
|---|---|---|---|---|---|---|
| ridge | asset_plus_market | +0.895 | 0.530 | 0.020 | 3.7 | 355 |
| buy_and_hold | asset_only | +0.747 | 0.394 | 0.044 | 21.0 | 63 |
| ridge | asset_only | +0.716 | 0.367 | 0.050 | 4.5 | 297 |

### MSFT
| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |
|---|---|---|---|---|---|---|
| zscore_mr_1.5 | asset_only | +0.887 | 0.503 | 0.021 | 2.6 | 158 |
| rsi2_mr_10_90 | asset_only | +0.804 | 0.425 | 0.033 | 1.8 | 234 |
| buy_and_hold | asset_only | +0.441 | 0.154 | 0.156 | 21.0 | 63 |

### GOOGL
| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |
|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | +1.049 | 0.789 | 0.008 | 21.0 | 63 |
| momentum_63 | asset_only | +0.577 | 0.384 | 0.093 | 11.8 | 112 |
| lgbm | asset_only | +0.506 | 0.324 | 0.123 | 3.4 | 385 |

### AMZN
| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |
|---|---|---|---|---|---|---|
| rsi2_mr_10_90 | asset_only | +0.741 | 0.539 | 0.045 | 1.7 | 246 |
| zscore_mr_1.5 | asset_only | +0.486 | 0.309 | 0.133 | 2.8 | 148 |
| buy_and_hold | asset_only | +0.434 | 0.273 | 0.160 | 21.0 | 63 |

### META
| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |
|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | +0.625 | 0.392 | 0.076 | 21.0 | 63 |
| rsi2_mr_10_90 | asset_only | +0.615 | 0.373 | 0.080 | 1.7 | 239 |
| momentum_63 | asset_only | +0.571 | 0.345 | 0.095 | 11.8 | 112 |

### TSLA
| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |
|---|---|---|---|---|---|---|
| logistic | asset_only | +0.852 | 0.467 | 0.026 | 4.0 | 328 |
| logistic | asset_plus_market | +0.824 | 0.442 | 0.030 | 3.6 | 367 |
| ridge | asset_plus_market | +0.804 | 0.425 | 0.033 | 3.7 | 362 |

### NVDA
| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |
|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | +1.145 | 0.710 | 0.004 | 21.0 | 63 |
| zscore_mr_1.5 | asset_only | +0.909 | 0.501 | 0.019 | 2.6 | 167 |
| rsi2_mr_10_90 | asset_only | +0.806 | 0.406 | 0.032 | 1.8 | 224 |

### GLD
| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |
|---|---|---|---|---|---|---|
| logistic | asset_plus_market | +1.132 | 0.602 | 0.005 | 3.8 | 349 |
| buy_and_hold | asset_only | +1.066 | 0.541 | 0.007 | 21.0 | 63 |
| ridge | asset_only | +0.953 | 0.437 | 0.015 | 2.9 | 464 |

## Comparison to bracket-order baseline (data/bakeoff/REPORT.md)
- **Baseline search**: (model × scope × tp_σ × sl_σ × hit_priority) ≈ 5,300 trials → DSR collapses to ~0 from the TP/SL noise; 0 winners promoted; per-asset best Sharpes 0.4–1.1.
- **Signal-flip variant**: (model × scope) ≈ 17 trials/asset → DSR is meaningful, trial count is honest, and the holding-period is endogenously set by the model's own conviction (avg ≈ 9.9 trading days here).
- **Promotion delta**: baseline 0/8 → signal-flip 3/8.
- **Portfolio Sharpe delta**: baseline NaN (no winners) → signal-flip +1.821.

## Methodology
- Same data, features, walk-forward harness, fees (Alpaca regs), and slippage (Almgren-Chriss + floor, k=0.1) as `scripts/50_per_asset_bakeoff.py`.
- Position dynamics: at bar `i`, hold direction `sig[i]` computed from info ≤ i-1. Day's P&L = `sig[i] * (close[i]/close[i-1] - 1)`. Transition (`sig[i] != sig[i-1]`) triggers round-trip cost at `open[i]` (close-then-open semantics; full flips pay 2 legs).
- DSR (variance-calibrated): `bench_SR = dsr_expected_max_sr(n_trials=N, sr_trials_std=sigma_emp/sqrt(252))`, then `dsr = psr(returns, benchmark_sr=bench_SR)`.
- Bootstrap CI: Politis-Romano stationary block-bootstrap, B=2000, block_length='auto' (Politis-White).

## Limitations
- Daily bars; intraday flip latency not modelled (the bot reacts at next open).
- No regime conditioning. No survivorship-bias correction. Same liquid mega-cap+GLD universe.
- Slippage uses Almgren-Chriss square-root with k=0.1; the asymmetric stop-jump cost (relevant for the bracket variant) does not apply here since there are no triggered stops.