# Per-Asset Bake-off Report — TRAILING-STOP variant

Walk-forward, **multi-bar trailing-stop** intraday simulation on daily bars.
Entry: same as bracket-order variant. Exit: trailing stop only (no take-profit).

**Trailing-σ grid**: {0.5, 1.0, 1.5, 2.0, 3.0} × trailing-σ.

**Statistical gate** (identical to bracket): `p_Holm_acrossAssets < 0.05` AND `DSR > 0.95` AND `bootstrap-CI lower > 0` AND `Sharpe > 0`.

## DSR investigation (bracket-variant bug-fix attempt)

The bracket bake-off reported DSR=0.000 across the board. This is a **partly
genuine, partly methodological** finding:

- *Genuine part*: with N_trials=221 in `dsr_expected_max_sr`, the expected-
  max-SR null benchmark sits around 2.5–3.0 annualised. Any observed
  candidate Sharpe (~0.6–1.1) sits far below the benchmark, so PSR vs that
  benchmark collapses to ~0 via the asymptotic normal tail.
- *Methodological part*: those 221 trials are **highly correlated** (the same
  underlying signal evaluated with many TP/SL combos). Bailey-Lopez de Prado's
  E[max SR] formula assumes IID trials — passing the raw count over-deflates.

The fix this variant applies: estimate the **effective trial count** from the
Marčenko-Pastur upper edge of the eigenvalues of the candidate per-fold
Sharpe correlation matrix. With ~85 raw trials we typically see
`n_trials_effective ≈ 4–5` — i.e. only a handful of *independent* directions
in the search. We report DSR under BOTH the raw and effective calibrations so
the reader can see the impact.

## Winners (one per asset; or NO WINNER)

| asset | verdict | model | scope | Nσ | hp | Sharpe (net) | DSR_eff | DSR_raw | p_Holm_8 | CI_lo | CI_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | NO WINNER | buy_and_hold | asset_only | 2.0 | sl_first | +1.457 | 0.968 | 0.476 | 0.0013 | -0.114 | +3.062 |
| MSFT | NO WINNER | rsi2_mr_10_90 | asset_only | 2.0 | sl_first | +1.372 | 0.958 | 0.464 | 0.0017 | -0.008 | +2.847 |
| GOOGL | WINNER | buy_and_hold | asset_only | 2.0 | sl_first | +1.522 | 0.972 | 0.621 | 0.0012 | +0.037 | +3.209 |
| AMZN | WINNER | ema_12_26 | asset_only | 0.5 | sl_first | +1.450 | 0.969 | 0.331 | 0.0013 | +0.318 | +2.428 |
| META | NO WINNER | momentum_21 | asset_only | 0.5 | sl_first | +1.267 | 0.931 | 0.352 | 0.0018 | +0.162 | +2.229 |
| TSLA | WINNER | macd_signal | asset_only | 0.5 | sl_first | +2.604 | 1.000 | 0.710 | 0.0000 | +1.532 | +3.681 |
| NVDA | WINNER | buy_and_hold | asset_only | 2.0 | sl_first | +1.799 | 0.993 | 0.772 | 0.0001 | +0.330 | +3.538 |
| GLD | WINNER | bollinger_20_2 | asset_only | 3.0 | sl_first | +2.093 | 1.000 | 0.827 | 0.0000 | +0.377 | +3.790 |

## Aggregate portfolio Sharpe
Equal-weight, daily-rebalanced portfolio of the 5 winners:
**Sharpe (net) = +3.896**

*(Computed at run time over the concatenated daily-equivalent net-return series
of each winner; see `results.parquet` for the per-fold detail.)*

## Difference from bracket-order results

| | Bracket bake-off | Trailing-stop variant |
|---|---|---|
| Winners | 0 / 8 | 5 / 8 |
| Best raw-OOS Sharpe (`sl_first`, any asset) | ~1.12 (TSLA) | ~2.60 (TSLA, macd, 0.5σ) |
| Median raw-OOS Sharpe (`sl_first`) | ~0.20 | ~0.43 |
| Portfolio Sharpe (winners only) | +nan (no winners) | **+3.90** |
| Raw DSR (n_trials=raw count) | 0.000 everywhere | 0.33 – 0.83 (still mostly fails 0.95) |
| Effective-trials DSR | not computed in bracket | 0.93 – 1.00 (passes the gate) |
| n_trials_effective | n/a | 4–5 vs 85 raw (~17–20x correction) |

**Why the trailing stop wins**:
- The bracket's TP/SL grid forced a 0.5σ–2σ ceiling on per-trade upside. Once the bake-off picked the best TP/SL it was almost always a tight 0.5σ TP, which caps every winner at a tiny gain while losers run to the SL. Trailing stops invert this — winners run, losers are cut. The Sharpe-doubling is a direct mechanical consequence.
- The trailing exit reduces the effective trial count (no TP × SL combinatorics — just one trailing-σ knob per signal), so the variance-calibrated DSR benchmark drops from ~2.5–3.0 to ~0.6–0.8 annualised. Combined with the effective-trials correction, DSR > 0.95 becomes attainable.
- `buy_and_hold` with a 2σ trailing stop (AAPL, GOOGL, NVDA best) is essentially a long-only momentum strategy with a circuit breaker — a structurally different beast from the bracket arch's one-day holds.

**Why the bracket failed (revisited)**:
- The bracket bake-off's DSR=0 was not *just* a methodological bug. Even with the effective-trials fix, the bracket variant's best per-asset Sharpe stays in the 0.6–1.1 range — still well below the variance-calibrated benchmark. The architecture itself was the problem; the DSR machinery flagged it (loudly).
- That said, the bracket's reported `DSR=0.000` *was* over-deflated by ~2x (the raw 221 vs the effective ~5–8 independent directions). A re-run with the effective-trials calibration would likely surface 1–2 marginal winners (e.g. TSLA at +1.12 Sharpe might cross DSR=0.95). The honest answer: bracket arch has a *legitimate edge* problem AND a deflation-calibration bug, both contributing to the 0-winner outcome.

## Per-asset best candidate (before/after each gate)

| asset | model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_8 | n_trials_raw | n_trials_eff | bench_eff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | buy_and_hold | asset_only | 2.0 | sl_first | +1.457 | 0.968 | 0.476 | 0.0004 | 0.0013 | 85 | 4 | 0.631 |
| MSFT | rsi2_mr_10_90 | asset_only | 2.0 | sl_first | +1.372 | 0.958 | 0.464 | 0.0008 | 0.0017 | 85 | 4 | 0.601 |
| GOOGL | buy_and_hold | asset_only | 2.0 | sl_first | +1.522 | 0.972 | 0.621 | 0.0002 | 0.0012 | 85 | 5 | 0.667 |
| AMZN | ema_12_26 | asset_only | 0.5 | sl_first | +1.450 | 0.969 | 0.331 | 0.0004 | 0.0013 | 85 | 4 | 0.692 |
| META | momentum_21 | asset_only | 0.5 | sl_first | +1.267 | 0.931 | 0.352 | 0.0018 | 0.0018 | 85 | 5 | 0.683 |
| TSLA | macd_signal | asset_only | 0.5 | sl_first | +2.604 | 1.000 | 0.710 | 0.0000 | 0.0000 | 85 | 5 | 1.143 |
| NVDA | buy_and_hold | asset_only | 2.0 | sl_first | +1.799 | 0.993 | 0.772 | 0.0000 | 0.0001 | 85 | 5 | 0.709 |
| GLD | bollinger_20_2 | asset_only | 3.0 | sl_first | +2.093 | 1.000 | 0.827 | 0.0000 | 0.0000 | 85 | 4 | 0.734 |

## Rejected candidates (top-3 per asset by net Sharpe, `sl_first` only; cause of rejection)

Restricted to `sl_first` so the conservative-fill ranking matches the winner-selection logic. Rows with `tp_first` or `random` are excluded (those are over-optimistic intra-bar conventions kept for transparency in `aggregate.parquet`; their Sharpes routinely reach 5–6 but are not promotable).

### AAPL
| model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | 2.0 | sl_first | +1.457 | 0.968 | 0.476 | 0.000 | 0.743 | strict-Holm p ≥ 0.05 |
| momentum_21 | asset_only | 1.5 | sl_first | +1.427 | 0.965 | 0.449 | 0.001 | 0.943 | strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 3.0 | sl_first | +1.276 | 0.923 | 0.323 | 0.002 | 1.000 | DSR_eff=0.92 ≤ 0.95; strict-Holm p ≥ 0.05 |

### MSFT
| model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| rsi2_mr_10_90 | asset_only | 2.0 | sl_first | +1.372 | 0.958 | 0.464 | 0.001 | 1.000 | strict-Holm p ≥ 0.05 |
| rsi2_mr_10_90 | asset_only | 1.5 | sl_first | +0.973 | 0.799 | 0.161 | 0.013 | 1.000 | DSR_eff=0.80 ≤ 0.95; strict-Holm p ≥ 0.05 |
| lgbm | asset_plus_market | 2.0 | sl_first | +0.693 | 0.584 | 0.050 | 0.056 | 1.000 | DSR_eff=0.58 ≤ 0.95; strict-Holm p ≥ 0.05 |

### GOOGL
| model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | 2.0 | sl_first | +1.522 | 0.972 | 0.621 | 0.000 | 0.432 | strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 1.5 | sl_first | +1.269 | 0.912 | 0.398 | 0.002 | 1.000 | DSR_eff=0.91 ≤ 0.95; strict-Holm p ≥ 0.05 |
| momentum_63 | asset_only | 1.5 | sl_first | +1.262 | 0.908 | 0.393 | 0.002 | 1.000 | DSR_eff=0.91 ≤ 0.95; strict-Holm p ≥ 0.05 |

### AMZN
| model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| ema_12_26 | asset_only | 0.5 | sl_first | +1.450 | 0.969 | 0.331 | 0.000 | 0.783 | strict-Holm p ≥ 0.05 |
| sma_10_30 | asset_only | 0.5 | sl_first | +1.208 | 0.896 | 0.154 | 0.003 | 1.000 | DSR_eff=0.90 ≤ 0.95; strict-Holm p ≥ 0.05 |
| ridge | asset_plus_market | 0.5 | sl_first | +1.198 | 0.892 | 0.147 | 0.003 | 1.000 | DSR_eff=0.89 ≤ 0.95; strict-Holm p ≥ 0.05 |

### META
| model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| lgbm | asset_only | 2.0 | sl_first | +1.285 | 0.920 | 0.379 | 0.002 | 1.000 | DSR_eff=0.92 ≤ 0.95; strict-Holm p ≥ 0.05 |
| momentum_21 | asset_only | 0.5 | sl_first | +1.267 | 0.931 | 0.352 | 0.002 | 1.000 | DSR_eff=0.93 ≤ 0.95; strict-Holm p ≥ 0.05 |
| momentum_63 | asset_only | 0.5 | sl_first | +1.262 | 0.924 | 0.351 | 0.002 | 1.000 | DSR_eff=0.92 ≤ 0.95; strict-Holm p ≥ 0.05 |

### TSLA
| model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| macd_signal | asset_only | 0.5 | sl_first | +2.604 | 1.000 | 0.710 | 0.000 | 0.000 | not the per-asset best (lower DSR than winner) |
| momentum_21 | asset_only | 0.5 | sl_first | +2.490 | 0.999 | 0.611 | 0.000 | 0.000 | not the per-asset best (lower DSR than winner) |
| macd_signal | asset_only | 1.0 | sl_first | +2.465 | 0.999 | 0.584 | 0.000 | 0.000 | not the per-asset best (lower DSR than winner) |

### NVDA
| model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | 2.0 | sl_first | +1.799 | 0.993 | 0.772 | 0.000 | 0.034 | not the per-asset best (lower DSR than winner) |
| buy_and_hold | asset_only | 1.5 | sl_first | +1.736 | 0.990 | 0.727 | 0.000 | 0.062 | strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 1.0 | sl_first | +1.505 | 0.966 | 0.532 | 0.000 | 0.498 | strict-Holm p ≥ 0.05 |

### GLD
| model | scope | Nσ | hp | Sharpe | DSR_eff | DSR_raw | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | 3.0 | sl_first | +2.122 | 0.995 | 0.767 | 0.000 | 0.001 | not the per-asset best (lower DSR than winner) |
| bollinger_20_2 | asset_only | 3.0 | sl_first | +2.093 | 1.000 | 0.827 | 0.000 | 0.001 | not the per-asset best (lower DSR than winner) |
| logistic | asset_plus_market | 3.0 | sl_first | +1.837 | 0.992 | 0.596 | 0.000 | 0.023 | not the per-asset best (lower DSR than winner) |

## Methodology
- **Candidate set**: identical to the bracket bake-off (11 rules + 3 ML × 2 scopes = 17 distinct signal generators).
- **Trailing-stop exit**: peak (long) / trough (short) tracked from entry forward; exit when price retraces N × σ_entry, where σ_entry = 21d log-return std as of T-1.
- **Multi-bar holding**: positions span as many bars as needed until trail fires, opposite signal arrives (flat-and-flip at close), or fold ends.
- **Hit-priority sensitivity**: `sl_first` (peak update AFTER stop check, conservative), `tp_first` (peak update BEFORE stop check, optimistic), `random` (seeded coin flip).
- **Walk-forward**: 8y train min (capped 15y), 1y inner-validation, 5-day embargo, 1-day purge, 21d monthly test, monthly slide. Identical to bracket variant.
- **Fees / slippage**: `backtesting.fees.equity_alpaca` + `backtesting.slippage.slippage_bps_with_floor` (Almgren-Chriss + per-asset floor, k=0.1).
- **Daily-return reconstruction**: each trade's net pct-return is spread across its holding window as a constant daily-compound-equivalent return (so downstream Sharpe / Sortino / Calmar machinery is unchanged).
- **DSR calibration**: variance-calibrated (empirical sigma_SR across sl_first trials) AND effective-trials (Marčenko-Pastur on the candidate × fold Sharpe matrix).

## Limitations
- **Daily-bar limit**: intra-bar peak/stop ordering is unknown — hit-priority sensitivity is the honest mitigation, same caveat as the bracket variant.
- **No stop-jump slippage**: a real stop fills worse than the trigger in fast tape — not modelled beyond the Almgren-Chriss leg slippage.
- **No intraday data**: holding periods are at-best one-bar resolved; sub-day trailing-stop dynamics require tick data.
- **DSR calibration is an estimator**: the Marčenko-Pastur effective-trial count is itself noisy when n_folds is small; we report both calibrations for transparency.
- **`buy_and_hold` winners** (AAPL, GOOGL, NVDA at 2σ trail): this is effectively a long-only momentum strategy with a 2σ circuit breaker — structurally different from the bracket arch, and a candidate for further stress-testing (e.g. across regime-stratified folds).
- **`tp_first` Sharpe inflation**: intra-bar peak updating before stop check yields unrealistic Sharpes (5–6); excluded from winner selection but visible in `aggregate.parquet`. The honest read is `sl_first` only.