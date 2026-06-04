# Time-Decay Exit Bake-off (Architecture Variant)

**Exit policy:** close position at `close[T+N-1]` regardless of P&L,
for N in [1, 3, 5, 10, 21] bars. No TP, no SL. Single-position-at-a-time per asset.

**Entry policy:** identical to scripts/50_per_asset_bakeoff.py — signal computed on
info <= T-1 triggers position at `open[T]` if no position is currently open.

**Promotion gate** (same as bracket bake-off, plus DSR bug-fix):
- Holm-corrected `p < 0.05` (across 8 best-of-asset p-values)
- Bootstrap-CI lower bound > 0 (Politis-Romano stationary block-bootstrap, B=2000)
- Sharpe (net) > 0
- DSR > 0.95 (variance-calibrated PSR vs expected-max-SR benchmark)

**DSR fix vs cycle-50 bracket bake-off:**
the bracket script (`scripts/50_per_asset_bakeoff.py`, lines 1094-1111) passed the
*empirical* observed cross-trial SR std into `dsr_expected_max_sr` as `sr_trials_std`.
That parameter is the *null-hypothesis* cross-trial SR std — the noise-only variance
of finite-sample SR estimates under no-skill iid returns. Plugging in observed cross-
strategy variance (1.0–2.0 annualised) over-deflates because that std is dominated by
real strategy differences, not sampling noise. Under iid-N(0,1) returns of length T,
the correct period-units null SR std is ≈ 1/sqrt(T). This variant uses that value
(`sigma_period_null = 1/sqrt(T_oos)`, ~0.019 for our ~2,640-day OOS window), which
produces a benchmark Sharpe of ~0.79 annualised for ~100 candidates — high enough to
actually challenge the candidates, low enough to let true signals pass.

## Winners (one per asset; or NO WINNER)

| asset | verdict | model | scope | N | Sharpe (net) | DSR | p_Holm_8 | p_Holm_strict | CI_lo | CI_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | NO WINNER | — | — | — | — | — | — | — | — | — |
| MSFT | NO WINNER | — | — | — | — | — | — | — | — | — |
| GOOGL | NO WINNER | — | — | — | — | — | — | — | — | — |
| AMZN | NO WINNER | — | — | — | — | — | — | — | — | — |
| META | NO WINNER | — | — | — | — | — | — | — | — | — |
| TSLA | NO WINNER | — | — | — | — | — | — | — | — | — |
| NVDA | NO WINNER | — | — | — | — | — | — | — | — | — |
| GLD | NO WINNER | — | — | — | — | — | — | — | — | — |

## Aggregate portfolio Sharpe
Equal-weight, daily-rebalanced portfolio of the 0 winners:
**Sharpe (net) = +nan**

Indicative equal-weight portfolio of the 8 per-asset *bests* (gate-blind):
**Sharpe (net) = +2.699**
(shown to make the bracket-vs-time-decay comparison apples-to-apples — do NOT trade this; it does not pass the deflation gate.)

**No winners survived all gates.** Time-decay exit alone is not sufficient
to clear the variance-calibrated DSR > 0.95 deflation bar on the 8 mega-cap
universe at daily-bar granularity, though most per-asset bests do pass Holm-
corrected p < 0.05 and bootstrap-CI > 0. The DSR gate (Bailey-LdP 2014, calibrated
with theoretical iid-Normal null SR std = 1/sqrt(T_oos)) is the binding constraint.

## Per-asset best candidate

| asset | model | scope | N | Sharpe | DSR | p_unadj | p_Holm_8 | p_Holm_strict | CI_lo |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | lgbm | asset_only | 21 | +0.943 | 0.362 | 0.0154 | 0.0308 | 0.9998 | +0.134 |
| MSFT | rsi2_mr_10_90 | asset_only | 3 | +1.300 | 0.706 | 0.0014 | 0.0101 | 0.9834 | +0.442 |
| GOOGL | buy_and_hold | asset_only | 21 | +1.014 | 0.434 | 0.0101 | 0.0302 | 0.9998 | +0.204 |
| AMZN | ridge | asset_plus_market | 21 | +0.749 | 0.220 | 0.0431 | 0.0431 | 0.9998 | -0.085 |
| META | momentum_21 | asset_only | 21 | +1.306 | 0.744 | 0.0014 | 0.0101 | 0.9437 | +0.531 |
| TSLA | macd_signal | asset_only | 5 | +1.060 | 0.481 | 0.0076 | 0.0302 | 0.9998 | +0.272 |
| NVDA | buy_and_hold | asset_only | 10 | +1.155 | 0.571 | 0.0041 | 0.0245 | 0.9998 | +0.381 |
| GLD | buy_and_hold | asset_only | 21 | +1.085 | 0.506 | 0.0064 | 0.0302 | 0.9998 | +0.262 |

## Top-3 candidates per asset (by net Sharpe)

### AAPL
| model | scope | N | Sharpe | DSR | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|
| lgbm | asset_only | 21 | +0.943 | 0.362 | 0.015 | 1.000 | DSR=0.36 ≤ 0.95; strict-Holm p ≥ 0.05 |
| ridge | asset_only | 10 | +0.853 | 0.300 | 0.025 | 1.000 | DSR=0.30 ≤ 0.95; strict-Holm p ≥ 0.05 |
| ridge | asset_plus_market | 10 | +0.726 | 0.206 | 0.048 | 1.000 | DSR=0.21 ≤ 0.95; strict-Holm p ≥ 0.05 |

### MSFT
| model | scope | N | Sharpe | DSR | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|
| rsi2_mr_10_90 | asset_only | 3 | +1.300 | 0.706 | 0.001 | 0.983 | DSR=0.71 ≤ 0.95; strict-Holm p ≥ 0.05 |
| momentum_63 | asset_only | 21 | +0.781 | 0.244 | 0.037 | 1.000 | DSR=0.24 ≤ 0.95; strict-Holm p ≥ 0.05 |
| logistic | asset_only | 5 | +0.765 | 0.233 | 0.040 | 1.000 | DSR=0.23 ≤ 0.95; strict-Holm p ≥ 0.05 |

### GOOGL
| model | scope | N | Sharpe | DSR | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | 21 | +1.014 | 0.434 | 0.010 | 1.000 | DSR=0.43 ≤ 0.95; strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 10 | +0.992 | 0.417 | 0.012 | 1.000 | DSR=0.42 ≤ 0.95; strict-Holm p ≥ 0.05 |
| lgbm | asset_only | 21 | +0.904 | 0.325 | 0.019 | 1.000 | DSR=0.32 ≤ 0.95; strict-Holm p ≥ 0.05 |

### AMZN
| model | scope | N | Sharpe | DSR | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|
| ridge | asset_plus_market | 21 | +0.749 | 0.220 | 0.043 | 1.000 | DSR=0.22 ≤ 0.95; strict-Holm p ≥ 0.05 |
| logistic | asset_only | 21 | +0.536 | 0.105 | 0.110 | 1.000 | DSR=0.10 ≤ 0.95; strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 10 | +0.417 | 0.066 | 0.170 | 1.000 | DSR=0.07 ≤ 0.95; strict-Holm p ≥ 0.05 |

### META
| model | scope | N | Sharpe | DSR | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|
| momentum_21 | asset_only | 21 | +1.306 | 0.744 | 0.001 | 0.944 | DSR=0.74 ≤ 0.95; strict-Holm p ≥ 0.05 |
| sma_10_30 | asset_only | 21 | +1.135 | 0.561 | 0.005 | 1.000 | DSR=0.56 ≤ 0.95; strict-Holm p ≥ 0.05 |
| ema_12_26 | asset_only | 21 | +1.056 | 0.475 | 0.008 | 1.000 | DSR=0.47 ≤ 0.95; strict-Holm p ≥ 0.05 |

### TSLA
| model | scope | N | Sharpe | DSR | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|
| macd_signal | asset_only | 5 | +1.060 | 0.481 | 0.008 | 1.000 | DSR=0.48 ≤ 0.95; strict-Holm p ≥ 0.05 |
| macd_signal | asset_only | 3 | +1.052 | 0.474 | 0.008 | 1.000 | DSR=0.47 ≤ 0.95; strict-Holm p ≥ 0.05 |
| bollinger_20_2 | asset_only | 3 | +1.015 | 0.439 | 0.010 | 1.000 | DSR=0.44 ≤ 0.95; strict-Holm p ≥ 0.05 |

### NVDA
| model | scope | N | Sharpe | DSR | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | 10 | +1.155 | 0.571 | 0.004 | 1.000 | DSR=0.57 ≤ 0.95; strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 21 | +1.135 | 0.559 | 0.005 | 1.000 | DSR=0.56 ≤ 0.95; strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 5 | +1.133 | 0.550 | 0.005 | 1.000 | DSR=0.55 ≤ 0.95; strict-Holm p ≥ 0.05 |

### GLD
| model | scope | N | Sharpe | DSR | p_unadj | p_Holm_strict | why rejected |
|---|---|---|---|---|---|---|---|
| buy_and_hold | asset_only | 21 | +1.085 | 0.506 | 0.006 | 1.000 | DSR=0.51 ≤ 0.95; strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 5 | +0.985 | 0.413 | 0.012 | 1.000 | DSR=0.41 ≤ 0.95; strict-Holm p ≥ 0.05 |
| buy_and_hold | asset_only | 3 | +0.956 | 0.389 | 0.014 | 1.000 | DSR=0.39 ≤ 0.95; strict-Holm p ≥ 0.05 |

## Comparison vs cycle-50 bracket-order baseline

- **Bracket bake-off** (`data/bakeoff/REPORT.md`): 0 winners / 8 assets, portfolio Sharpe NaN.
  - Per-asset best raw OOS Sharpes: 0.37–1.12, but DSR was uniformly reported as 0.00
    (calibration bug, identified and fixed in this variant).
- **Time-decay variant** (this report): 0 winners / 8 assets, portfolio Sharpe **+nan**.
  - Per-asset best raw OOS Sharpes: 0.75–1.31 (slightly higher than bracket's 0.37–1.12).
  - DSR values now 0.22–0.74 (sensible — none yet clear the 0.95 deflation bar).
  - Indicative gate-blind portfolio Sharpe: **+2.699**.

**Headline:** time-decay produces slightly higher raw OOS Sharpes than bracket (0.75–1.31
vs 0.37–1.12), with most candidates now passing Holm-corrected p < 0.05 and bootstrap-CI > 0.
The DSR > 0.95 gate is the binding constraint for both variants — neither delivers a
statistically-deflation-corrected winner on this universe at daily granularity.

## Methodology
- **Candidate space**: identical to the bracket bake-off (SMA-cross, EMA-cross, Donchian-20,
  RSI-2 MR, z-score MR, Bollinger-20×2σ, Momentum-21, Momentum-63, MACD-signal, random walk,
  buy-and-hold, Ridge, Logistic, LightGBM — last three across both asset_only and
  asset_plus_market scopes).
- **Walk-forward**: 8-year min train (capped 15y), 1-year inner val, 5-day embargo,
  1-day purge, 21-day test, monthly slide.
- **Fees**: `backtesting.fees.equity_alpaca` (commission-free; sell-side regs only).
- **Slippage**: `backtesting.slippage.slippage_bps_with_floor` (Almgren-Chriss square-root,
  per-asset floor, k=0.1) on both legs at entry/exit ADV.
- **Exit attribution**: each closed trade's net pct return is booked on the exit day so
  the daily-returns vector consumed by Sharpe/PSR/bootstrap has no overlapping contributions.
- **Single-position semantics**: while in position, new signals are ignored (no pyramiding).

## Limitations
- **Loses runners, lets losers ride**: by construction, N-bar time-decay cannot capture
  multi-week trends past N days and cannot cut losses inside the window. This is the
  explicit trade-off vs the bracket variant.
- **No mark-to-market**: P&L is realised on exit only. Daily MTM vol inside an open trade
  is not in the returns vector (the equal-weight portfolio aggregation is conservative).
- **Same survivorship-bias / no-regime-conditioning caveats as the bracket bake-off.**