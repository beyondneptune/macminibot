# Per-Asset Bake-off Report

Walk-forward, bracket-order intraday simulation on daily bars.

**Two-stage statistical gate:**
1. *Within-asset* multiple-comparison correction: Deflated Sharpe Ratio
   (Bailey & López de Prado 2014) with `n_trials = #candidates_searched_for_asset`,
   accounting for the full (model × scope × tp_σ × sl_σ × hit_priority) search.
2. *Across-asset* family-wise correction: Bonferroni-Holm step-down over the 8
   best-of-asset p-values (controls FWER on the portfolio of promotion decisions).

**Promotion gate**: `p_Holm_acrossAssets < 0.05` AND `DSR > 0.95` AND `bootstrap-CI lower > 0` AND `Sharpe > 0`.

A more conservative *strict* gate is also reported: Bonferroni-Holm across the FULL
(asset × candidate × scope × tp × sl × hp) family (~5,300 hypotheses); this is the
hardest-possible-to-pass calibration and is usually too punitive.

## Winners (one per asset; or NO WINNER)

| asset | verdict | model | scope | tp_σ | sl_σ | hp | Sharpe (net) | DSR | p_Holm_8 | p_Holm_strict | CI_lo | CI_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | NO WINNER | — | — | — | — | — | — | — | — | — | — | — |
| MSFT | NO WINNER | — | — | — | — | — | — | — | — | — | — | — |
| GOOGL | NO WINNER | — | — | — | — | — | — | — | — | — | — | — |
| AMZN | NO WINNER | — | — | — | — | — | — | — | — | — | — | — |
| META | NO WINNER | — | — | — | — | — | — | — | — | — | — | — |
| TSLA | NO WINNER | — | — | — | — | — | — | — | — | — | — | — |
| NVDA | NO WINNER | — | — | — | — | — | — | — | — | — | — | — |
| GLD | NO WINNER | — | — | — | — | — | — | — | — | — | — | — |

## Aggregate portfolio Sharpe
Equal-weight, daily-rebalanced portfolio of the 0 winners:
**Sharpe (net) = +nan**

**No winners survived all gates.** The honest answer is to NOT trade any
of the 8 assets actively with daily-bar bracket orders. The best raw OOS
Sharpes are in the 1.6–1.9 range (per-asset, gross of FWER), but DSR and
the bootstrap CI cut these down once the size of the candidate search is
accounted for. See the *Per-asset best* table below for the runners-up.

## Per-asset best candidate (before/after each gate)

| asset | model | scope | tp_σ | sl_σ | hp | Sharpe | DSR | p_unadj | p_Holm_8 | p_Holm_strict | CI_lo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | ridge | asset_plus_market | None | None | sl_first | +0.637 | 0.000 | 0.0722 | 0.1996 | 1.0000 | -0.221 |
| MSFT | zscore_mr_1.5 | asset_only | None | None | sl_first | +0.808 | 0.000 | 0.0320 | 0.1600 | 1.0000 | -0.030 |
| GOOGL | momentum_63 | asset_only | 2.0 | 1.5 | sl_first | +0.442 | 0.000 | 0.1558 | 0.1996 | 1.0000 | -0.406 |
| AMZN | rsi2_mr_10_90 | asset_only | None | None | sl_first | +0.368 | 0.000 | 0.1996 | 0.1996 | 1.0000 | -0.444 |
| META | momentum_63 | asset_only | None | None | sl_first | +0.487 | 0.000 | 0.1324 | 0.1996 | 1.0000 | -0.396 |
| TSLA | macd_signal | asset_only | 2.0 | 1.0 | sl_first | +1.022 | 0.000 | 0.0096 | 0.0768 | 1.0000 | +0.196 |
| NVDA | zscore_mr_1.5 | asset_only | None | None | sl_first | +0.870 | 0.000 | 0.0231 | 0.1600 | 1.0000 | +0.045 |
| GLD | logistic | asset_plus_market | None | None | sl_first | +0.823 | 0.000 | 0.0297 | 0.1600 | 1.0000 | -0.017 |

## Hit-priority sensitivity — the headline finding

The intra-bar TP/SL ordering convention is the single largest swing in the bake-off.
For each asset, here is the SAME (model, scope, tp_σ, sl_σ) configuration evaluated
under each of the three priorities — `tp_first` (optimistic), `random` (neutral), and
`sl_first` (conservative). The selection takes the top `random`-priority config and
re-evaluates it under the other two:

| asset | model | tp_σ | sl_σ | Sharpe (sl_first) | Sharpe (random) | Sharpe (tp_first) |
|---|---|---|---|---|---|---|
| AAPL | bollinger_20_2 | 0.5 | 0.5 | -1.410 | +1.901 | +1.901 |
| MSFT | zscore_mr_1.5 | 0.5 | 0.5 | -1.861 | +2.441 | +3.158 |
| GOOGL | zscore_mr_1.5 | 0.5 | 0.5 | -2.235 | +1.927 | +2.348 |
| AMZN | bollinger_20_2 | 0.5 | 0.5 | -1.173 | +1.742 | +1.871 |
| META | momentum_63 | 1.0 | 0.5 | -0.429 | +1.872 | +2.216 |
| TSLA | bollinger_20_2 | 0.5 | 0.5 | -0.698 | +2.550 | +2.678 |
| NVDA | zscore_mr_1.5 | 0.5 | 0.5 | -1.847 | +1.636 | +2.213 |
| GLD | logistic | 0.5 | 0.5 | -2.679 | +1.675 | +3.440 |

Sharpe swings of 3–4 units between `tp_first` and `sl_first` for the same exact
strategy. **This is not edge — it is intra-bar fill-convention bias.** Any winner
that requires `tp_first` to look good is rejected.

## Rejected candidates (top-3 per asset under the CONSERVATIVE sl_first gate)

### AAPL (sl_first only)
| model | scope | tp_σ | sl_σ | Sharpe | DSR | p_unadj | p_Holm_8 | why rejected |
|---|---|---|---|---|---|---|---|---|
| ridge | asset_plus_market | None | None | +0.637 | 0.000 | 0.072 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | 1.5 | 1.5 | +0.528 | 0.000 | 0.113 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | 2.0 | 1.5 | +0.528 | 0.000 | 0.113 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | None | None | +0.528 | 0.000 | 0.113 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | 1.5 | 0.5 | +0.518 | 0.000 | 0.118 | 1.000 | DSR=0.00 ≤ 0.95 |

### MSFT (sl_first only)
| model | scope | tp_σ | sl_σ | Sharpe | DSR | p_unadj | p_Holm_8 | why rejected |
|---|---|---|---|---|---|---|---|---|
| zscore_mr_1.5 | asset_only | None | None | +0.808 | 0.000 | 0.032 | 1.000 | DSR=0.00 ≤ 0.95 |
| rsi2_mr_10_90 | asset_only | None | None | +0.685 | 0.000 | 0.058 | 1.000 | DSR=0.00 ≤ 0.95 |
| zscore_mr_1.5 | asset_only | 1.0 | 1.5 | +0.573 | 0.000 | 0.095 | 1.000 | DSR=0.00 ≤ 0.95 |
| zscore_mr_1.5 | asset_only | 2.0 | 1.5 | +0.399 | 0.000 | 0.180 | 1.000 | DSR=0.00 ≤ 0.95 |
| zscore_mr_1.5 | asset_only | 1.5 | 1.5 | +0.335 | 0.000 | 0.221 | 1.000 | DSR=0.00 ≤ 0.95 |

### GOOGL (sl_first only)
| model | scope | tp_σ | sl_σ | Sharpe | DSR | p_unadj | p_Holm_8 | why rejected |
|---|---|---|---|---|---|---|---|---|
| momentum_63 | asset_only | 2.0 | 1.5 | +0.442 | 0.000 | 0.156 | 1.000 | DSR=0.00 ≤ 0.95 |
| buy_and_hold | asset_only | None | None | +0.442 | 0.000 | 0.156 | 1.000 | DSR=0.00 ≤ 0.95 |
| lgbm | asset_only | None | None | +0.373 | 0.000 | 0.196 | 1.000 | DSR=0.00 ≤ 0.95 |
| lgbm | asset_only | 2.0 | 1.5 | +0.373 | 0.000 | 0.197 | 1.000 | DSR=0.00 ≤ 0.95 |
| momentum_63 | asset_only | None | None | +0.368 | 0.000 | 0.200 | 1.000 | DSR=0.00 ≤ 0.95 |

### AMZN (sl_first only)
| model | scope | tp_σ | sl_σ | Sharpe | DSR | p_unadj | p_Holm_8 | why rejected |
|---|---|---|---|---|---|---|---|---|
| rsi2_mr_10_90 | asset_only | None | None | +0.368 | 0.000 | 0.200 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | 2.0 | 1.5 | +0.293 | 0.000 | 0.251 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | 1.5 | 1.5 | +0.194 | 0.000 | 0.329 | 1.000 | DSR=0.00 ≤ 0.95 |
| bollinger_20_2 | asset_only | 2.0 | 1.0 | +0.179 | 0.000 | 0.341 | 1.000 | DSR=0.00 ≤ 0.95 |
| bollinger_20_2 | asset_only | 2.0 | 1.5 | +0.146 | 0.000 | 0.369 | 1.000 | DSR=0.00 ≤ 0.95 |

### META (sl_first only)
| model | scope | tp_σ | sl_σ | Sharpe | DSR | p_unadj | p_Holm_8 | why rejected |
|---|---|---|---|---|---|---|---|---|
| momentum_63 | asset_only | None | None | +0.487 | 0.000 | 0.132 | 1.000 | DSR=0.00 ≤ 0.95 |
| lgbm | asset_plus_market | None | None | +0.468 | 0.000 | 0.142 | 1.000 | DSR=0.00 ≤ 0.95 |
| rsi2_mr_10_90 | asset_only | None | None | +0.462 | 0.000 | 0.145 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | 1.0 | 0.5 | +0.436 | 0.000 | 0.159 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | 1.0 | 1.0 | +0.436 | 0.000 | 0.159 | 1.000 | DSR=0.00 ≤ 0.95 |

### TSLA (sl_first only)
| model | scope | tp_σ | sl_σ | Sharpe | DSR | p_unadj | p_Holm_8 | why rejected |
|---|---|---|---|---|---|---|---|---|
| bollinger_20_2 | asset_only | 2.0 | 1.0 | +1.118 | 0.000 | 0.005 | 1.000 | DSR=0.00 ≤ 0.95 |
| bollinger_20_2 | asset_only | 1.5 | 1.0 | +1.075 | 0.000 | 0.007 | 1.000 | DSR=0.00 ≤ 0.95 |
| macd_signal | asset_only | 2.0 | 1.0 | +1.022 | 0.000 | 0.010 | 1.000 | DSR=0.00 ≤ 0.95 |
| bollinger_20_2 | asset_only | 2.0 | 1.5 | +0.983 | 0.000 | 0.012 | 1.000 | DSR=0.00 ≤ 0.95 |
| logistic | asset_only | 2.0 | 1.0 | +0.931 | 0.000 | 0.016 | 1.000 | DSR=0.00 ≤ 0.95 |

### NVDA (sl_first only)
| model | scope | tp_σ | sl_σ | Sharpe | DSR | p_unadj | p_Holm_8 | why rejected |
|---|---|---|---|---|---|---|---|---|
| zscore_mr_1.5 | asset_only | None | None | +0.870 | 0.000 | 0.023 | 1.000 | DSR=0.00 ≤ 0.95 |
| rsi2_mr_10_90 | asset_only | None | None | +0.576 | 0.000 | 0.093 | 1.000 | DSR=0.00 ≤ 0.95 |
| zscore_mr_1.5 | asset_only | 2.0 | 1.5 | +0.542 | 0.000 | 0.107 | 1.000 | DSR=0.00 ≤ 0.95 |
| buy_and_hold | asset_only | None | None | +0.523 | 0.000 | 0.115 | 1.000 | DSR=0.00 ≤ 0.95 |
| donchian_20 | asset_only | 1.0 | 1.5 | +0.382 | 0.000 | 0.190 | 1.000 | DSR=0.00 ≤ 0.95 |

### GLD (sl_first only)
| model | scope | tp_σ | sl_σ | Sharpe | DSR | p_unadj | p_Holm_8 | why rejected |
|---|---|---|---|---|---|---|---|---|
| logistic | asset_plus_market | None | None | +0.823 | 0.000 | 0.030 | 1.000 | DSR=0.00 ≤ 0.95 |
| logistic | asset_plus_market | 2.0 | 1.5 | +0.753 | 0.000 | 0.042 | 1.000 | DSR=0.00 ≤ 0.95 |
| logistic | asset_plus_market | 1.5 | 1.5 | +0.750 | 0.000 | 0.043 | 1.000 | DSR=0.00 ≤ 0.95 |
| ridge | asset_only | None | None | +0.687 | 0.000 | 0.058 | 1.000 | DSR=0.00 ≤ 0.95 |
| logistic | asset_only | 1.5 | 1.5 | +0.675 | 0.000 | 0.061 | 1.000 | DSR=0.00 ≤ 0.95 |

## Top-3 surprising findings

1. **Intra-bar fill convention dominates everything.** Switching the tie-breaking rule
   when both TP and SL touch within a daily bar from `sl_first` (conservative) to
   `tp_first` (optimistic) lifts net Sharpe by 3–4 units for the same exact strategy
   parameters. The tight 0.5σ × 0.5σ bracket configurations look like Sharpe-3+ printing
   machines under `tp_first` and collapse to Sharpe -1 to -3 under `sl_first`. This is
   not a real edge — it's the implicit look-ahead in the daily-bar bracket simulation.
   The headline result of the bake-off is: **daily-bar bracket-order backtests are
   structurally unreliable for tight stops; we need intraday data to resolve.**

2. **The `asset_only` feature scope wins everywhere it matters.** Of the per-asset best
   candidates under `sl_first`, 6 of 8 are `asset_only` (no market context features).
   Adding SPY/QQQ/IWM benchmark returns as features did NOT systematically improve the
   per-asset OOS Sharpe — in fact for AAPL the `asset_plus_market` ridge was best, but
   for everything else the cross-asset context was either neutral or noise. This is
   evidence against the cross-sectional architecture for these specific 8 names.

3. **No-bracket (EOD close) candidates often beat the bracket variants under `sl_first`.**
   For 5 of 8 assets (MSFT, AMZN, META, NVDA, GLD), the best `sl_first` candidate has
   TP=None / SL=None — i.e. the trade just runs to the close. This makes sense in
   reverse: any tight stop is hit by noise on a typical 2σ-range day, so cutting losses
   intra-day on daily-bar signals is destructive without finer-grained price information.
   The brief asked for bracket-order TP/SL configs — the honest answer is that on daily
   bars, the best 'bracket' is no bracket.


## Methodology
- **Feature scopes tested**: `asset_only` and `asset_plus_market` (lagged SPY/QQQ/IWM returns + 5/21d rolling sums).
  Rule-based candidates use `asset_only` only (no benefit from market features for a threshold rule).
- **Candidates**: SMA cross (10/30), EMA cross (12/26), Donchian-20 breakout, RSI-2 mean-reversion, z-score-21 mean-reversion, Bollinger-20×2σ breakout, Momentum-21, Momentum-63, MACD signal, Random walk (sanity), Buy-and-hold (control), Ridge regression, Logistic regression, LightGBM classifier.
- **TP/SL grid**: TP ∈ {0.5σ, 1.0σ, 1.5σ, 2.0σ}, SL ∈ {0.5σ, 1.0σ, 1.5σ}, plus an EOD-close no-bracket control. σ = trailing 21d realised log-return std.
- **Walk-forward**: 8-year minimum training (capped at 15y), 1-year inner-validation, 5-day embargo, 1-day purge, 21-day (monthly) test, monthly slide.
- **Fees**: `backtesting.fees.equity_alpaca` (commission-free; sell-side regs only).
- **Slippage**: `backtesting.slippage.slippage_bps_with_floor` (Almgren-Chriss square-root + per-asset floor, k=0.1).
- **Path-dependent bracket simulation**: Entry at `open[T]` if signal computed on info ≤ T-1 is non-zero. Exit at TP (≤ high) or SL (≤ low) intra-bar T; else at `close[T]`. Hit-priority sensitivity tested with three conventions (`sl_first`, `tp_first`, `random`).
- **FWER correction (two-stage)**:
  1. *Within-asset*: Deflated Sharpe (Bailey & López de Prado 2014) with variance-calibrated
     `sr_trials_std` (empirical std across the asset's sl_first sub-family) and `n_trials` =
     the sl_first sub-family size (≈ 221 per asset).
  2. *Across-asset*: Bonferroni-Holm step-down over the 8 per-asset-best p-values.
  A more conservative variant — strict Holm over the full (asset × cand × scope × tp × sl × hp)
  family of ≈ 5,304 hypotheses — is also reported (`p_Holm_strict`) but is dominated by the
  tp_first / 0.5σ-bracket outliers and is rarely informative for sl_first candidates.
- **Hit-priority robustness**: per-asset ranking uses `sl_first` ONLY. Candidates that look
  good under `tp_first` but collapse under `sl_first` are excluded by construction.
- **Promotion gate**: Holm-adjusted (across-asset) p < 0.05  AND  Deflated Sharpe > 0.95
  AND  block-bootstrap 95% CI lower bound > 0  AND  raw Sharpe > 0.

## Limitations
- **Daily-bar intraday simulation**: TP/SL hits are inferred from the bar's high/low; the *order* in which TP and SL get hit within a bar is unknown from daily data — the `hit_priority` sensitivity check is the honest mitigation.
- **No slippage on TP/SL stop fills**: a real stop fills at a worse price than the trigger in fast tape; this version assumes fills at the trigger price. The fee+slippage on both legs is charged using ADV-based Almgren-Chriss, but the additional 'stop-jump' cost (1–5 bps in fast tape) is *not* modelled.
- **No intraday data**: we cannot test sub-daily holding periods or true intra-day momentum/mean-reversion microstructure. This must be revisited once tick-level data is backfilled.
- **Survivorship bias**: the 8 contracts here are all currently-listed; no de-listings are simulated. Acceptable given the universe is hand-picked liquid mega-caps + GLD.
- **No regime conditioning**: VIX-regime stratification (per `backtesting.regime`) is supported by the driver but not turned on here — left for a follow-up cycle since the bake-off is wide enough already.