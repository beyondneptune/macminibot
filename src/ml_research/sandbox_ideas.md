# Sandbox Research Report — Overnight Campaign 2026-05-27
## Frameworks evaluated
1. **Volatility Targeting** — inverse-vol position sizing
2. **Kalman Filter** — local linear trend state-space model
3. **Multi-Timeframe Momentum Coherence** — 4-SMA alignment

All evaluated on SOXL + NVDL (Alpaca IEX daily, ~750 bars each, 2023-05 → 2026-05). Walk-forward 60/40 train/test split.

Costs: IBKR Pro tiered commission + Almgren-Chriss √-impact + SEC §31 + FINRA TAF.

---

## SOXL

B&H train: $-6,178  |  B&H test: $+97,143

### VolTarget

Best (ranked by min(train_α, test_α)):

- Params: `tv=0.5 sma=0 ml=1.0`
- TRAIN: net=$-3,051  α=$+3,127  Sharpe=-0.38  Omega=0.93  DD=$-11,461  trades=50
- TEST: net=$+18,069  α=$-79,074  Sharpe=+2.99  Omega=1.65  DD=$-2,831  trades=31

  ⚠️ OVERFIT: alpha positive on train but negative on test

### Kalman

Best (ranked by min(train_α, test_α)):

- Params: `q=1e-06 r=0.001 vthr=0.0001`
- TRAIN: net=$+864  α=$+7,042  Sharpe=+0.07  Omega=1.02  DD=$-18,205  trades=24
- TEST: net=$+34,325  α=$-62,817  Sharpe=+2.24  Omega=1.62  DD=$-7,120  trades=11

  ⚠️ OVERFIT: alpha positive on train but negative on test

### MTF-Coherence

Best (ranked by min(train_α, test_α)):

- Params: `sma=(5, 20, 50, 200) strict=False`
- TRAIN: net=$-3,739  α=$+2,439  Sharpe=-0.47  Omega=0.84  DD=$-9,139  trades=15
- TEST: net=$+52,031  α=$-45,112  Sharpe=+2.26  Omega=1.50  DD=$-15,166  trades=3

  ⚠️ OVERFIT: alpha positive on train but negative on test

#### Top-3 per framework (raw)

**VolTarget**

| Rank | Params | TR α | TR Sh | TE α | TE Sh | TE Omega | TE DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `tv=0.5 sma=0 ml=1.0` | $+3,127 | -0.38 | $-79,074 | +2.99 | 1.65 | $-2,831 |
| 2 | `tv=0.5 sma=200 ml=1.0` | $+5,262 | -0.21 | $-79,435 | +3.05 | 1.66 | $-2,831 |
| 3 | `tv=0.5 sma=100 ml=1.0` | $+4,152 | -0.41 | $-81,565 | +2.71 | 1.59 | $-3,202 |

**Kalman**

| Rank | Params | TR α | TR Sh | TE α | TE Sh | TE Omega | TE DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `q=1e-06 r=0.001 vthr=0.0001` | $+7,042 | +0.07 | $-62,817 | +2.24 | 1.62 | $-7,120 |
| 2 | `q=1e-05 r=0.01 vthr=0.0001` | $+7,042 | +0.07 | $-62,817 | +2.24 | 1.62 | $-7,120 |
| 3 | `q=0.0001 r=0.1 vthr=0.0001` | $+7,042 | +0.07 | $-62,817 | +2.24 | 1.62 | $-7,120 |

**MTF-Coherence**

| Rank | Params | TR α | TR Sh | TE α | TE Sh | TE Omega | TE DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `sma=(5, 20, 50, 200) strict=False` | $+2,439 | -0.47 | $-45,112 | +2.26 | 1.50 | $-15,166 |
| 2 | `sma=(10, 30, 90, 200) strict=False` | $+2,439 | -0.47 | $-45,112 | +2.26 | 1.50 | $-15,166 |
| 3 | `sma=(20, 50, 100, 200) strict=False` | $+2,439 | -0.47 | $-45,112 | +2.26 | 1.50 | $-15,166 |

## NVDL

B&H train: $+7,115  |  B&H test: $+1,880

### VolTarget

Best (ranked by min(train_α, test_α)):

- Params: `tv=0.5 sma=0 ml=1.0`
- TRAIN: net=$+7,157  α=$+42  Sharpe=+0.90  Omega=1.16  DD=$-6,517  trades=38
- TEST: net=$+3,252  α=$+1,372  Sharpe=+0.61  Omega=1.11  DD=$-4,020  trades=36

### Kalman

Best (ranked by min(train_α, test_α)):

- Params: `q=1e-06 r=0.1 vthr=0.0001`
- TRAIN: net=$+4,726  α=$-2,389  Sharpe=+0.50  Omega=1.12  DD=$-8,545  trades=7
- TEST: net=$-826  α=$-2,706  Sharpe=-0.15  Omega=0.97  DD=$-5,685  trades=5

  ⚠️ OVERFIT: Sharpe collapses from train (+0.50) to test (-0.15)

### MTF-Coherence

Best (ranked by min(train_α, test_α)):

- Params: `sma=(20, 50, 100, 200) strict=True`
- TRAIN: net=$+0  α=$-7,115  Sharpe=+nan  Omega=inf  DD=$+0  trades=0
- TEST: net=$-3,900  α=$-5,780  Sharpe=-1.29  Omega=0.56  DD=$-4,268  trades=10

#### Top-3 per framework (raw)

**VolTarget**

| Rank | Params | TR α | TR Sh | TE α | TE Sh | TE Omega | TE DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `tv=0.5 sma=0 ml=1.0` | $+42 | +0.90 | $+1,372 | +0.61 | 1.11 | $-4,020 |
| 2 | `tv=0.5 sma=0 ml=0.5` | $-1,447 | +0.72 | $-972 | +0.27 | 1.05 | $-3,215 |
| 3 | `tv=0.3 sma=0 ml=0.5` | $-2,946 | +0.87 | $-574 | +0.45 | 1.08 | $-2,649 |

**Kalman**

| Rank | Params | TR α | TR Sh | TE α | TE Sh | TE Omega | TE DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `q=1e-06 r=0.1 vthr=0.0001` | $-2,389 | +0.50 | $-2,706 | -0.15 | 0.97 | $-5,685 |
| 2 | `q=1e-06 r=0.1 vthr=0.0005` | $-3,800 | +0.35 | $-2,253 | -0.07 | 0.98 | $-5,186 |
| 3 | `q=1e-06 r=0.1 vthr=0.0` | $-4,017 | +0.32 | $-2,068 | -0.03 | 0.99 | $-5,076 |

**MTF-Coherence**

| Rank | Params | TR α | TR Sh | TE α | TE Sh | TE Omega | TE DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `sma=(20, 50, 100, 200) strict=True` | $-7,115 | +nan | $-5,780 | -1.29 | 0.56 | $-4,268 |
| 2 | `sma=(3, 10, 30, 100) strict=False` | $-7,166 | -0.01 | $-3,813 | -0.33 | 0.93 | $-7,768 |
| 3 | `sma=(10, 30, 90, 200) strict=True` | $-7,259 | -1.28 | $-5,322 | -1.50 | 0.48 | $-3,442 |

---

## Overall ranking (top 10 by min(train_α, test_α))

| Rank | Asset | Framework | Params | TR α | TR Sh | TE α | TE Sh | TE Omega |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | NVDL | VolTarget | `tv=0.5 sma=0 ml=1.0` | $+42 | +0.90 | $+1,372 | +0.61 | 1.11 |
| 2 | NVDL | VolTarget | `tv=0.5 sma=0 ml=0.5` | $-1,447 | +0.72 | $-972 | +0.27 | 1.05 |
| 3 | NVDL | Kalman | `q=1e-06 r=0.1 vthr=0.0001` | $-2,389 | +0.50 | $-2,706 | -0.15 | 0.97 |
| 4 | NVDL | VolTarget | `tv=0.3 sma=0 ml=0.5` | $-2,946 | +0.87 | $-574 | +0.45 | 1.08 |
| 5 | NVDL | VolTarget | `tv=0.3 sma=0 ml=1.0` | $-3,088 | +0.84 | $-635 | +0.40 | 1.07 |
| 6 | NVDL | Kalman | `q=1e-06 r=0.1 vthr=0.0005` | $-3,800 | +0.35 | $-2,253 | -0.07 | 0.98 |
| 7 | NVDL | Kalman | `q=1e-06 r=0.1 vthr=0.0` | $-4,017 | +0.32 | $-2,068 | -0.03 | 0.99 |
| 8 | NVDL | VolTarget | `tv=0.2 sma=0 ml=0.5` | $-4,142 | +0.91 | $+364 | +0.97 | 1.18 |
| 9 | NVDL | VolTarget | `tv=0.2 sma=0 ml=1.0` | $-4,142 | +0.91 | $+364 | +0.97 | 1.18 |
| 10 | NVDL | VolTarget | `tv=0.5 sma=100 ml=1.0` | $-5,225 | +0.32 | $-2,479 | -0.14 | 0.97 |

## Robust candidates (train_α > 0 AND test_α > 0)

Found 1 robust configs:

| Asset | Framework | Params | TR α | TE α | min α | TE Sh | TE Omega | TE DD |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NVDL | VolTarget | `tv=0.5 sma=0 ml=1.0` | $+42 | $+1,372 | $+42 | +0.61 | 1.11 | $-4,020 |

---

## Conclusions & Citations

**Methodology**: Walk-forward 60/40 split, IBKR-tiered commission + Almgren-Chriss √-impact slippage + SEC §31 + FINRA TAF. Position sizing via fractional replay (rebalance when |Δfrac| > 0.05).

**Sample size caveat**: SOXL and NVDL only have ~750 daily bars (launched 2023-05). Statistical power is limited; results should be treated as exploratory rather than predictive.

**Key sources** (web research 2026-05-27):

- Volatility Targeting: [QuantPedia — An Introduction to Volatility Targeting](https://quantpedia.com/an-introduction-to-volatility-targeting/), [QuantPedia — Leveraged ETFs in Low-Volatility Environments](https://quantpedia.com/leveraged-etfs-in-low-volatility-environments/), [arXiv 2603.01298 — Single-Asset Adaptive Leveraged Volatility Control](https://arxiv.org/pdf/2603.01298)
- Kalman Filter: [PyQuantLab — Adaptive Kalman Trend Execution](https://pyquantlab.medium.com/adaptive-kalman-trend-execution-auto-tuned-state-estimation-with-velocity-triggers-78438b7bd5f7), [QuantInsti — Kalman Filter Python Tutorial](https://blog.quantinsti.com/kalman-filter/), [Forecastegy — Kalman Filter for Time Series Forecasting](https://forecastegy.com/posts/kalman-filter-for-time-series-forecasting-in-python/)
- Multi-Timeframe Momentum: [AlinaKhay — Tri-Timeframe Trend-Following System](https://alinakhay.com/p/mastering-the-tri-timeframe-trend), [Sword Red Medium — Multi-Timeframe EMA Trend Momentum](https://medium.com/@redsword_23261/multi-timeframe-ema-trend-momentum-trading-strategy-612d1df08d79), [Mind Math Money — Multi-Timeframe Analysis 2026](https://www.mindmathmoney.com/articles/multi-timeframe-analysis-trading-strategy-the-complete-guide-to-trading-multiple-timeframes)

