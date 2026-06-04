"""Experimental signal class for 3× leveraged ETFs.

Synthesized from live research (2026-05-27):

  1. Connors RSI(2) mean-reversion (Connors 2008, "Short Term Trading
     Strategies That Work") — entry when RSI(2) < threshold, exit when
     close > 5-day SMA. No fixed stop (Connors finding: stops hurt MR).

  2. 200-day SMA regime filter (Brightwork Research; Bogleheads thread;
     academic 26.7% annual on 3× LETFs) — only enable long entries when
     close > SMA(200). Below SMA(200), strategy stays flat (or goes to
     cash equivalent in production).

  3. Optional volatility-target sizing — scale position size inversely
     to recent realized vol via L × R − ½ × L² × σ² volatility drag math
     (Menthor-Q LETF guide). Higher vol → smaller position. Implemented
     as a multiplier {0, 0.33, 0.67, 1.0} based on vol percentile.

The combined signal generator returns a *position level* in
{0, 0.33, 0.67, 1.0} representing fraction of full notional, so the
replay engine can size positions accordingly.

References (gathered via WebSearch/WebFetch 2026-05-27):
  - https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2
  - https://medium.com/@FMZQuant/larry-connors-rsi2-mean-reversion-strategy-861f5a3579e3
  - https://www.brightworkresearch.com/using-letfs-combined-with-the-200-day-moving-average-trading-approach/
  - https://menthorq.com/guide/leveraged-etfs-and-volatility-spxl-and-tqqq/
"""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "compute_rsi",
    "signal_connors_rsi_regime",
    "signal_connors_rsi_voltarget",
]


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def signal_connors_rsi_regime(
    bars: pd.DataFrame,
    *,
    rsi_period: int = 2,
    rsi_entry_threshold: float = 10.0,
    regime_sma_period: int = 200,
    exit_sma_period: int = 5,
    use_regime: bool = True,
) -> np.ndarray:
    """Connors RSI mean-reversion with 200-day MA regime filter.

    Returns a position-level numpy array, values in {0, 1}:
      1 = hold long position
      0 = flat / cash

    Logic (vectorised, no look-ahead — uses shift(1)):
      regime[i] = (close[i-1] > SMA(close, regime_sma_period)[i-1])
      entry[i]  = regime[i] AND RSI(period)[i-1] < entry_threshold
      exit[i]   = close[i-1] > SMA(close, exit_sma_period)[i-1]

      state[i] = 1 if (state[i-1]==1 AND not exit[i]) OR entry[i] else 0
    """
    close = bars["close"].astype(float)
    n = len(bars)
    if n < max(regime_sma_period, 50):
        return np.zeros(n, dtype=np.int8)

    rsi = compute_rsi(close, rsi_period).shift(1).fillna(50.0).values
    regime_sma = close.rolling(regime_sma_period).mean().shift(1).fillna(method="bfill").values
    exit_sma = close.rolling(exit_sma_period).mean().shift(1).fillna(method="bfill").values
    prev_close = close.shift(1).fillna(method="bfill").values

    regime_ok = (prev_close > regime_sma) if use_regime else np.ones(n, dtype=bool)
    entry_mask = regime_ok & (rsi < rsi_entry_threshold)
    exit_mask = prev_close > exit_sma

    out = np.zeros(n, dtype=np.int8)
    in_position = False
    for i in range(regime_sma_period, n):
        if in_position:
            if exit_mask[i]:
                in_position = False
        else:
            if entry_mask[i]:
                in_position = True
        out[i] = 1 if in_position else 0
    return out


def signal_connors_rsi_voltarget(
    bars: pd.DataFrame,
    *,
    rsi_period: int = 2,
    rsi_entry_threshold: float = 10.0,
    regime_sma_period: int = 200,
    exit_sma_period: int = 5,
    vol_lookback: int = 21,
    vol_percentile_caps: tuple[float, float, float] = (0.40, 0.70, 0.90),
) -> np.ndarray:
    """RSI/regime entry, vol-target sizing.

    Returns numpy float array of position FRACTIONS in {0.0, 0.33, 0.67, 1.0}:
      0.00: flat (no entry signal OR regime off OR vol percentile > 0.90)
      1.00: vol percentile ≤ 0.40 (calm — full size)
      0.67: vol percentile ≤ 0.70 (medium)
      0.33: vol percentile ≤ 0.90 (elevated)

    The replay engine reads this as the desired position fraction at each
    bar; deltas drive trades.
    """
    base_signal = signal_connors_rsi_regime(
        bars,
        rsi_period=rsi_period,
        rsi_entry_threshold=rsi_entry_threshold,
        regime_sma_period=regime_sma_period,
        exit_sma_period=exit_sma_period,
        use_regime=True,
    )
    close = bars["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(vol_lookback).std()
    # Trailing-window rolling percentile (no look-ahead): use expanding rank
    vol_pct = realized_vol.rolling(252, min_periods=vol_lookback).rank(pct=True).shift(1).fillna(0.5).values

    out = np.zeros(len(bars), dtype=np.float32)
    cap_calm, cap_mid, cap_elev = vol_percentile_caps
    for i in range(len(bars)):
        if base_signal[i] == 0:
            out[i] = 0.0
            continue
        v = vol_pct[i]
        if v <= cap_calm:
            out[i] = 1.0
        elif v <= cap_mid:
            out[i] = 0.67
        elif v <= cap_elev:
            out[i] = 0.33
        else:
            out[i] = 0.0
    return out
