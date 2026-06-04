"""Momentum / trend-following signals for leveraged ETFs.

Designed for 3× ETFs (SOXL, NVDL, TQQQ) where the established mean-reversion
library (rsi2, macd) cuts winning trends prematurely. The core idea is:

  1. Confirm a TREND exists (ADX filter).
  2. Enter only on a strong directional BREAKOUT (Donchian high).
  3. PYRAMID — add notional in fixed increments as price progresses,
     instead of taking a profit target and exiting.
  4. Exit only on trailing stop or ADX collapse.

Three exported functions:

  * :func:`compute_adx` — pure ADX(14) indicator.
  * :func:`compute_donchian_breakout` — boolean breakout series.
  * :func:`signal_momentum_pyramid_entry` — combined ADX-gated breakout
    entry signal. The PYRAMID adds and trail/exit logic live in the
    replay loop because they depend on running PnL state, not on the
    feature panel.

API contract for the sweep:
    entry = signal_momentum_pyramid_entry(bars, adx_threshold, breakout_window)
    # entry[i] ∈ {0, 1, -1}: 0=no signal, 1=long breakout, -1=short breakout
"""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "compute_adx",
    "compute_donchian_breakout",
    "signal_momentum_pyramid_entry",
]


def compute_adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength regardless of direction.

    Returns a Series aligned with ``bars.index``. NaN for the first
    ``period * 2`` bars (warmup). Values typically 0-100; >25 indicates
    a strong trend.
    """
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # True range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional movements
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=bars.index)
    minus_dm = pd.Series(minus_dm, index=bars.index)

    # Wilder's smoothing (EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, adjust=False,
                                     min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, adjust=False,
                                       min_periods=period).mean() / atr.replace(0, np.nan))

    dx_den = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / dx_den
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx


def compute_donchian_breakout(bars: pd.DataFrame, window: int) -> pd.Series:
    """Returns a Series of breakout direction:
        +1 if close > rolling(window).high.shift(1)  (upside breakout)
        -1 if close < rolling(window).low.shift(1)   (downside breakout)
         0 otherwise

    Shifted by 1 so the comparison uses YESTERDAY's window (no look-ahead).
    """
    close = bars["close"].astype(float)
    upper = bars["high"].astype(float).rolling(window).max().shift(1)
    lower = bars["low"].astype(float).rolling(window).min().shift(1)
    out = pd.Series(0, index=bars.index, dtype=np.int8)
    out[close > upper] = 1
    out[close < lower] = -1
    return out


def signal_momentum_pyramid_entry(bars: pd.DataFrame, *,
                                    adx_threshold: float,
                                    breakout_window: int,
                                    adx_period: int = 14) -> np.ndarray:
    """ADX-gated Donchian breakout entry signal.

    Returns numpy int8 array, same length as bars:
      +1: long entry (ADX > threshold AND upside breakout)
      -1: short entry (ADX > threshold AND downside breakout)
       0: no entry (trend too weak OR no breakout)

    The PYRAMID levels and trailing-stop exits are managed by the
    replay engine, not here — this just generates entry triggers.
    """
    adx = compute_adx(bars, period=adx_period)
    breakout = compute_donchian_breakout(bars, window=breakout_window)
    strong_trend = adx > adx_threshold
    out = np.zeros(len(bars), dtype=np.int8)
    long_mask = strong_trend & (breakout == 1)
    short_mask = strong_trend & (breakout == -1)
    out[long_mask.values] = 1
    out[short_mask.values] = -1
    # Warmup: first max(adx_period*2, breakout_window) bars are NaN/zero
    out[: max(adx_period * 2, breakout_window)] = 0
    return out
