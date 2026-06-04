"""ml_research.sandbox_signals — research/sandbox signal generators.

Three frameworks for high-volatility 3× leveraged ETFs, synthesized
from live research (2026-05-27, see sandbox_ideas.md for citations):

  1. ``signal_volatility_target`` — Inverse-vol position sizing
     (target_vol / realized_vol_21d, capped). Reduces exposure when
     market is choppy → mitigates LETF daily-reset decay.

  2. ``signal_kalman_trend`` — Local linear trend Kalman filter
     state-space model. Estimates hidden price level + velocity from
     noisy daily closes. Signal = sign(velocity).

  3. ``signal_mtf_coherence`` — Multi-timeframe moving-average
     alignment. Long only when SMA(5), SMA(20), SMA(50), and
     SMA(200) are in bullish order (perfect alignment).

All three are PURE FUNCTIONS over daily OHLCV — no production runner
modification, no minute-bar dependency. Returns numpy arrays aligned
with ``bars.index``.

Designed for sandbox use only. NOT exported to asset specialist modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "signal_volatility_target",
    "signal_kalman_trend",
    "signal_mtf_coherence",
]


# ===========================================================================
# 1. Volatility Targeting
#
# Position fraction = target_vol_annual / realized_vol_annual_21d, clipped to
# [0, max_leverage]. When realized vol > target, position scales down.
# Source: AQR "Volatility Targeting" framework, quantpedia.com/an-introduction-to-volatility-targeting/
# ===========================================================================


def signal_volatility_target(bars: pd.DataFrame, *,
                              target_vol_annual: float = 0.20,
                              vol_lookback: int = 21,
                              max_leverage: float = 1.0,
                              entry_filter_sma_period: int = 200) -> np.ndarray:
    """Inverse-vol position sizing.

    Returns float fractions in [0, max_leverage]. 1.0 = full notional,
    0.0 = no position.

    target_vol_annual:
        annualized vol the strategy targets (e.g. 0.20 = 20%/year).
    vol_lookback:
        days of returns used for realized-vol estimate.
    max_leverage:
        upper cap on the fraction.
    entry_filter_sma_period:
        if >0, gate position by close > SMA(close, period). Disabled
        when 0.

    The filter ensures we don't go full-vol-target during downtrends.
    """
    close = bars["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(vol_lookback).std() * np.sqrt(252)
    pos = (target_vol_annual / realized_vol).clip(upper=max_leverage)
    pos = pos.fillna(0.0)
    pos[pos < 0.05] = 0.0   # don't bother with tiny positions

    if entry_filter_sma_period > 0:
        sma = close.rolling(entry_filter_sma_period).mean().shift(1)
        regime_off = close.shift(1) < sma
        pos[regime_off] = 0.0

    # First-period warmup
    pos[:max(vol_lookback, entry_filter_sma_period)] = 0.0
    return pos.values.astype(np.float32)


# ===========================================================================
# 2. Kalman Filter — local linear trend state-space model
#
# State x_t = [level, velocity]^T evolves as:
#     x_t = F @ x_{t-1} + w_t,  w_t ~ N(0, Q)
#     y_t = H @ x_t   + v_t,   v_t ~ N(0, R)
#   with F = [[1, 1], [0, 1]], H = [1, 0]
#
# Recursion at each bar (predict + update):
#     x_pred  = F @ x_post
#     P_pred  = F @ P_post @ F^T + Q
#     S       = H @ P_pred @ H^T + R
#     K       = P_pred @ H^T / S
#     innov   = y - H @ x_pred
#     x_post  = x_pred + K @ innov
#     P_post  = (I - K @ H) @ P_pred
#
# Signal = sign(velocity_t) — long when hidden trend slope > 0.
# Source: PyQuantLab "Adaptive Kalman Trend Execution", quantinsti Kalman tutorial.
# ===========================================================================


def signal_kalman_trend(bars: pd.DataFrame, *,
                        process_noise_q: float = 1e-4,
                        observation_noise_r: float = 1e-2,
                        velocity_threshold: float = 0.0) -> np.ndarray:
    """Kalman-filtered hidden-trend long/short/flat signal.

    process_noise_q:
        diagonal of Q. Smaller → smoother trend. Sweep range: 1e-6 .. 1e-2.
    observation_noise_r:
        scalar R. Larger → more trust in model, less trust in observation.
        Sweep range: 1e-4 .. 1.0.
    velocity_threshold:
        |velocity| must exceed this to take a position. 0.0 = always trade.

    Returns int8 in {-1, 0, +1}.
    """
    log_close = np.log(bars["close"].astype(float).values)
    n = len(log_close)
    out = np.zeros(n, dtype=np.int8)

    # State
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * process_noise_q
    R_scalar = observation_noise_r

    # Initialize from first observation
    x_post = np.array([log_close[0], 0.0])
    P_post = np.eye(2) * 1.0  # broad prior

    velocity_series = np.zeros(n)
    for i in range(1, n):
        x_pred = F @ x_post
        P_pred = F @ P_post @ F.T + Q
        innov = log_close[i] - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T)[0, 0] + R_scalar
        K = (P_pred @ H.T).flatten() / S  # 2-vector
        x_post = x_pred + K * innov
        P_post = (np.eye(2) - np.outer(K, H[0])) @ P_pred
        velocity_series[i] = x_post[1]

    # Use velocity at i-1 to make signal at i (no look-ahead)
    velocity_lag = np.concatenate([[0.0], velocity_series[:-1]])
    out[velocity_lag > velocity_threshold] = 1
    out[velocity_lag < -velocity_threshold] = -1
    return out


# ===========================================================================
# 3. Multi-Timeframe Momentum Coherence
#
# Uses four daily SMAs to approximate macro/meso/micro alignment within
# the daily-bar substrate available. Full 4-MA bullish alignment requires:
#     SMA_5 > SMA_20 > SMA_50 > SMA_200
# All four must be in ascending order AND close > SMA_5.
#
# In future when minute-bar SOXL/NVDL substrate is available, the SMA_5
# can be replaced with a true intraday EMA for genuine multi-timeframe.
# Source: alinakhay.com "Mastering the Tri-Timeframe Trend-Following System",
#         medium.com Multi-Timeframe EMA Trend Momentum (Sword Red).
# ===========================================================================


def signal_mtf_coherence(bars: pd.DataFrame, *,
                          sma_periods: tuple[int, int, int, int] = (5, 20, 50, 200),
                          require_strict: bool = True) -> np.ndarray:
    """Multi-timeframe SMA-alignment momentum signal.

    Returns int8 in {-1, 0, +1}.

    require_strict:
        if True, requires STRICT ordering (each < next).
        if False, allows any close > SMA_200 (weakest filter).
    """
    close = bars["close"].astype(float)
    p1, p2, p3, p4 = sma_periods
    sma_a = close.rolling(p1).mean().shift(1)
    sma_b = close.rolling(p2).mean().shift(1)
    sma_c = close.rolling(p3).mean().shift(1)
    sma_d = close.rolling(p4).mean().shift(1)
    prev_close = close.shift(1)

    if require_strict:
        long_mask = ((prev_close > sma_a) & (sma_a > sma_b) &
                      (sma_b > sma_c) & (sma_c > sma_d))
        short_mask = ((prev_close < sma_a) & (sma_a < sma_b) &
                       (sma_b < sma_c) & (sma_c < sma_d))
    else:
        long_mask = prev_close > sma_d
        short_mask = prev_close < sma_d

    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    out[long_mask.fillna(False).values] = 1
    out[short_mask.fillna(False).values] = -1
    return out
