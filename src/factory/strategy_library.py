"""factory.strategy_library — a zoo of signal templates for the
Autonomous Quant Factory.

Each entry produces a numpy int8 array of {-1, 0, +1} per bar. Designed
to be called from the Crucible's replay engine. NO look-ahead — all
indicators are computed with .shift(1).

Library categories:
  * Mean reversion: bollinger, rsi2, rsi14, zscore_30, williams_r,
    stochastic_k
  * Momentum: macd_hist, donchian_breakout, atr_breakout,
    triple_ma_align, linreg_slope, obv_momentum
  * Volatility-aware: vol_target_sma, atr_filtered_bah
  * Hybrid: bollinger_with_rsi_filter, momentum_after_dip
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                     (high - prev_close).abs(),
                     (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Strategy generators — each returns np.ndarray int8 aligned with bars.index
# ---------------------------------------------------------------------------


def strat_bollinger(bars: pd.DataFrame, *, window: int = 20,
                     entry_z: float = 2.0) -> np.ndarray:
    close = bars["close"].astype(float)
    ma = _sma(close, window).shift(1)
    sd = close.rolling(window).std().shift(1)
    z = (close.shift(1) - ma) / sd.replace(0, np.nan)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(z < -entry_z).fillna(False).values] = 1
    out[(z > entry_z).fillna(False).values] = -1
    return out


def strat_rsi2(bars: pd.DataFrame, *, low: float = 10.0,
                high: float = 90.0) -> np.ndarray:
    r = _rsi(bars["close"].astype(float), 2).shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(r < low).fillna(False).values] = 1
    out[(r > high).fillna(False).values] = -1
    return out


def strat_rsi14(bars: pd.DataFrame, *, low: float = 30.0,
                 high: float = 70.0) -> np.ndarray:
    r = _rsi(bars["close"].astype(float), 14).shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(r < low).fillna(False).values] = 1
    out[(r > high).fillna(False).values] = -1
    return out


def strat_zscore_reversion(bars: pd.DataFrame, *, window: int = 30,
                             entry_z: float = 2.5) -> np.ndarray:
    close = bars["close"].astype(float)
    ma = close.rolling(window).mean().shift(1)
    sd = close.rolling(window).std().shift(1)
    z = (close.shift(1) - ma) / sd.replace(0, np.nan)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(z < -entry_z).fillna(False).values] = 1
    out[(z > entry_z).fillna(False).values] = -1
    return out


def strat_williams_r(bars: pd.DataFrame, *, period: int = 14,
                       low: float = -80.0, high: float = -20.0) -> np.ndarray:
    high_n = bars["high"].rolling(period).max().shift(1)
    low_n = bars["low"].rolling(period).min().shift(1)
    close = bars["close"].shift(1)
    wr = -100.0 * (high_n - close) / (high_n - low_n).replace(0, np.nan)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(wr < low).fillna(False).values] = 1
    out[(wr > high).fillna(False).values] = -1
    return out


def strat_stochastic(bars: pd.DataFrame, *, period: int = 14,
                       low: float = 20.0, high: float = 80.0) -> np.ndarray:
    high_n = bars["high"].rolling(period).max().shift(1)
    low_n = bars["low"].rolling(period).min().shift(1)
    close = bars["close"].shift(1)
    k = 100.0 * (close - low_n) / (high_n - low_n).replace(0, np.nan)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(k < low).fillna(False).values] = 1
    out[(k > high).fillna(False).values] = -1
    return out


def strat_macd_hist(bars: pd.DataFrame, *, fast: int = 12, slow: int = 26,
                      signal: int = 9) -> np.ndarray:
    close = bars["close"].astype(float)
    macd = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd, signal)
    hist = (macd - signal_line).shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(hist > 0).fillna(False).values] = 1
    out[(hist < 0).fillna(False).values] = -1
    return out


def strat_donchian_breakout(bars: pd.DataFrame, *, window: int = 20) -> np.ndarray:
    close = bars["close"].astype(float)
    upper = bars["high"].rolling(window).max().shift(1)
    lower = bars["low"].rolling(window).min().shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(close > upper).fillna(False).values] = 1
    out[(close < lower).fillna(False).values] = -1
    return out


def strat_atr_breakout(bars: pd.DataFrame, *, atr_period: int = 14,
                         multiplier: float = 2.0) -> np.ndarray:
    close = bars["close"].astype(float)
    atr = _atr(bars, atr_period).shift(1)
    ma = close.rolling(atr_period).mean().shift(1)
    upper = ma + multiplier * atr
    lower = ma - multiplier * atr
    out = np.zeros(len(bars), dtype=np.int8)
    out[(close.shift(1) > upper).fillna(False).values] = 1
    out[(close.shift(1) < lower).fillna(False).values] = -1
    return out


def strat_triple_ma_align(bars: pd.DataFrame, *,
                            fast: int = 5, mid: int = 20,
                            slow: int = 50) -> np.ndarray:
    close = bars["close"].astype(float)
    f = _sma(close, fast).shift(1)
    m = _sma(close, mid).shift(1)
    s = _sma(close, slow).shift(1)
    long_mask = (f > m) & (m > s)
    short_mask = (f < m) & (m < s)
    out = np.zeros(len(bars), dtype=np.int8)
    out[long_mask.fillna(False).values] = 1
    out[short_mask.fillna(False).values] = -1
    return out


def strat_linreg_slope(bars: pd.DataFrame, *, window: int = 20,
                        slope_threshold: float = 0.0) -> np.ndarray:
    """Sign of rolling least-squares slope on log-close."""
    close = np.log(bars["close"].astype(float))
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = (x_centered ** 2).sum()

    def slope(arr):
        if len(arr) < window:
            return np.nan
        y_centered = arr - arr.mean()
        return (x_centered * y_centered).sum() / denom

    slopes = close.rolling(window).apply(slope, raw=True).shift(1).values
    out = np.zeros(len(bars), dtype=np.int8)
    out[slopes > slope_threshold] = 1
    out[slopes < -slope_threshold] = -1
    return out


def strat_obv_momentum(bars: pd.DataFrame, *, window: int = 20) -> np.ndarray:
    """On-Balance Volume momentum: long when OBV's 20-day slope is up."""
    close = bars["close"].astype(float)
    vol = bars["volume"].astype(float)
    dir_ = np.sign(close.diff().fillna(0.0))
    obv = (dir_ * vol).cumsum()
    obv_ma = obv.rolling(window).mean().shift(1)
    obv_lag = obv.shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(obv_lag > obv_ma).fillna(False).values] = 1
    out[(obv_lag < obv_ma).fillna(False).values] = -1
    return out


def strat_vol_target_sma(bars: pd.DataFrame, *, sma_period: int = 200,
                          atr_period: int = 21,
                          atr_pct_max: float = 0.03) -> np.ndarray:
    """Long when above SMA200 AND realized ATR% is below atr_pct_max."""
    close = bars["close"].astype(float)
    sma = _sma(close, sma_period).shift(1)
    atr = _atr(bars, atr_period).shift(1)
    atr_pct = atr / close.shift(1)
    regime_on = (close.shift(1) > sma) & (atr_pct < atr_pct_max)
    out = np.zeros(len(bars), dtype=np.int8)
    out[regime_on.fillna(False).values] = 1
    return out


def strat_bollinger_with_rsi_filter(bars: pd.DataFrame, *,
                                      window: int = 20,
                                      entry_z: float = 2.0,
                                      rsi_period: int = 14,
                                      rsi_long_max: float = 50.0,
                                      rsi_short_min: float = 50.0) -> np.ndarray:
    """Bollinger MR, but require RSI confirmation (long only when RSI<50, etc.)."""
    base = strat_bollinger(bars, window=window, entry_z=entry_z)
    r = _rsi(bars["close"].astype(float), rsi_period).shift(1).values
    out = base.copy()
    out[(base == 1) & ~(r < rsi_long_max)] = 0
    out[(base == -1) & ~(r > rsi_short_min)] = 0
    return out


def strat_momentum_after_dip(bars: pd.DataFrame, *,
                                rsi_period: int = 14,
                                dip_rsi: float = 30.0,
                                recovery_bars: int = 3,
                                trend_sma: int = 50) -> np.ndarray:
    """Buy when RSI dipped below threshold within last N bars AND price is
    now above SMA(trend_sma). Exit when price drops below SMA."""
    close = bars["close"].astype(float)
    r = _rsi(close, rsi_period)
    sma = _sma(close, trend_sma).shift(1)
    dip_window = (r.shift(1) < dip_rsi).rolling(recovery_bars).max()
    long_mask = (close.shift(1) > sma) & (dip_window > 0)
    out = np.zeros(len(bars), dtype=np.int8)
    out[long_mask.fillna(False).values] = 1
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategySpec:
    name: str
    fn: Callable
    param_grid: dict   # dict of param-name → list of values


# ---------------------------------------------------------------------------
# Extended zoo additions (round 2)
# ---------------------------------------------------------------------------


def strat_kama(bars: pd.DataFrame, *, period: int = 10) -> np.ndarray:
    """Kaufman Adaptive Moving Average crossover."""
    close = bars["close"].astype(float)
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0)
    fast_alpha = 2.0 / (2 + 1)
    slow_alpha = 2.0 / (30 + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    kama = close.copy().astype(float)
    for i in range(1, len(close)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i - 1])
    out = np.zeros(len(bars), dtype=np.int8)
    kama_lag = kama.shift(1).values
    close_lag = close.shift(1).values
    out[close_lag > kama_lag] = 1
    out[close_lag < kama_lag] = -1
    return out


def strat_cci(bars: pd.DataFrame, *, period: int = 20,
                threshold: float = 100.0) -> np.ndarray:
    """Commodity Channel Index reversal signal."""
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    sma = tp.rolling(period).mean().shift(1)
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(),
                                    raw=True).shift(1)
    cci = (tp.shift(1) - sma) / (0.015 * md.replace(0, np.nan))
    out = np.zeros(len(bars), dtype=np.int8)
    out[(cci < -threshold).fillna(False).values] = 1
    out[(cci > threshold).fillna(False).values] = -1
    return out


def strat_dpo(bars: pd.DataFrame, *, period: int = 20) -> np.ndarray:
    """Detrended Price Oscillator — sign of detrended price."""
    close = bars["close"].astype(float)
    sma = close.rolling(period).mean().shift(period // 2 + 1)
    dpo = (close.shift(1) - sma)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(dpo > 0).fillna(False).values] = 1
    out[(dpo < 0).fillna(False).values] = -1
    return out


def strat_trix(bars: pd.DataFrame, *, period: int = 15) -> np.ndarray:
    """TRIX — rate-of-change of triple-EMA."""
    close = bars["close"].astype(float)
    ema1 = _ema(close, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    trix = ema3.pct_change().shift(1) * 1e4
    out = np.zeros(len(bars), dtype=np.int8)
    out[(trix > 0).fillna(False).values] = 1
    out[(trix < 0).fillna(False).values] = -1
    return out


def strat_aroon(bars: pd.DataFrame, *, period: int = 14) -> np.ndarray:
    """Aroon crossover: aroon_up - aroon_down."""
    h = bars["high"]
    l = bars["low"]
    up = h.rolling(period + 1).apply(lambda x: 100 * (period - x.argmax()) / period,
                                       raw=True).shift(1)
    dn = l.rolling(period + 1).apply(lambda x: 100 * (period - x.argmin()) / period,
                                       raw=True).shift(1)
    osc = up - dn
    out = np.zeros(len(bars), dtype=np.int8)
    out[(osc > 50).fillna(False).values] = 1
    out[(osc < -50).fillna(False).values] = -1
    return out


def strat_vortex(bars: pd.DataFrame, *, period: int = 14) -> np.ndarray:
    """Vortex Indicator — +VI vs -VI crossover."""
    h = bars["high"]
    l = bars["low"]
    c = bars["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    vm_pos = (h - l.shift(1)).abs()
    vm_neg = (l - h.shift(1)).abs()
    vi_pos = vm_pos.rolling(period).sum() / tr.rolling(period).sum()
    vi_neg = vm_neg.rolling(period).sum() / tr.rolling(period).sum()
    diff = (vi_pos - vi_neg).shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(diff > 0).fillna(False).values] = 1
    out[(diff < 0).fillna(False).values] = -1
    return out


def strat_chaikin_money_flow(bars: pd.DataFrame, *, period: int = 20) -> np.ndarray:
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = (mfm * v).rolling(period).sum()
    cmf = (mfv / v.rolling(period).sum()).shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(cmf > 0.05).fillna(False).values] = 1
    out[(cmf < -0.05).fillna(False).values] = -1
    return out


def strat_awesome_osc(bars: pd.DataFrame) -> np.ndarray:
    median = (bars["high"] + bars["low"]) / 2
    ao = (_sma(median, 5) - _sma(median, 34)).shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(ao > 0).fillna(False).values] = 1
    out[(ao < 0).fillna(False).values] = -1
    return out


def strat_bollinger_percentb(bars: pd.DataFrame, *,
                                window: int = 20, low: float = 0.0,
                                high: float = 1.0) -> np.ndarray:
    """Bollinger %B reversion: long when %B < low, short when > high."""
    close = bars["close"].astype(float)
    sma = close.rolling(window).mean().shift(1)
    sd = close.rolling(window).std().shift(1)
    upper = sma + 2 * sd
    lower = sma - 2 * sd
    pb = (close.shift(1) - lower) / (upper - lower).replace(0, np.nan)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(pb < low).fillna(False).values] = 1
    out[(pb > high).fillna(False).values] = -1
    return out


def strat_keltner_breakout(bars: pd.DataFrame, *, period: int = 20,
                              multiplier: float = 1.5) -> np.ndarray:
    close = bars["close"].astype(float)
    ema = _ema(close, period).shift(1)
    atr = _atr(bars, period).shift(1)
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    out = np.zeros(len(bars), dtype=np.int8)
    out[(close.shift(1) > upper).fillna(False).values] = 1
    out[(close.shift(1) < lower).fillna(False).values] = -1
    return out


def strat_ttm_squeeze(bars: pd.DataFrame, *, period: int = 20) -> np.ndarray:
    """TTM Squeeze: long when BB inside Keltner AND momentum positive."""
    close = bars["close"].astype(float)
    sma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    bb_upper = sma + 2 * sd
    bb_lower = sma - 2 * sd
    ema = _ema(close, period)
    atr = _atr(bars, period)
    kc_upper = ema + 1.5 * atr
    kc_lower = ema - 1.5 * atr
    squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).shift(1)
    momentum = (close - close.rolling(period).mean()).shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(squeeze.fillna(False) & (momentum > 0)).values] = 1
    out[(squeeze.fillna(False) & (momentum < 0)).values] = -1
    return out


def strat_ultimate_osc(bars: pd.DataFrame, *,
                          low: float = 30.0, high: float = 70.0) -> np.ndarray:
    h, l, c = bars["high"], bars["low"], bars["close"]
    bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([h, c.shift(1)], axis=1).max(axis=1) - pd.concat(
        [l, c.shift(1)], axis=1).min(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    uo = (100 * (4 * avg7 + 2 * avg14 + avg28) / 7).shift(1)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(uo < low).fillna(False).values] = 1
    out[(uo > high).fillna(False).values] = -1
    return out


def strat_fisher_transform(bars: pd.DataFrame, *, period: int = 10) -> np.ndarray:
    """Fisher Transform — converts price to Gaussian distribution for sharp signals."""
    median = (bars["high"] + bars["low"]) / 2
    min_l = median.rolling(period).min()
    max_h = median.rolling(period).max()
    raw = 2 * ((median - min_l) / (max_h - min_l).replace(0, np.nan) - 0.5)
    raw = raw.clip(-0.999, 0.999).fillna(0)
    fish = pd.Series(0.0, index=raw.index)
    for i in range(1, len(raw)):
        fish.iloc[i] = 0.5 * np.log((1 + raw.iloc[i]) / (1 - raw.iloc[i])) + 0.5 * fish.iloc[i - 1]
    fish_lag = fish.shift(1).values
    out = np.zeros(len(bars), dtype=np.int8)
    out[fish_lag > 0] = 1
    out[fish_lag < 0] = -1
    return out


def strat_zscore_ema(bars: pd.DataFrame, *, window: int = 20,
                       entry_z: float = 2.0) -> np.ndarray:
    """Z-score reversion using EMA instead of SMA (more responsive)."""
    close = bars["close"].astype(float)
    ema = _ema(close, window).shift(1)
    sd = close.rolling(window).std().shift(1)
    z = (close.shift(1) - ema) / sd.replace(0, np.nan)
    out = np.zeros(len(bars), dtype=np.int8)
    out[(z < -entry_z).fillna(False).values] = 1
    out[(z > entry_z).fillna(False).values] = -1
    return out


def strat_volume_breakout(bars: pd.DataFrame, *,
                             vol_window: int = 20,
                             vol_mult: float = 2.0,
                             price_window: int = 20) -> np.ndarray:
    """Long when volume spike AND price closes above price_window high."""
    close = bars["close"]
    vol = bars["volume"]
    high_n = bars["high"].rolling(price_window).max().shift(1)
    low_n = bars["low"].rolling(price_window).min().shift(1)
    avg_vol = vol.rolling(vol_window).mean().shift(1)
    spike = (vol.shift(1) > vol_mult * avg_vol)
    long_mask = spike & (close.shift(1) > high_n)
    short_mask = spike & (close.shift(1) < low_n)
    out = np.zeros(len(bars), dtype=np.int8)
    out[long_mask.fillna(False).values] = 1
    out[short_mask.fillna(False).values] = -1
    return out


def strat_relative_volume_filter(bars: pd.DataFrame, *,
                                    rsi_period: int = 2,
                                    rsi_low: float = 10.0,
                                    rsi_high: float = 90.0,
                                    rvol_min: float = 1.5,
                                    rvol_window: int = 50) -> np.ndarray:
    """RSI signal gated by relative volume."""
    base = strat_rsi2(bars, low=rsi_low, high=rsi_high)
    rvol = (bars["volume"] / bars["volume"].rolling(rvol_window).mean()).shift(1).fillna(1.0).values
    out = base.copy()
    out[rvol < rvol_min] = 0
    return out


def strat_double_bottom(bars: pd.DataFrame, *, lookback: int = 10,
                           tolerance_pct: float = 0.02) -> np.ndarray:
    """Detect approximate double-bottom: two local minima within tolerance."""
    low = bars["low"].astype(float)
    out = np.zeros(len(bars), dtype=np.int8)
    for i in range(lookback * 2 + 1, len(bars)):
        window = low.iloc[i - lookback * 2 - 1:i]
        if len(window) < lookback * 2:
            continue
        # Find two halves
        first = window.iloc[:lookback]
        second = window.iloc[lookback:]
        f_min = first.min()
        s_min = second.min()
        if f_min > 0 and abs(s_min - f_min) / f_min < tolerance_pct:
            if low.iloc[i - 1] > s_min * (1 + tolerance_pct):
                out[i] = 1
    return out


def strat_gap_fade(bars: pd.DataFrame, *, gap_pct: float = 0.01) -> np.ndarray:
    """Fade overnight gaps (open vs prior close)."""
    o = bars["open"]
    c_prev = bars["close"].shift(1)
    gap = (o - c_prev) / c_prev
    out = np.zeros(len(bars), dtype=np.int8)
    out[(gap < -gap_pct).fillna(False).values] = 1   # gap down → fade long
    out[(gap > gap_pct).fillna(False).values] = -1   # gap up → fade short
    return out


def strat_ema_ribbon(bars: pd.DataFrame) -> np.ndarray:
    """EMA ribbon: 6 EMAs aligned bullishly or bearishly."""
    close = bars["close"].astype(float)
    emas = [_ema(close, p).shift(1) for p in (5, 10, 15, 20, 30, 50)]
    long_mask = np.all([emas[i] > emas[i + 1] for i in range(5)], axis=0)
    short_mask = np.all([emas[i] < emas[i + 1] for i in range(5)], axis=0)
    out = np.zeros(len(bars), dtype=np.int8)
    out[long_mask] = 1
    out[short_mask] = -1
    return out


STRATEGY_ZOO: list[StrategySpec] = [
    StrategySpec("bollinger", strat_bollinger,
                  {"window": [10, 20, 30], "entry_z": [1.5, 2.0, 2.5]}),
    StrategySpec("rsi2", strat_rsi2,
                  {"low": [5, 10, 15], "high": [85, 90, 95]}),
    StrategySpec("rsi14", strat_rsi14,
                  {"low": [25, 30, 35], "high": [65, 70, 75]}),
    StrategySpec("zscore_reversion", strat_zscore_reversion,
                  {"window": [20, 30, 50], "entry_z": [2.0, 2.5, 3.0]}),
    StrategySpec("williams_r", strat_williams_r,
                  {"period": [10, 14, 21]}),
    StrategySpec("stochastic", strat_stochastic,
                  {"period": [10, 14, 21]}),
    StrategySpec("macd_hist", strat_macd_hist,
                  {"fast": [8, 12], "slow": [21, 26], "signal": [9]}),
    StrategySpec("donchian_breakout", strat_donchian_breakout,
                  {"window": [10, 20, 40, 55]}),
    StrategySpec("atr_breakout", strat_atr_breakout,
                  {"atr_period": [14, 21], "multiplier": [1.5, 2.0, 3.0]}),
    StrategySpec("triple_ma_align", strat_triple_ma_align,
                  {"fast": [5, 10], "mid": [20, 30], "slow": [50, 100]}),
    StrategySpec("linreg_slope", strat_linreg_slope,
                  {"window": [15, 30, 60]}),
    StrategySpec("obv_momentum", strat_obv_momentum,
                  {"window": [10, 20, 40]}),
    StrategySpec("vol_target_sma", strat_vol_target_sma,
                  {"sma_period": [100, 200], "atr_pct_max": [0.02, 0.03, 0.05]}),
    StrategySpec("bollinger_with_rsi", strat_bollinger_with_rsi_filter,
                  {"window": [20], "entry_z": [2.0, 2.5],
                   "rsi_long_max": [40, 50]}),
    StrategySpec("momentum_after_dip", strat_momentum_after_dip,
                  {"dip_rsi": [25, 30, 35], "recovery_bars": [2, 4, 6]}),
    # ---- round-2 expansion ----
    StrategySpec("kama", strat_kama,
                  {"period": [10, 20, 30]}),
    StrategySpec("cci", strat_cci,
                  {"period": [14, 20, 30], "threshold": [100.0, 150.0, 200.0]}),
    StrategySpec("dpo", strat_dpo,
                  {"period": [10, 20, 30]}),
    StrategySpec("trix", strat_trix,
                  {"period": [9, 15, 25]}),
    StrategySpec("aroon", strat_aroon,
                  {"period": [10, 14, 25]}),
    StrategySpec("vortex", strat_vortex,
                  {"period": [10, 14, 21]}),
    StrategySpec("chaikin_money_flow", strat_chaikin_money_flow,
                  {"period": [10, 20, 40]}),
    StrategySpec("awesome_osc", strat_awesome_osc, {}),
    StrategySpec("bollinger_percentb", strat_bollinger_percentb,
                  {"window": [20], "low": [0.0, 0.1], "high": [0.9, 1.0]}),
    StrategySpec("keltner_breakout", strat_keltner_breakout,
                  {"period": [14, 20], "multiplier": [1.0, 1.5, 2.0]}),
    StrategySpec("ttm_squeeze", strat_ttm_squeeze,
                  {"period": [14, 20]}),
    StrategySpec("ultimate_osc", strat_ultimate_osc,
                  {"low": [25.0, 30.0], "high": [70.0, 75.0]}),
    StrategySpec("fisher_transform", strat_fisher_transform,
                  {"period": [9, 13, 20]}),
    StrategySpec("zscore_ema", strat_zscore_ema,
                  {"window": [20, 30, 50], "entry_z": [2.0, 2.5]}),
    StrategySpec("volume_breakout", strat_volume_breakout,
                  {"vol_window": [20], "vol_mult": [1.5, 2.0, 3.0],
                   "price_window": [10, 20]}),
    StrategySpec("relative_volume_rsi", strat_relative_volume_filter,
                  {"rsi_low": [5, 10], "rvol_min": [1.5, 2.0]}),
    StrategySpec("double_bottom", strat_double_bottom,
                  {"lookback": [5, 10], "tolerance_pct": [0.01, 0.02]}),
    StrategySpec("gap_fade", strat_gap_fade,
                  {"gap_pct": [0.005, 0.01, 0.02]}),
    StrategySpec("ema_ribbon", strat_ema_ribbon, {}),
]


def iter_param_combinations(grid: dict) -> list[dict]:
    """Cartesian product over a param grid."""
    keys = list(grid.keys())
    if not keys:
        return [{}]
    out = [{}]
    for k in keys:
        new = []
        for combo in out:
            for v in grid[k]:
                merged = combo.copy()
                merged[k] = v
                new.append(merged)
        out = new
    return out


__all__ = ["StrategySpec", "STRATEGY_ZOO", "iter_param_combinations"]
