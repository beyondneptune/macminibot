"""Per-asset signal generators used by the deployment specialists.

These are the four signal families that won the trailing-stop bake-off
(per ``data/bakeoff_variants/FINAL_ANALYSIS.md``):

  - ``buy_and_hold``   — always +1 (used with a circuit-breaker trail).
  - ``rsi2_mr_10_90``  — Connors RSI(2) mean-reversion.
  - ``macd_signal``    — sign of the MACD histogram.
  - ``lgbm``           — LightGBM classifier on sign(next-day return).

Each generator takes a feature-enriched OHLCV bar frame (the output of
:func:`add_per_asset_features`) and returns the latest *causal* signal
in ``{-1, 0, +1}`` — i.e. the position to take at the upcoming open[T],
computed using only information dated ``<= T-1``.

The implementations mirror the bake-off candidate code in
``scripts/50_per_asset_bakeoff.py`` so the deploy path is byte-equivalent
to what the harmonised harness evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering — duplicated from the bake-off so the deploy runtime
# does not import a script-package directly. Keeping these in lockstep is a
# project-level invariant; the unit tests for the runner pin the equivalence.
# ---------------------------------------------------------------------------


def add_per_asset_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Causal per-asset features: returns, momentum, vol, range, technicals.

    Mirror of ``scripts/50_per_asset_bakeoff.py::add_per_asset_features``.
    All features use closed-bar info only; the consumer shifts by one bar
    where a t-1 alignment is required.
    """
    out = bars.copy()
    out["ret_1"] = np.log(out["close"] / out["close"].shift(1))
    out["ret_5"] = np.log(out["close"] / out["close"].shift(5))
    out["ret_21"] = np.log(out["close"] / out["close"].shift(21))
    out["ret_63"] = np.log(out["close"] / out["close"].shift(63))
    out["vol_21"] = out["ret_1"].rolling(21, min_periods=10).std(ddof=0)
    out["vol_63"] = out["ret_1"].rolling(63, min_periods=20).std(ddof=0)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - out["close"].shift(1)).abs()
    tr3 = (out["low"] - out["close"].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_norm"] = out["atr_14"] / out["close"]
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta.clip(upper=0)).rolling(2).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["rsi_2"] = 100.0 - 100.0 / (1.0 + rs)
    out["zscore_21"] = (
        (out["close"] - out["close"].rolling(21, min_periods=10).mean())
        / out["close"].rolling(21, min_periods=10).std(ddof=0)
    )
    ema_12 = out["close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_sig"]
    return out


ASSET_FEATURE_COLS: tuple[str, ...] = (
    "a_ret_1",
    "a_ret_5",
    "a_ret_21",
    "a_ret_63",
    "a_vol_21",
    "a_vol_63",
    "a_atr_norm",
    "a_rsi_2",
    "a_zscore_21",
    "a_macd_hist",
)


def asset_only_feature_matrix(bars_feat: pd.DataFrame) -> pd.DataFrame:
    """Build the asset_only ML feature matrix (lagged by 1 bar).

    Mirror of the ``asset_only`` branch of
    ``scripts/50_per_asset_bakeoff.py::feature_matrix``.
    """
    base_cols = [c[2:] for c in ASSET_FEATURE_COLS]  # strip "a_" prefix
    shifted = bars_feat[base_cols].shift(1).rename(columns=lambda c: f"a_{c}")
    return shifted.dropna()


# ---------------------------------------------------------------------------
# Rule-based signals
# ---------------------------------------------------------------------------


def _ensure_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return ``bars`` with feature columns present; idempotent."""
    if "macd_hist" in bars.columns and "rsi_2" in bars.columns and "vol_21" in bars.columns:
        return bars
    return add_per_asset_features(bars)


def _last_finite_int(series: pd.Series) -> int:
    """Return the most recent finite int-castable value; 0 if none."""
    s = series.dropna()
    if s.empty:
        return 0
    val = s.iloc[-1]
    if isinstance(val, float) and not math.isfinite(val):
        return 0
    return int(val)


def signal_buy_and_hold(bars: pd.DataFrame) -> int:
    """Always long. Mirror of ``_buy_and_hold`` in the bake-off."""
    if bars.empty:
        return 0
    return 1


def signal_rsi2_mr_10_90(
    bars: pd.DataFrame, *, low: float = 10.0, high: float = 90.0
) -> int:
    """RSI(2) mean-reversion: long < ``low``, short > ``high``.

    Mirror of ``_rsi2_mr`` in the bake-off — uses RSI(2) at T-1.
    """
    feat = _ensure_features(bars)
    if "rsi_2" not in feat.columns or len(feat) < 2:
        return 0
    rsi_prev = feat["rsi_2"].shift(1)
    sig = pd.Series(0, index=feat.index, dtype="int8")
    sig = sig.mask(rsi_prev < low, 1).mask(rsi_prev > high, -1)
    return _last_finite_int(sig)


def signal_macd(bars: pd.DataFrame) -> int:
    """Sign of MACD histogram (12/26/9), lagged by one bar.

    Mirror of ``_macd_signal`` in the bake-off.
    """
    feat = _ensure_features(bars)
    if "macd_hist" not in feat.columns or len(feat) < 2:
        return 0
    h = feat["macd_hist"].shift(1)
    sig = pd.Series(0, index=feat.index, dtype="int8")
    sig = sig.mask(h > 0, 1).mask(h < 0, -1)
    return _last_finite_int(sig)


# ---------------------------------------------------------------------------
# LightGBM signal — trained lazily, cached at module scope per asset.
# ---------------------------------------------------------------------------


@dataclass
class _LGBMSignalModel:
    """A trained LightGBM classifier and its feature column order.

    Wraps the in-process model so callers can re-use the cached instance
    across many daily wakes without retraining (we retrain on the auto-
    retrain cadence elsewhere; here we just need a deterministic latest
    signal).
    """

    model: object  # lightgbm.LGBMClassifier
    feature_cols: tuple[str, ...]
    training_window_start: pd.Timestamp
    training_window_end: pd.Timestamp
    n_rows: int

    def predict_latest_signal(self, bars: pd.DataFrame) -> int:
        feat = _ensure_features(bars)
        X = asset_only_feature_matrix(feat)
        if X.empty:
            return 0
        # Keep only the columns the model was trained on (defensive).
        X = X[list(self.feature_cols)]
        latest = X.iloc[[-1]]
        try:
            pred = self.model.predict(latest)
        except Exception:  # pragma: no cover — model would have to be corrupted
            return 0
        return int(pred[0] * 2 - 1)  # {0,1} -> {-1,+1}


_LGBM_TRAINING_YEARS = 8


def train_lgbm_signal(
    bars: pd.DataFrame,
    *,
    training_years: int = _LGBM_TRAINING_YEARS,
    random_state: int = 42,
) -> _LGBMSignalModel:
    """Train a LightGBM classifier on the latest N years of daily bars.

    Mirror of the bake-off's ``_ml_signal_lgbm`` setup. The label is
    ``sign(next-day log return)`` mapped to ``{0, 1}``. Returns a wrapper
    that exposes ``predict_latest_signal(bars)``.
    """
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as e:  # pragma: no cover — lightgbm is in pyproject
        raise ImportError("lightgbm required for LGBM signal training") from e

    feat = _ensure_features(bars)
    X_all = asset_only_feature_matrix(feat)
    # Next-day return is target (use the feature frame's own close).
    next_ret = np.log(feat["close"] / feat["close"].shift(1)).shift(-1)
    y_all = (next_ret > 0).astype(np.int8)
    aligned = X_all.join(y_all.rename("y"), how="inner").dropna()
    if training_years is not None and not aligned.empty:
        cutoff = aligned.index.max() - pd.DateOffset(years=training_years)
        aligned = aligned.loc[aligned.index >= cutoff]
    if aligned.empty or aligned["y"].nunique() < 2:
        raise RuntimeError(
            "LGBM training data is empty or single-class — cannot fit a classifier."
        )
    X_df = aligned.drop(columns=["y"])
    y = aligned["y"].values
    model = lgb.LGBMClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_data_in_leaf=20,
        verbose=-1,
        random_state=random_state,
    )
    model.fit(X_df, y)
    return _LGBMSignalModel(
        model=model,
        feature_cols=tuple(X_df.columns),
        training_window_start=aligned.index.min(),
        training_window_end=aligned.index.max(),
        n_rows=int(len(aligned)),
    )


# ---------------------------------------------------------------------------
# Dispatch table — what the per-asset modules import.
# ---------------------------------------------------------------------------


SignalFn = Callable[[pd.DataFrame], int]


__all__ = [
    "ASSET_FEATURE_COLS",
    "SignalFn",
    "add_per_asset_features",
    "asset_only_feature_matrix",
    "signal_buy_and_hold",
    "signal_macd",
    "signal_rsi2_mr_10_90",
    "train_lgbm_signal",
]
