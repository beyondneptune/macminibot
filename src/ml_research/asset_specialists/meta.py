"""Bake-off winner for META: LightGBM classifier + 2.0σ trailing stop.

Source: ``data/bakeoff_variants/FINAL_ANALYSIS.md`` (harmonised trailing-stop
variant winner; Sharpe +1.285, DSR_harm 0.971, Holm-p 0.0017, CI_lo +0.93).

The LightGBM model is trained lazily on the latest 8 years of META daily
bars (the same window used by the bake-off candidate). The trained model
is cached at module scope so the daily 09:35 ET wake reuses it across
calls; the auto-retrain pipeline (``ml_research.auto_retrain``) is the
expected refresher.
"""

from __future__ import annotations

import threading
from typing import Optional

import pandas as pd

from ml_research.asset_specialists._signals import (
    _LGBMSignalModel,
    train_lgbm_signal,
)

WINNER = {
    "model": "lgbm",
    "scope": "asset_only",
    "trail_sigma": 2.0,
    "exit_kind": "trailing_stop",
    "oos_sharpe_net": 1.285,
    "dsr_harm": 0.971,
    "p_holm": 0.0017,
    "ci_lower": 0.93,
}


# Module-level cache so the model is trained once per process (or refreshed
# explicitly by the auto-retrain pipeline). The lock keeps cold-start
# concurrent callers from racing on a duplicate fit.
_MODEL: Optional[_LGBMSignalModel] = None
_MODEL_LOCK = threading.Lock()


def get_trained_model(bars: pd.DataFrame) -> _LGBMSignalModel:
    """Return the cached LGBM model, training it from ``bars`` on first call."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = train_lgbm_signal(bars, training_years=8)
    return _MODEL


def reset_model_cache() -> None:
    """Discard the cached model — used by tests and by auto-retrain."""
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in ``{-1, 0, +1}`` (LightGBM classifier)."""
    if bars is None or len(bars) < 100:
        # Need a reasonable history to compute features + predict.
        return 0.0
    model = get_trained_model(bars)
    return float(model.predict_latest_signal(bars))
