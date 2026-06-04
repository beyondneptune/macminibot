"""ml_research — walk-forward, dollar-bar-fed, gradient-boosted return predictor.

Public surface (training + inference parity):
    features.compute_features              — pure functions, daily bars
    labels.next_day_open_close_log_return  — continuous regression target
    labels.winsorize_train_fold            — train-only ±5σ winsorization
    labels.cross_sectional_zscore          — daily xs z-score
    cv.WalkForwardHarness                  — canonical splitter (binding)
    cv.FoldSpec                            — fold metadata (mirrors shared-core)
    trainer.LightGBMTrainer                — primary baseline
    trainer.RidgeTrainer                   — sanity-check floor
    inference.predict                      — stateless adapter

See `.octogent/tentacles/ml-research/CONTEXT.md` for the ratified spec.
"""

from ml_research.cv import FoldSpec, WalkForwardHarness
from ml_research.features import FEATURE_NAMES, compute_features
from ml_research.inference import Signal, predict
from ml_research.labels import (
    cross_sectional_zscore,
    next_day_open_close_log_return,
    winsorize_train_fold,
)
from ml_research.trainer import LightGBMTrainer, RidgeTrainer

__all__ = [
    "FEATURE_NAMES",
    "FoldSpec",
    "LightGBMTrainer",
    "RidgeTrainer",
    "Signal",
    "WalkForwardHarness",
    "compute_features",
    "cross_sectional_zscore",
    "next_day_open_close_log_return",
    "predict",
    "winsorize_train_fold",
]
