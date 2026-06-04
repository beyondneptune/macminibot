"""Tests: LightGBM + Ridge train and predict without error on synthetic data."""

from __future__ import annotations

import numpy as np

from ml_research.trainer import (
    LightGBMConfig,
    LightGBMTrainer,
    RidgeConfig,
    RidgeTrainer,
    evaluate_fold,
)


def _synthetic_regression(n: int = 1000, d: int = 9, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # Build a linear-ish signal so both Ridge and LGBM have something to learn.
    w = rng.normal(size=d) * 0.3
    y = X @ w + rng.normal(scale=0.5, size=n)
    return X, y


def test_lightgbm_trainer_fits_and_predicts():
    X, y = _synthetic_regression()
    split = int(0.8 * len(y))
    model = LightGBMTrainer(config=LightGBMConfig(n_estimators=50, early_stopping_rounds=None))
    model.fit(X[:split], y[:split])
    pred = model.predict(X[split:])
    assert pred.shape == y[split:].shape
    assert np.all(np.isfinite(pred))


def test_lightgbm_trainer_with_validation_set_runs():
    X, y = _synthetic_regression()
    s1, s2 = 600, 800
    model = LightGBMTrainer(config=LightGBMConfig(n_estimators=50, early_stopping_rounds=10))
    model.fit(X[:s1], y[:s1], X_val=X[s1:s2], y_val=y[s1:s2])
    pred = model.predict(X[s2:])
    assert np.all(np.isfinite(pred))


def test_ridge_trainer_fits_and_predicts():
    X, y = _synthetic_regression()
    split = int(0.8 * len(y))
    model = RidgeTrainer(config=RidgeConfig(alpha=1.0))
    model.fit(X[:split], y[:split])
    pred = model.predict(X[split:])
    assert pred.shape == y[split:].shape
    assert np.all(np.isfinite(pred))


def test_evaluate_fold_returns_expected_keys():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.normal(size=n)
    pred = y * 0.5 + rng.normal(scale=0.5, size=n)
    dates = np.tile(np.arange(50), 10)
    m = evaluate_fold(y, pred, dates)
    for k in ("ic", "rank_ic", "hit_rate", "sharpe_daily", "n"):
        assert k in m
    assert m["n"] == n
    # Pearson IC of a positively-correlated signal should be > 0.
    assert m["ic"] > 0


def test_evaluate_fold_handles_all_nan_gracefully():
    y = np.full(10, np.nan)
    pred = np.full(10, np.nan)
    dates = np.arange(10)
    m = evaluate_fold(y, pred, dates)
    assert m["n"] == 0
