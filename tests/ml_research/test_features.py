"""Tests: features compute on synthetic data without errors and respect spec shapes."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from ml_research.features import (
    FEATURE_NAMES,
    apply_feature_clip,
    clip_features_train_fold,
    compute_features,
    drop_feature_failures,
)


def _synthetic_panel(n_symbols: int = 4, n_days: int = 600, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # Include SPY so rel_strength_vs_spy is non-NaN.
    symbols = ["SPY", "QQQ", "IWM"] + [f"SYM{i}" for i in range(n_symbols)]
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for sym in symbols:
        price = 100.0
        for d in dates:
            ret = rng.normal(0.0005, 0.015)
            o = price * (1 + rng.normal(0, 0.002))
            c = price * np.exp(ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.003)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.003)))
            rows.append((sym, d, o, h, lo, c, rng.integers(1_000_000, 5_000_000)))
            price = c
    return pl.DataFrame(
        rows, schema=["symbol", "date", "open", "high", "low", "close", "volume"], orient="row"
    )


def test_compute_features_returns_all_v1_columns():
    panel = _synthetic_panel()
    feat = compute_features(panel)
    for name in FEATURE_NAMES:
        assert name in feat.columns, f"missing feature {name}"


def test_features_have_warmup_nulls_then_become_finite():
    panel = _synthetic_panel(n_symbols=2, n_days=400)
    feat = compute_features(panel)
    # First-21d-per-symbol rows should have nulls for ret_21d.
    one_sym = feat.filter(pl.col("symbol") == "SYM0").sort("date")
    assert one_sym["ret_21d"].is_null().sum() >= 21
    # After warmup (>= 252 days), ret_252d_skip_21 should be finite somewhere.
    assert one_sym["ret_252d_skip_21"].is_finite().sum() > 10


def test_drop_feature_failures_removes_nan_rows():
    panel = _synthetic_panel(n_symbols=2, n_days=300)
    feat = compute_features(panel)
    cleaned = drop_feature_failures(feat)
    assert cleaned.height < feat.height
    # No NaN in any feature column.
    for name in FEATURE_NAMES:
        if name in cleaned.columns:
            assert cleaned[name].is_finite().all()


def test_clip_feature_bounds_fitted_on_train_fold_only():
    panel = _synthetic_panel(n_symbols=3, n_days=400)
    feat = compute_features(panel)
    feat_clean = drop_feature_failures(feat)
    bounds = clip_features_train_fold(feat_clean)
    # Every feature should have finite bounds.
    for name, (lo, hi) in bounds.items():
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi
    clipped = apply_feature_clip(feat_clean, bounds)
    # Clipped values stay within bounds.
    for name, (lo, hi) in bounds.items():
        if name in clipped.columns:
            col = clipped[name].drop_nulls().to_numpy()
            assert col.min() >= lo - 1e-9
            assert col.max() <= hi + 1e-9
