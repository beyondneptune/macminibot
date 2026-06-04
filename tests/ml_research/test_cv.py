"""Tests: WalkForwardHarness never overlaps train/test, respects embargo+purge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_research.cv import FoldSpec, WalkForwardHarness


def _trading_day_index(n_years: int) -> pd.DatetimeIndex:
    # ~252 trading days/year, just dense business days for the test.
    return pd.bdate_range(start="2000-01-03", periods=n_years * 252)


def test_harness_yields_at_least_one_fold():
    idx = _trading_day_index(7)
    h = WalkForwardHarness(train_min_years=5)
    folds = list(h.split(idx))
    assert len(folds) > 0


def test_train_and_test_never_overlap():
    idx = _trading_day_index(10)
    h = WalkForwardHarness(train_min_years=5, embargo_days=5, purge_days=1, test_days=21)
    for fold in h.split(idx):
        train = set(fold.train_idx.tolist())
        test = set(fold.test_idx.tolist())
        assert not (train & test), f"fold {fold.fold_id}: train/test overlap"


def test_embargo_and_purge_gap_respected():
    idx = _trading_day_index(10)
    embargo, purge, test_days = 5, 1, 21
    h = WalkForwardHarness(
        train_min_years=5, embargo_days=embargo, purge_days=purge, test_days=test_days
    )
    for fold in h.split(idx):
        max_train_idx = int(fold.train_idx.max())
        min_test_idx = int(fold.test_idx.min())
        gap = min_test_idx - max_train_idx - 1  # number of dropped days between
        assert gap == embargo + purge, f"fold {fold.fold_id}: gap={gap}, expected {embargo + purge}"


def test_val_is_subset_of_train():
    idx = _trading_day_index(8)
    h = WalkForwardHarness(train_min_years=5)
    for fold in h.split(idx):
        train = set(fold.train_idx.tolist())
        val = set(fold.val_idx.tolist())
        assert val.issubset(train)


def test_test_size_matches_test_days():
    idx = _trading_day_index(8)
    test_days = 21
    h = WalkForwardHarness(train_min_years=5, test_days=test_days)
    for fold in h.split(idx):
        assert len(fold.test_idx) == test_days


def test_walk_forward_monotonic_in_time():
    idx = _trading_day_index(10)
    h = WalkForwardHarness(train_min_years=5)
    prev_test_max = -1
    for fold in h.split(idx):
        assert int(fold.test_idx.min()) > prev_test_max
        prev_test_max = int(fold.test_idx.max())


def test_train_window_capped_at_max_years():
    idx = _trading_day_index(25)
    h = WalkForwardHarness(train_min_years=5, train_max_years=10)
    for fold in h.split(idx):
        assert len(fold.train_idx) <= 10 * 252


def test_foldspec_has_required_attrs():
    f = FoldSpec(
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([2]),
        test_idx=np.array([3]),
        fold_id=0,
        fold_meta={},
    )
    assert hasattr(f, "train_idx")
    assert hasattr(f, "val_idx")
    assert hasattr(f, "test_idx")
    assert hasattr(f, "fold_id")
    assert hasattr(f, "fold_meta")
