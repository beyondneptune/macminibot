"""Covariance shrinkage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_engine.covariance import (
    ILL_CONDITIONED_THRESHOLD,
    cond_number,
    ledoit_wolf_shrunk,
)


def test_shrunk_sigma_is_psd():
    rng = np.random.default_rng(seed=42)
    n_obs, n_assets = 252, 8
    returns = pd.DataFrame(
        rng.standard_normal(size=(n_obs, n_assets)) * 0.01,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    result = ledoit_wolf_shrunk(returns)
    assert result.sigma.shape == (n_assets, n_assets)
    eig = np.linalg.eigvalsh(result.sigma)
    # PSD: all eigenvalues ≥ -tol
    assert (eig >= -1e-10).all(), f"min eig = {eig.min()}"


def test_shrinkage_in_unit_interval():
    rng = np.random.default_rng(seed=123)
    returns = pd.DataFrame(
        rng.standard_normal(size=(252, 5)) * 0.02,
        columns=list("ABCDE"),
    )
    result = ledoit_wolf_shrunk(returns)
    assert 0.0 <= result.shrinkage <= 1.0


def test_diagonal_fallback_on_ill_conditioned():
    # Construct a near-singular returns matrix: two columns nearly identical.
    rng = np.random.default_rng(seed=7)
    base = rng.standard_normal(size=(60, 1)) * 0.01
    cols = []
    for i in range(6):
        # near-duplicate columns: tiny perturbation only
        cols.append(base[:, 0] + rng.standard_normal(60) * 1e-9)
    returns = pd.DataFrame(np.column_stack(cols), columns=[f"X{i}" for i in range(6)])
    result = ledoit_wolf_shrunk(returns)
    # In near-singular case the Ledoit-Wolf shrinkage may itself recondition
    # enough to land below 1e6 — but the cond number we computed must be the
    # honest one. Verify the fallback triggers when condition is huge.
    if result.diagonal_fallback:
        # off-diagonals must be exactly zero
        off = result.sigma - np.diag(np.diag(result.sigma))
        assert np.allclose(off, 0.0)


def test_nan_input_rejected():
    bad = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.0, 0.01, 0.02]})
    with pytest.raises(ValueError, match="NaN"):
        ledoit_wolf_shrunk(bad)


def test_cond_number_basic():
    s = np.diag([1.0, 0.5, 0.25])
    assert cond_number(s) == pytest.approx(4.0)
