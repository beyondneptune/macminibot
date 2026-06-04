"""Ledoit-Wolf shrinkage covariance with conditioning fallback.

Per CONTEXT.md "Optimizer" key decision:
    Σ_shrunk = (1 - δ) · Σ_sample + δ · F

Plus the edge-case rule book entry:
    Σ ill-conditioned (cond(Σ_shrunk) > 1e6) → fall back to diagonal Σ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ILL_CONDITIONED_THRESHOLD: float = 1e6


@dataclass(frozen=True)
class ShrinkageResult:
    """Σ plus the diagnostics that go to observability."""

    sigma: np.ndarray
    shrinkage: float
    cond_number: float
    diagonal_fallback: bool


def ledoit_wolf_shrunk(returns: pd.DataFrame) -> ShrinkageResult:
    """Shrink the sample covariance of a returns frame toward a structured target.

    Parameters
    ----------
    returns:
        Wide DataFrame, one column per name, one row per session. Index is
        ignored. Float values, NaNs not permitted (caller drops stale-bar
        names per the edge-case rule book before calling us).

    Returns
    -------
    ShrinkageResult
        Always returns a PSD matrix. If the shrunk Σ has ``cond > 1e6`` we
        substitute ``diag(var)`` and flag ``diagonal_fallback=True`` so
        observability can alert.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a DataFrame (got {type(returns).__name__})")
    if returns.isna().any().any():
        raise ValueError("returns contains NaN; drop stale-bar names before shrinkage")
    if returns.shape[0] < 2:
        raise ValueError("need at least 2 observations for covariance")

    arr = returns.to_numpy(dtype=np.float64)
    estimator = LedoitWolf().fit(arr)
    sigma = np.asarray(estimator.covariance_, dtype=np.float64)
    shrinkage = float(estimator.shrinkage_)

    # Force exact symmetry (sklearn occasionally returns 1e-18 asymmetry).
    sigma = 0.5 * (sigma + sigma.T)

    cond = cond_number(sigma)
    if not np.isfinite(cond) or cond > ILL_CONDITIONED_THRESHOLD:
        diag = np.diag(np.diag(sigma))
        # Floor variance so all-zero columns don't blow up downstream solvers.
        diag_vals = np.diag(diag).copy()
        floor = max(1e-12, float(np.median(diag_vals[diag_vals > 0]) * 1e-6)) if (diag_vals > 0).any() else 1e-12
        diag_vals = np.where(diag_vals > 0, diag_vals, floor)
        sigma_fb = np.diag(diag_vals)
        return ShrinkageResult(
            sigma=sigma_fb,
            shrinkage=shrinkage,
            cond_number=cond,
            diagonal_fallback=bool(True),
        )

    return ShrinkageResult(
        sigma=sigma,
        shrinkage=shrinkage,
        cond_number=cond,
        diagonal_fallback=bool(False),
    )


def cond_number(sigma: np.ndarray) -> float:
    """Spectral condition number (largest / smallest eigenvalue, absolute)."""
    eig = np.linalg.eigvalsh(sigma)
    eig = np.abs(eig)
    eig_min = float(eig.min())
    eig_max = float(eig.max())
    if eig_min <= 0:
        return float("inf")
    return eig_max / eig_min


__all__ = [
    "ILL_CONDITIONED_THRESHOLD",
    "ShrinkageResult",
    "cond_number",
    "ledoit_wolf_shrunk",
]
