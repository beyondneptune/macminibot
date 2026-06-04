"""Mean-variance optimizer with hybrid L1+L2 turnover penalty.

CONTEXT.md "Optimizer" + "Solver selection" (Improvement #3):
    n ≤ 50          → cvxpy + ECOS  (interior-point, cleanest small-dense duality)
    50 < n ≤ 500    → cvxpy + OSQP  (ADMM, warm-startable)
    n > 500         → HRP            (no Σ inversion; López de Prado 2016)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

SolverTier = Literal["ecos", "osqp", "hrp"]


@dataclass(frozen=True)
class OptimizerResult:
    """Optimizer return envelope. ``w`` is in risk-units (dimensionless weights)."""

    w: np.ndarray
    solver_status: str
    solver_tier: SolverTier
    iterations: Optional[int]
    objective: Optional[float]


def _solver_tier_for(n: int) -> SolverTier:
    if n <= 50:
        return "ecos"
    if n <= 500:
        return "osqp"
    return "hrp"


def mean_variance_optimize(
    r_hat: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    *,
    lambda_risk: float = 5.0,
    gamma_turnover: float = 0.002,
    gamma_turnover_l2: Optional[float] = None,
    L_gross: float = 2.0,
    c_max: float = 0.05,
    net_cap: float = 1.0,
    warm_start: Optional[np.ndarray] = None,
) -> OptimizerResult:
    """Solve the ratified MV problem with hybrid L1+L2 turnover penalty.

    .. code-block:: text

        max  r' w - (λ/2) w' Σ w - γ_L1 ||w - w_prev||_1 - γ_L2 ||w - w_prev||_2^2
        s.t. ||w||_1 ≤ L_gross
             |w_i|   ≤ c_max
             sum(w) ∈ [-net_cap, +net_cap]

    ``gamma_turnover_l2`` defaults to ``0.1 * gamma_turnover`` per Improvement #1
    (DeMiguel et al. 2014; Olivares-Nadal & DeMiguel 2018).
    """
    r_hat = np.asarray(r_hat, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    w_prev = np.asarray(w_prev, dtype=np.float64)

    n = r_hat.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma shape {Sigma.shape} does not match r_hat length {n}")
    if w_prev.shape != (n,):
        raise ValueError(f"w_prev shape {w_prev.shape} does not match r_hat length {n}")

    if gamma_turnover_l2 is None:
        gamma_turnover_l2 = 0.1 * gamma_turnover

    tier = _solver_tier_for(n)

    if tier == "hrp":
        w = _hrp_allocate(Sigma)
        return OptimizerResult(
            w=w,
            solver_status="hrp_ok",
            solver_tier="hrp",
            iterations=None,
            objective=None,
        )

    # cvxpy path (ECOS or OSQP). Symmetrize Σ before handing to PSD wrap so a
    # micro-asymmetry from a Ledoit-Wolf round-trip doesn't blow up validation.
    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    w_var = cp.Variable(n)

    risk_term = 0.5 * lambda_risk * cp.quad_form(w_var, cp.psd_wrap(Sigma_sym))
    l1_turnover = gamma_turnover * cp.norm1(w_var - w_prev)
    l2_turnover = gamma_turnover_l2 * cp.sum_squares(w_var - w_prev)

    objective = cp.Maximize(r_hat @ w_var - risk_term - l1_turnover - l2_turnover)
    constraints = [
        cp.norm1(w_var) <= L_gross,
        w_var <= c_max,
        w_var >= -c_max,
        cp.sum(w_var) <= net_cap,
        cp.sum(w_var) >= -net_cap,
    ]
    prob = cp.Problem(objective, constraints)

    solver_kwargs: dict = {}
    if tier == "ecos":
        solver = cp.ECOS
        solver_kwargs = {"abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9}
    else:  # osqp
        solver = cp.OSQP
        solver_kwargs = {
            "eps_abs": 1e-7,
            "eps_rel": 1e-7,
            "max_iter": 50_000,
            "warm_start": warm_start is not None,
        }
        if warm_start is not None:
            w_var.value = np.asarray(warm_start, dtype=np.float64)

    prob.solve(solver=solver, **solver_kwargs)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Fallback to HRP for the day per CONTEXT.md.
        w_fb = _hrp_allocate(Sigma_sym)
        return OptimizerResult(
            w=w_fb,
            solver_status=f"fallback_hrp_from_{prob.status}",
            solver_tier="hrp",
            iterations=None,
            objective=None,
        )

    w_opt = np.asarray(w_var.value, dtype=np.float64).reshape(-1)
    iters = prob.solver_stats.num_iters if prob.solver_stats else None
    obj_val = float(prob.value) if prob.value is not None else None
    return OptimizerResult(
        w=w_opt,
        solver_status=prob.status,
        solver_tier=tier,
        iterations=iters,
        objective=obj_val,
    )


def _hrp_allocate(Sigma: np.ndarray) -> np.ndarray:
    """López de Prado 2016 HRP. Inverse-variance allocation inside each cluster.

    Returns a long-only weight vector summing to 1. The ratified scope uses
    HRP only as a fallback (n > 500 or solver failure) so we don't try to
    overlay the MV objective on top of it; risk-units interpretation is the
    same as the MV path.
    """
    n = Sigma.shape[0]
    if n == 1:
        return np.array([1.0])

    var = np.diag(Sigma).copy()
    var = np.where(var <= 0, 1e-12, var)
    std = np.sqrt(var)
    denom = np.outer(std, std)
    corr = np.clip(Sigma / denom, -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    # Squareform requires zero diagonal and exact symmetry.
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    order = _quasi_diag(link, n)
    w = np.ones(n)
    _recursive_bisect(w, var, order)
    # Reorder back to the original index ordering.
    out = np.zeros(n)
    for slot, original in enumerate(order):
        out[original] = w[slot]
    return out


def _quasi_diag(link: np.ndarray, n: int) -> list[int]:
    """Quasi-diagonalize the linkage matrix into a leaf ordering."""
    link = link.astype(int)
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    while max(sort_ix) >= n:
        new = []
        for idx in sort_ix:
            if idx < n:
                new.append(idx)
            else:
                row = link[idx - n]
                new.extend([int(row[0]), int(row[1])])
        sort_ix = new
    return sort_ix


def _recursive_bisect(w: np.ndarray, var: np.ndarray, order: list[int]) -> None:
    """In-place recursive bisection; assigns inverse-variance weights into ``w``."""
    clusters = [list(range(len(order)))]
    while clusters:
        next_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            v_left = _cluster_var(var, [order[i] for i in left])
            v_right = _cluster_var(var, [order[i] for i in right])
            alpha = 1.0 - v_left / (v_left + v_right)
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= 1.0 - alpha
            next_clusters.append(left)
            next_clusters.append(right)
        clusters = next_clusters


def _cluster_var(var: np.ndarray, idx: list[int]) -> float:
    """Inverse-variance portfolio variance for a cluster (diagonal approximation)."""
    v = var[idx]
    inv = 1.0 / v
    w = inv / inv.sum()
    return float((w**2 * v).sum())


def hrp_allocate(Sigma: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Public wrapper for HRP — used by callers that explicitly want HRP."""
    if isinstance(Sigma, pd.DataFrame):
        Sigma = Sigma.to_numpy(dtype=np.float64)
    return _hrp_allocate(np.asarray(Sigma, dtype=np.float64))


__all__ = [
    "OptimizerResult",
    "SolverTier",
    "hrp_allocate",
    "mean_variance_optimize",
]
