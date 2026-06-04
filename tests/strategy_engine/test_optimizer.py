"""Optimizer tests: closed-form sanity + turnover penalty behavior."""

from __future__ import annotations

import numpy as np
import pytest

from strategy_engine.optimizer import mean_variance_optimize


def test_two_asset_closed_form_matches_cvxpy():
    """2-asset MV with no turnover penalty + slack constraints reduces to:
        w* = (1/λ) · Σ^{-1} · r_hat

    Constraints are picked loose enough not to bind; verify cvxpy reproduces
    the analytical solution within solver tolerance.
    """
    # Use an annualized-style Σ so weights land in a sane numerical range.
    Sigma = np.array([[0.04, 0.01], [0.01, 0.06]])  # daily-equivalent, well scaled
    r_hat = np.array([0.005, 0.003])
    lambda_risk = 5.0

    analytical = (1.0 / lambda_risk) * np.linalg.solve(Sigma, r_hat)
    # Sanity: ||analytical||_1 should be small enough not to bind L_gross=2.0.
    assert float(np.sum(np.abs(analytical))) < 1.0

    res = mean_variance_optimize(
        r_hat,
        Sigma,
        w_prev=np.zeros(2),
        lambda_risk=lambda_risk,
        gamma_turnover=0.0,
        gamma_turnover_l2=0.0,
        L_gross=2.0,
        c_max=1.0,
        net_cap=2.0,
    )
    assert res.solver_tier == "ecos"
    np.testing.assert_allclose(res.w, analytical, atol=1e-5)


def test_turnover_penalty_reduces_churn():
    """Increasing gamma_turnover from 0 → 0.1 must reduce ||w - w_prev||_1."""
    rng = np.random.default_rng(seed=11)
    n = 8
    A = rng.standard_normal(size=(n, n)) * 0.01
    Sigma = A @ A.T + np.eye(n) * 1e-4
    r_hat = rng.standard_normal(n) * 0.001
    w_prev = rng.standard_normal(n) * 0.02
    # Project w_prev into feasible region (per-name cap)
    w_prev = np.clip(w_prev, -0.04, 0.04)

    res_lo = mean_variance_optimize(
        r_hat, Sigma, w_prev,
        lambda_risk=5.0,
        gamma_turnover=0.0,
        gamma_turnover_l2=0.0,
        L_gross=2.0, c_max=0.05,
    )
    res_hi = mean_variance_optimize(
        r_hat, Sigma, w_prev,
        lambda_risk=5.0,
        gamma_turnover=0.1,
        gamma_turnover_l2=0.01,
        L_gross=2.0, c_max=0.05,
    )
    churn_lo = float(np.sum(np.abs(res_lo.w - w_prev)))
    churn_hi = float(np.sum(np.abs(res_hi.w - w_prev)))
    assert churn_hi < churn_lo, f"expected high-γ churn ({churn_hi}) < low-γ churn ({churn_lo})"


def test_constraints_respected():
    """Per-name cap and gross cap are honored."""
    rng = np.random.default_rng(seed=99)
    n = 10
    A = rng.standard_normal(size=(n, n)) * 0.01
    Sigma = A @ A.T + np.eye(n) * 1e-4
    r_hat = rng.standard_normal(n) * 0.005
    w_prev = np.zeros(n)

    res = mean_variance_optimize(
        r_hat, Sigma, w_prev,
        lambda_risk=1.0,  # low → wants big positions, constraints must bind
        gamma_turnover=0.001,
        L_gross=1.5, c_max=0.05,
    )
    assert np.all(np.abs(res.w) <= 0.05 + 1e-4)
    assert float(np.sum(np.abs(res.w))) <= 1.5 + 1e-4
