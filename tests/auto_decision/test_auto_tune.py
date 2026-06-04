"""Unit tests for ``strategy_engine.auto_tune``.

These tests use synthetic, well-conditioned inputs so the optimizer is
deterministic — no I/O, no ArcticDB, no Alpaca.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from strategy_engine.auto_tune import (
    AutoTuneResult,
    DEFAULT_LINF_TARGET,
    LAMBDA_RISK_BOUNDS,
    TARGET_VOL_BOUNDS,
    _clamp_lambda,
    _clamp_target_vol,
    derive_target_vol,
    load_latest_tuning,
    persist_tuning_record,
    run_auto_tune,
    solve_lambda_for_linf,
)


@pytest.fixture
def synth_inputs():
    """8-name flat-book signal with ±2% predicted log returns, diagonal-ish Σ."""
    rng = np.random.default_rng(42)
    n = 8
    r_hat = np.array([+0.020, +0.015, +0.010, +0.005,
                      -0.005, -0.010, -0.015, -0.020])
    # Daily Σ with ~2% per-name vol and modest correlation.
    base = np.eye(n) * (0.02 ** 2)
    noise = rng.standard_normal((n, n)) * 1e-5
    Sigma = base + 0.5 * (noise + noise.T)
    # Project to PSD (clip negative eigenvalues).
    eigvals, eigvecs = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    Sigma = eigvecs @ np.diag(np.clip(eigvals, 1e-10, None)) @ eigvecs.T
    w_prev = np.zeros(n)
    return r_hat, Sigma, w_prev


def test_solve_lambda_for_linf_hits_target(synth_inputs):
    """Bisection should either land within tolerance of the target OR cleanly
    saturate one of the binding constraints (c_max cap, signal-bound).
    """
    r_hat, Sigma, w_prev = synth_inputs
    linf_target = 0.04  # below c_max=0.05 so the bisection can actually reach it
    lam, achieved, w_star, n_iter = solve_lambda_for_linf(
        r_hat=r_hat, Sigma=Sigma, w_prev=w_prev,
        linf_target=linf_target,
        tol_rel=0.10,
    )
    assert LAMBDA_RISK_BOUNDS[0] <= lam <= LAMBDA_RISK_BOUNDS[1]
    assert w_star.shape == r_hat.shape
    assert n_iter >= 0
    # On a moderate-magnitude signal we should land within ~20% of the target.
    assert abs(achieved - linf_target) / linf_target < 0.30, (
        f"achieved {achieved} far from target {linf_target}; lam={lam}"
    )


def test_solve_lambda_signal_bound_returns_lambda_lo(synth_inputs):
    """When the target is so high that even lambda_lo can't reach it (c_max
    binds), we return lambda_lo and the achieved linf saturates at c_max."""
    r_hat, Sigma, w_prev = synth_inputs
    linf_target = 0.10  # above c_max=0.05 — unreachable
    lam, achieved, w_star, n_iter = solve_lambda_for_linf(
        r_hat=r_hat, Sigma=Sigma, w_prev=w_prev,
        linf_target=linf_target,
    )
    # Cap binds → linf saturates near c_max=0.05.
    assert achieved <= 0.05 + 1e-6
    # Bisection short-circuits to the endpoint, n_iter == 0.
    assert n_iter == 0
    assert np.isclose(lam, LAMBDA_RISK_BOUNDS[0])


def test_solve_lambda_monotone(synth_inputs):
    """||w*||_inf must be (weakly) decreasing in lambda_risk."""
    from strategy_engine.auto_tune import _linf_for_lambda

    r_hat, Sigma, w_prev = synth_inputs
    lambdas = [0.01, 0.1, 1.0, 10.0, 50.0]
    linfs = [
        _linf_for_lambda(
            r_hat=r_hat, Sigma=Sigma, w_prev=w_prev,
            lambda_risk=lam, gamma_turnover=0.002, L_gross=2.0, c_max=0.05,
        )[0]
        for lam in lambdas
    ]
    # Allow a tiny non-monotone wiggle from solver numerics (1e-4).
    for a, b in zip(linfs, linfs[1:]):
        assert b <= a + 1e-4, f"non-monotone: linfs={linfs}"


def test_clamp_lambda_floor(caplog):
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("auto_tune_test")
    val, was_clamped = _clamp_lambda(1e-6, logger)
    assert was_clamped is True
    assert val == LAMBDA_RISK_BOUNDS[0]


def test_clamp_lambda_ceiling(caplog):
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("auto_tune_test")
    val, was_clamped = _clamp_lambda(1e6, logger)
    assert was_clamped is True
    assert val == LAMBDA_RISK_BOUNDS[1]


def test_clamp_target_vol_floor():
    logger = logging.getLogger("auto_tune_test")
    val, was_clamped = _clamp_target_vol(1e-5, logger)
    assert was_clamped is True
    assert val == TARGET_VOL_BOUNDS[0]


def test_clamp_target_vol_ceiling():
    logger = logging.getLogger("auto_tune_test")
    val, was_clamped = _clamp_target_vol(5.0, logger)
    assert was_clamped is True
    assert val == TARGET_VOL_BOUNDS[1]


def test_derive_target_vol_degenerate_falls_back():
    """Zero-weight portfolio → falls back to the default vol target."""
    w = np.zeros(8)
    Sigma = np.eye(8) * 1e-4
    tv, realized = derive_target_vol(w_star=w, Sigma_daily=Sigma)
    assert realized == 0.0
    assert tv > 0.0  # Returns the default, not zero.


def test_run_auto_tune_end_to_end(synth_inputs):
    r_hat, Sigma, w_prev = synth_inputs
    result = run_auto_tune(
        r_hat=r_hat,
        Sigma_daily=Sigma,
        w_prev=w_prev,
        as_of_date=date(2026, 5, 18),
    )
    assert isinstance(result, AutoTuneResult)
    assert LAMBDA_RISK_BOUNDS[0] <= result.lambda_risk <= LAMBDA_RISK_BOUNDS[1]
    assert TARGET_VOL_BOUNDS[0] <= result.target_vol_annualized <= TARGET_VOL_BOUNDS[1]
    assert result.as_of_date == date(2026, 5, 18)


def test_persist_and_load_roundtrip(synth_inputs, tmp_path: Path):
    r_hat, Sigma, w_prev = synth_inputs
    result = run_auto_tune(
        r_hat=r_hat, Sigma_daily=Sigma, w_prev=w_prev,
        as_of_date=date(2026, 5, 17),
    )
    persist_tuning_record(result, project_root=tmp_path)

    cfg_path = tmp_path / "data" / "config" / "strategy_tuning.json"
    assert cfg_path.exists()
    payload = json.loads(cfg_path.read_text())
    assert "2026-05-17" in payload
    assert payload["2026-05-17"]["lambda_risk"] == pytest.approx(result.lambda_risk)

    # Second write on the next day → both keys present, latest is the new one.
    result2 = run_auto_tune(
        r_hat=r_hat, Sigma_daily=Sigma, w_prev=w_prev,
        as_of_date=date(2026, 5, 18),
    )
    persist_tuning_record(result2, project_root=tmp_path)
    payload = json.loads(cfg_path.read_text())
    assert {"2026-05-17", "2026-05-18"}.issubset(payload.keys())

    latest = load_latest_tuning(project_root=tmp_path)
    assert latest is not None
    assert latest["as_of_date"] == "2026-05-18"
    assert latest["lambda_risk"] == pytest.approx(result2.lambda_risk)


def test_load_latest_tuning_missing_returns_none(tmp_path: Path):
    assert load_latest_tuning(project_root=tmp_path) is None


def test_persist_overwrites_same_day(synth_inputs, tmp_path: Path):
    """Two writes on the same calendar day → second one wins."""
    r_hat, Sigma, w_prev = synth_inputs
    r1 = run_auto_tune(
        r_hat=r_hat, Sigma_daily=Sigma, w_prev=w_prev,
        as_of_date=date(2026, 5, 18),
    )
    persist_tuning_record(r1, project_root=tmp_path)

    # Run again — same date.
    r2 = run_auto_tune(
        r_hat=r_hat * 2.0,  # different inputs → different lambda
        Sigma_daily=Sigma, w_prev=w_prev,
        as_of_date=date(2026, 5, 18),
    )
    persist_tuning_record(r2, project_root=tmp_path)
    payload = json.loads((tmp_path / "data" / "config" / "strategy_tuning.json").read_text())
    assert list(payload.keys()) == ["2026-05-18"]
    assert payload["2026-05-18"]["lambda_risk"] == pytest.approx(r2.lambda_risk)


def test_solver_returns_within_hard_bounds_for_extreme_signal():
    """Stress: huge r_hat must still produce lambda within hard bounds (no NaNs)."""
    n = 8
    r_hat = np.ones(n) * 0.5  # 50% predicted return — wildly out of range
    Sigma = np.eye(n) * (0.05 ** 2)
    w_prev = np.zeros(n)
    result = run_auto_tune(
        r_hat=r_hat, Sigma_daily=Sigma, w_prev=w_prev,
        as_of_date=date(2026, 5, 18),
    )
    assert LAMBDA_RISK_BOUNDS[0] <= result.lambda_risk <= LAMBDA_RISK_BOUNDS[1]
    assert TARGET_VOL_BOUNDS[0] <= result.target_vol_annualized <= TARGET_VOL_BOUNDS[1]


def test_default_linf_target_is_sane():
    assert 0.01 < DEFAULT_LINF_TARGET < 0.20
