"""Tests for performance metrics: Sharpe / Sortino / Calmar / MaxDD /
PSR / DSR / PBO."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtesting.metrics import (
    BootstrapSharpeResult,
    bootstrap_sharpe_ci,
    calmar,
    dsr,
    dsr_expected_max_sr,
    max_drawdown,
    pbo,
    politis_white_block_length,
    psr,
    sharpe,
    sortino,
)


# ---- sharpe ---------------------------------------------------------------


def test_sharpe_constant_positive_is_inf() -> None:
    assert sharpe([0.01, 0.01, 0.01]) == math.inf


def test_sharpe_constant_negative_is_neg_inf() -> None:
    assert sharpe([-0.01, -0.01, -0.01]) == -math.inf


def test_sharpe_alternating_known_value() -> None:
    # returns = [0.01, -0.01, 0.01]
    # mean = 0.01/3, std (ddof=1) = sqrt( ( (0.01 - 1/300)^2*2 + (-0.01 - 1/300)^2 ) / 2 )
    r = np.array([0.01, -0.01, 0.01])
    expected = float(r.mean()) / float(r.std(ddof=1)) * math.sqrt(252)
    got = sharpe(r)
    assert got == pytest.approx(expected, rel=1e-9)


def test_sharpe_all_zero_is_zero() -> None:
    assert sharpe([0.0, 0.0, 0.0]) == 0.0


def test_sharpe_short_series_is_nan() -> None:
    assert math.isnan(sharpe([0.01]))


# ---- sortino --------------------------------------------------------------


def test_sortino_no_downside_is_inf() -> None:
    assert sortino([0.01, 0.02, 0.005]) == math.inf


def test_sortino_with_downside_finite_positive() -> None:
    val = sortino([0.01, -0.005, 0.02, -0.001, 0.015])
    assert math.isfinite(val) and val > 0


# ---- max_drawdown ---------------------------------------------------------


def test_max_drawdown_monotonic_zero() -> None:
    assert max_drawdown([1.0, 1.1, 1.2, 1.3]) == 0.0


def test_max_drawdown_known_30_percent() -> None:
    # peak at 1.5, trough at 1.05 → DD = (1.05 - 1.5) / 1.5 = -0.30
    assert max_drawdown([1.0, 1.5, 1.05, 1.2]) == pytest.approx(-0.30)


def test_max_drawdown_empty_zero() -> None:
    assert max_drawdown([]) == 0.0


# ---- calmar ---------------------------------------------------------------


def test_calmar_no_drawdown_is_nan() -> None:
    assert math.isnan(calmar([0.001] * 252))


def test_calmar_finite_for_realistic_returns() -> None:
    rng = np.random.default_rng(0xB10CC0DE)
    r = rng.normal(0.0005, 0.01, size=252)
    val = calmar(r)
    assert math.isfinite(val) or math.isnan(val)


# ---- PSR ------------------------------------------------------------------


def test_psr_in_unit_interval() -> None:
    rng = np.random.default_rng(42)
    r = rng.normal(0.0005, 0.01, size=252)
    val = psr(r, benchmark_sr=0.0)
    assert 0.0 <= val <= 1.0


def test_psr_high_for_strong_signal() -> None:
    # Constant positive returns + tiny noise → very high PSR.
    rng = np.random.default_rng(1)
    r = 0.005 + rng.normal(0, 1e-6, size=252)
    val = psr(r, benchmark_sr=0.0)
    assert val > 0.99


# ---- DSR ------------------------------------------------------------------


def test_dsr_equals_psr_when_n_trials_1() -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(0.001, 0.01, size=300)
    sr = (r.mean() / r.std(ddof=1)) * math.sqrt(252)
    assert dsr(sr, n_trials=1, returns=r) == pytest.approx(psr(r, 0.0))


def test_dsr_less_than_psr_for_n_trials_many() -> None:
    rng = np.random.default_rng(11)
    r = rng.normal(0.0008, 0.01, size=300)
    sr = (r.mean() / r.std(ddof=1)) * math.sqrt(252)
    p = psr(r, benchmark_sr=0.0)
    d = dsr(sr, n_trials=100, returns=r)
    # Deflation MUST not increase the probability.
    assert d <= p


# ---- PBO ------------------------------------------------------------------


def test_pbo_perfect_persistence_is_zero() -> None:
    # IS-best is also OOS-best → PBO = 0.
    is_sr = [0.5, 1.0, 1.5, 2.0]
    oos_sr = [0.4, 0.9, 1.4, 1.9]
    val = pbo(is_sr, oos_sr)
    assert val == pytest.approx(0.0)


def test_pbo_complete_inversion_is_high() -> None:
    is_sr = [0.5, 1.0, 1.5, 2.0]
    oos_sr = [2.0, 1.5, 1.0, 0.5]  # best-IS is worst-OOS
    val = pbo(is_sr, oos_sr)
    assert val == pytest.approx(0.75)  # 1 - (1/4)


def test_pbo_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        pbo([1, 2, 3], [1, 2])


# ---- DSR expected-max benchmark (calibration helper) ---------------------


def test_dsr_expected_max_sr_n_eq_1_is_zero() -> None:
    """Single trial → no deflation, benchmark collapses to 0."""
    assert dsr_expected_max_sr(n_trials=1) == 0.0


def test_dsr_expected_max_sr_monotone_in_n_trials() -> None:
    """E[max SR] strictly increases as more trials are searched."""
    vals = [dsr_expected_max_sr(n_trials=n) for n in (2, 10, 100, 1000)]
    for a, b in zip(vals, vals[1:]):
        assert a < b


def test_dsr_expected_max_sr_scales_linearly_with_sigma() -> None:
    """Benchmark is sigma * (z-mixture) — linear in sr_trials_std."""
    base = dsr_expected_max_sr(n_trials=50, sr_trials_std=1.0)
    scaled = dsr_expected_max_sr(n_trials=50, sr_trials_std=0.25)
    assert scaled == pytest.approx(base * 0.25, rel=1e-12)


def test_dsr_expected_max_sr_matches_dsr_default() -> None:
    """The default dsr() path uses sigma=1.0 — re-deriving via psr() must
    give the same number, confirming the refactor preserves behaviour."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0008, 0.01, size=400)
    sr = (r.mean() / r.std(ddof=1)) * math.sqrt(252)
    bench = dsr_expected_max_sr(n_trials=20, sr_trials_std=1.0)
    via_psr = psr(r, benchmark_sr=bench)
    via_dsr = dsr(sr, n_trials=20, returns=r)
    assert via_psr == pytest.approx(via_dsr, rel=1e-12)


def test_dsr_expected_max_sr_tightens_with_low_fold_variance() -> None:
    """Variance-calibrated gate: when observed cross-fold Sharpe std is
    small (the strategy is stable), the deflation benchmark is *lower*
    than the generic sigma=1 default — i.e. the DSR=0.95 hurdle is in
    fact looser than the CONTEXT's generic gate would imply. This is the
    cycle-6 calibration improvement the todo flagged."""
    generic = dsr_expected_max_sr(n_trials=18, sr_trials_std=1.0)
    calibrated = dsr_expected_max_sr(n_trials=18, sr_trials_std=0.4)
    assert calibrated < generic


def test_dsr_expected_max_sr_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="sr_trials_std"):
        dsr_expected_max_sr(n_trials=5, sr_trials_std=-0.1)


# ---- Politis-White block length ------------------------------------------


def test_politis_white_iid_returns_short_block() -> None:
    """IID Gaussian → block-length should be very small (≈1–4): no
    autocorrelation to preserve, so the block size collapses."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=500)
    L = politis_white_block_length(r)
    assert 1.0 <= L <= 8.0


def test_politis_white_ar1_returns_longer_block() -> None:
    """Strongly autocorrelated AR(1) returns → longer block-length than IID."""
    rng = np.random.default_rng(1)
    n = 500
    eps = rng.normal(0.0, 0.01, size=n)
    r = np.empty(n)
    r[0] = eps[0]
    phi = 0.7
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    L_ar = politis_white_block_length(r)
    L_iid = politis_white_block_length(rng.normal(0.0, 0.01, size=n))
    assert L_ar > L_iid


def test_politis_white_handles_short_series() -> None:
    """Pathological short input must not raise."""
    L = politis_white_block_length([0.01, -0.005, 0.003])
    assert L >= 1.0


# ---- bootstrap_sharpe_ci -------------------------------------------------


def test_bootstrap_sharpe_ci_returns_dataclass() -> None:
    rng_returns = np.random.default_rng(2024)
    r = rng_returns.normal(0.0008, 0.012, size=252)
    res = bootstrap_sharpe_ci(r, B=500, seed=7)
    assert isinstance(res, BootstrapSharpeResult)
    assert res.n_resamples == 500
    assert res.alpha == 0.05
    assert res.bootstrap_distribution.shape == (500,)


def test_bootstrap_sharpe_ci_brackets_point_estimate() -> None:
    """With a strong-signal series the point estimate must lie inside
    the 95% CI (it's a percentile of B≥10 resamples)."""
    rng_returns = np.random.default_rng(2025)
    r = rng_returns.normal(0.002, 0.01, size=400)
    res = bootstrap_sharpe_ci(r, B=2000, seed=11)
    assert res.ci_lower <= res.point_estimate <= res.ci_upper


def test_bootstrap_sharpe_ci_p_positive_high_for_strong_signal() -> None:
    """Sharpe ≈ 3 series → p(SR > 0) should be ≈ 1."""
    rng_returns = np.random.default_rng(3)
    r = rng_returns.normal(0.002, 0.01, size=500)
    res = bootstrap_sharpe_ci(r, B=2000, seed=1)
    assert res.p_positive > 0.99


def test_bootstrap_sharpe_ci_p_positive_low_for_negative_signal() -> None:
    """Negative-drift series → p(SR > 0) should be low."""
    rng_returns = np.random.default_rng(4)
    r = rng_returns.normal(-0.001, 0.01, size=500)
    res = bootstrap_sharpe_ci(r, B=2000, seed=2)
    assert res.p_positive < 0.05


def test_bootstrap_sharpe_ci_deterministic_with_seed() -> None:
    """Same seed + same input → byte-identical bootstrap distribution
    (CONTEXT.md cycle-4 RNG contract)."""
    rng_returns = np.random.default_rng(5)
    r = rng_returns.normal(0.001, 0.01, size=250)
    a = bootstrap_sharpe_ci(r, B=500, seed=42)
    b = bootstrap_sharpe_ci(r, B=500, seed=42)
    assert a.ci_lower == b.ci_lower
    assert a.ci_upper == b.ci_upper
    np.testing.assert_array_equal(a.bootstrap_distribution, b.bootstrap_distribution)


def test_bootstrap_sharpe_ci_explicit_block_length() -> None:
    rng_returns = np.random.default_rng(6)
    r = rng_returns.normal(0.001, 0.01, size=250)
    res = bootstrap_sharpe_ci(r, B=300, block_length=10.0, seed=99)
    assert res.block_length == 10.0


def test_bootstrap_sharpe_ci_rejects_both_rng_and_seed() -> None:
    rng_returns = np.random.default_rng(7)
    r = rng_returns.normal(0.001, 0.01, size=100)
    with pytest.raises(ValueError, match="rng or seed"):
        bootstrap_sharpe_ci(r, B=100, rng=np.random.default_rng(0), seed=0)


def test_bootstrap_sharpe_ci_rejects_bad_alpha() -> None:
    r = np.array([0.001] * 100)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_sharpe_ci(r, B=100, alpha=1.5)


def test_bootstrap_sharpe_ci_rejects_bad_B() -> None:
    r = np.array([0.001] * 100)
    with pytest.raises(ValueError, match="B"):
        bootstrap_sharpe_ci(r, B=0)


def test_bootstrap_sharpe_ci_rejects_nonpositive_block_length() -> None:
    r = np.array([0.001] * 100)
    with pytest.raises(ValueError, match="block_length"):
        bootstrap_sharpe_ci(r, B=100, block_length=0.0)


def test_bootstrap_sharpe_ci_short_series_returns_nan() -> None:
    res = bootstrap_sharpe_ci([0.01], B=100)
    assert math.isnan(res.point_estimate)
    assert math.isnan(res.ci_lower)
    assert math.isnan(res.ci_upper)
    assert math.isnan(res.p_positive)
    assert res.n_resamples == 0
