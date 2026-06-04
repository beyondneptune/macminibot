"""Vol-target scaler and signal-winsorization tests."""

from __future__ import annotations

import numpy as np
import pytest

from strategy_engine.sizing import (
    TRADING_DAYS_PER_YEAR,
    VIX_DAMPEN_INTERMEDIATE_MULT,
    VIX_MID_ONLY_PCT_HIGH,
    VIX_MID_ONLY_PCT_LOW,
    signal_winsorize,
    vol_target_scale,
    vol_target_with_regime_dampening,
    vol_target_with_regime_mid_only_mode,
)


def test_vol_target_matches_within_1pct():
    rng = np.random.default_rng(seed=33)
    n = 10
    A = rng.standard_normal(size=(n, n)) * 0.01
    Sigma = A @ A.T + np.eye(n) * 1e-4  # daily covariance
    w = rng.standard_normal(n) * 0.03

    target = 0.12  # 12% annualized
    res = vol_target_scale(w, Sigma, target_vol_annualized=target)
    # post-scale realized vol must be within 1% of target
    rel_err = abs(res.realized_vol_post - target) / target
    assert rel_err < 0.01, f"rel err = {rel_err:.4f}"


def test_vol_target_handles_degenerate_zero_portfolio():
    Sigma = np.eye(3) * 1e-4
    w = np.zeros(3)
    res = vol_target_scale(w, Sigma)
    assert res.scaling_factor == 1.0
    assert res.clamped is True


def test_vol_target_clamps_at_leverage_cap():
    # Tiny portfolio vol → demand for huge leverage → clamp.
    Sigma = np.eye(2) * 1e-8
    w = np.array([0.001, -0.001])
    res = vol_target_scale(w, Sigma, target_vol_annualized=0.12, max_leverage_multiplier=5.0)
    assert res.clamped is True
    assert res.scaling_factor == pytest.approx(5.0)


def test_winsorize_clips_at_3sigma():
    rng = np.random.default_rng(seed=4)
    base = rng.standard_normal(50) * 0.001
    # inject one outlier well beyond 3σ
    base[10] = 100.0
    clipped, mask = signal_winsorize(base, n_sigma=3.0)
    assert mask[10] == True
    mu = float(np.mean(base))
    sd = float(np.std(base, ddof=0))
    hi = mu + 3.0 * sd
    assert clipped[10] == pytest.approx(hi, rel=1e-9)
    # untouched elements unchanged
    for i in [0, 5, 25, 40]:
        if not mask[i]:
            assert clipped[i] == base[i]


def test_winsorize_handles_zero_std():
    """When all signals are identical, nothing to clip."""
    arr = np.full(20, 0.001)
    clipped, mask = signal_winsorize(arr, n_sigma=3.0)
    assert not mask.any()
    np.testing.assert_array_equal(clipped, arr)


def test_winsorize_handles_empty():
    arr = np.array([], dtype=np.float64)
    clipped, mask = signal_winsorize(arr)
    assert clipped.shape == (0,)
    assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# vol_target_with_regime_dampening — cycle-10 soft degradation
# ---------------------------------------------------------------------------


def _toy_portfolio(seed: int = 7, n: int = 6):
    rng = np.random.default_rng(seed=seed)
    A = rng.standard_normal(size=(n, n)) * 0.01
    Sigma = A @ A.T + np.eye(n) * 1e-4
    w = rng.standard_normal(n) * 0.03
    return w, Sigma


def test_regime_dampen_normal_matches_plain_vol_target():
    """Below the 60th percentile, behaviour is identical to vol_target_scale."""
    w, Sigma = _toy_portfolio()
    vix_dist = np.linspace(10.0, 50.0, 1000)
    current = 15.0  # well below the 60th pct
    plain = vol_target_scale(w, Sigma, target_vol_annualized=0.12)
    res = vol_target_with_regime_dampening(w, Sigma, current, vix_dist)
    assert res.regime_label == "normal"
    assert res.effective_target_vol == pytest.approx(0.12)
    np.testing.assert_allclose(res.w_scaled, plain.w_scaled, rtol=1e-12)
    assert res.realized_vol_post == pytest.approx(plain.realized_vol_post, rel=1e-12)


def test_regime_dampen_intermediate_halves_target_vol():
    """In the 60–80 pct band, realized vol should sit at 6%, not 12%."""
    w, Sigma = _toy_portfolio()
    vix_dist = np.linspace(10.0, 50.0, 1000)
    # 70th-pct reading → 10 + 0.70*(50-10) = 38
    res = vol_target_with_regime_dampening(w, Sigma, 38.0, vix_dist)
    assert res.regime_label == "intermediate"
    assert res.effective_target_vol == pytest.approx(0.12 * VIX_DAMPEN_INTERMEDIATE_MULT)
    rel_err = abs(res.realized_vol_post - 0.06) / 0.06
    assert rel_err < 0.01, f"intermediate realized vol off-target: {res.realized_vol_post}"
    assert 0.60 <= res.vix_percentile < 0.80


def test_regime_dampen_halt_zeros_weights():
    """At/above the 80th pct, weights are zeroed (paper-mode halt equivalent)."""
    w, Sigma = _toy_portfolio()
    vix_dist = np.linspace(10.0, 50.0, 1000)
    # 90th-pct reading → 10 + 0.90*40 = 46
    res = vol_target_with_regime_dampening(w, Sigma, 46.0, vix_dist)
    assert res.regime_label == "halt"
    assert res.effective_target_vol == 0.0
    assert res.scaling_factor == 0.0
    assert np.all(res.w_scaled == 0.0)
    assert res.realized_vol_post == 0.0
    # realized_vol_pre is informational — should equal the un-scaled vol
    assert res.realized_vol_pre > 0.0
    assert res.vix_percentile >= 0.80


def test_regime_dampen_exact_threshold_is_inclusive():
    """At pct == 0.80 we must halt, not stay in intermediate."""
    w, Sigma = _toy_portfolio()
    # Construct a distribution where exactly 80% of samples are < current.
    vix_dist = np.arange(100, dtype=float)  # values 0..99
    current = 80.0  # mean(dist < 80) = 0.80 exactly
    res = vol_target_with_regime_dampening(w, Sigma, current, vix_dist)
    assert res.regime_label == "halt"
    assert np.all(res.w_scaled == 0.0)


def test_regime_dampen_empty_distribution_falls_back_to_normal():
    """No history → behave like the plain scaler at base target."""
    w, Sigma = _toy_portfolio()
    res = vol_target_with_regime_dampening(
        w, Sigma, 100.0, np.array([], dtype=float)
    )
    assert res.regime_label == "normal"
    assert res.effective_target_vol == pytest.approx(0.12)
    # vix_percentile is NaN sentinel when distribution is empty
    assert np.isnan(res.vix_percentile)


def test_regime_dampen_nonfinite_reading_falls_back_to_normal():
    w, Sigma = _toy_portfolio()
    vix_dist = np.linspace(10.0, 50.0, 1000)
    res = vol_target_with_regime_dampening(w, Sigma, float("nan"), vix_dist)
    assert res.regime_label == "normal"
    assert np.isnan(res.vix_percentile)


def test_regime_dampen_filters_nonfinite_from_distribution():
    """NaNs/infs in the historical sample must not poison the percentile."""
    w, Sigma = _toy_portfolio()
    clean = np.linspace(10.0, 50.0, 1000)
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    res_clean = vol_target_with_regime_dampening(w, Sigma, 38.0, clean)
    res_dirty = vol_target_with_regime_dampening(w, Sigma, 38.0, dirty)
    assert res_clean.regime_label == res_dirty.regime_label == "intermediate"
    assert res_clean.vix_percentile == pytest.approx(res_dirty.vix_percentile)
    np.testing.assert_allclose(res_clean.w_scaled, res_dirty.w_scaled, rtol=1e-12)


# ---------------------------------------------------------------------------
# vol_target_with_regime_mid_only_mode — cycle-13 opt-in mid-band-only mode
# ---------------------------------------------------------------------------


def test_mid_only_low_band_zeros_weights():
    """Below the 40th percentile, weights are zeroed (vix_low → flat)."""
    w, Sigma = _toy_portfolio()
    vix_dist = np.linspace(10.0, 50.0, 1000)
    # 20th-pct reading → 10 + 0.20 * 40 = 18
    res = vol_target_with_regime_mid_only_mode(w, Sigma, 18.0, vix_dist)
    assert res.regime_label == "vix_low"
    assert res.effective_target_vol == 0.0
    assert res.scaling_factor == 0.0
    assert np.all(res.w_scaled == 0.0)
    assert res.realized_vol_post == 0.0
    # Pre-scale vol is reported for logging.
    assert res.realized_vol_pre > 0.0
    assert res.vix_percentile < VIX_MID_ONLY_PCT_LOW


def test_mid_only_mid_band_matches_plain_vol_target():
    """In the 40–80 pct band, behaviour equals vol_target_scale at 12%."""
    w, Sigma = _toy_portfolio()
    vix_dist = np.linspace(10.0, 50.0, 1000)
    # 60th-pct reading → 10 + 0.60 * 40 = 34
    plain = vol_target_scale(w, Sigma, target_vol_annualized=0.12)
    res = vol_target_with_regime_mid_only_mode(w, Sigma, 34.0, vix_dist)
    assert res.regime_label == "vix_mid"
    assert res.effective_target_vol == pytest.approx(0.12)
    # Mid-band behaviour must be bit-identical to the plain scaler at the
    # base target — including leverage-cap clamping, scaling factor, and
    # realized post-vol — so the only difference vs. always-on sizing is
    # the regime gate.
    np.testing.assert_allclose(res.w_scaled, plain.w_scaled, rtol=1e-12)
    assert res.scaling_factor == pytest.approx(plain.scaling_factor)
    assert res.clamped == plain.clamped
    assert res.realized_vol_post == pytest.approx(plain.realized_vol_post, rel=1e-12)
    assert VIX_MID_ONLY_PCT_LOW <= res.vix_percentile < VIX_MID_ONLY_PCT_HIGH


def test_mid_only_high_band_zeros_weights():
    """At/above the 80th percentile, weights are zeroed (vix_high → flat)."""
    w, Sigma = _toy_portfolio()
    vix_dist = np.linspace(10.0, 50.0, 1000)
    # 90th-pct reading → 10 + 0.90 * 40 = 46
    res = vol_target_with_regime_mid_only_mode(w, Sigma, 46.0, vix_dist)
    assert res.regime_label == "vix_high"
    assert res.effective_target_vol == 0.0
    assert res.scaling_factor == 0.0
    assert np.all(res.w_scaled == 0.0)
    assert res.realized_vol_post == 0.0
    assert res.realized_vol_pre > 0.0
    assert res.vix_percentile >= VIX_MID_ONLY_PCT_HIGH


def test_mid_only_lower_threshold_is_inclusive():
    """At pct == 0.40 we must enter the mid band, not stay in vix_low."""
    w, Sigma = _toy_portfolio()
    vix_dist = np.arange(100, dtype=float)  # values 0..99
    current = 40.0  # mean(dist < 40) == 0.40 exactly
    res = vol_target_with_regime_mid_only_mode(w, Sigma, current, vix_dist)
    assert res.regime_label == "vix_mid"
    assert not np.all(res.w_scaled == 0.0)


def test_mid_only_upper_threshold_is_exclusive():
    """At pct == 0.80 we must flip to vix_high (zero weights)."""
    w, Sigma = _toy_portfolio()
    vix_dist = np.arange(100, dtype=float)  # values 0..99
    current = 80.0  # mean(dist < 80) == 0.80 exactly
    res = vol_target_with_regime_mid_only_mode(w, Sigma, current, vix_dist)
    assert res.regime_label == "vix_high"
    assert np.all(res.w_scaled == 0.0)


def test_mid_only_empty_distribution_falls_back_to_mid_band():
    """No history → trade at base target (fail-to-trade is worse for one bar)."""
    w, Sigma = _toy_portfolio()
    res = vol_target_with_regime_mid_only_mode(
        w, Sigma, 100.0, np.array([], dtype=float)
    )
    assert res.regime_label == "vix_mid"
    assert res.effective_target_vol == pytest.approx(0.12)
    assert np.isnan(res.vix_percentile)
    # And the scaled weights must actually be deployed.
    assert not np.all(res.w_scaled == 0.0)


def test_mid_only_nonfinite_reading_falls_back_to_mid_band():
    w, Sigma = _toy_portfolio()
    vix_dist = np.linspace(10.0, 50.0, 1000)
    res = vol_target_with_regime_mid_only_mode(w, Sigma, float("nan"), vix_dist)
    assert res.regime_label == "vix_mid"
    assert np.isnan(res.vix_percentile)
    assert not np.all(res.w_scaled == 0.0)


def test_mid_only_filters_nonfinite_from_distribution():
    """NaNs/infs in the historical sample must not poison the percentile."""
    w, Sigma = _toy_portfolio()
    clean = np.linspace(10.0, 50.0, 1000)
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    res_clean = vol_target_with_regime_mid_only_mode(w, Sigma, 34.0, clean)
    res_dirty = vol_target_with_regime_mid_only_mode(w, Sigma, 34.0, dirty)
    assert res_clean.regime_label == res_dirty.regime_label == "vix_mid"
    assert res_clean.vix_percentile == pytest.approx(res_dirty.vix_percentile)
    np.testing.assert_allclose(res_clean.w_scaled, res_dirty.w_scaled, rtol=1e-12)


def test_mid_only_documented_as_opt_in_alternative():
    """Module-level docstring must flag mid-only mode as the opt-in alternative."""
    from strategy_engine import sizing as sizing_module

    fn = sizing_module.vol_target_with_regime_mid_only_mode
    assert fn.__doc__ is not None
    doc = fn.__doc__.lower()
    assert "opt-in" in doc
    assert "alternative" in doc
    # Cross-reference to the existing dampening path so future readers can
    # find the other option.
    assert "vol_target_with_regime_dampening" in fn.__doc__
