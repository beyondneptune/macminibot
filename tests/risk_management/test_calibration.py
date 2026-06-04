"""Tests for risk_management.calibration (Epic 5 derivation from backtest).

Cycle 8 / CONTEXT.md I2: drawdown breakers should be derived from the actual
backtest fold-return distribution, not copied from the placeholder
-2% / -10% prior. These tests exercise the calibration helper against
synthetic distributions whose tails we can compute by hand.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal

import pytest

from risk_management.calibration import (
    DrawdownCalibration,
    calibrate_drawdown_thresholds,
)


def test_rejects_empty_returns():
    with pytest.raises(ValueError, match="non-empty"):
        calibrate_drawdown_thresholds([])


def test_rejects_all_non_finite():
    with pytest.raises(ValueError, match="no finite values"):
        calibrate_drawdown_thresholds([float("nan"), float("inf"), -float("inf")])


def test_low_variance_strategy_hits_daily_floor():
    """A near-deterministic small positive drift hits the daily floor clamp.

    Without the clamp the calibrated daily breaker would be ~0bps and HALT
    at the slightest noise.
    """

    returns = [0.0001] * 500
    cal = calibrate_drawdown_thresholds(returns, floor_daily_pct=-0.005)
    # Floor wins; daily exactly at the floor.
    assert cal.dd_daily_pct == Decimal("-0.0050")
    # Weekly / PTT scale off the floored daily.
    assert cal.dd_weekly_pct < cal.dd_daily_pct  # more negative
    assert cal.dd_ptt_pct < cal.dd_weekly_pct


def test_fat_tail_strategy_hits_ceiling_clamp():
    """A return series with a huge left tail is clamped at the ceiling.

    Operator's risk appetite caps how loose the breaker may go, regardless
    of what the backtest shows.
    """

    # 10% of days lose 20% — target_daily_tail_pct=0.05 lands deep in the
    # cluster (q ~ -0.20). Even with no skew adjustment the result would
    # be -0.20, well past the operator's -0.05 ceiling.
    returns = [0.001] * 90 + [-0.20] * 10
    cal = calibrate_drawdown_thresholds(
        returns, ceiling_daily_pct=-0.05, target_daily_tail_pct=0.05
    )
    assert cal.dd_daily_pct == Decimal("-0.0500")


def test_typical_strategy_calibrates_between_floor_and_ceiling():
    """Normal-ish returns => calibrated breaker lands in the operator band."""

    random.seed(42)
    returns = [random.gauss(0.0008, 0.012) for _ in range(2_000)]
    cal = calibrate_drawdown_thresholds(returns)
    # Daily breaker should land between -1% and -5%, comfortably in band.
    assert Decimal("-0.05") <= cal.dd_daily_pct <= Decimal("-0.005")
    # Provenance fields populated.
    assert cal.n_obs == 2_000
    assert cal.daily_std > 0.0
    assert "Calibrated from n=2000" in cal.rationale


def test_calibrated_thresholds_are_quantized_to_1bp():
    random.seed(0)
    returns = [random.gauss(0.0, 0.01) for _ in range(500)]
    cal = calibrate_drawdown_thresholds(returns)
    # All three thresholds quantize to exactly 4 decimal places (1bp grid).
    for val in (cal.dd_daily_pct, cal.dd_weekly_pct, cal.dd_ptt_pct):
        # Decimal exponent of -4 means 4 fractional digits.
        assert val.as_tuple().exponent == -4


def test_threshold_ordering_daily_less_negative_than_ptt():
    """PTT must be the most negative (largest loss tolerated before HALT)."""

    random.seed(1)
    returns = [random.gauss(0.0, 0.01) for _ in range(500)]
    cal = calibrate_drawdown_thresholds(returns)
    assert cal.dd_ptt_pct < cal.dd_weekly_pct <= cal.dd_daily_pct


def test_negative_skew_tightens_threshold_vs_symmetric():
    """A left-skewed distribution should produce a tighter (more negative)
    daily breaker than a symmetric one with the same stddev.

    This is the Sortino & van der Meer 1991 asymmetric-loss adjustment.
    """

    random.seed(7)
    symmetric = [random.gauss(0.0, 0.01) for _ in range(2_000)]
    # Inject a fat negative tail: 5% of the days are 4x worse than gauss.
    skewed = list(symmetric)
    for i in range(0, 100):  # 100 of 2100 days
        skewed.append(-0.04)

    cal_sym = calibrate_drawdown_thresholds(symmetric)
    cal_skew = calibrate_drawdown_thresholds(skewed)

    assert cal_skew.daily_skew < cal_sym.daily_skew  # confirms inputs differ
    # Skewed result is no looser than the symmetric one.
    assert cal_skew.dd_daily_pct <= cal_sym.dd_daily_pct


def test_drops_non_finite_values():
    returns = [0.01, float("nan"), -0.01, float("inf"), 0.005]
    cal = calibrate_drawdown_thresholds(returns)
    assert cal.n_obs == 3


def test_as_dict_serializes_decimals_as_str():
    cal = calibrate_drawdown_thresholds([0.001, -0.01, 0.002, -0.005])
    d = cal.as_dict()
    assert isinstance(d["dd_daily_pct"], str)
    assert isinstance(d["dd_weekly_pct"], str)
    assert isinstance(d["dd_ptt_pct"], str)
    # Round-trip: strings parse back to Decimal cleanly.
    assert Decimal(d["dd_daily_pct"]) == cal.dd_daily_pct


def test_dataclass_is_frozen():
    cal = calibrate_drawdown_thresholds([0.01, -0.01])
    with pytest.raises(Exception):
        cal.dd_daily_pct = Decimal("0")  # type: ignore[misc]
