"""Slippage: Almgren-Chriss sqrt scaling + spread-half floor by asset class."""

from __future__ import annotations

import math

import pytest

from backtesting.slippage import (
    SPREAD_HALF_FLOOR_BPS,
    almgren_chriss,
    slippage_bps_with_floor,
)


def test_almgren_chriss_zero_volume_is_zero() -> None:
    assert almgren_chriss(0, adv=1_000_000, daily_vol_bps=100.0) == 0.0


def test_almgren_chriss_sqrt_scaling() -> None:
    # Doubling Q at fixed ADV multiplies impact by sqrt(2).
    base = almgren_chriss(volume_traded=10_000, adv=1_000_000, daily_vol_bps=100.0)
    doubled = almgren_chriss(volume_traded=20_000, adv=1_000_000, daily_vol_bps=100.0)
    assert doubled == pytest.approx(base * math.sqrt(2), rel=1e-9)


def test_almgren_chriss_formula() -> None:
    # 100 bps daily vol, k=0.1, Q/ADV = 0.01 → 100 * 0.1 * sqrt(0.01) = 1.0 bps.
    got = almgren_chriss(
        volume_traded=10_000, adv=1_000_000, daily_vol_bps=100.0, k=0.1
    )
    assert got == pytest.approx(1.0, rel=1e-9)


def test_almgren_chriss_rejects_zero_adv() -> None:
    with pytest.raises(ValueError):
        almgren_chriss(volume_traded=100, adv=0, daily_vol_bps=100.0)


def test_almgren_chriss_rejects_negative_k() -> None:
    with pytest.raises(ValueError):
        almgren_chriss(volume_traded=100, adv=1_000, daily_vol_bps=100.0, k=-0.1)


def test_floor_binds_for_tiny_orders() -> None:
    # 10 shares vs 10MM ADV → impact ~negligible; floor 1 bp must bind.
    sb = slippage_bps_with_floor(
        volume_traded=10, adv=10_000_000, daily_vol_bps=100.0, asset_class="equity"
    )
    assert sb.impact_bps < sb.floor_bps
    assert sb.total_bps == pytest.approx(SPREAD_HALF_FLOOR_BPS["equity"])


def test_floor_does_not_bind_for_large_orders() -> None:
    # 100,000 shares vs 100,000 ADV — impact >> floor.
    sb = slippage_bps_with_floor(
        volume_traded=100_000, adv=100_000, daily_vol_bps=100.0, asset_class="equity"
    )
    assert sb.impact_bps > sb.floor_bps
    assert sb.total_bps == sb.impact_bps


def test_floor_per_asset_class() -> None:
    assert SPREAD_HALF_FLOOR_BPS["equity"] == 1.0
    assert SPREAD_HALF_FLOOR_BPS["future"] == 0.5
    assert SPREAD_HALF_FLOOR_BPS["fx"] == 0.1


def test_unknown_asset_class_raises() -> None:
    with pytest.raises(ValueError):
        slippage_bps_with_floor(
            volume_traded=100, adv=1_000, daily_vol_bps=50.0, asset_class="crypto"
        )
