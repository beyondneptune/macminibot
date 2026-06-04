"""Edge-case rule book tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from strategy_engine.edge_cases import (
    HaltAction,
    StaleBarsTier,
    apply_no_trade_band,
    apply_turnover_ceiling,
    check_sigma_conditioning,
    classify_universe_staleness,
    decay_to_cash,
    handle_halt,
    is_market_holiday,
    is_nav_jump,
    is_stale_bars,
    is_zero_signal_day,
    validate_signal_scale,
)


def test_christmas_2025_is_holiday():
    assert is_market_holiday(date(2025, 12, 25)) is True


def test_2025_11_04_tuesday_is_not_holiday():
    # Tuesday — regular session, no observance
    assert is_market_holiday(date(2025, 11, 4)) is False


def test_handle_halt_one_session_freezes():
    d = handle_halt("AAPL", w_prev_i=0.03, sessions_halted=1)
    assert d.action == HaltAction.FREEZE
    assert d.target_weight == pytest.approx(0.03)
    assert d.hold_reason == "halt.luld"


def test_handle_halt_multi_session_flattens():
    d = handle_halt("AAPL", w_prev_i=0.03, sessions_halted=3)
    assert d.action == HaltAction.FLATTEN_TWAP
    assert d.target_weight == 0.0


def test_validate_signal_scale_accepts_small():
    v = validate_signal_scale(Decimal("0.002"))
    assert v.accepted is True


def test_validate_signal_scale_rejects_blowup():
    v = validate_signal_scale(Decimal("0.10"))
    assert v.accepted is False
    assert "blowup" in v.reason


def test_validate_signal_scale_rejects_nan():
    v = validate_signal_scale(float("nan"))
    assert v.accepted is False


def test_check_sigma_conditioning_well_conditioned():
    s = np.diag([1.0, 0.5, 0.25])
    out = check_sigma_conditioning(s)
    assert out.ill_conditioned is False


def test_check_sigma_conditioning_singular():
    s = np.diag([1.0, 0.0, 1e-15])
    out = check_sigma_conditioning(s)
    assert out.ill_conditioned is True


def test_is_stale_bars():
    now = 10 * 24 * 3600 * 1_000_000_000  # day 10 in ns
    fresh_ts = now - (3 * 3600 * 1_000_000_000)  # 3h old
    stale_ts = now - (72 * 3600 * 1_000_000_000)  # 72h old
    assert is_stale_bars("X", fresh_ts, now_ns=now) is False
    assert is_stale_bars("X", stale_ts, now_ns=now) is True


def test_classify_universe_staleness():
    assert classify_universe_staleness(0, 100) == StaleBarsTier.OK
    assert classify_universe_staleness(5, 100) == StaleBarsTier.OK
    assert classify_universe_staleness(20, 100) == StaleBarsTier.HALVED_TARGET
    assert classify_universe_staleness(40, 100) == StaleBarsTier.HALT


def test_is_nav_jump():
    assert is_nav_jump(Decimal("110000"), Decimal("100000")) is True  # 10%
    assert is_nav_jump(Decimal("105000"), Decimal("100000")) is False  # 5%


def test_zero_signal_day_and_decay():
    quiet = np.array([0.0001, -0.00005, 0.0002])
    loud = np.array([0.001, -0.002, 0.0005])
    assert is_zero_signal_day(quiet) is True
    assert is_zero_signal_day(loud) is False
    w_prev = np.array([0.05, -0.04, 0.02])
    decayed = decay_to_cash(w_prev, rate=0.2)
    np.testing.assert_allclose(decayed, w_prev * 0.8)


def test_no_trade_band_freezes_small_changes():
    w_prev = np.array([0.05, 0.04, 0.03])
    # name 0: 1bp change (frozen), name 1: well above band (trades), name 2: 0bp
    w_new = np.array([0.0501, 0.05, 0.030])
    out = apply_no_trade_band(w_new, w_prev, band=0.005)
    assert out[0] == pytest.approx(0.05)  # frozen back to w_prev
    assert out[1] == pytest.approx(0.05)  # passes through (|Δ|=0.01 > band)
    assert out[2] == pytest.approx(0.03)  # unchanged (Δ=0 < band → also frozen, but already w_prev)


def test_turnover_ceiling_caps_excessive_trade():
    w_prev = np.zeros(3)
    w_new = np.array([0.6, 0.6, 0.6])  # L1 = 1.8 > 1.0 ceiling
    capped, was_capped = apply_turnover_ceiling(w_new, w_prev, ceiling=1.0)
    assert was_capped is True
    assert float(np.sum(np.abs(capped))) == pytest.approx(1.0)


def test_turnover_ceiling_passes_through_when_under():
    w_prev = np.zeros(3)
    w_new = np.array([0.2, 0.2, 0.2])  # L1 = 0.6
    capped, was_capped = apply_turnover_ceiling(w_new, w_prev, ceiling=1.0)
    assert was_capped is False
    np.testing.assert_array_equal(capped, w_new)
