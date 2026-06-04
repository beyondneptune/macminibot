"""Test IBKR Pro Tiered fee formulas — match CONTEXT.md §Fee model."""

from __future__ import annotations

import math

import pytest

from backtesting.fees import equity_tiered, fx_idealpro, future_cme


# ---- equity ---------------------------------------------------------------


def test_equity_floor_binds_on_tiny_orders() -> None:
    # 10 shares × $100 → IBKR commission 10*0.0035 = $0.035 → floor to $0.35.
    fb = equity_tiered(shares=10, price=100.0)
    assert fb.commission == pytest.approx(0.35)
    assert fb.passthrough == pytest.approx(10 * 0.003)


def test_equity_per_share_above_floor() -> None:
    # 1000 shares × $100 → 1000*0.0035 = $3.50 commission + $3.00 passthrough.
    fb = equity_tiered(shares=1000, price=100.0)
    assert fb.commission == pytest.approx(3.50)
    assert fb.passthrough == pytest.approx(3.00)
    assert fb.total == pytest.approx(6.50)


def test_equity_one_percent_value_cap() -> None:
    # 1 share @ $0.10 → trade value $0.10; cap = $0.001. Floor would put us
    # at $0.35 + tiny — so the cap MUST clamp total to $0.001.
    fb = equity_tiered(shares=1, price=0.10)
    assert fb.total == pytest.approx(0.001, rel=1e-9)


def test_equity_zero_shares() -> None:
    fb = equity_tiered(shares=0, price=100.0)
    assert fb.total == 0.0


def test_equity_rejects_negative() -> None:
    with pytest.raises(ValueError):
        equity_tiered(shares=-1, price=100.0)
    with pytest.raises(ValueError):
        equity_tiered(shares=10, price=-1.0)


# ---- future ---------------------------------------------------------------


def test_future_es_flat_per_contract() -> None:
    ff = future_cme(contracts=1)
    assert ff.ibkr == pytest.approx(0.85)
    assert ff.exchange == pytest.approx(1.38)
    assert ff.reg == pytest.approx(0.02)
    assert ff.total == pytest.approx(2.25)


def test_future_es_scales_linearly() -> None:
    ff = future_cme(contracts=10)
    assert ff.total == pytest.approx(22.50)


def test_future_zero() -> None:
    assert future_cme(0).total == 0.0


# ---- FX -------------------------------------------------------------------


def test_fx_floor_binds_small_notional() -> None:
    # $10,000 × 0.20 bps = $0.20 → floor binds at $2.00.
    fxb = fx_idealpro(notional_usd=10_000.0)
    assert fxb.commission == pytest.approx(2.00)


def test_fx_bps_above_floor() -> None:
    # $10,000,000 × 0.20 bps = $200.
    fxb = fx_idealpro(notional_usd=10_000_000.0)
    assert fxb.commission == pytest.approx(200.0)


def test_fx_zero() -> None:
    assert fx_idealpro(0).commission == 0.0


def test_fx_rejects_negative() -> None:
    with pytest.raises(ValueError):
        fx_idealpro(-1.0)
