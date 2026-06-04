"""Tests for ``strategy_engine.long_only`` — top-N long-only sleeve.

Coverage:
- Top-N selection correctness (descending by r_hat, deterministic ties).
- Equal-weighting sums to 1.0 long exposure pre-vol-scaling.
- Vol-target scaling produces realized vol ~target (diagonal approx).
- ``n`` defaulting to ``ceil(len(universe) / 2)``.
- Defensive: nav<=0, vol_target<=0, missing sigma, empty universe.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from shared_core.contracts import Symbol, Venue
from strategy_engine.long_only import (
    DEFAULT_VOL_TARGET,
    long_only_top_n,
)


def _mksym(ticker: str) -> Symbol:
    return Symbol(ticker=ticker, venue=Venue.NASDAQ, currency="USD")


def _uniform_sigma(symbols, sigma_value: float) -> dict[Symbol, Decimal]:
    return {s: Decimal(str(sigma_value)) for s in symbols}


# ---------------------------------------------------------------------------
# Selection correctness
# ---------------------------------------------------------------------------


def test_top_n_picks_highest_r_hat():
    syms = [_mksym(t) for t in ("AAA", "BBB", "CCC", "DDD")]
    r_hat = {
        syms[0]: Decimal("0.01"),
        syms[1]: Decimal("0.05"),  # highest
        syms[2]: Decimal("-0.02"),
        syms[3]: Decimal("0.03"),  # second
    }
    sigma = _uniform_sigma(syms, 0.20)
    out = long_only_top_n(
        r_hat, nav=Decimal("100000"), sigma_annualized=sigma, n=2
    )
    assert set(out.keys()) == {syms[1], syms[3]}, "must pick top-2 by r_hat"
    # Long-only: every allocation is non-negative.
    assert all(v >= Decimal(0) for v in out.values())


def test_top_n_deterministic_tie_break_by_ticker():
    """Ties on r_hat should break by ticker (alphabetical) for replay."""
    syms = [_mksym(t) for t in ("ZZZ", "AAA", "MMM")]
    r_hat = {s: Decimal("0.05") for s in syms}  # all tied
    sigma = _uniform_sigma(syms, 0.20)
    out = long_only_top_n(
        r_hat, nav=Decimal("100000"), sigma_annualized=sigma, n=2
    )
    # Tie-break: AAA, MMM before ZZZ.
    assert set(out.keys()) == {_mksym("AAA"), _mksym("MMM")}


# ---------------------------------------------------------------------------
# Equal-weighting sums to 1.0 long exposure (pre-scaling check)
# ---------------------------------------------------------------------------


def test_equal_weight_sums_to_one_when_vol_target_matches_universe_vol():
    """Pre-scaling weights are 1/N each, summing to 1.0.

    With a single uniform σ across the picked names and the vol target
    set to exactly the diagonal-portfolio vol, the scaling factor is
    1.0 and the resulting dollar allocations sum to ``nav``.
    """
    syms = [_mksym(t) for t in ("A", "B", "C", "D")]
    n = 4
    sigma_value = 0.20
    # σ_p^2 = Σ (1/N)^2 σ_i^2 = N * (1/N)^2 * σ^2 = σ^2 / N
    # σ_p = σ / sqrt(N) = 0.20 / 2 = 0.10
    target = sigma_value / math.sqrt(n)
    r_hat = {s: Decimal("0.01") for s in syms}
    sigma = _uniform_sigma(syms, sigma_value)
    nav = Decimal("100000")

    out = long_only_top_n(
        r_hat, nav=nav, sigma_annualized=sigma, n=n, vol_target=target
    )
    total = sum(out.values(), Decimal(0))
    # Allocations should sum to nav (scaling factor = 1.0).
    assert float(total) == pytest.approx(float(nav), rel=1e-6)
    # And each name gets equal weight = nav/N.
    expected_each = nav / Decimal(n)
    for v in out.values():
        assert float(v) == pytest.approx(float(expected_each), rel=1e-6)


# ---------------------------------------------------------------------------
# Vol-target scaling
# ---------------------------------------------------------------------------


def test_vol_target_scaling_hits_target_within_1pct():
    """Realized diagonal-portfolio vol matches the requested target."""
    syms = [_mksym(t) for t in ("A", "B", "C", "D", "E")]
    sigma_value = 0.30
    target = 0.12
    n = 3
    r_hat = {s: Decimal(str(0.10 - 0.01 * i)) for i, s in enumerate(syms)}
    sigma = _uniform_sigma(syms, sigma_value)
    nav = Decimal("1000000")

    out = long_only_top_n(
        r_hat, nav=nav, sigma_annualized=sigma, n=n, vol_target=target
    )
    # Recover weights: w_i = alloc_i / nav.
    weights = [float(v / nav) for v in out.values()]
    # Diagonal portfolio vol: sqrt(Σ w_i^2 σ_i^2).
    realized = math.sqrt(sum((w * sigma_value) ** 2 for w in weights))
    rel_err = abs(realized - target) / target
    assert rel_err < 0.01, f"vol target miss: realized={realized:.4f}, rel_err={rel_err:.4f}"


def test_vol_target_scales_up_when_universe_is_low_vol():
    """Low-vol universe → scaling factor > 1 (leverage up to hit target)."""
    syms = [_mksym(t) for t in ("A", "B")]
    r_hat = {s: Decimal("0.01") for s in syms}
    sigma = _uniform_sigma(syms, 0.05)  # 5% annual vol
    nav = Decimal("100000")

    out = long_only_top_n(
        r_hat, nav=nav, sigma_annualized=sigma, n=2, vol_target=0.12
    )
    total = sum(out.values(), Decimal(0))
    # Diagonal σ_p with N=2, σ=0.05 → 0.05/sqrt(2) ≈ 0.0354
    # Scaling = 0.12 / 0.0354 ≈ 3.39 → total allocation > nav.
    assert float(total) > float(nav), "low-vol universe should leverage up"


# ---------------------------------------------------------------------------
# N defaulting
# ---------------------------------------------------------------------------


def test_n_defaults_to_ceil_half():
    syms = [_mksym(t) for t in ("A", "B", "C", "D", "E")]  # 5 names
    r_hat = {s: Decimal(str(0.10 - 0.01 * i)) for i, s in enumerate(syms)}
    sigma = _uniform_sigma(syms, 0.20)
    out = long_only_top_n(
        r_hat, nav=Decimal("100000"), sigma_annualized=sigma  # n omitted
    )
    # ceil(5 / 2) = 3
    assert len(out) == 3


def test_n_defaults_handle_even_universe():
    syms = [_mksym(t) for t in ("A", "B", "C", "D")]  # 4 names
    r_hat = {s: Decimal(str(0.10 - 0.01 * i)) for i, s in enumerate(syms)}
    sigma = _uniform_sigma(syms, 0.20)
    out = long_only_top_n(
        r_hat, nav=Decimal("100000"), sigma_annualized=sigma  # n omitted
    )
    # ceil(4 / 2) = 2
    assert len(out) == 2


def test_n_clamped_to_universe_size():
    syms = [_mksym(t) for t in ("A", "B")]
    r_hat = {s: Decimal("0.05") for s in syms}
    sigma = _uniform_sigma(syms, 0.20)
    out = long_only_top_n(
        r_hat, nav=Decimal("100000"), sigma_annualized=sigma, n=99
    )
    assert len(out) == 2


def test_n_clamped_to_minimum_one():
    syms = [_mksym(t) for t in ("A", "B", "C")]
    r_hat = {s: Decimal("0.05") for s in syms}
    sigma = _uniform_sigma(syms, 0.20)
    out = long_only_top_n(
        r_hat, nav=Decimal("100000"), sigma_annualized=sigma, n=0
    )
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Defensive guards
# ---------------------------------------------------------------------------


def test_empty_universe_returns_empty():
    out = long_only_top_n(
        {}, nav=Decimal("100000"), sigma_annualized={}
    )
    assert out == {}


def test_invalid_nav_raises():
    syms = [_mksym("A")]
    r_hat = {syms[0]: Decimal("0.05")}
    sigma = _uniform_sigma(syms, 0.20)
    with pytest.raises(ValueError, match="nav must be positive"):
        long_only_top_n(r_hat, nav=Decimal("0"), sigma_annualized=sigma)


def test_invalid_vol_target_raises():
    syms = [_mksym("A")]
    r_hat = {syms[0]: Decimal("0.05")}
    sigma = _uniform_sigma(syms, 0.20)
    with pytest.raises(ValueError, match="vol_target must be positive"):
        long_only_top_n(
            r_hat, nav=Decimal("100000"), sigma_annualized=sigma, vol_target=0.0
        )


def test_missing_sigma_raises():
    syms = [_mksym(t) for t in ("A", "B")]
    r_hat = {s: Decimal("0.05") for s in syms}
    sigma = {syms[0]: Decimal("0.20")}  # missing syms[1]
    with pytest.raises(KeyError, match="sigma_annualized missing"):
        long_only_top_n(
            r_hat, nav=Decimal("100000"), sigma_annualized=sigma, n=2
        )


def test_default_vol_target_constant_matches_sizing_module():
    """Long-only default must track the MV path's default so regime
    switches don't change the risk envelope unexpectedly."""
    from strategy_engine.sizing import DEFAULT_TARGET_VOL

    assert DEFAULT_VOL_TARGET == DEFAULT_TARGET_VOL
