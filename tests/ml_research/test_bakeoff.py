"""Tests for the per-asset bake-off (scripts/50_per_asset_bakeoff.py).

We test the pure-function pieces — feature builders, signal generators, the
bracket simulator, FWER correction — without touching ArcticDB. The
end-to-end orchestrator is exercised by the `--quick` smoke run separately.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "50_per_asset_bakeoff.py"


def _load_bakeoff_module():
    """Load `50_per_asset_bakeoff.py` as a module despite the leading digit."""
    spec = importlib.util.spec_from_file_location("bakeoff", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bakeoff"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bakeoff():
    return _load_bakeoff_module()


@pytest.fixture(scope="module")
def synthetic_bars() -> pd.DataFrame:
    """Synthetic 4-year OHLCV — geometric Brownian motion + intra-day range."""
    rng = np.random.default_rng(7)
    n = 1000
    dates = pd.bdate_range("2020-01-02", periods=n)
    daily_ret = rng.normal(0.0005, 0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    open_ = close * np.exp(rng.normal(0.0, 0.003, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.008, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.008, size=n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return bars


def test_add_per_asset_features_columns(bakeoff, synthetic_bars):
    feat = bakeoff.add_per_asset_features(synthetic_bars)
    for col in [
        "ret_1", "ret_5", "ret_21", "ret_63",
        "vol_21", "vol_63",
        "atr_14", "atr_norm",
        "adv_21d_shares", "daily_vol_bps",
        "rsi_2", "zscore_21",
        "macd", "macd_sig", "macd_hist",
    ]:
        assert col in feat.columns, f"missing feature column {col}"
    # vol_21 must be non-negative everywhere it's defined.
    assert (feat["vol_21"].dropna() >= 0).all()


def test_rule_signal_returns_in_minus1_0_plus1(bakeoff, synthetic_bars):
    feat = bakeoff.add_per_asset_features(synthetic_bars).dropna()
    for name, fn in bakeoff.RULE_CANDIDATES:
        sig = fn(feat)
        assert sig.dtype in (np.int8, np.int64, np.int32, int), name
        vals = set(np.unique(sig.values))
        assert vals.issubset({-1, 0, 1}), f"{name} produced {vals}"


def test_simulate_bracket_long_tp_hit(bakeoff):
    """Long with TP that's clearly hit and SL not — must exit at TP."""
    # bars where the bar's high spikes well above entry → TP hit.
    idx = pd.date_range("2024-01-02", periods=1)
    bars = pd.DataFrame({
        "open": [100.0],
        "high": [104.0],
        "low": [99.5],
        "close": [101.0],
        "vol_21": [0.02],  # 2% σ → 0.5σ TP at +1.0; SL at -1.0
        "adv_21d_shares": [1e6],
        "daily_vol_bps": [200.0],
    }, index=idx)
    # Shift so vol_21 is what was known "yesterday" — we need a synthetic prior bar.
    bars_with_prior = pd.concat([bars.iloc[[0]].assign(open=99.5).copy(), bars]).reset_index(drop=True)
    bars_with_prior.index = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=2))
    sig = pd.Series([0, 1], index=bars_with_prior.index, dtype=np.int8)
    cfg = bakeoff.TPSLConfig(tp_sigma=1.0, sl_sigma=1.0)
    sim = bakeoff.simulate_bracket(bars_with_prior, sig, cfg, hit_priority="sl_first")
    # On day 2: σ_log = 0.02 → σ_price = 100 * 0.02 = 2.0
    # TP at 100 + 2.0 = 102, SL at 100 - 2.0 = 98. High=104 (TP hit), low=99.5 (SL miss).
    row = sim.iloc[1]
    assert row["entered"] == 1
    assert row["exit_kind"] == "tp"
    assert row["exit_px"] == pytest.approx(102.0, abs=0.01)


def test_simulate_bracket_short_sl_hit(bakeoff):
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=2))
    bars = pd.DataFrame({
        "open": [99.5, 100.0],
        "high": [100.0, 104.0],  # spikes up → bad for short
        "low": [99.0, 99.5],
        "close": [99.7, 103.5],
        "vol_21": [0.02, 0.02],
        "adv_21d_shares": [1e6, 1e6],
        "daily_vol_bps": [200.0, 200.0],
    }, index=idx)
    sig = pd.Series([0, -1], index=idx, dtype=np.int8)
    cfg = bakeoff.TPSLConfig(tp_sigma=2.0, sl_sigma=1.0)
    sim = bakeoff.simulate_bracket(bars, sig, cfg, hit_priority="sl_first")
    # Short at 100. σ_price=2.0. TP @ 96, SL @ 102. high=104 → SL hit, low=99.5 → TP miss.
    row = sim.iloc[1]
    assert row["entered"] == 1
    assert row["exit_kind"] == "sl"
    assert row["gross_pct_ret"] == pytest.approx(-0.02, abs=1e-6)


def test_simulate_bracket_eod_close(bakeoff):
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=2))
    bars = pd.DataFrame({
        "open": [99.5, 100.0],
        "high": [100.0, 100.5],   # neither TP nor SL hit at 1σ
        "low": [99.0, 99.5],
        "close": [99.7, 100.3],
        "vol_21": [0.02, 0.02],
        "adv_21d_shares": [1e6, 1e6],
        "daily_vol_bps": [200.0, 200.0],
    }, index=idx)
    sig = pd.Series([0, 1], index=idx, dtype=np.int8)
    cfg = bakeoff.TPSLConfig(tp_sigma=1.0, sl_sigma=1.0)
    sim = bakeoff.simulate_bracket(bars, sig, cfg, hit_priority="sl_first")
    row = sim.iloc[1]
    assert row["entered"] == 1
    assert row["exit_kind"] == "eod"
    assert row["exit_px"] == pytest.approx(100.3, abs=1e-6)


def test_holm_adjust_basic(bakeoff):
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = bakeoff.holm_adjust(p)
    # Smallest p: 0.01 × 4 = 0.04. Adjusted values must be monotone in the
    # sorted ranking and bounded ≤ 1.
    assert all(0.0 <= v <= 1.0 for v in adj)
    # The largest unadjusted p (0.5) at rank 4 → multiplier 1 → stays at 0.5.
    assert adj[3] == pytest.approx(0.5, abs=1e-9)


def test_holm_adjust_monotonicity(bakeoff):
    rng = np.random.default_rng(5)
    p = rng.uniform(0.0, 1.0, size=20)
    adj = bakeoff.holm_adjust(p)
    # When sorted in the same order as the original p, adj should also be
    # weakly monotone.
    order = np.argsort(p)
    sorted_adj = adj[order]
    assert np.all(np.diff(sorted_adj) >= -1e-12)


def test_sharpe_to_pvalue_one_sided(bakeoff):
    p_zero = bakeoff.sharpe_to_pvalue(0.0, 252)
    assert p_zero == pytest.approx(0.5, abs=1e-6)
    p_pos = bakeoff.sharpe_to_pvalue(2.0, 252)
    assert p_pos < 0.05
    p_neg = bakeoff.sharpe_to_pvalue(-2.0, 252)
    assert p_neg > 0.95


def test_build_tpsl_grid_shape(bakeoff):
    grid = bakeoff.build_tpsl_grid()
    assert len(grid) == 4 * 3 + 1  # 12 + control
    # Last entry must be the no-bracket control.
    assert grid[-1].tp_sigma is None and grid[-1].sl_sigma is None


def test_build_candidate_list_has_rules_and_ml(bakeoff):
    cands = bakeoff.build_candidate_list(("asset_only", "asset_plus_market"))
    rule_names = {c.name for c in cands if c.family == "rule"}
    ml_names = {c.name for c in cands if c.family == "ml"}
    # Every rule must appear exactly once (asset_only only).
    for name, _ in bakeoff.RULE_CANDIDATES:
        assert name in rule_names
    # Every ML must appear under each scope.
    for name, _ in bakeoff.ML_CANDIDATES:
        assert name in ml_names
    n_ml_scopes = sum(1 for c in cands if c.family == "ml")
    assert n_ml_scopes == 2 * len(bakeoff.ML_CANDIDATES)
