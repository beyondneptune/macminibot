"""Tests for regime tagging + driver integration (Epic 17, cycle 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.driver import RegimeMetrics, WalkForwardDriver
from backtesting.regime import REGIME_LABELS, RegimeTagger, tag_regime


def _make_synthetic_vix(n_days: int = 600, seed: int = 0xB10CC0DE) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    # Geometric-ish vol proxy in [0, 1] range — enough spread to populate
    # 20th and 80th rolling quintiles cleanly.
    vals = np.abs(rng.normal(0.03, 0.015, size=n_days))
    return pd.Series(vals, index=dates, name="vix_proxy")


def test_tag_regime_returns_valid_labels() -> None:
    vix = _make_synthetic_vix()
    tagger = RegimeTagger.from_series(vix)
    # Tag a warm-cutoff date (well past the 252-day warmup).
    d = vix.index[400]
    label = tag_regime(d, float(vix.loc[d]), tagger=tagger)
    assert label in REGIME_LABELS


def test_tag_regime_buckets_match_quintile_definition() -> None:
    vix = _make_synthetic_vix()
    tagger = RegimeTagger.from_series(vix)
    d = vix.index[400]
    # Synthetic high reading → must land in vix_high.
    assert tag_regime(d, 1.0, tagger=tagger) == "vix_high"
    # Synthetic near-zero reading → vix_low.
    assert tag_regime(d, 1e-6, tagger=tagger) == "vix_low"


def test_tag_regime_defaults_mid_during_warmup() -> None:
    vix = _make_synthetic_vix()
    tagger = RegimeTagger.from_series(vix)
    # First date — no rolling window history → cutoffs NaN → defaults mid.
    d = vix.index[0]
    assert tag_regime(d, float(vix.iloc[0]), tagger=tagger) == "vix_mid"


def test_driver_emits_regime_metrics_when_tagger_provided() -> None:
    # Build synthetic bars over the same dates as a warm vix series.
    vix = _make_synthetic_vix(n_days=600)
    tagger = RegimeTagger.from_series(vix)

    # Use only the last 300 days (post-warmup) for the bar panel.
    bar_dates = vix.index[300:]
    rng = np.random.default_rng(42)
    rows = []
    for sym in ("AAPL", "MSFT"):
        rets = rng.normal(0.0005, 0.01, size=len(bar_dates))
        close = 100.0 * np.cumprod(1.0 + rets)
        open_ = np.concatenate([[100.0], close[:-1]])
        for d, o, c in zip(bar_dates, open_, close):
            rows.append({
                "date": d, "symbol": sym, "open": float(o), "close": float(c),
                "volume": 1_000_000, "adv_21d_shares": 50_000_000,
                "daily_vol_bps": 100.0,
            })
    bars = pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()

    def always_long(bar: pd.Series, _tw: pd.DataFrame) -> dict[str, float]:
        return {s: 1.0 / len(bar.index) for s in bar.index}

    driver = WalkForwardDriver(
        bars=bars,
        signal_generator=always_long,
        asset_class="equity",
        regime_tagger=tagger,
        vix_proxy=vix,
    )
    result = driver.run()
    # Three buckets, each is a RegimeMetrics.
    assert len(result.regime_metrics) == 3
    assert {rm.regime for rm in result.regime_metrics} == set(REGIME_LABELS)
    for rm in result.regime_metrics:
        assert isinstance(rm, RegimeMetrics)


def test_driver_rejects_one_sided_tagger_arg() -> None:
    bars = pd.DataFrame(
        {"open": [1.0], "close": [1.0], "volume": [1],
         "adv_21d_shares": [1], "daily_vol_bps": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-02"), "AAPL")], names=["date", "symbol"]
        ),
    )
    def sg(bar, tw):  # noqa: ANN001
        return {}
    import pytest
    with pytest.raises(ValueError, match="both be provided or both omitted"):
        WalkForwardDriver(
            bars=bars, signal_generator=sg,
            regime_tagger=RegimeTagger.from_series(_make_synthetic_vix()),
            vix_proxy=None,
        )
