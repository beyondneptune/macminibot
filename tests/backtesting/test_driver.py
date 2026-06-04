"""End-to-end WalkForwardDriver tests on synthetic daily bars."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtesting.driver import (
    ExpandingWindowFolder,
    FoldSpec,
    WalkForwardDriver,
)


def _make_synthetic_bars(
    n_days: int = 100,
    symbols: tuple[str, ...] = ("AAPL", "MSFT"),
    seed: int = 0xB10CC0DE,
) -> pd.DataFrame:
    """Geometric Brownian-ish daily bars with consistent ADV + vol_bps."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-02", periods=n_days, freq="B", tz="UTC")
    rows = []
    for sym in symbols:
        # Tiny positive drift so always-long has some signal.
        drift = 0.0005
        sigma = 0.01
        rets = rng.normal(drift, sigma, size=n_days)
        close = 100.0 * np.cumprod(1.0 + rets)
        open_ = np.concatenate([[100.0], close[:-1] * (1.0 + rng.normal(0, 0.002, n_days - 1))])
        for d, o, c in zip(dates, open_, close):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": float(o),
                    "close": float(c),
                    "volume": 1_000_000,
                    "adv_21d_shares": 50_000_000,  # very liquid → floor will bind
                    "daily_vol_bps": 100.0,
                }
            )
    df = pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
    return df


def always_long(bar: pd.Series, train_window: pd.DataFrame) -> dict[str, float]:
    """Constant equal-weight long signal across the bar's symbols."""
    if len(bar) == 0:
        return {}
    w = 1.0 / len(bar)
    return {sym: w for sym in bar.index}


def test_expanding_window_folder_yields_canonical_foldspec() -> None:
    """Splitter yields shared_core.contracts.FoldSpec with the cycle-3-
    ratified field order: (fold_id, train_idx, val_idx, test_idx, fold_meta)
    and ``np.ndarray`` index fields."""
    dates = pd.date_range("2025-01-02", periods=100, freq="B")
    folder = ExpandingWindowFolder(train_min_days=30, test_days=10, embargo_days=2, purge_days=1)
    folds = list(folder.split(dates))
    assert len(folds) > 0
    fs = folds[0]
    assert fs.fold_id == 0
    assert isinstance(fs.train_idx, np.ndarray)
    assert isinstance(fs.val_idx, np.ndarray)
    assert isinstance(fs.test_idx, np.ndarray)
    assert fs.train_idx.ndim == 1
    assert fs.test_idx.ndim == 1
    assert isinstance(fs.fold_meta, dict)
    assert {"embargo_days", "purge_days"} <= set(fs.fold_meta)


def test_expanding_window_folder_no_train_test_overlap() -> None:
    dates = pd.date_range("2025-01-02", periods=100, freq="B")
    folder = ExpandingWindowFolder(train_min_days=20, test_days=10, embargo_days=2, purge_days=1)
    for fold in folder.split(dates):
        if fold.train_idx.size > 0:
            assert fold.train_idx.max() < fold.test_idx.min()


def test_driver_validates_bars_multiindex() -> None:
    bad = pd.DataFrame({"open": [1, 2], "close": [1, 2]})
    with pytest.raises(ValueError, match="MultiIndex"):
        WalkForwardDriver(bars=bad, signal_generator=always_long)


def test_driver_validates_required_columns() -> None:
    bars = _make_synthetic_bars(n_days=50)
    bars = bars.drop(columns=["adv_21d_shares"])
    with pytest.raises(ValueError, match="adv_21d_shares"):
        WalkForwardDriver(bars=bars, signal_generator=always_long)


def test_driver_end_to_end_always_long() -> None:
    bars = _make_synthetic_bars(n_days=100, symbols=("AAPL", "MSFT", "SPY"))
    driver = WalkForwardDriver(
        bars=bars,
        signal_generator=always_long,
        splitter=ExpandingWindowFolder(
            train_min_days=30, test_days=14, embargo_days=2, purge_days=1
        ),
        asset_class="equity",
        initial_capital_usd=100_000.0,
    )
    result = driver.run()
    assert len(result.fold_metrics) >= 1
    # Every fold must have a finite (or 0-obs) fee total.
    for fm in result.fold_metrics:
        assert math.isfinite(fm.total_fees_usd)
        assert fm.total_fees_usd >= 0
        assert fm.total_slippage_usd >= 0
        assert fm.turnover >= 0
    assert result.equity_curve.shape[0] > 0
    assert result.equity_curve[0] > 0


def test_driver_runs_with_flat_signal() -> None:
    """A flat (no-position) generator → zero fees, zero PnL."""
    bars = _make_synthetic_bars(n_days=60, symbols=("AAPL",))

    def flat(bar: pd.Series, train_window: pd.DataFrame) -> dict[str, float]:
        return {}

    driver = WalkForwardDriver(
        bars=bars,
        signal_generator=flat,
        splitter=ExpandingWindowFolder(train_min_days=20, test_days=10),
    )
    result = driver.run()
    for fm in result.fold_metrics:
        assert fm.total_fees_usd == 0.0
        assert fm.total_slippage_usd == 0.0
        # No trades → flat returns → either NaN sharpe (0-obs) or 0.0.
        assert fm.turnover == 0.0


def test_driver_broker_rejects_unknown() -> None:
    bars = _make_synthetic_bars(n_days=60, symbols=("AAPL",))
    with pytest.raises(ValueError, match="unknown broker"):
        WalkForwardDriver(bars=bars, signal_generator=always_long, broker="schwab")


def test_driver_broker_alpaca_rejects_non_equity() -> None:
    """Alpaca commission-free tier is equity-only (cycle-5 CONTEXT)."""
    bars = _make_synthetic_bars(n_days=60, symbols=("AAPL",))
    with pytest.raises(ValueError, match="alpaca"):
        WalkForwardDriver(
            bars=bars, signal_generator=always_long,
            asset_class="future", broker="alpaca",
        )


def test_driver_alpaca_has_lower_fee_drag_than_ibkr() -> None:
    """Cycle-5 'run twice' architecture: Alpaca commission-free must charge
    strictly less than IBKR Pro Tiered on identical fills, surfacing the
    'broker subsidy' the CONTEXT reports as a first-class HTML line."""
    bars = _make_synthetic_bars(n_days=80, symbols=("AAPL", "MSFT"))
    splitter = ExpandingWindowFolder(
        train_min_days=30, test_days=10, embargo_days=2, purge_days=1
    )

    ibkr_total = sum(
        fm.total_fees_usd
        for fm in WalkForwardDriver(
            bars=bars, signal_generator=always_long, splitter=splitter,
            broker="ibkr_pro",
        ).run().fold_metrics
    )
    alpaca_total = sum(
        fm.total_fees_usd
        for fm in WalkForwardDriver(
            bars=bars, signal_generator=always_long, splitter=splitter,
            broker="alpaca",
        ).run().fold_metrics
    )
    # Both non-zero (there are trades), and Alpaca strictly cheaper.
    assert ibkr_total > 0
    assert alpaca_total >= 0
    assert alpaca_total < ibkr_total, (
        f"broker subsidy missing: ibkr={ibkr_total:.4f} alpaca={alpaca_total:.4f}"
    )
