"""Tests: labels respect winsorization, xs z-score is per-day."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from ml_research.labels import (
    cross_sectional_zscore,
    fit_winsorization_bounds,
    next_day_open_close_log_return,
    winsorize_train_fold,
)


def _toy_bars() -> pl.DataFrame:
    dates = [dt.date(2024, 1, i) for i in range(2, 12)]
    rows = []
    for sym in ("A", "B", "C"):
        for d in dates:
            rows.append((sym, d, 100.0, 101.0, 99.0, 100.5, 1_000_000))
    return pl.DataFrame(rows, schema=["symbol", "date", "open", "high", "low", "close", "volume"], orient="row")


def test_next_day_open_close_log_return_shifts_correctly():
    bars = _toy_bars()
    out = next_day_open_close_log_return(bars, label_col="y_raw")
    assert "y_raw" in out.columns
    # Last row per symbol must be null (no t+1 bar).
    last = out.sort(["symbol", "date"]).group_by("symbol").last()
    assert last["y_raw"].is_null().all()


def test_winsorization_bounds_clipped_in_5sigma():
    rng = np.random.default_rng(42)
    sample = rng.normal(0, 0.01, 10_000)
    sample[0] = 100.0  # outlier
    bounds = fit_winsorization_bounds(sample, sigma=5.0)
    df = pl.DataFrame({"y_raw": sample})
    out = winsorize_train_fold(df, bounds, label_col="y_raw", out_col="y_wins")
    assert out["y_wins"].max() <= bounds.upper + 1e-9
    assert out["y_wins"].min() >= bounds.lower - 1e-9
    assert bounds.upper < 100.0  # outlier was clipped


def test_xs_zscore_is_per_day_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    rows = []
    for d in [dt.date(2024, 1, i) for i in range(1, 6)]:
        for sym in ("A", "B", "C", "D", "E", "F"):
            rows.append((sym, d, float(rng.normal(0, 0.01))))
    df = pl.DataFrame(rows, schema=["symbol", "date", "y_wins"], orient="row")
    out = cross_sectional_zscore(df, label_col="y_wins", out_col="y_xs_z")
    for d in df["date"].unique():
        day = out.filter(pl.col("date") == d)
        z = day["y_xs_z"].to_numpy()
        z = z[np.isfinite(z)]
        assert abs(z.mean()) < 1e-9
        assert abs(z.std(ddof=0) - 1.0) < 1e-9
