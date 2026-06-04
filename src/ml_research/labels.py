"""ml_research.labels — next-day open-to-close log return, train-fold winsorized + xs z-scored.

Spec (CONTEXT line 71):
    r_{t+1} = log(close_{t+1} / open_{t+1})
    → winsorize at ±5σ on the TRAIN fold's distribution
    → daily cross-sectional z-score

Critical: winsorization bounds and (if used) z-score statistics are fit on
the TRAIN fold ONLY (CONTEXT line 73 — "biggest leakage risk: cross-sectional
z-score contamination"). The z-score itself is computed per-day so it's
unaffected by the train/test boundary, but the winsorization sigma absolutely
is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class WinsorizationBounds:
    """Train-fold-fit winsorization parameters. Persisted with the model artifact."""

    mean: float
    std: float
    sigma: float
    lower: float
    upper: float


def next_day_open_close_log_return(
    bars_df: pl.DataFrame, *, label_col: str = "y_raw"
) -> pl.DataFrame:
    """Compute r_{t+1} = log(close_{t+1} / open_{t+1}) per symbol.

    The label is attached to row t (the as-of date), so a feature row dated t
    joins a label that uses bars from t+1. There is no peek: at as-of t the
    label is computed from t+1's bar, which is exactly what we'd be predicting.

    Output: input columns + `label_col` (next-day open-to-close log return).
    """
    df = bars_df.sort(["symbol", "date"])
    return df.with_columns(
        [
            (
                pl.col("close").shift(-1).over("symbol")
                / pl.col("open").shift(-1).over("symbol")
            )
            .log()
            .alias(label_col)
        ]
    )


def fit_winsorization_bounds(
    train_labels: pl.Series | np.ndarray, *, sigma: float = 5.0
) -> WinsorizationBounds:
    """Fit ±`sigma`σ winsorization on the TRAIN fold only."""
    arr = np.asarray(train_labels.to_numpy() if isinstance(train_labels, pl.Series) else train_labels, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("fit_winsorization_bounds: empty / all-NaN train labels")
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=0))
    return WinsorizationBounds(
        mean=mu,
        std=sd,
        sigma=sigma,
        lower=mu - sigma * sd,
        upper=mu + sigma * sd,
    )


def winsorize_train_fold(
    df: pl.DataFrame,
    bounds: WinsorizationBounds,
    *,
    label_col: str = "y_raw",
    out_col: str = "y_wins",
) -> pl.DataFrame:
    """Apply train-fold-fit ±5σ winsorization to a label column."""
    return df.with_columns(
        pl.col(label_col).clip(lower_bound=bounds.lower, upper_bound=bounds.upper).alias(out_col)
    )


def cross_sectional_zscore(
    df: pl.DataFrame,
    *,
    label_col: str = "y_wins",
    out_col: str = "y_xs_z",
    date_col: str = "date",
) -> pl.DataFrame:
    """Per-day cross-sectional z-score of the label column.

    Computed PER DAY so the operation does not leak across the train/test
    boundary in time. Each date's mean/std uses only that date's universe.
    """
    return df.with_columns(
        (
            (pl.col(label_col) - pl.col(label_col).mean().over(date_col))
            / pl.col(label_col).std(ddof=0).over(date_col)
        ).alias(out_col)
    )


def build_labels(
    bars_df: pl.DataFrame,
    train_dates: tuple,  # (start_date, end_date) inclusive — for bounds fitting
    *,
    sigma: float = 5.0,
) -> tuple[pl.DataFrame, WinsorizationBounds]:
    """End-to-end label pipeline: raw → winsorized (train-fold-fit) → xs z-score.

    Returns (df_with_labels, bounds). df has columns y_raw, y_wins, y_xs_z.
    """
    df = next_day_open_close_log_return(bars_df, label_col="y_raw")
    train_mask = (pl.col("date") >= train_dates[0]) & (pl.col("date") <= train_dates[1])
    train_labels = df.filter(train_mask)["y_raw"].drop_nulls().drop_nans()
    bounds = fit_winsorization_bounds(train_labels, sigma=sigma)
    df = winsorize_train_fold(df, bounds, label_col="y_raw", out_col="y_wins")
    df = cross_sectional_zscore(df, label_col="y_wins", out_col="y_xs_z")
    return df, bounds
