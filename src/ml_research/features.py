"""ml_research.features — canonical feature implementations.

Per CONTEXT.md the feature module is imported by BOTH training and
`strategy-engine` runtime. Duplicating feature code is the cardinal sin
(training-serving skew). Every function here is a pure function over a
(symbol, date)-keyed Polars DataFrame so it can be reused at inference
unchanged.

This module implements the daily-bar-computable subset of the 13-element
v1 feature spec:
    ret_1d, ret_5d, ret_21d, ret_252d_skip_21,
    realized_vol_21d, parkinson_vol_21d,
    cross_sectional_rank_mom, rel_strength_vs_spy

The dollar-bar-specific features (db_duration_mean_21d,
db_signed_vol_imb_21d, db_kyle_lambda_21d, gap_overnight) are deferred
until the dollar-bar substrate has ≥6 months of history. `vix_proxy`
needs the SPY/QQQ/IWM ETF universe and is broadcast across symbols.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import polars as pl

# Public feature list (v1 daily-bar subset). The CONTEXT-locked dollar-bar
# features are listed in DEFERRED_FEATURE_NAMES so a future PR adds them
# without renaming or reordering this tuple.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_252d_skip_21",
    "realized_vol_21d",
    "parkinson_vol_21d",
    "cross_sectional_rank_mom",
    "rel_strength_vs_spy",
    "vix_proxy",
)

# Cycle-7 candidate: cross-asset spread features (CONTEXT improvement #12,
# family a). Each is a *spread* between this symbol's value and the mean of
# the same metric across the benchmark ETF basket {SPY, QQQ, IWM}. Ridge-
# friendly (linear-additive lift) and orthogonal to the locked 9; trees
# would need a split per spread component. Reference: Asness/Moskowitz/
# Pedersen 2013, "Value and Momentum Everywhere" (JF) — cross-asset spreads
# carry independent risk premia. NOT promoted to FEATURE_NAMES until the
# bake-off in scripts/11_train_with_xs_spreads.py demonstrates lift per
# CONTEXT #12 promotion rule (≥+0.005 IC on ≥4/6 folds, shuffle-canary
# pass, Sharpe ≥ +0.452).
CANDIDATE_XS_SPREAD_FEATURES: Final[tuple[str, ...]] = (
    "ret_21d_xs_spread",          # ret_21d(symbol) - mean(ret_21d of {SPY,QQQ,IWM})
    "ret_5d_xs_spread",           # ret_5d(symbol)  - mean(ret_5d of {SPY,QQQ,IWM})
    "realized_vol_21d_xs_spread", # vol_21d(symbol) - mean(vol_21d of {SPY,QQQ,IWM})
)

# Cycle-8 candidate: VIX-proxy tercile one-hot encoding + momentum interactions
# (CONTEXT improvement #12, family b). The proxy is the rolling 21d xs vol of
# {SPY, QQQ, IWM} (same series as `vix_proxy`). Each trading day is bucketed
# by the 33rd / 67th percentile of the trailing 252d distribution of that
# series. Three 1-hot columns and three ret_21d × tercile interactions let
# Ridge learn vol-regime-conditional momentum without sacrificing linearity.
# Reference: Moreira & Muir 2017 ("Volatility-Managed Portfolios", JF) —
# returns conditional on vol-regime carry independent expected-return signal.
# NOT promoted to FEATURE_NAMES until the bake-off below meets the CONTEXT
# #12 promotion rule.
CANDIDATE_VIX_TERCILE_FEATURES: Final[tuple[str, ...]] = (
    "vix_tercile_low",
    "vix_tercile_mid",
    "vix_tercile_high",
    "ret_21d_x_vix_low",
    "ret_21d_x_vix_mid",
    "ret_21d_x_vix_high",
)

# Cycle-9 candidate: lagged interaction features (CONTEXT improvement #12,
# family c). Lo/MacKinlay 1990 ("When are contrarian profits due to stock
# market overreaction?") showed that cross-products of lagged returns &
# vol-conditional returns expose autocovariance structure ridge cannot reach
# through main effects alone. Three new ridge-friendly columns:
#   ret_5d_x_ret_21d     : short × medium momentum interaction
#   vol_21d_x_ret_5d     : vol-conditional short momentum
#   vol_21d_x_ret_21d    : vol-conditional medium momentum
# All inputs are already in FEATURE_NAMES, so no new data dependencies.
# NOT promoted to FEATURE_NAMES until the bake-off in
# scripts/14_train_with_lagged_interactions.py meets the CONTEXT #12 rule.
CANDIDATE_LAGGED_INTERACTION_FEATURES: Final[tuple[str, ...]] = (
    "ret_5d_x_ret_21d",
    "vol_21d_x_ret_5d",
    "vol_21d_x_ret_21d",
)

DEFERRED_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "gap_overnight",
    "db_duration_mean_21d",
    "db_signed_vol_imb_21d",
    "db_kyle_lambda_21d",
)

# Reference symbol for relative-strength feature. The relative-strength
# benchmark is SPY by spec (CONTEXT line 68: `rel_strength_vs_spy`).
REL_STRENGTH_BENCHMARK: Final[str] = "SPY"

# Symbol set used to compute the vix_proxy broadcast feature (CONTEXT
# line 66: "cross-sectional realized vol of {SPY, QQQ, IWM} over 21d").
VIX_PROXY_SYMBOLS: Final[tuple[str, ...]] = ("SPY", "QQQ", "IWM")


def _require_columns(df: pl.DataFrame, required: tuple[str, ...]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"features: missing required columns {missing}")


def _log_return(close: pl.Expr, periods: int) -> pl.Expr:
    """log(close_t / close_{t-periods}), grouped per symbol externally."""
    return (close / close.shift(periods)).log()


def _ret_skip(close: pl.Expr, total: int, skip: int) -> pl.Expr:
    """12-1 momentum: ret over `total` periods, skipping the last `skip`.

    Jegadeesh & Titman 1993 (JF). Computed as log(close_{t-skip} / close_{t-total}).
    """
    return (close.shift(skip) / close.shift(total)).log()


def _realized_vol(close: pl.Expr, window: int) -> pl.Expr:
    """Annualized close-to-close realized vol over `window` trading days."""
    log_ret = (close / close.shift(1)).log()
    # Population std (ddof=0) — large sample, matches numpy.std default.
    return log_ret.rolling_std(window_size=window, ddof=0) * math.sqrt(252.0)


def _parkinson_vol(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    """Parkinson 1980 high-low estimator over `window` days, annualized.

    σ_P = sqrt( 1/(4*ln 2) * mean_t (ln(H_t/L_t))^2 ) * sqrt(252)
    """
    rng = (high / low).log()
    rng_sq = rng * rng
    mean_sq = rng_sq.rolling_mean(window_size=window)
    coef = 1.0 / (4.0 * math.log(2.0))
    return (mean_sq * coef).sqrt() * math.sqrt(252.0)


def _per_symbol_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-symbol time-series features. Input must be sorted by (symbol, date)."""
    _require_columns(df, ("symbol", "date", "open", "high", "low", "close"))
    return df.with_columns(
        [
            _log_return(pl.col("close"), 1).over("symbol").alias("ret_1d"),
            _log_return(pl.col("close"), 5).over("symbol").alias("ret_5d"),
            _log_return(pl.col("close"), 21).over("symbol").alias("ret_21d"),
            _ret_skip(pl.col("close"), total=252, skip=21)
            .over("symbol")
            .alias("ret_252d_skip_21"),
            _realized_vol(pl.col("close"), 21)
            .over("symbol")
            .alias("realized_vol_21d"),
            _parkinson_vol(pl.col("high"), pl.col("low"), 21)
            .over("symbol")
            .alias("parkinson_vol_21d"),
        ]
    )


def _cross_sectional_features(
    df: pl.DataFrame, benchmark_symbol: str
) -> pl.DataFrame:
    """Daily cross-sectional features: xs rank-of-momentum + rel-strength-vs-benchmark.

    Both depend on `ret_21d` already being computed.
    """
    # XS rank: rank within each date, normalized to [0, 1]. Method "average"
    # (tie-aware) prevents jitter when prices tie at decimal granularity.
    df = df.with_columns(
        pl.col("ret_21d")
        .rank(method="average")
        .over("date")
        .alias("_xs_rank_raw")
    )
    df = df.with_columns(
        (
            (pl.col("_xs_rank_raw") - 1.0)
            / (pl.col("ret_21d").count().over("date") - 1.0).clip(lower_bound=1.0)
        ).alias("cross_sectional_rank_mom")
    ).drop("_xs_rank_raw")

    # Relative strength: subtract benchmark's ret_21d (broadcast by date).
    bench = (
        df.filter(pl.col("symbol") == benchmark_symbol)
        .select(["date", pl.col("ret_21d").alias("_bench_ret_21d")])
    )
    df = df.join(bench, on="date", how="left").with_columns(
        (pl.col("ret_21d") - pl.col("_bench_ret_21d")).alias("rel_strength_vs_spy")
    ).drop("_bench_ret_21d")
    return df


def _vix_proxy(
    df: pl.DataFrame, proxy_symbols: tuple[str, ...] = VIX_PROXY_SYMBOLS
) -> pl.DataFrame:
    """vix_proxy — xs std of 21d realized vol across {SPY, QQQ, IWM}.

    Broadcast across all rows on a given date (every symbol sees the same
    market-wide vol-of-vol number for that day).
    """
    available = [s for s in proxy_symbols if s in df["symbol"].unique().to_list()]
    if not available:
        # Universe lacks proxy ETFs — emit zeros rather than NaN so downstream
        # clipping doesn't drop every row. Caller can detect & log.
        return df.with_columns(pl.lit(0.0).alias("vix_proxy"))
    proxy_rv = (
        df.filter(pl.col("symbol").is_in(available))
        .group_by("date")
        .agg(pl.col("realized_vol_21d").std(ddof=0).alias("vix_proxy"))
    )
    return df.join(proxy_rv, on="date", how="left")


def _xs_spread_features(
    df: pl.DataFrame,
    *,
    basket_symbols: tuple[str, ...] = VIX_PROXY_SYMBOLS,
) -> pl.DataFrame:
    """Cross-asset spread features (cycle-7 candidate; CONTEXT improvement #12a).

    For each (symbol, date) row, computes the spread between the symbol's
    value and the mean of the same metric across the benchmark basket.
    Depends on `ret_5d`, `ret_21d`, `realized_vol_21d` already being computed.
    """
    available = [s for s in basket_symbols if s in df["symbol"].unique().to_list()]
    if not available:
        return df.with_columns(
            [
                pl.lit(0.0).alias("ret_21d_xs_spread"),
                pl.lit(0.0).alias("ret_5d_xs_spread"),
                pl.lit(0.0).alias("realized_vol_21d_xs_spread"),
            ]
        )
    basket_means = (
        df.filter(pl.col("symbol").is_in(available))
        .group_by("date")
        .agg(
            [
                pl.col("ret_21d").mean().alias("_basket_ret_21d"),
                pl.col("ret_5d").mean().alias("_basket_ret_5d"),
                pl.col("realized_vol_21d").mean().alias("_basket_rv_21d"),
            ]
        )
    )
    df = df.join(basket_means, on="date", how="left").with_columns(
        [
            (pl.col("ret_21d") - pl.col("_basket_ret_21d")).alias("ret_21d_xs_spread"),
            (pl.col("ret_5d") - pl.col("_basket_ret_5d")).alias("ret_5d_xs_spread"),
            (pl.col("realized_vol_21d") - pl.col("_basket_rv_21d")).alias("realized_vol_21d_xs_spread"),
        ]
    ).drop(["_basket_ret_21d", "_basket_ret_5d", "_basket_rv_21d"])
    return df


def _vix_tercile_features(
    df: pl.DataFrame,
    *,
    lookback_days: int = 252,
) -> pl.DataFrame:
    """vix_tercile_{low,mid,high} 1-hot + ret_21d × tercile interactions.

    Bucketing is causal: the 33rd / 67th percentiles are computed over the
    rolling `lookback_days` trailing window of the `vix_proxy` series. The
    buckets are then broadcast across all (symbol, date) rows for that date
    and one-hot encoded. ret_21d × tercile interactions multiply the (already
    computed) per-symbol `ret_21d` with each one-hot column.

    Depends on `vix_proxy` and `ret_21d` already being present.
    """
    _require_columns(df, ("date", "vix_proxy", "ret_21d"))
    # Build a *date-level* series from vix_proxy (which is broadcast across
    # symbols on each date, so first() == any()).
    date_series = (
        df.group_by("date")
        .agg(pl.col("vix_proxy").first().alias("vix_proxy"))
        .sort("date")
    )
    # Trailing percentiles — use shift(1) so today's bucket uses ONLY history
    # ending yesterday (no leakage of today's vix value into today's bucket).
    date_series = date_series.with_columns(
        [
            pl.col("vix_proxy")
            .shift(1)
            .rolling_quantile(
                quantile=0.3333,
                window_size=lookback_days,
                min_samples=lookback_days // 2,
                interpolation="linear",
            )
            .alias("_vix_p33"),
            pl.col("vix_proxy")
            .shift(1)
            .rolling_quantile(
                quantile=0.6667,
                window_size=lookback_days,
                min_samples=lookback_days // 2,
                interpolation="linear",
            )
            .alias("_vix_p67"),
        ]
    )
    date_series = date_series.with_columns(
        [
            (pl.col("vix_proxy") <= pl.col("_vix_p33"))
            .cast(pl.Float64)
            .alias("vix_tercile_low"),
            (
                (pl.col("vix_proxy") > pl.col("_vix_p33"))
                & (pl.col("vix_proxy") <= pl.col("_vix_p67"))
            )
            .cast(pl.Float64)
            .alias("vix_tercile_mid"),
            (pl.col("vix_proxy") > pl.col("_vix_p67"))
            .cast(pl.Float64)
            .alias("vix_tercile_high"),
        ]
    ).select(["date", "vix_tercile_low", "vix_tercile_mid", "vix_tercile_high"])

    df = df.join(date_series, on="date", how="left")
    # Interactions.
    df = df.with_columns(
        [
            (pl.col("ret_21d") * pl.col("vix_tercile_low")).alias("ret_21d_x_vix_low"),
            (pl.col("ret_21d") * pl.col("vix_tercile_mid")).alias("ret_21d_x_vix_mid"),
            (pl.col("ret_21d") * pl.col("vix_tercile_high")).alias("ret_21d_x_vix_high"),
        ]
    )
    return df


def _lagged_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Lagged interaction features (cycle-9 candidate; CONTEXT improvement #12c).

    Pure cross-products of already-computed per-symbol features:
        ret_5d_x_ret_21d  = ret_5d * ret_21d
        vol_21d_x_ret_5d  = realized_vol_21d * ret_5d
        vol_21d_x_ret_21d = realized_vol_21d * ret_21d

    All inputs are causal (rolling windows ending today), so the products
    inherit causality. No look-ahead.
    """
    _require_columns(df, ("ret_5d", "ret_21d", "realized_vol_21d"))
    return df.with_columns(
        [
            (pl.col("ret_5d") * pl.col("ret_21d")).alias("ret_5d_x_ret_21d"),
            (pl.col("realized_vol_21d") * pl.col("ret_5d")).alias("vol_21d_x_ret_5d"),
            (pl.col("realized_vol_21d") * pl.col("ret_21d")).alias("vol_21d_x_ret_21d"),
        ]
    )


def compute_features(
    bars_df: pl.DataFrame,
    *,
    benchmark_symbol: str = REL_STRENGTH_BENCHMARK,
    proxy_symbols: tuple[str, ...] = VIX_PROXY_SYMBOLS,
    include_xs_spreads: bool = False,
    include_vix_terciles: bool = False,
    include_lagged_interactions: bool = False,
) -> pl.DataFrame:
    """Compute the v1 daily-bar feature set.

    Parameters
    ----------
    bars_df : pl.DataFrame
        Schema: symbol (str), date (datetime/date), open, high, low, close (f64).
        Must contain at least one row per (symbol, date). Multi-symbol panel
        layout (long-form), not wide.
    benchmark_symbol : str
        Symbol used as the rel_strength_vs_spy denominator.
    proxy_symbols : tuple[str, ...]
        Symbols whose realized vols are used to compute vix_proxy.

    Returns
    -------
    pl.DataFrame
        Original columns + the FEATURE_NAMES columns. Rows where any feature
        is NaN (typically the first 252 trading days per symbol) are NOT
        dropped here — that is the caller's responsibility (drop after the
        join with labels so train/inference apply the same rule).
    """
    _require_columns(bars_df, ("symbol", "date", "open", "high", "low", "close"))
    df = bars_df.sort(["symbol", "date"])
    df = _per_symbol_features(df)
    df = _cross_sectional_features(df, benchmark_symbol)
    df = _vix_proxy(df, proxy_symbols)
    if include_xs_spreads:
        df = _xs_spread_features(df, basket_symbols=proxy_symbols)
    if include_vix_terciles:
        df = _vix_tercile_features(df)
    if include_lagged_interactions:
        df = _lagged_interaction_features(df)

    # Reorder so feature columns appear in canonical FEATURE_NAMES order.
    all_features = (
        FEATURE_NAMES
        + (CANDIDATE_XS_SPREAD_FEATURES if include_xs_spreads else ())
        + (CANDIDATE_VIX_TERCILE_FEATURES if include_vix_terciles else ())
        + (CANDIDATE_LAGGED_INTERACTION_FEATURES if include_lagged_interactions else ())
    )
    keep_meta = [c for c in df.columns if c not in all_features]
    ordered = [c for c in all_features if c in df.columns]
    return df.select(keep_meta + ordered)


def clip_features_train_fold(
    train_df: pl.DataFrame,
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> dict[str, tuple[float, float]]:
    """Fit per-feature clipping bounds on the train fold (CONTEXT cycle-4 improvement 7).

    Returns a dict mapping feature_name → (lower, upper) for `apply_clip`.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for name in feature_names:
        if name not in train_df.columns:
            continue
        col = train_df[name].drop_nulls()
        if col.len() == 0:
            bounds[name] = (float("-inf"), float("inf"))
            continue
        lo = float(col.quantile(lower_quantile, interpolation="linear") or 0.0)
        hi = float(col.quantile(upper_quantile, interpolation="linear") or 0.0)
        bounds[name] = (lo, hi)
    return bounds


def apply_feature_clip(
    df: pl.DataFrame, bounds: dict[str, tuple[float, float]]
) -> pl.DataFrame:
    """Apply train-fold clipping bounds in-place (returns new DF, no mutation)."""
    exprs = []
    for name, (lo, hi) in bounds.items():
        if name in df.columns:
            # NaN/inf rows are NOT clipped (NaN propagates) — drop policy is
            # handled at the row level by the caller per CONTEXT line 104.
            exprs.append(pl.col(name).clip(lower_bound=lo, upper_bound=hi).alias(name))
    return df.with_columns(exprs) if exprs else df


def drop_feature_failures(
    df: pl.DataFrame, feature_names: tuple[str, ...] = FEATURE_NAMES
) -> pl.DataFrame:
    """Drop rows where ANY feature is NaN or +/-inf (per CONTEXT line 104).

    Never silently imputes — phantom information. `predict()` would simply
    emit no Signal for those (symbol, date) pairs.
    """
    cols = [c for c in feature_names if c in df.columns]
    if not cols:
        return df
    # is_finite covers NaN and ±inf for floats.
    mask = pl.all_horizontal([pl.col(c).is_finite() for c in cols])
    return df.filter(mask)


# Convenience for tests/tooling that prefer numpy.
def features_to_numpy(df: pl.DataFrame, feature_names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    cols = [c for c in feature_names if c in df.columns]
    if not cols:
        raise ValueError("no feature columns present in DataFrame")
    return df.select(cols).to_numpy().astype(np.float64, copy=False)
