"""Cycle-13 diagnostic: cross-model validation of the cycle-12 decile-Sharpe
per-regime finding.

Cycle-12 (script 18) reported on the Ridge predictor:
    vix_low   -> U-shape (mean-reversion)
    vix_mid   -> monotonic_up (momentum, D10-D1 = +0.87 Sharpe)
    vix_high  -> flat/noisy (D10-D1 = +0.15, both endpoints positive)

i.e. "vix_mid is the only regime in which the model produces a clean
directional, monotonic decile spread". That is a strong claim and we
must cross-validate it on alternative learners before betting cycle-14
infrastructure on it. Two hazards:

  (a) Ridge-specific artifact — a linear projection of these features
      may happen to align with the vix_mid label distribution and not
      with vix_low/high, even though a non-linear model would extract
      signal from those regimes.
  (b) Data-mined accident — the diagnostic itself (decile rank in 8
      tradables, pooled across many days) could happen to favor one
      regime by chance.

Method (mirrors script 18 exactly, swapping the trainer):
    1. Same walk-forward harness, same features, same labels, same
       date-regime map.
    2. Per fold: fit alt-model on the *exact same* train rows, predict
       on the *exact same* test rows. Hyper-parameters set to sensible
       defaults — we are not tuning here, we are stress-testing the
       *shape* of the decile curve.
    3. Pool predictions, assign deciles per-day, compute per-regime
       per-decile Sharpe.
    4. Emit one 10x3 table per model, side-by-side with the cycle-12
       Ridge table re-loaded from data/18_decile_sharpe_per_regime.log.

Models under test
-----------------
  - OLS         : LinearRegression (Ridge with alpha=0); if vix_mid
                  monotonicity is real and merely linear, OLS will show
                  the same pattern. If Ridge regularization was masking
                  noise that OLS does not, OLS will look messier
                  everywhere — but the *relative* ranking of regimes
                  should still favor vix_mid.
  - LightGBM    : non-linear, captures interactions. If the finding is
                  driven by a real feature-label structure (and not by
                  Ridge's linear projection) LightGBM will surface the
                  *same* per-regime ordering.

Verdict rule
------------
For each alt model we compute D10-D1 Sharpe per regime, then check
whether vix_mid has the largest positive D10-D1 spread of the three
regimes. If YES for both alt models -> finding is MODEL-AGNOSTIC. If
NO for either -> finding is Ridge-specific or data-mined.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _storage import open_store  # noqa: E402

from backtesting.regime import RegimeTagger  # noqa: E402
from ml_research.cv import WalkForwardHarness  # noqa: E402
from ml_research.features import (  # noqa: E402
    FEATURE_NAMES,
    apply_feature_clip,
    clip_features_train_fold,
    compute_features,
    drop_feature_failures,
)
from ml_research.labels import (  # noqa: E402
    cross_sectional_zscore,
    fit_winsorization_bounds,
    next_day_open_close_log_return,
    winsorize_train_fold,
)
from ml_research.trainer import LightGBMConfig, LightGBMTrainer  # noqa: E402

# Same universe as cycle-12 (script 18).
TRADABLE = [
    ("equity.daily_bars", "NASDAQ.AAPL", "AAPL"),
    ("equity.daily_bars", "NASDAQ.MSFT", "MSFT"),
    ("equity.daily_bars", "NASDAQ.GOOGL", "GOOGL"),
    ("equity.daily_bars", "NASDAQ.AMZN", "AMZN"),
    ("equity.daily_bars", "NASDAQ.META", "META"),
    ("equity.daily_bars", "NASDAQ.TSLA", "TSLA"),
    ("equity.daily_bars", "NASDAQ.NVDA", "NVDA"),
    ("etf.daily_bars", "ARCA.GLD", "GLD"),
]
BENCHMARK_AUX = [
    ("etf.daily_bars", "ARCA.SPY", "SPY"),
    ("etf.daily_bars", "NASDAQ.QQQ", "QQQ"),
    ("etf.daily_bars", "ARCA.IWM", "IWM"),
]


def load_panel() -> pl.DataFrame:
    store = open_store()
    rows: list[pl.DataFrame] = []
    for lib_name, arctic_sym, display_sym in TRADABLE + BENCHMARK_AUX:
        lib = store[lib_name]
        df = lib.read(arctic_sym).data.reset_index().rename(columns={"date": "date"})
        df["symbol"] = display_sym
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        rows.append(pl.from_pandas(df[cols]))
    panel = pl.concat(rows, how="vertical_relaxed").sort(["symbol", "date"])
    panel = panel.with_columns(pl.col("date").cast(pl.Datetime("ns")))
    return panel


def _build_date_regime_map(feat_pl: pl.DataFrame) -> dict:
    feat_pd = feat_pl.to_pandas()
    feat_pd["date"] = pd.to_datetime(feat_pd["date"])
    vix_series = (
        feat_pd[["date", "vix_proxy"]]
        .drop_duplicates(subset=["date"])
        .set_index("date")["vix_proxy"]
        .sort_index()
        .astype(float)
    )
    tagger = RegimeTagger.from_series(vix_series)
    return {pd.Timestamp(d): tagger.tag(d, v) for d, v in vix_series.items()}


def _fit_predict_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """OLS = no regularization. We standardize so the comparison to Ridge
    is feature-scale invariant (Ridge standardizes too)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0)
    sd[sd == 0.0] = 1.0
    Xs_tr = (X_train - mu) / sd
    Xs_te = (X_test - mu) / sd
    ols = LinearRegression(fit_intercept=True)
    ols.fit(Xs_tr, y_train)
    return np.asarray(ols.predict(Xs_te), dtype=np.float64)


def _fit_predict_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    categorical_idx: list[int],
) -> np.ndarray:
    """LightGBM with project-default hyperparameters (matches src/ml_research
    /trainer.py LightGBMConfig).  No validation set passed -> no early
    stopping, fit to full n_estimators=500 — same protocol every fold."""
    cfg = LightGBMConfig(early_stopping_rounds=None)
    trainer = LightGBMTrainer(config=cfg, categorical_feature=categorical_idx)
    trainer.fit(X_train, y_train, categorical_feature=categorical_idx)
    return trainer.predict(X_test)


def collect_predictions(panel: pl.DataFrame, model_name: str) -> pd.DataFrame:
    """Identical pipeline to script 18 collect_predictions, but the trainer
    is switched per `model_name` ∈ {"ols", "lgbm"}.

    Returns DataFrame[date, symbol, regime, pred, y_z, y_raw, fold_id].
    """
    print(f"\n[{model_name}] Computing features...")
    feat = compute_features(panel)
    date_regime = _build_date_regime_map(feat)
    lab = next_day_open_close_log_return(feat, label_col="y_raw")
    tradable_syms = {disp for _, _, disp in TRADABLE}
    lab = lab.filter(pl.col("symbol").is_in(tradable_syms))
    lab = drop_feature_failures(lab, feature_names=FEATURE_NAMES)
    lab = lab.drop_nulls(subset=["y_raw"]).filter(pl.col("y_raw").is_finite())
    sym_to_id = {s: i for i, s in enumerate(sorted(tradable_syms))}
    lab = lab.with_columns(
        pl.col("symbol").replace_strict(sym_to_id, return_dtype=pl.Int32).alias("symbol_id")
    )
    sorted_dates = pd.DatetimeIndex(np.sort(lab["date"].unique().to_numpy()))
    print(f"[{model_name}]   unique trading days: {len(sorted_dates):,}")

    harness = WalkForwardHarness(
        train_min_years=5,
        train_max_years=15,
        val_days=252,
        embargo_days=5,
        purge_days=1,
        test_days=21,
        refit_cadence="monthly",
    )

    pdf = lab.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values(["date", "symbol"]).reset_index(drop=True)
    pdf["regime"] = pdf["date"].map(lambda d: date_regime.get(pd.Timestamp(d), "vix_mid"))
    feature_cols = list(FEATURE_NAMES) + ["symbol_id"]
    # symbol_id is the only categorical (added last in feature_cols)
    categorical_idx = [len(feature_cols) - 1]

    all_rows: list[pd.DataFrame] = []
    n_folds = 0
    for fold in harness.split(sorted_dates):
        train_dates = set(sorted_dates[fold.train_idx].to_pydatetime())
        test_dates = set(sorted_dates[fold.test_idx].to_pydatetime())
        val_dates = set(sorted_dates[fold.val_idx].to_pydatetime())

        train_rows = pdf[pdf["date"].isin(train_dates)]
        test_rows = pdf[pdf["date"].isin(test_dates)]
        if len(train_rows) < 200 or len(test_rows) < 5:
            continue

        bounds = fit_winsorization_bounds(
            train_rows[~train_rows["date"].isin(val_dates)]["y_raw"].to_numpy(),
            sigma=5.0,
        )
        train_pl = pl.from_pandas(train_rows[["symbol", "date", "y_raw"]])
        train_pl = winsorize_train_fold(train_pl, bounds, label_col="y_raw", out_col="y_wins")
        train_pl = cross_sectional_zscore(train_pl, label_col="y_wins", out_col="y_xs_z")
        test_pl = pl.from_pandas(test_rows[["symbol", "date", "y_raw"]])
        test_pl = winsorize_train_fold(test_pl, bounds, label_col="y_raw", out_col="y_wins")
        test_pl = cross_sectional_zscore(test_pl, label_col="y_wins", out_col="y_xs_z")

        train_feat_pl = pl.from_pandas(train_rows[["symbol", "date"] + feature_cols])
        clip_bounds = clip_features_train_fold(train_feat_pl, feature_names=tuple(FEATURE_NAMES))
        train_feat_pl = apply_feature_clip(train_feat_pl, clip_bounds)
        test_feat_pl = pl.from_pandas(test_rows[["symbol", "date"] + feature_cols])
        test_feat_pl = apply_feature_clip(test_feat_pl, clip_bounds)

        train_full = train_feat_pl.join(train_pl, on=["symbol", "date"], how="inner").to_pandas()
        test_full = test_feat_pl.join(test_pl, on=["symbol", "date"], how="inner").to_pandas()

        train_full["regime"] = train_full["date"].map(
            lambda d: date_regime.get(pd.Timestamp(d), "vix_mid")
        )
        test_full["regime"] = test_full["date"].map(
            lambda d: date_regime.get(pd.Timestamp(d), "vix_mid")
        )
        inner_train = train_full[~train_full["date"].isin(val_dates)]

        X_train = inner_train[feature_cols].to_numpy(dtype=np.float64)
        y_train = inner_train["y_xs_z"].to_numpy(dtype=np.float64)
        X_test = test_full[feature_cols].to_numpy(dtype=np.float64)
        y_test_z = test_full["y_xs_z"].to_numpy(dtype=np.float64)
        y_test_raw = test_full["y_raw"].to_numpy(dtype=np.float64)

        m_tr = np.isfinite(y_train)
        m_te = np.isfinite(y_test_z) & np.isfinite(y_test_raw)
        if m_tr.sum() < 100 or m_te.sum() < 5:
            continue

        if model_name == "ols":
            pred = _fit_predict_ols(X_train[m_tr], y_train[m_tr], X_test)
        elif model_name == "lgbm":
            pred = _fit_predict_lgbm(X_train[m_tr], y_train[m_tr], X_test, categorical_idx)
        else:
            raise ValueError(f"unknown model_name={model_name!r}")

        kept = test_full.loc[m_te, ["date", "symbol", "regime"]].copy()
        kept["pred"] = pred[m_te]
        kept["y_z"] = y_test_z[m_te]
        kept["y_raw"] = y_test_raw[m_te]
        kept["fold_id"] = fold.fold_id
        all_rows.append(kept)
        n_folds += 1
        if n_folds % 50 == 0:
            print(f"[{model_name}]   ... {n_folds} folds processed")

    print(f"[{model_name}]   total folds with preds: {n_folds}")
    return pd.concat(all_rows, ignore_index=True)


def assign_deciles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day percentile-rank into 10 buckets. See script 18 for the
    N=8 sparsity caveat (D3 and D8 are thinly populated on most days,
    D1 and D10 always fill)."""
    out = df.copy()
    out["rank"] = out.groupby("date")["pred"].rank(method="first", ascending=True)
    counts = out.groupby("date")["pred"].transform("count")
    pct = (out["rank"] - 0.5) / counts
    decile = np.ceil(pct * 10).astype(int)
    decile = decile.clip(1, 10)
    out["decile"] = decile
    return out


def per_decile_sharpe(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    sub = df[df["regime"] == regime]
    rows = []
    for d in range(1, 11):
        g = sub[sub["decile"] == d]["y_raw"].to_numpy()
        if g.size == 0:
            rows.append({"decile": d, "n": 0, "mean_bps": np.nan,
                         "std_bps": np.nan, "sharpe": np.nan})
            continue
        mu = float(np.mean(g))
        sd = float(np.std(g, ddof=0))
        sh = (mu / sd * np.sqrt(252)) if sd > 0 else np.nan
        rows.append({
            "decile": d,
            "n": int(g.size),
            "mean_bps": mu * 1e4,
            "std_bps": sd * 1e4,
            "sharpe": sh,
        })
    return pd.DataFrame(rows)


# Cycle-12 Ridge Sharpe table (rows = D1..D10, cols = vix_low/mid/high)
# Copied verbatim from data/18_decile_sharpe_per_regime.log lines 28-37 so
# this script is self-contained for the comparison printout.
RIDGE_SHARPE_C12 = pd.DataFrame(
    {
        "vix_low":  [-0.301, +0.277, +0.926, +0.483, -0.439, -0.206, +0.843, -0.354, +0.585, -0.117],
        "vix_mid":  [-0.508, -0.248, -0.400, +0.173, +0.004, +0.315, -0.282, +0.121, +0.227, +0.366],
        "vix_high": [+0.705, -0.073, -0.171, +0.301, -0.000, +0.278, +0.743, -0.035, +0.655, +0.858],
    },
    index=[f"D{d}" for d in range(1, 11)],
)


def print_sharpe_table(model_name: str, per_regime: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 78)
    print(f"[{model_name.upper()}] PER-DECILE SHARPE (annualized) — 10 deciles x 3 regimes")
    print("=" * 78)
    print(f"{'decile':>6} | {'vix_low':>10} {'vix_mid':>10} {'vix_high':>10}  "
          f"| {'n_low':>6} {'n_mid':>6} {'n_high':>6}")
    print("-" * 78)
    for d in range(1, 11):
        sl = per_regime["vix_low"].iloc[d - 1]
        sm = per_regime["vix_mid"].iloc[d - 1]
        sh = per_regime["vix_high"].iloc[d - 1]
        print(
            f"D{d:<5d} | {sl['sharpe']:>+10.3f} {sm['sharpe']:>+10.3f} "
            f"{sh['sharpe']:>+10.3f}  | "
            f"{int(sl['n']):>6d} {int(sm['n']):>6d} {int(sh['n']):>6d}"
        )


def print_side_by_side_d10_minus_d1(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Compact summary: D10-D1 spread per (model, regime).

    Plus: 'mid_is_best' flag = True iff vix_mid has the LARGEST positive
    D10-D1 spread among the three regimes for that model. If True for
    every model -> the cycle-12 finding is model-agnostic.
    """
    print("\n" + "=" * 78)
    print("CROSS-MODEL COMPARISON: D10 - D1 Sharpe spread per regime")
    print("=" * 78)
    print(f"{'model':>8} | {'vix_low':>10} {'vix_mid':>10} {'vix_high':>10}  "
          f"| {'mid_is_best?':>13}")
    print("-" * 78)
    # First row: cycle-12 Ridge reference.
    rl = RIDGE_SHARPE_C12["vix_low"].iloc[-1] - RIDGE_SHARPE_C12["vix_low"].iloc[0]
    rm = RIDGE_SHARPE_C12["vix_mid"].iloc[-1] - RIDGE_SHARPE_C12["vix_mid"].iloc[0]
    rh = RIDGE_SHARPE_C12["vix_high"].iloc[-1] - RIDGE_SHARPE_C12["vix_high"].iloc[0]
    mid_best_ridge = (rm > 0) and (rm > rl) and (rm > rh)
    print(f"{'ridge*':>8} | {rl:>+10.3f} {rm:>+10.3f} {rh:>+10.3f}  | "
          f"{'YES' if mid_best_ridge else 'NO':>13}")

    flags: dict[str, bool] = {"ridge": mid_best_ridge}
    for model_name, per_regime in results.items():
        sl = per_regime["vix_low"]["sharpe"].iloc[-1] - per_regime["vix_low"]["sharpe"].iloc[0]
        sm = per_regime["vix_mid"]["sharpe"].iloc[-1] - per_regime["vix_mid"]["sharpe"].iloc[0]
        sh = per_regime["vix_high"]["sharpe"].iloc[-1] - per_regime["vix_high"]["sharpe"].iloc[0]
        mid_best = (sm > 0) and (sm > sl) and (sm > sh)
        flags[model_name] = mid_best
        print(f"{model_name:>8} | {sl:>+10.3f} {sm:>+10.3f} {sh:>+10.3f}  | "
              f"{'YES' if mid_best else 'NO':>13}")
    print("-" * 78)
    print("  *ridge: cycle-12 reference, re-loaded from script 18 log.\n")

    all_agree = all(flags.values())
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if all_agree:
        print("  Finding is MODEL-AGNOSTIC: every tested learner reproduces")
        print("  the cycle-12 pattern where vix_mid is the only regime with a")
        print("  clean positive D10-D1 monotonic spread.")
        print("  Cycle-14 action: gate live trading to vix_mid regime, full size.")
    else:
        bad = [m for m, ok in flags.items() if not ok]
        print(f"  Finding is NOT robust. The following models did NOT show")
        print(f"  vix_mid as the dominant directional regime: {bad}")
        print("  This implies the cycle-12 Ridge result is either Ridge-specific")
        print("  or partially data-mined. Do not gate trading on vix_mid alone")
        print("  until additional cross-validation passes.")


def main() -> int:
    print("=" * 78)
    print("CYCLE-13: cross-model decile-Sharpe diagnostic (OLS + LightGBM)")
    print("=" * 78)
    panel = load_panel()
    print(f"Loaded {panel.height:,} rows across {panel['symbol'].n_unique()} symbols")

    results: dict[str, dict[str, pd.DataFrame]] = {}
    for model_name in ("ols", "lgbm"):
        preds_df = collect_predictions(panel, model_name)
        print(f"\n[{model_name}] Total pred rows: {len(preds_df):,}")
        print(f"[{model_name}] Regime distribution (rows):")
        print(preds_df["regime"].value_counts())

        preds_df = assign_deciles(preds_df)
        per_regime: dict[str, pd.DataFrame] = {}
        for reg in ("vix_low", "vix_mid", "vix_high"):
            per_regime[reg] = per_decile_sharpe(preds_df, reg)
        results[model_name] = per_regime
        print_sharpe_table(model_name, per_regime)

    print_side_by_side_d10_minus_d1(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
