"""Walk-forward baseline trainer for the ml-research tentacle.

Loads daily bars from ArcticDB for the 8 Alpaca-tradable contracts
(AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, GLD), computes the v1
feature set, builds next-day open-to-close log-return labels (winsorized
on the train fold + xs z-scored), and runs walk-forward LightGBM with
Ridge as the sanity floor.

Reports per-fold OOS IC, rank IC, hit rate, and daily-Sharpe (long-short
sized by sign of prediction). Honest reporting — Sharpe is whatever it is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _storage import open_store  # noqa: E402

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
from ml_research.trainer import (  # noqa: E402
    LightGBMConfig,
    LightGBMTrainer,
    RidgeConfig,
    RidgeTrainer,
    evaluate_fold,
)

# Spec says: 8 Alpaca-tradable contracts. SPY/QQQ/IWM included for vix_proxy
# + rel_strength benchmark (added implicitly to the universe; their predictions
# are evaluated too — they're just market-cap ETFs we'd happily trade).
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
    # Cast date to a clean pl.Date for consistent joins (input is datetime64[ns]).
    panel = panel.with_columns(pl.col("date").cast(pl.Datetime("ns")))
    return panel


def main() -> int:
    print("=" * 78)
    print("ml-research baseline training run")
    print("=" * 78)

    panel = load_panel()
    print(
        f"Loaded {panel.height:,} rows across {panel['symbol'].n_unique()} symbols"
    )
    print(
        f"Date range: {panel['date'].min()}  →  {panel['date'].max()}"
    )

    # Build features + labels on the FULL panel (label leakage is prevented by
    # fitting winsorization bounds on train-fold rows only; xs z-score is
    # per-day so there's no temporal leak).
    print("\nComputing features...")
    feat = compute_features(panel)
    print(f"  rows with all features finite: {drop_feature_failures(feat).height:,} / {feat.height:,}")

    print("Computing labels (next-day open-close log return)...")
    lab = next_day_open_close_log_return(feat, label_col="y_raw")

    # Restrict to TRADABLE universe for training/eval — benchmark ETFs were
    # only loaded to compute the broadcast features (vix_proxy, rel_strength).
    tradable_syms = {disp for _, _, disp in TRADABLE}
    lab = lab.filter(pl.col("symbol").is_in(tradable_syms))

    # Drop rows with any NaN/inf feature OR null label.
    lab = drop_feature_failures(lab)
    lab = lab.drop_nulls(subset=["y_raw"]).filter(pl.col("y_raw").is_finite())
    print(f"  training-eligible rows: {lab.height:,}")

    # Add a one-hot-ish contract ID (LightGBM handles categorical natively).
    sym_to_id = {s: i for i, s in enumerate(sorted(tradable_syms))}
    lab = lab.with_columns(
        pl.col("symbol").map_elements(lambda s: sym_to_id[s], return_dtype=pl.Int32).alias("symbol_id")
    )

    # Build the date axis for the walk-forward splitter.
    sorted_dates = (
        pd.DatetimeIndex(np.sort(lab["date"].unique().to_numpy()))
    )
    print(f"  unique trading days: {len(sorted_dates):,}")

    harness = WalkForwardHarness(
        train_min_years=5,
        train_max_years=15,
        val_days=252,
        embargo_days=5,
        purge_days=1,
        test_days=21,
        refit_cadence="monthly",
    )

    # Materialize the rows as a single pandas frame indexed by date for cheap
    # positional slicing per fold.
    pdf = lab.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values(["date", "symbol"]).reset_index(drop=True)

    feature_cols = list(FEATURE_NAMES) + ["symbol_id"]
    categorical_feature = [feature_cols.index("symbol_id")]

    folds_metrics: list[dict] = []
    for fold in harness.split(sorted_dates):
        train_dates = set(sorted_dates[fold.train_idx].to_pydatetime())
        test_dates = set(sorted_dates[fold.test_idx].to_pydatetime())
        val_dates = set(sorted_dates[fold.val_idx].to_pydatetime())

        train_rows = pdf[pdf["date"].isin(train_dates)]
        test_rows = pdf[pdf["date"].isin(test_dates)]
        # Split inner-val out of train for early stopping.
        inner_val = train_rows[train_rows["date"].isin(val_dates)]
        inner_train = train_rows[~train_rows["date"].isin(val_dates)]

        if len(inner_train) < 200 or len(test_rows) < 5:
            continue  # skip degenerate fold

        # Train-fold winsorization + per-day xs z-score.
        bounds = fit_winsorization_bounds(inner_train["y_raw"].to_numpy(), sigma=5.0)
        # Build labels using polars for the xs zscore.
        train_pl = pl.from_pandas(train_rows[["symbol", "date", "y_raw"]])
        train_pl = winsorize_train_fold(train_pl, bounds, label_col="y_raw", out_col="y_wins")
        train_pl = cross_sectional_zscore(train_pl, label_col="y_wins", out_col="y_xs_z")
        test_pl = pl.from_pandas(test_rows[["symbol", "date", "y_raw"]])
        test_pl = winsorize_train_fold(test_pl, bounds, label_col="y_raw", out_col="y_wins")
        test_pl = cross_sectional_zscore(test_pl, label_col="y_wins", out_col="y_xs_z")

        # Feature clipping on train fold.
        train_feat_pl = pl.from_pandas(train_rows[["symbol", "date"] + feature_cols])
        clip_bounds = clip_features_train_fold(train_feat_pl, feature_names=FEATURE_NAMES)
        train_feat_pl = apply_feature_clip(train_feat_pl, clip_bounds)
        test_feat_pl = pl.from_pandas(test_rows[["symbol", "date"] + feature_cols])
        test_feat_pl = apply_feature_clip(test_feat_pl, clip_bounds)

        # Rejoin label columns.
        train_full = train_feat_pl.join(train_pl, on=["symbol", "date"], how="inner")
        test_full = test_feat_pl.join(test_pl, on=["symbol", "date"], how="inner")

        # Split inner train/val.
        train_full_pd = train_full.to_pandas()
        is_val = train_full_pd["date"].isin(val_dates)
        inner_train_pd = train_full_pd[~is_val]
        inner_val_pd = train_full_pd[is_val]

        X_train = inner_train_pd[feature_cols].to_numpy(dtype=np.float64)
        y_train = inner_train_pd["y_xs_z"].to_numpy(dtype=np.float64)
        X_val = inner_val_pd[feature_cols].to_numpy(dtype=np.float64)
        y_val = inner_val_pd["y_xs_z"].to_numpy(dtype=np.float64)

        test_pd = test_full.to_pandas()
        X_test = test_pd[feature_cols].to_numpy(dtype=np.float64)
        y_test = test_pd["y_xs_z"].to_numpy(dtype=np.float64)
        # For sharpe we want the raw return (not z) so the long-short PnL is
        # in actual return units.
        y_test_raw = test_pd["y_raw"].to_numpy(dtype=np.float64)
        test_dates_arr = test_pd["date"].to_numpy()

        # Drop any rows where z-score is undefined (single-symbol day → std=0).
        m_train = np.isfinite(y_train)
        m_val = np.isfinite(y_val)
        m_test = np.isfinite(y_test)
        if m_train.sum() < 100 or m_test.sum() < 5:
            continue

        # LightGBM.
        lgbm = LightGBMTrainer(config=LightGBMConfig())
        lgbm.fit(
            X_train[m_train],
            y_train[m_train],
            X_val=X_val[m_val] if m_val.sum() > 0 else None,
            y_val=y_val[m_val] if m_val.sum() > 0 else None,
            feature_names=feature_cols,
            categorical_feature=categorical_feature,
        )
        pred_lgbm = lgbm.predict(X_test)
        m_test_lgbm = m_test & np.isfinite(pred_lgbm)
        # Evaluate against the raw return (sharpe) but report IC on z-score
        # since that's what the model is fit on.
        ic_metrics = evaluate_fold(
            y_test[m_test_lgbm], pred_lgbm[m_test_lgbm], test_dates_arr[m_test_lgbm]
        )
        sharpe_metrics = evaluate_fold(
            y_test_raw[m_test_lgbm], pred_lgbm[m_test_lgbm], test_dates_arr[m_test_lgbm]
        )

        # Ridge.
        ridge = RidgeTrainer(config=RidgeConfig(alpha=1.0))
        ridge.fit(X_train[m_train], y_train[m_train])
        pred_ridge = ridge.predict(X_test)
        m_test_r = m_test & np.isfinite(pred_ridge)
        ic_r = evaluate_fold(y_test[m_test_r], pred_ridge[m_test_r], test_dates_arr[m_test_r])
        sh_r = evaluate_fold(y_test_raw[m_test_r], pred_ridge[m_test_r], test_dates_arr[m_test_r])

        fold_row = {
            "fold_id": fold.fold_id,
            "test_start": fold.fold_meta["test_start_date"],
            "test_end": fold.fold_meta["test_end_date"],
            "n_train": int(m_train.sum()),
            "n_test": int(m_test_lgbm.sum()),
            "lgbm_ic": ic_metrics["ic"],
            "lgbm_rank_ic": ic_metrics["rank_ic"],
            "lgbm_hit_rate": sharpe_metrics["hit_rate"],
            "lgbm_sharpe": sharpe_metrics["sharpe_daily"],
            "ridge_ic": ic_r["ic"],
            "ridge_sharpe": sh_r["sharpe_daily"],
            "ridge_hit_rate": sh_r["hit_rate"],
            "lgbm_beats_ridge": sharpe_metrics["sharpe_daily"] > sh_r["sharpe_daily"],
        }
        folds_metrics.append(fold_row)
        print(
            f"fold {fold.fold_id:>3d}  {fold_row['test_start']}→{fold_row['test_end']}  "
            f"n_test={fold_row['n_test']:>4d}  "
            f"LGBM ic={fold_row['lgbm_ic']:+.4f} rIC={fold_row['lgbm_rank_ic']:+.4f} "
            f"hit={fold_row['lgbm_hit_rate']:.3f} Sh={fold_row['lgbm_sharpe']:+.2f}  "
            f"| Ridge ic={fold_row['ridge_ic']:+.4f} hit={fold_row['ridge_hit_rate']:.3f} "
            f"Sh={fold_row['ridge_sharpe']:+.2f}  "
            f"{'WIN' if fold_row['lgbm_beats_ridge'] else 'lose'}"
        )

    if not folds_metrics:
        print("\nNo folds completed (insufficient history).")
        return 1

    fm = pd.DataFrame(folds_metrics)
    print("\n" + "=" * 78)
    print(f"AGGREGATE across {len(fm)} folds:")
    print("=" * 78)
    print(f"  LightGBM    mean IC = {fm['lgbm_ic'].mean():+.4f}   "
          f"median IC = {fm['lgbm_ic'].median():+.4f}   "
          f"std IC = {fm['lgbm_ic'].std():.4f}")
    print(f"  LightGBM    mean rank-IC = {fm['lgbm_rank_ic'].mean():+.4f}")
    print(f"  LightGBM    mean hit rate = {fm['lgbm_hit_rate'].mean():.4f}   "
          f"mean Sharpe = {fm['lgbm_sharpe'].mean():+.3f}   "
          f"median Sharpe = {fm['lgbm_sharpe'].median():+.3f}")
    print(f"  Ridge       mean IC = {fm['ridge_ic'].mean():+.4f}   "
          f"mean Sharpe = {fm['ridge_sharpe'].mean():+.3f}   "
          f"hit = {fm['ridge_hit_rate'].mean():.4f}")
    win_rate = float(fm['lgbm_beats_ridge'].mean())
    print(f"  LGBM beats Ridge (Sharpe) in {fm['lgbm_beats_ridge'].sum()}/{len(fm)} folds = {win_rate:.1%}")

    # CONTEXT decision rule (line 119): LightGBM must beat Ridge OOS Sharpe
    # per fold or it's rejected as overfit. We report; we don't gate the script.
    if win_rate < 0.5:
        print("\n  WARNING: LightGBM fails to beat Ridge in majority of folds.")
        print("  Per CONTEXT.md decision rule, this baseline would be rejected as overfit.")
        print("  Honest read: with only 8 symbols and 9 features (no dollar-bar features yet),")
        print("  the tree model has very little to chew on vs the linear floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
