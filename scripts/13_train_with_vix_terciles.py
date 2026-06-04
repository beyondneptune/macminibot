"""Cycle-8 training run: 9 baseline features + 6 VIX-tercile candidates.

CONTEXT improvement #12, family (b): VIX-proxy tercile one-hot encoding +
ret_21d × tercile interactions. Six new ridge-friendly columns:
    vix_tercile_low, vix_tercile_mid, vix_tercile_high,
    ret_21d_x_vix_low, ret_21d_x_vix_mid, ret_21d_x_vix_high

Anchor (cycle-6 baseline, `manifest.baselines.cycle6_ridge_v0`):
    Ridge mean IC = +0.021, mean Sharpe = +0.452  (9 features, 8 symbols)

Cycle-7 rejected family (a) (cross-asset spreads, dSharpe -0.151). This script
tests family (b). The bucketing is causal — trailing 252d percentiles of the
vix_proxy series, shifted by one day so today's bucket uses only history
ending yesterday.

Promotion rule (CONTEXT improvement #12): the candidate family is adopted iff
  (a) aggregate IC delta >= +0.005 on >= 67% of walk-forward folds, AND
  (b) shuffle-label canary passes (not run here — CI gate), AND
  (c) Ridge OOS Sharpe does NOT drop vs +0.452 baseline.
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
    CANDIDATE_VIX_TERCILE_FEATURES,
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
    RidgeConfig,
    RidgeTrainer,
    evaluate_fold,
)

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

BASELINE_IC = 0.021
BASELINE_SHARPE = 0.452


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


def run_walk_forward(
    panel: pl.DataFrame,
    feature_set: tuple[str, ...],
    *,
    include_vix_terciles: bool,
) -> pd.DataFrame:
    print(f"\nComputing features (include_vix_terciles={include_vix_terciles})...")
    feat = compute_features(panel, include_vix_terciles=include_vix_terciles)

    lab = next_day_open_close_log_return(feat, label_col="y_raw")

    tradable_syms = {disp for _, _, disp in TRADABLE}
    lab = lab.filter(pl.col("symbol").is_in(tradable_syms))

    lab = drop_feature_failures(lab, feature_names=feature_set)
    lab = lab.drop_nulls(subset=["y_raw"]).filter(pl.col("y_raw").is_finite())
    print(f"  training-eligible rows: {lab.height:,}")

    sym_to_id = {s: i for i, s in enumerate(sorted(tradable_syms))}
    lab = lab.with_columns(
        pl.col("symbol")
        .map_elements(lambda s: sym_to_id[s], return_dtype=pl.Int32)
        .alias("symbol_id")
    )

    sorted_dates = pd.DatetimeIndex(np.sort(lab["date"].unique().to_numpy()))
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

    pdf = lab.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values(["date", "symbol"]).reset_index(drop=True)

    feature_cols = list(feature_set) + ["symbol_id"]

    folds_metrics: list[dict] = []
    for fold in harness.split(sorted_dates):
        train_dates = set(sorted_dates[fold.train_idx].to_pydatetime())
        test_dates = set(sorted_dates[fold.test_idx].to_pydatetime())
        val_dates = set(sorted_dates[fold.val_idx].to_pydatetime())

        train_rows = pdf[pdf["date"].isin(train_dates)]
        test_rows = pdf[pdf["date"].isin(test_dates)]
        inner_train = train_rows[~train_rows["date"].isin(val_dates)]

        if len(inner_train) < 200 or len(test_rows) < 5:
            continue

        bounds = fit_winsorization_bounds(inner_train["y_raw"].to_numpy(), sigma=5.0)
        train_pl = pl.from_pandas(train_rows[["symbol", "date", "y_raw"]])
        train_pl = winsorize_train_fold(train_pl, bounds, label_col="y_raw", out_col="y_wins")
        train_pl = cross_sectional_zscore(train_pl, label_col="y_wins", out_col="y_xs_z")
        test_pl = pl.from_pandas(test_rows[["symbol", "date", "y_raw"]])
        test_pl = winsorize_train_fold(test_pl, bounds, label_col="y_raw", out_col="y_wins")
        test_pl = cross_sectional_zscore(test_pl, label_col="y_wins", out_col="y_xs_z")

        train_feat_pl = pl.from_pandas(train_rows[["symbol", "date"] + feature_cols])
        # Don't clip the one-hot indicators or interactions (already bounded).
        clip_targets = tuple(c for c in feature_set if c in FEATURE_NAMES)
        clip_bounds = clip_features_train_fold(train_feat_pl, feature_names=clip_targets)
        train_feat_pl = apply_feature_clip(train_feat_pl, clip_bounds)
        test_feat_pl = pl.from_pandas(test_rows[["symbol", "date"] + feature_cols])
        test_feat_pl = apply_feature_clip(test_feat_pl, clip_bounds)

        train_full = train_feat_pl.join(train_pl, on=["symbol", "date"], how="inner")
        test_full = test_feat_pl.join(test_pl, on=["symbol", "date"], how="inner")

        train_full_pd = train_full.to_pandas()
        is_val = train_full_pd["date"].isin(val_dates)
        inner_train_pd = train_full_pd[~is_val]

        X_train = inner_train_pd[feature_cols].to_numpy(dtype=np.float64)
        y_train = inner_train_pd["y_xs_z"].to_numpy(dtype=np.float64)
        test_pd = test_full.to_pandas()
        X_test = test_pd[feature_cols].to_numpy(dtype=np.float64)
        y_test = test_pd["y_xs_z"].to_numpy(dtype=np.float64)
        y_test_raw = test_pd["y_raw"].to_numpy(dtype=np.float64)
        test_dates_arr = test_pd["date"].to_numpy()

        m_train = np.isfinite(y_train)
        m_test = np.isfinite(y_test)
        if m_train.sum() < 100 or m_test.sum() < 5:
            continue

        ridge = RidgeTrainer(config=RidgeConfig(alpha=1.0))
        ridge.fit(X_train[m_train], y_train[m_train])
        pred = ridge.predict(X_test)
        m = m_test & np.isfinite(pred)
        ic = evaluate_fold(y_test[m], pred[m], test_dates_arr[m])
        sh = evaluate_fold(y_test_raw[m], pred[m], test_dates_arr[m])

        folds_metrics.append(
            {
                "fold_id": fold.fold_id,
                "test_start": fold.fold_meta["test_start_date"],
                "test_end": fold.fold_meta["test_end_date"],
                "n_test": int(m.sum()),
                "ridge_ic": ic["ic"],
                "ridge_rank_ic": ic["rank_ic"],
                "ridge_hit_rate": sh["hit_rate"],
                "ridge_sharpe": sh["sharpe_daily"],
            }
        )

    return pd.DataFrame(folds_metrics)


def summarize(label: str, fm: pd.DataFrame) -> dict:
    mean_ic = float(fm["ridge_ic"].mean())
    median_ic = float(fm["ridge_ic"].median())
    std_ic = float(fm["ridge_ic"].std())
    mean_sharpe = float(fm["ridge_sharpe"].mean())
    median_sharpe = float(fm["ridge_sharpe"].median())
    mean_hit = float(fm["ridge_hit_rate"].mean())
    print(f"\n{label}: {len(fm)} folds")
    print(f"  Ridge mean IC      = {mean_ic:+.4f}   median IC = {median_ic:+.4f}   std IC = {std_ic:.4f}")
    print(f"  Ridge mean Sharpe  = {mean_sharpe:+.3f}   median Sharpe = {median_sharpe:+.3f}")
    print(f"  Ridge mean hit     = {mean_hit:.4f}")
    return {
        "label": label,
        "n_folds": len(fm),
        "mean_ic": mean_ic,
        "median_ic": median_ic,
        "std_ic": std_ic,
        "mean_sharpe": mean_sharpe,
        "median_sharpe": median_sharpe,
    }


def main() -> int:
    print("=" * 78)
    print("Cycle-8 bake-off: 9 baseline features  vs  9 + 6 VIX-tercile features")
    print(f"Anchor (cycle-6 Ridge v0):  mean IC = +{BASELINE_IC:.3f}   mean Sharpe = +{BASELINE_SHARPE:.3f}")
    print("=" * 78)

    panel = load_panel()
    print(f"Loaded {panel.height:,} rows across {panel['symbol'].n_unique()} symbols")
    print(f"Date range: {panel['date'].min()}  ->  {panel['date'].max()}")

    fm_base = run_walk_forward(panel, FEATURE_NAMES, include_vix_terciles=False)
    fm_vix = run_walk_forward(
        panel,
        FEATURE_NAMES + CANDIDATE_VIX_TERCILE_FEATURES,
        include_vix_terciles=True,
    )

    if fm_base.empty or fm_vix.empty:
        print("\nERROR: at least one walk-forward run produced no folds.")
        return 1

    print("\n" + "=" * 78)
    print("PER-FOLD COMPARISON")
    print("=" * 78)
    joined = fm_base.merge(
        fm_vix, on="fold_id", how="inner", suffixes=("_base", "_vix")
    )
    joined["ic_delta"] = joined["ridge_ic_vix"] - joined["ridge_ic_base"]
    joined["sharpe_delta"] = joined["ridge_sharpe_vix"] - joined["ridge_sharpe_base"]
    for _, row in joined.iterrows():
        print(
            f"fold {int(row['fold_id']):>3d}  "
            f"{row['test_start_base']}->{row['test_end_base']}  "
            f"base IC={row['ridge_ic_base']:+.4f} Sh={row['ridge_sharpe_base']:+.2f}  |  "
            f"vix IC={row['ridge_ic_vix']:+.4f} Sh={row['ridge_sharpe_vix']:+.2f}  "
            f"|  dIC={row['ic_delta']:+.4f} dSh={row['sharpe_delta']:+.2f}"
        )

    base_summary = summarize("BASELINE (9 features, recomputed)", fm_base)
    vix_summary = summarize("CANDIDATE (9 + 6 VIX-tercile features)", fm_vix)

    print("\n" + "=" * 78)
    print("PROMOTION RULE (CONTEXT improvement #12)")
    print("=" * 78)
    delta_ic = vix_summary["mean_ic"] - base_summary["mean_ic"]
    delta_sharpe = vix_summary["mean_sharpe"] - base_summary["mean_sharpe"]
    n_folds_ic_gain = int((joined["ic_delta"] >= 0.005).sum())
    total_folds = len(joined)
    pct_folds_ic_gain = n_folds_ic_gain / max(total_folds, 1)
    sharpe_no_drop = vix_summary["mean_sharpe"] >= base_summary["mean_sharpe"]
    sharpe_anchor_ok = vix_summary["mean_sharpe"] >= BASELINE_SHARPE

    print(f"  mean IC delta vs baseline rerun     : {delta_ic:+.4f}")
    print(f"  mean Sharpe delta vs baseline rerun : {delta_sharpe:+.3f}")
    print(
        f"  folds with IC delta >= +0.005       : {n_folds_ic_gain} / {total_folds} "
        f"({pct_folds_ic_gain:.0%}, need >= 67%)"
    )
    print(f"  no Sharpe drop vs baseline rerun    : {'PASS' if sharpe_no_drop else 'FAIL'} ({delta_sharpe:+.3f})")
    print(f"  Sharpe floor vs anchor (>= +0.452)  : {'PASS' if sharpe_anchor_ok else 'FAIL'} ({vix_summary['mean_sharpe']:+.3f})")
    print(f"  shuffle-label canary                : (CI gate, not run in this script)")

    rule_a = pct_folds_ic_gain >= 0.67
    rule_c = sharpe_no_drop
    measurable_pass = rule_a and rule_c
    if measurable_pass:
        print("\n  RESULT: measurable rules (a) and (c) PASS — pending shuffle-canary in CI.")
        print("  Recommend promotion to FEATURE_NAMES + bump feature_spec_hash.")
    else:
        print("\n  RESULT: measurable rules FAIL — candidate rejected at this universe size.")
        print("  Document as negative result; do NOT bump feature_spec_hash.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
