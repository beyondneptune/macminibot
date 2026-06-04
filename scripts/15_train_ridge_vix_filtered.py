"""Cycle-10 training run: Ridge trained ONLY on {vix_low, vix_mid} days.

Background (cycle 9, Epic 17):
    The cycle-7 Ridge backtest is structurally broken in vix_high regime
    (Sharpe = -0.252 per scripts/14_regime_stratified_ridge.py). The
    aggregate +0.452 / +2.03 net Sharpe is bull-window-driven. Three
    feature-family lifts (cycles 7-9: xs-spreads, vix terciles, lagged
    interactions) were all rejected — adding features cannot rescue a
    model whose conditional mean function reverses sign under vol.

This cycle (10) inverts the question: instead of *expanding* the feature
set, *contract* the training set. Drop vix_high training days entirely
and let Ridge fit the conditional mean over the calm-regime subspace
where it demonstrably works. The model is still EVALUATED on every test
day (incl. vix_high) so we can quantify two distinct effects:
    1. Does dropping vix_high training noise lift the Ridge fit on
       low/mid days? (cleaner training distribution → tighter coefs)
    2. Does the calm-fit model collapse harder, hold, or recover on
       vix_high test days? (model now extrapolates outside training
       support — a model risk we must measure, not assume.)

This is variant (1) from the cycle-10 spec: same Ridge deployed
everywhere with a universe-time-filter on training. Variant (2) — a
two-headed regime-switch model — is deferred until variant (1)
results in to keep the bake-off isolated to one variable change.

Promotion rule (CONTEXT improvement #12, applied to a TRAINING-FILTER
rather than feature change — same arithmetic, same anchor):
    (a) aggregate IC delta >= +0.005 on >= 67% of walk-forward folds, AND
    (b) shuffle-label canary passes (not run here — CI gate), AND
    (c) Ridge OOS Sharpe (aggregate) does NOT drop vs +0.452 baseline.

The filter is causal by construction: the RegimeTagger fits its
quintile cutoffs on shifted vix_proxy history (.shift(1)), so
membership of date t in {vix_low, vix_mid} uses only data ending at
t-1. No look-ahead is introduced.
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


def _build_date_regime_map(feat_pl: pl.DataFrame) -> dict:
    """Build a {pd.Timestamp -> regime_label} map via RegimeTagger.

    Uses the broadcast `vix_proxy` series already in the feature panel.
    """
    feat_pd = feat_pl.to_pandas()
    feat_pd["date"] = pd.to_datetime(feat_pd["date"])
    if "vix_proxy" not in feat_pd.columns:
        raise RuntimeError("vix_proxy column missing — compute_features must include it")
    vix_series = (
        feat_pd[["date", "vix_proxy"]]
        .drop_duplicates(subset=["date"])
        .set_index("date")["vix_proxy"]
        .sort_index()
        .astype(float)
    )
    tagger = RegimeTagger.from_series(vix_series)
    out: dict = {}
    for d, v in vix_series.items():
        out[pd.Timestamp(d)] = tagger.tag(d, v)
    return out


def run_walk_forward(
    panel: pl.DataFrame,
    *,
    drop_vix_high_train: bool,
) -> pd.DataFrame:
    print(f"\nComputing features (drop_vix_high_train={drop_vix_high_train})...")
    feat = compute_features(panel)

    date_regime = _build_date_regime_map(feat)
    n_high = sum(1 for r in date_regime.values() if r == "vix_high")
    n_low = sum(1 for r in date_regime.values() if r == "vix_low")
    n_mid = sum(1 for r in date_regime.values() if r == "vix_mid")
    print(
        f"  regime distribution: low={n_low}, mid={n_mid}, "
        f"high={n_high} (total={len(date_regime)})"
    )

    lab = next_day_open_close_log_return(feat, label_col="y_raw")

    tradable_syms = {disp for _, _, disp in TRADABLE}
    lab = lab.filter(pl.col("symbol").is_in(tradable_syms))

    lab = drop_feature_failures(lab, feature_names=FEATURE_NAMES)
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
    pdf["regime"] = pdf["date"].map(lambda d: date_regime.get(pd.Timestamp(d), "vix_mid"))

    feature_cols = list(FEATURE_NAMES) + ["symbol_id"]

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

        # Stats before any regime filtering — needed to compute bounds on
        # the same support the baseline sees. We will only filter rows
        # at the X/y materialization step so winsorization bounds and
        # clip bounds remain comparable across baseline vs filtered.
        bounds = fit_winsorization_bounds(inner_train["y_raw"].to_numpy(), sigma=5.0)
        train_pl = pl.from_pandas(train_rows[["symbol", "date", "y_raw"]])
        train_pl = winsorize_train_fold(train_pl, bounds, label_col="y_raw", out_col="y_wins")
        train_pl = cross_sectional_zscore(train_pl, label_col="y_wins", out_col="y_xs_z")
        test_pl = pl.from_pandas(test_rows[["symbol", "date", "y_raw"]])
        test_pl = winsorize_train_fold(test_pl, bounds, label_col="y_raw", out_col="y_wins")
        test_pl = cross_sectional_zscore(test_pl, label_col="y_wins", out_col="y_xs_z")

        train_feat_pl = pl.from_pandas(train_rows[["symbol", "date"] + feature_cols])
        clip_targets = tuple(c for c in FEATURE_NAMES)
        clip_bounds = clip_features_train_fold(train_feat_pl, feature_names=clip_targets)
        train_feat_pl = apply_feature_clip(train_feat_pl, clip_bounds)
        test_feat_pl = pl.from_pandas(test_rows[["symbol", "date"] + feature_cols])
        test_feat_pl = apply_feature_clip(test_feat_pl, clip_bounds)

        train_full = train_feat_pl.join(train_pl, on=["symbol", "date"], how="inner")
        test_full = test_feat_pl.join(test_pl, on=["symbol", "date"], how="inner")

        train_full_pd = train_full.to_pandas()
        train_full_pd["regime"] = train_full_pd["date"].map(
            lambda d: date_regime.get(pd.Timestamp(d), "vix_mid")
        )
        is_val = train_full_pd["date"].isin(val_dates)
        inner_train_pd = train_full_pd[~is_val]

        n_inner_pre = len(inner_train_pd)
        if drop_vix_high_train:
            inner_train_pd = inner_train_pd[inner_train_pd["regime"] != "vix_high"]
        n_inner_post = len(inner_train_pd)

        X_train = inner_train_pd[feature_cols].to_numpy(dtype=np.float64)
        y_train = inner_train_pd["y_xs_z"].to_numpy(dtype=np.float64)
        test_pd = test_full.to_pandas()
        test_pd["regime"] = test_pd["date"].map(
            lambda d: date_regime.get(pd.Timestamp(d), "vix_mid")
        )
        X_test = test_pd[feature_cols].to_numpy(dtype=np.float64)
        y_test = test_pd["y_xs_z"].to_numpy(dtype=np.float64)
        y_test_raw = test_pd["y_raw"].to_numpy(dtype=np.float64)
        test_dates_arr = test_pd["date"].to_numpy()
        test_regime_arr = test_pd["regime"].to_numpy()

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

        # Per-regime test-side sharpe (so we can see whether the calm-fit
        # model is actually better on vix_high test days, not just on
        # aggregate).
        per_regime: dict[str, dict] = {}
        for reg_label in ("vix_low", "vix_mid", "vix_high"):
            reg_mask = m & (test_regime_arr == reg_label)
            if reg_mask.sum() >= 3:
                reg_sh = evaluate_fold(
                    y_test_raw[reg_mask], pred[reg_mask], test_dates_arr[reg_mask]
                )
                per_regime[reg_label] = {
                    "n": int(reg_mask.sum()),
                    "sharpe": reg_sh["sharpe_daily"],
                }
            else:
                per_regime[reg_label] = {"n": int(reg_mask.sum()), "sharpe": np.nan}

        folds_metrics.append(
            {
                "fold_id": fold.fold_id,
                "test_start": fold.fold_meta["test_start_date"],
                "test_end": fold.fold_meta["test_end_date"],
                "n_train_pre": int(n_inner_pre),
                "n_train_post": int(n_inner_post),
                "n_test": int(m.sum()),
                "ridge_ic": ic["ic"],
                "ridge_rank_ic": ic["rank_ic"],
                "ridge_hit_rate": sh["hit_rate"],
                "ridge_sharpe": sh["sharpe_daily"],
                "sharpe_low": per_regime["vix_low"]["sharpe"],
                "sharpe_mid": per_regime["vix_mid"]["sharpe"],
                "sharpe_high": per_regime["vix_high"]["sharpe"],
                "n_low": per_regime["vix_low"]["n"],
                "n_mid": per_regime["vix_mid"]["n"],
                "n_high": per_regime["vix_high"]["n"],
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

    # Weighted per-regime sharpe (aggregate over folds, weighted by n).
    def _weighted(reg: str) -> float:
        n_col = f"n_{reg}"
        s_col = f"sharpe_{reg}"
        mask = fm[s_col].notna() & (fm[n_col] > 0)
        if not mask.any():
            return float("nan")
        sub = fm[mask]
        w = sub[n_col].astype(float)
        return float((sub[s_col] * w).sum() / w.sum())

    sh_low = _weighted("low")
    sh_mid = _weighted("mid")
    sh_high = _weighted("high")

    print(f"\n{label}: {len(fm)} folds")
    print(f"  Ridge mean IC      = {mean_ic:+.4f}   median IC = {median_ic:+.4f}   std IC = {std_ic:.4f}")
    print(f"  Ridge mean Sharpe  = {mean_sharpe:+.3f}   median Sharpe = {median_sharpe:+.3f}")
    print(f"  Ridge mean hit     = {mean_hit:.4f}")
    print(f"  Per-regime weighted Sharpe (test side):")
    print(f"    vix_low  : {sh_low:+.3f}")
    print(f"    vix_mid  : {sh_mid:+.3f}")
    print(f"    vix_high : {sh_high:+.3f}")
    return {
        "label": label,
        "n_folds": len(fm),
        "mean_ic": mean_ic,
        "median_ic": median_ic,
        "std_ic": std_ic,
        "mean_sharpe": mean_sharpe,
        "median_sharpe": median_sharpe,
        "sharpe_low": sh_low,
        "sharpe_mid": sh_mid,
        "sharpe_high": sh_high,
    }


def main() -> int:
    print("=" * 78)
    print("Cycle-10 bake-off: Ridge on FULL train set vs Ridge on {low,mid}-only train")
    print(f"Anchor (cycle-6 Ridge v0): mean IC = +{BASELINE_IC:.3f}   mean Sharpe = +{BASELINE_SHARPE:.3f}")
    print("Cycle-9 found vix_high Sharpe = -0.252 in the baseline — this script")
    print("tests whether removing vix_high TRAINING days lifts OOS performance.")
    print("=" * 78)

    panel = load_panel()
    print(f"Loaded {panel.height:,} rows across {panel['symbol'].n_unique()} symbols")
    print(f"Date range: {panel['date'].min()}  ->  {panel['date'].max()}")

    fm_base = run_walk_forward(panel, drop_vix_high_train=False)
    fm_filt = run_walk_forward(panel, drop_vix_high_train=True)

    if fm_base.empty or fm_filt.empty:
        print("\nERROR: at least one walk-forward run produced no folds.")
        return 1

    print("\n" + "=" * 78)
    print("PER-FOLD COMPARISON (aggregate ridge Sharpe per fold)")
    print("=" * 78)
    joined = fm_base.merge(
        fm_filt, on="fold_id", how="inner", suffixes=("_base", "_filt")
    )
    joined["ic_delta"] = joined["ridge_ic_filt"] - joined["ridge_ic_base"]
    joined["sharpe_delta"] = joined["ridge_sharpe_filt"] - joined["ridge_sharpe_base"]
    # Print only first 12 and last 6 to keep log readable.
    head = joined.head(12)
    tail = joined.tail(6)
    for _, row in head.iterrows():
        print(
            f"fold {int(row['fold_id']):>3d}  "
            f"{row['test_start_base']}->{row['test_end_base']}  "
            f"train n: {int(row['n_train_pre_base'])}->{int(row['n_train_post_filt'])}  "
            f"base IC={row['ridge_ic_base']:+.4f} Sh={row['ridge_sharpe_base']:+.2f}  |  "
            f"filt IC={row['ridge_ic_filt']:+.4f} Sh={row['ridge_sharpe_filt']:+.2f}  "
            f"|  dIC={row['ic_delta']:+.4f} dSh={row['sharpe_delta']:+.2f}"
        )
    if len(joined) > 18:
        print(f"  ... ({len(joined) - 18} folds omitted from middle of run) ...")
    for _, row in tail.iterrows():
        print(
            f"fold {int(row['fold_id']):>3d}  "
            f"{row['test_start_base']}->{row['test_end_base']}  "
            f"train n: {int(row['n_train_pre_base'])}->{int(row['n_train_post_filt'])}  "
            f"base IC={row['ridge_ic_base']:+.4f} Sh={row['ridge_sharpe_base']:+.2f}  |  "
            f"filt IC={row['ridge_ic_filt']:+.4f} Sh={row['ridge_sharpe_filt']:+.2f}  "
            f"|  dIC={row['ic_delta']:+.4f} dSh={row['sharpe_delta']:+.2f}"
        )

    base_summary = summarize("BASELINE (full train set)", fm_base)
    filt_summary = summarize("FILTERED (drop vix_high training days)", fm_filt)

    print("\n" + "=" * 78)
    print("PROMOTION RULE (CONTEXT improvement #12, training-filter variant)")
    print("=" * 78)
    delta_ic = filt_summary["mean_ic"] - base_summary["mean_ic"]
    delta_sharpe = filt_summary["mean_sharpe"] - base_summary["mean_sharpe"]
    n_folds_ic_gain = int((joined["ic_delta"] >= 0.005).sum())
    total_folds = len(joined)
    pct_folds_ic_gain = n_folds_ic_gain / max(total_folds, 1)
    sharpe_no_drop = filt_summary["mean_sharpe"] >= base_summary["mean_sharpe"]
    sharpe_anchor_ok = filt_summary["mean_sharpe"] >= BASELINE_SHARPE

    print(f"  mean IC delta vs baseline rerun     : {delta_ic:+.4f}")
    print(f"  mean Sharpe delta vs baseline rerun : {delta_sharpe:+.3f}")
    print(
        f"  folds with IC delta >= +0.005       : {n_folds_ic_gain} / {total_folds} "
        f"({pct_folds_ic_gain:.0%}, need >= 67%)"
    )
    print(f"  no Sharpe drop vs baseline rerun    : {'PASS' if sharpe_no_drop else 'FAIL'} ({delta_sharpe:+.3f})")
    print(f"  Sharpe floor vs anchor (>= +0.452)  : {'PASS' if sharpe_anchor_ok else 'FAIL'} ({filt_summary['mean_sharpe']:+.3f})")
    print(f"  shuffle-label canary                : (CI gate, not run in this script)")

    print("\n  Per-regime weighted-Sharpe delta (filtered minus baseline):")
    for reg in ("low", "mid", "high"):
        b = base_summary[f"sharpe_{reg}"]
        f = filt_summary[f"sharpe_{reg}"]
        delta = f - b if (np.isfinite(b) and np.isfinite(f)) else float("nan")
        print(f"    vix_{reg:<4}: base={b:+.3f}  filt={f:+.3f}  delta={delta:+.3f}")

    rule_a = pct_folds_ic_gain >= 0.67
    rule_c = sharpe_no_drop
    measurable_pass = rule_a and rule_c
    if measurable_pass:
        print("\n  RESULT: measurable rules (a) and (c) PASS — pending shuffle-canary in CI.")
        print("  Recommend promoting training-time vix_high filter to production.")
    else:
        print("\n  RESULT: measurable rules FAIL — training-filter rejected.")
        print("  Document as negative result; baseline Ridge stays on full train set.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
