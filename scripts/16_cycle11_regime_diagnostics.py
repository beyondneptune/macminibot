"""Cycle-11 diagnostic: per-regime residual std, hit rate, decile spread.

Investigates why vix_high has +0.611 raw Sharpe (sign correct) but
collapses to -0.252 net after fees in cycle 9 / cycle 10. Hypotheses:
    (1) prediction-residual std is wider in vix_high → predictions noisier
    (2) hit-rate is worse in vix_high even with positive rank-IC
    (3) decile-1 vs decile-10 spread is small in vix_high → fees eat edge

No new model training: re-uses the same baseline Ridge walk-forward as
cycle 10 (script 15, drop_vix_high_train=False), but captures every
(date, pred, y_raw, regime) tuple so we can compute the three
diagnostics post-hoc per regime.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

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
from ml_research.trainer import RidgeConfig, RidgeTrainer  # noqa: E402

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


def collect_predictions(panel: pl.DataFrame) -> pd.DataFrame:
    print("Computing features...")
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

        train_full["regime"] = train_full["date"].map(lambda d: date_regime.get(pd.Timestamp(d), "vix_mid"))
        test_full["regime"] = test_full["date"].map(lambda d: date_regime.get(pd.Timestamp(d), "vix_mid"))
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

        ridge = RidgeTrainer(config=RidgeConfig(alpha=1.0))
        ridge.fit(X_train[m_tr], y_train[m_tr])
        pred = ridge.predict(X_test)

        kept = test_full.loc[m_te, ["date", "symbol", "regime"]].copy()
        kept["pred"] = pred[m_te]
        kept["y_z"] = y_test_z[m_te]
        kept["y_raw"] = y_test_raw[m_te]
        kept["fold_id"] = fold.fold_id
        all_rows.append(kept)
        n_folds += 1
        if n_folds % 50 == 0:
            print(f"  ... {n_folds} folds processed")

    print(f"  total folds with preds: {n_folds}")
    return pd.concat(all_rows, ignore_index=True)


def _decile_spread(df: pd.DataFrame) -> tuple[float, float, float]:
    """Per-day form decile-1 (lowest pred) and decile-10 (highest pred)
    portfolios; return mean (d10 - d1) y_raw spread plus daily-Sharpe of
    the spread plus number of decile-spread days."""
    daily_spreads = []
    for _, g in df.groupby("date"):
        if len(g) < 5:
            continue
        # With 8 tradables, use top-2 vs bottom-2 as decile-proxy.
        n = len(g)
        k = max(1, n // 5)  # quintile when 8 syms (top/bot 2 each)
        top = g.nlargest(k, "pred")["y_raw"].mean()
        bot = g.nsmallest(k, "pred")["y_raw"].mean()
        daily_spreads.append(top - bot)
    if not daily_spreads:
        return float("nan"), float("nan"), 0
    a = np.array(daily_spreads)
    mean_spread = float(a.mean())
    sd = float(a.std(ddof=0))
    sh = float(a.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    return mean_spread, sh, len(a)


def diagnose_regime(df: pd.DataFrame, regime: str) -> dict:
    sub = df[df["regime"] == regime].copy()
    n = len(sub)
    if n < 5:
        return {"regime": regime, "n_obs": n}
    pred = sub["pred"].to_numpy()
    y_raw = sub["y_raw"].to_numpy()
    y_z = sub["y_z"].to_numpy()

    # 1. Prediction & residual std (on standardized label scale).
    pred_std = float(pred.std(ddof=0))
    residual = y_z - pred
    resid_std = float(residual.std(ddof=0))
    # signal-to-noise: pred_std / resid_std
    snr = pred_std / resid_std if resid_std > 0 else float("nan")

    # Pearson + Spearman on standardized label
    if pred.std() > 0 and y_z.std() > 0:
        ic = float(np.corrcoef(pred, y_z)[0, 1])
    else:
        ic = float("nan")
    rank_ic = float(stats.spearmanr(pred, y_z).correlation) if n > 2 else float("nan")

    # 2. Hit rate (sign agreement)
    hit_rate = float(np.mean(np.sign(pred) == np.sign(y_raw)))
    # Bootstrap CI on hit_rate
    rng = np.random.default_rng(seed=42)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, n, n)
        boot.append(np.mean(np.sign(pred[idx]) == np.sign(y_raw[idx])))
    hit_ci_lo, hit_ci_hi = np.percentile(boot, [2.5, 97.5])

    # 3. Decile-1 vs decile-10 spread (top vs bottom by pred, per day)
    mean_spread_bps, spread_sh, spread_days = _decile_spread(sub)

    # Mean |pred| — magnitude of bet sizing.
    mean_abs_pred = float(np.mean(np.abs(pred)))

    return {
        "regime": regime,
        "n_obs": n,
        "pred_std": pred_std,
        "resid_std": resid_std,
        "snr": snr,
        "ic": ic,
        "rank_ic": rank_ic,
        "hit_rate": hit_rate,
        "hit_ci": (float(hit_ci_lo), float(hit_ci_hi)),
        "mean_abs_pred": mean_abs_pred,
        "mean_spread_bps_per_day": mean_spread_bps,
        "spread_sharpe": spread_sh,
        "spread_days": spread_days,
    }


def main() -> int:
    print("=" * 78)
    print("CYCLE-11: per-regime diagnostics on cycle-9/10 Ridge baseline")
    print("=" * 78)
    panel = load_panel()
    print(f"Loaded {panel.height:,} rows across {panel['symbol'].n_unique()} symbols")
    preds_df = collect_predictions(panel)
    print(f"\nTotal pred rows: {len(preds_df):,}")
    print("Regime distribution:")
    print(preds_df["regime"].value_counts())

    print("\n" + "=" * 78)
    print("PER-REGIME DIAGNOSTIC TABLE")
    print("=" * 78)
    print(
        f"{'regime':>9} {'n':>6} {'pred_std':>9} {'resid_std':>9} "
        f"{'SNR':>6} {'rIC':>7} {'hit':>6} {'hit_CI':>16} "
        f"{'|pred|':>8} {'D10-D1':>9} {'spr_Sh':>7}"
    )
    print("-" * 110)
    rows = []
    for reg in ("vix_low", "vix_mid", "vix_high"):
        d = diagnose_regime(preds_df, reg)
        rows.append(d)
        if d.get("n_obs", 0) < 5:
            print(f"{reg:>9} {d['n_obs']:>6d}  (insufficient data)")
            continue
        ci = d["hit_ci"]
        ci_str = f"[{ci[0]:.3f},{ci[1]:.3f}]"
        print(
            f"{reg:>9} {d['n_obs']:>6d} {d['pred_std']:>9.4f} "
            f"{d['resid_std']:>9.4f} {d['snr']:>6.3f} "
            f"{d['rank_ic']:>+7.4f} {d['hit_rate']:>6.4f} {ci_str:>16} "
            f"{d['mean_abs_pred']:>8.4f} {d['mean_spread_bps_per_day']*1e4:>+9.1f} "
            f"{d['spread_sharpe']:>+7.3f}"
        )

    # Build the verdict per row.
    print("\n" + "=" * 78)
    print("PER-ROW VERDICT")
    print("=" * 78)
    by_reg = {r["regime"]: r for r in rows}
    low = by_reg["vix_low"]
    high = by_reg["vix_high"]
    for reg in ("vix_low", "vix_mid", "vix_high"):
        d = by_reg[reg]
        if d.get("n_obs", 0) < 5:
            print(f"{reg}: insufficient observations.")
            continue
        # Compose verdict.
        snr_str = f"SNR={d['snr']:.3f}"
        hit_str = f"hit={d['hit_rate']:.3f}"
        spread_str = f"D10-D1={d['mean_spread_bps_per_day']*1e4:+.1f}bps/day"
        if reg == "vix_high":
            ratio_snr = d["snr"] / low["snr"] if low["snr"] > 0 else float("nan")
            ratio_spr = (
                d["mean_spread_bps_per_day"] / low["mean_spread_bps_per_day"]
                if low["mean_spread_bps_per_day"] != 0 else float("nan")
            )
            print(
                f"{reg}: {snr_str}, {hit_str}, {spread_str}  "
                f"(SNR-vs-low={ratio_snr:.2f}x, spread-vs-low={ratio_spr:.2f}x)"
            )
        else:
            print(f"{reg}: {snr_str}, {hit_str}, {spread_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
