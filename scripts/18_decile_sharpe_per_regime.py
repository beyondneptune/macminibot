"""Cycle-12 diagnostic: per-decile Sharpe by VIX regime.

Cycle 11 showed vix_high has the highest rank-IC of the three regimes but
an *inverted* decile spread (top-pred minus bottom-pred is negative).
That is internally inconsistent for a "momentum-style" reading and
points at either (a) noise inflating the spread metric or (b) a real
non-monotonic relationship — the model picks signals that work in
vix_low/mid (where momentum / trend is intact) but the same signal
*inverts* in vix_high where prices mean-revert.

This cycle disambiguates by computing the *full decile-Sharpe curve*
per regime instead of just the D10-D1 spread.

Method
------
1. Re-uses the cycle-10/11 walk-forward Ridge prediction pipeline
   (script 16) and collects every (date, symbol, pred, y_raw, regime)
   tuple across all OOS folds.
2. For each (date), ranks symbols by predicted return (rank 1 = lowest
   prediction).  With ~8 tradables/day we map rank r in 1..N to decile
   d = ceil(r / N * 10) in 1..10 — most days only fill ~8 of 10
   buckets, but pooled across hundreds of days every bucket gets dense
   coverage.
3. Per (regime, decile) computes:
       n_obs, mean (bps), std (bps), daily-Sharpe (annualized * sqrt(252))
4. Emits a 10x3 Sharpe table + a 10x3 mean-return table + a verbal
   interpretation.

Hypothesis
----------
- vix_low/mid show monotonic decile-Sharpe: Sharpe(D10) > Sharpe(D1),
  ideally positive D10 and negative D1.  Standard momentum behavior.
- vix_high shows U-shape or *inverted* monotonic: Sharpe(D10) <= 0,
  Sharpe(D1) <= 0, middle buckets positive — i.e. mean-reversion
  regime where the extremes are the noisiest and most unreliable.
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
    """Mirrors cycle-11 collect_predictions exactly (so deciles are
    comparable to the cycle-11 D10-D1 numbers)."""
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


def assign_deciles(df: pd.DataFrame) -> pd.DataFrame:
    """For each date bucket symbols into 10 deciles by predicted return.

    Implementation note: with ~8 tradables/day we cannot fill all 10
    deciles on every date.  We use the within-day rank percentile
    p = (rank - 0.5) / N  in (0, 1)  and assign decile = ceil(p * 10).
    For N=8 this fills D1, D2, D4, D5, D6, D7, D9, D10 (skipping D3 and
    D8).  D3 / D8 will be populated only on days where N != 8 (rare,
    due to per-symbol feature dropouts) — those buckets will show low
    n_obs.  D1 and D10 always fill (rank 1 -> D1, rank N -> D10), so
    the D1-vs-D10 endpoints — the key test for momentum vs reversion —
    are well-powered.
    """
    out = df.copy()
    out["rank"] = out.groupby("date")["pred"].rank(method="first", ascending=True)
    counts = out.groupby("date")["pred"].transform("count")
    pct = (out["rank"] - 0.5) / counts
    decile = np.ceil(pct * 10).astype(int)
    decile = decile.clip(1, 10)
    out["decile"] = decile
    return out


def per_decile_sharpe(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Return one row per decile with realized mean (bps), std (bps),
    Sharpe, n_obs for the given regime."""
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


def print_sharpe_table(per_regime: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 78)
    print("PER-DECILE SHARPE (annualized) — 10 deciles x 3 regimes")
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


def print_mean_table(per_regime: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 78)
    print("PER-DECILE MEAN REALIZED RETURN (bps/day) — 10 deciles x 3 regimes")
    print("=" * 78)
    print(f"{'decile':>6} | {'vix_low':>10} {'vix_mid':>10} {'vix_high':>10}")
    print("-" * 60)
    for d in range(1, 11):
        sl = per_regime["vix_low"].iloc[d - 1]
        sm = per_regime["vix_mid"].iloc[d - 1]
        sh = per_regime["vix_high"].iloc[d - 1]
        print(
            f"D{d:<5d} | {sl['mean_bps']:>+10.2f} {sm['mean_bps']:>+10.2f} "
            f"{sh['mean_bps']:>+10.2f}"
        )


def classify_curve(sharpes: np.ndarray) -> str:
    """Heuristic shape classifier on a 10-element Sharpe curve.

    monotonic_up   : Sharpe rises from D1 to D10 (D10 > D1 by > 0.2 and
                     correlation with rank > 0.6)
    monotonic_down : inverse of above
    u_shape        : extremes (D1, D10) both negative, middle bucket
                     (D4..D7 mean) positive
    inverted_u     : extremes both positive, middle negative
    flat/noisy     : otherwise
    """
    s = np.asarray(sharpes, dtype=float)
    # Some deciles may be NaN when N=8 dominates (D3, D8 are sparse) —
    # ignore them for the shape heuristic but require D1 and D10 to be
    # finite, since those drive the momentum-vs-reversion verdict.
    if not (np.isfinite(s[0]) and np.isfinite(s[-1])):
        return "n/a"
    ranks = np.arange(1, 11)
    mask = np.isfinite(s)
    corr = float(np.corrcoef(ranks[mask], s[mask])[0, 1]) if mask.sum() >= 4 else float("nan")
    spread = s[-1] - s[0]
    extremes_mean = (s[0] + s[-1]) / 2
    middle = s[3:7]
    middle_mean = float(np.nanmean(middle)) if np.any(np.isfinite(middle)) else float("nan")
    if corr > 0.6 and spread > 0.2:
        return "monotonic_up (momentum)"
    if corr < -0.6 and spread < -0.2:
        return "monotonic_down (anti-momentum)"
    if extremes_mean < -0.1 and middle_mean > 0.1:
        return "U-shape (mean-reversion: extremes lose, middle wins)"
    if extremes_mean > 0.1 and middle_mean < -0.1:
        return "inverted-U (extremes win, middle loses)"
    return "flat/noisy (no exploitable shape)"


def main() -> int:
    print("=" * 78)
    print("CYCLE-12: per-decile Sharpe by VIX regime (10 buckets x 3 regimes)")
    print("=" * 78)
    panel = load_panel()
    print(f"Loaded {panel.height:,} rows across {panel['symbol'].n_unique()} symbols")
    preds_df = collect_predictions(panel)
    print(f"\nTotal pred rows: {len(preds_df):,}")
    print("Regime distribution (rows):")
    print(preds_df["regime"].value_counts())

    preds_df = assign_deciles(preds_df)

    per_regime: dict[str, pd.DataFrame] = {}
    for reg in ("vix_low", "vix_mid", "vix_high"):
        per_regime[reg] = per_decile_sharpe(preds_df, reg)

    print_sharpe_table(per_regime)
    print_mean_table(per_regime)

    print("\n" + "=" * 78)
    print("SHAPE CLASSIFICATION (per regime)")
    print("=" * 78)
    shape_map: dict[str, str] = {}
    for reg in ("vix_low", "vix_mid", "vix_high"):
        s = per_regime[reg]["sharpe"].to_numpy()
        shape = classify_curve(s)
        shape_map[reg] = shape
        d10_d1 = s[-1] - s[0]
        print(f"{reg:>9}: D1={s[0]:+.3f}  D10={s[-1]:+.3f}  D10-D1={d10_d1:+.3f}  "
              f"=> {shape}")

    # Quick interpretation block (the assistant will also write a prose
    # summary, but this gives the operator a self-contained verdict in
    # the log).
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    low_shape = shape_map["vix_low"]
    mid_shape = shape_map["vix_mid"]
    high_shape = shape_map["vix_high"]
    print(f"  vix_low  -> {low_shape}")
    print(f"  vix_mid  -> {mid_shape}")
    print(f"  vix_high -> {high_shape}")
    if "monotonic_up" in low_shape and "monotonic_up" in mid_shape:
        if "U-shape" in high_shape or "monotonic_down" in high_shape:
            print("\n  HYPOTHESIS CONFIRMED: vix_low/mid are momentum regimes,")
            print("  vix_high is a mean-reverting / signal-inverting regime.")
            print("  Cycle-13 action: gate vix_high either OFF entirely or")
            print("  invert the sign of predictions when regime==vix_high.")
        else:
            print("\n  PARTIAL: vix_low/mid behave as momentum; vix_high is")
            print("  noisy but not cleanly inverted — disable vix_high trading.")
    else:
        print("\n  Hypothesis NOT cleanly confirmed; inspect tables above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
