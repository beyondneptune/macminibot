"""End-to-end backtest of the Ridge sanity-floor model on real ArcticDB bars.

Cycle-7 deliverable: stitch ml_research.{features, trainer.RidgeTrainer, cv} into
backtesting.WalkForwardDriver with a real per-fold fit/predict, and report the
*net* (post-fee + post-slippage) Sharpe + DSR vs the gross (training-only)
proxy. Honestly reports whether DSR ≥ 0.95 promotion gate passes for paper
trading tomorrow.

Cycle-8 deliverable: add Politis-Romano stationary block-bootstrap CI on the
aggregate OOS daily returns (B=10,000) so the +2.03 net Sharpe carries an
honest 95% lower bound rather than an unqualified point estimate. The
cycle-7 verdict flagged +2.03 as "suspiciously high for tech mega-cap
2024-2026 bull window" — the CI lower bound + p(Sharpe > 0) are the
quantitative answer.

Cycle-8 also replaces the broken DSR gate. Cycle 7 used ``n_trials = n_folds``
which over-deflates (folds are sequential evaluations of ONE fitted model,
not independent hyperparameter trials). The new gate uses
``n_trials = n_alphas × n_seeds`` from the actual Ridge hyperparameter
search — i.e. the number of independent strategy variants we'd have
considered before settling on alpha=1.0. For a single-alpha-single-seed
Ridge baseline (this script) that's ``n_trials = 1`` and DSR collapses
to PSR vs zero, which is the honest answer for "we did not search".

Trades the 8 Alpaca-tradable contracts (SPY/QQQ/IWM are loaded as feature
benchmarks only — their weights are forced to 0 so we don't fee-up an ETF
basket the strategy was never meant to trade).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _storage import open_store  # noqa: E402

from backtesting.driver import FoldSpec, WalkForwardDriver  # noqa: E402
from backtesting import metrics as bt_metrics  # noqa: E402

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

# Same universe as scripts/10_train_baseline.py (8 tradable + 3 benchmark ETFs).
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
TRADABLE_SYMS = {disp for _, _, disp in TRADABLE}


def load_panel() -> pl.DataFrame:
    store = open_store()
    rows: list[pl.DataFrame] = []
    for lib_name, arctic_sym, display_sym in TRADABLE + BENCHMARK_AUX:
        df = store[lib_name].read(arctic_sym).data.reset_index()
        df["symbol"] = display_sym
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        rows.append(pl.from_pandas(df[cols]))
    panel = pl.concat(rows, how="vertical_relaxed").sort(["symbol", "date"])
    return panel.with_columns(pl.col("date").cast(pl.Datetime("ns")))


def build_bars_for_driver(panel: pl.DataFrame) -> pd.DataFrame:
    """Driver wants MultiIndex(date,symbol) with open/close/volume +
    adv_21d_shares + daily_vol_bps. Compute the two derived columns once
    over the full panel (no leakage: they're causal rolling stats)."""
    pdf = panel.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Per-symbol rolling stats.
    pdf["adv_21d_shares"] = (
        pdf.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(21, min_periods=5).mean()
        )
    )
    # daily_vol_bps: rolling 21d std of log returns × 1e4 (bps).
    pdf["log_ret"] = np.log(pdf["close"] / pdf.groupby("symbol")["close"].shift(1))
    pdf["daily_vol_bps"] = (
        pdf.groupby("symbol")["log_ret"]
        .transform(lambda s: s.rolling(21, min_periods=5).std(ddof=0))
        * 1e4
    )
    pdf = pdf.dropna(subset=["adv_21d_shares", "daily_vol_bps"])

    pdf = pdf.set_index(["date", "symbol"]).sort_index()
    return pdf[["open", "close", "volume", "adv_21d_shares", "daily_vol_bps"]]


class RidgeSignalAdapter:
    """Per-fold Ridge predict → target-weight dict consumed by the driver.

    The driver calls ``__call__(bar, train_window)`` once per test-fold bar.
    ``bar`` is a Series of *previous-close* prices indexed by symbol — we
    ignore it (we need the t-1 *features* row already living in the
    pre-computed feature panel, not the close price). Instead we look up
    the feature row for the per-fold ``feature_panel`` indexed by date.

    The adapter is *closed over* the date being predicted via a callable
    factory: the driver doesn't pass `date` so we expose a tiny stateful
    helper that pops the next test date in order.
    """

    def __init__(
        self,
        ridge: RidgeTrainer,
        feature_panel: pd.DataFrame,
        feature_cols: list[str],
        test_dates: list[pd.Timestamp],
        sorted_all_dates: pd.DatetimeIndex,
        max_gross: float = 1.0,
    ) -> None:
        self._ridge = ridge
        self._feat = feature_panel
        self._cols = feature_cols
        self._dates_iter = iter(test_dates)
        self._max_gross = max_gross
        # CRITICAL: at open[T] we know features as of close[T-1], NOT close[T].
        # Look up the prior-trading-day's features (avoids look-ahead leak).
        self._date_to_prev = {}
        prev = None
        for d in sorted_all_dates:
            self._date_to_prev[pd.Timestamp(d)] = prev
            prev = pd.Timestamp(d)

    def __call__(self, bar: pd.Series, train_window: pd.DataFrame) -> dict[str, float]:
        try:
            date = next(self._dates_iter)
        except StopIteration:
            return {}
        # Use T-1 features (causal — no peek at close[T]).
        feat_date = self._date_to_prev.get(date)
        if feat_date is None:
            return {}
        try:
            day_feat = self._feat.loc[feat_date]
        except KeyError:
            return {}
        if isinstance(day_feat, pd.Series):
            day_feat = day_feat.to_frame().T
        # Restrict to tradable symbols only — SPY/QQQ/IWM are feature inputs,
        # not trade candidates.
        day_feat = day_feat[day_feat["symbol"].isin(TRADABLE_SYMS)]
        if day_feat.empty:
            return {}
        X = day_feat[self._cols].to_numpy(dtype=np.float64)
        mask = np.all(np.isfinite(X), axis=1)
        if not mask.any():
            return {}
        preds = self._ridge.predict(X[mask])
        syms = day_feat["symbol"].to_numpy()[mask]
        # Cross-sectional demean → dollar-neutral long-short.
        preds = preds - preds.mean()
        gross = np.abs(preds).sum()
        if gross <= 0:
            return {}
        weights = preds / gross * self._max_gross  # |w| sums to max_gross
        return {str(s): float(w) for s, w in zip(syms, weights)}


class _PrefitFoldsSplitter:
    """Driver-compatible splitter: yields FoldSpec objects whose
    train_idx/test_idx are *date arrays* (what the driver indexes
    ``bars.loc[(dates, slice(None)), :]`` with).

    Pre-fits a Ridge per fold so the signal generator for that fold is
    locked in before the driver iterates the test days. The signal
    generator is injected via a setter on a wrapper driver below.
    """

    def __init__(self, folds: list[FoldSpec]) -> None:
        self._folds = folds

    def split(self, index: pd.DatetimeIndex) -> Iterator[FoldSpec]:
        yield from self._folds


def _prepare_folds_and_adapters(
    bars: pd.DataFrame,
    full_features_pd: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[FoldSpec], dict[int, RidgeSignalAdapter]]:
    """Walk the WalkForwardHarness once to (a) materialize per-fold date
    arrays and (b) fit Ridge per fold and stash adapters keyed by fold_id."""
    all_dates = pd.DatetimeIndex(np.sort(bars.index.get_level_values(0).unique()))
    harness = WalkForwardHarness(
        train_min_years=5, train_max_years=15,
        val_days=252, embargo_days=5, purge_days=1, test_days=21,
        refit_cadence="monthly",
    )

    fold_specs: list[FoldSpec] = []
    adapters: dict[int, RidgeSignalAdapter] = {}

    # Labels for fitting (raw next-day open→close log return).
    # full_features_pd already has features; build labels on it.
    panel_pl = pl.from_pandas(full_features_pd[["symbol", "date", "open", "close"]])
    panel_pl = next_day_open_close_log_return(panel_pl, label_col="y_raw")
    labels_pd = panel_pl.to_pandas()[["symbol", "date", "y_raw"]]
    labels_pd["date"] = pd.to_datetime(labels_pd["date"])
    full = full_features_pd.merge(labels_pd, on=["symbol", "date"], how="left")
    full = full.dropna(subset=["y_raw"])
    # Restrict training to tradable symbols.
    train_pool = full[full["symbol"].isin(TRADABLE_SYMS)].copy()

    for fold in harness.split(all_dates):
        train_dates = all_dates[fold.train_idx]
        test_dates = all_dates[fold.test_idx]

        train_rows = train_pool[train_pool["date"].isin(set(train_dates))]
        if len(train_rows) < 200:
            continue

        # Winsorize on TRAIN fold + per-day cross-sectional z-score.
        bounds = fit_winsorization_bounds(
            train_rows["y_raw"].to_numpy(), sigma=5.0
        )
        train_pl = pl.from_pandas(train_rows[["symbol", "date", "y_raw"]])
        train_pl = winsorize_train_fold(train_pl, bounds, label_col="y_raw", out_col="y_wins")
        train_pl = cross_sectional_zscore(train_pl, label_col="y_wins", out_col="y_xs_z")
        # Feature clipping on train fold.
        feat_pl = pl.from_pandas(train_rows[["symbol", "date"] + feature_cols])
        clip_bounds = clip_features_train_fold(feat_pl, feature_names=tuple(feature_cols))
        feat_pl_clipped = apply_feature_clip(feat_pl, clip_bounds)
        train_full = feat_pl_clipped.join(train_pl, on=["symbol", "date"], how="inner").to_pandas()

        X_train = train_full[feature_cols].to_numpy(dtype=np.float64)
        y_train = train_full["y_xs_z"].to_numpy(dtype=np.float64)
        m = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
        if m.sum() < 100:
            continue

        ridge = RidgeTrainer(config=RidgeConfig(alpha=1.0))
        ridge.fit(X_train[m], y_train[m])

        # Apply the SAME clip bounds to the panel for the test-date features
        # (so the adapter doesn't refit anything at inference).
        full_pl = pl.from_pandas(full_features_pd[["symbol", "date"] + feature_cols])
        full_clipped = apply_feature_clip(full_pl, clip_bounds).to_pandas()
        full_clipped["date"] = pd.to_datetime(full_clipped["date"])
        full_clipped = full_clipped.set_index("date").sort_index()

        adapter = RidgeSignalAdapter(
            ridge=ridge,
            feature_panel=full_clipped,
            feature_cols=feature_cols,
            test_dates=[pd.Timestamp(d) for d in test_dates],
            sorted_all_dates=all_dates,
            max_gross=1.0,
        )

        # Construct FoldSpec with date-array train_idx / test_idx (driver
        # contract — `bars.loc[(fold.train_idx, slice(None)), :]`).
        # Use pandas Timestamp objects (which carry `.year`) — driver iterates
        # `for date in test_dates` and the per-fold fee call needs `date.year`.
        fold_specs.append(
            FoldSpec(
                fold_id=fold.fold_id,
                train_idx=np.array(
                    [pd.Timestamp(d) for d in train_dates], dtype=object
                ),
                val_idx=np.array(
                    [pd.Timestamp(d) for d in all_dates[fold.val_idx]],
                    dtype=object,
                ),
                test_idx=np.array(
                    [pd.Timestamp(d) for d in test_dates], dtype=object
                ),
                fold_meta=fold.fold_meta,
            )
        )
        adapters[fold.fold_id] = adapter

    return fold_specs, adapters


class _PerFoldSignalDispatcher:
    """Driver receives ONE signal_generator, but we need a different
    Ridge-fitted adapter per fold. This dispatcher peeks at the dates passed
    to it and routes to the right pre-fit adapter."""

    def __init__(self, adapters: dict[int, RidgeSignalAdapter], folds: list[FoldSpec]) -> None:
        self._adapters = adapters
        # Map first-test-date → fold_id so we can pick the right adapter.
        self._date_to_fold = {
            pd.Timestamp(f.test_idx[0]): f.fold_id for f in folds
        }
        # Map every test-date → fold_id for the dispatch lookup.
        self._all_test_date_to_fold: dict[pd.Timestamp, int] = {}
        for f in folds:
            for d in f.test_idx:
                self._all_test_date_to_fold[pd.Timestamp(d)] = f.fold_id
        self._current_fold_id: int | None = None
        self._remaining: list[pd.Timestamp] = []

    def __call__(self, bar: pd.Series, train_window: pd.DataFrame) -> dict[str, float]:
        # On a fold boundary, the train_window's *last train date* is what's
        # visible. Simpler: track which fold we're in by peeking at the
        # driver-side iteration order: each call corresponds to *the next
        # test date* of the active fold. We rebuild a queue per fold by
        # detecting changes in the last train-window date.
        if not self._remaining:
            # New fold starting: identify it from the train_window's last row.
            last_train_date = (
                train_window.index.get_level_values(0).max()
                if len(train_window) else None
            )
            # The next test date should be the very next date after the last
            # train date + purge + embargo for the active fold. We instead
            # look up by checking which fold's first test date is > last_train.
            candidates = sorted(
                (first_d, fid) for first_d, fid in self._date_to_fold.items()
                if last_train_date is None or first_d > last_train_date
            )
            if not candidates:
                return {}
            _, fid = candidates[0]
            self._current_fold_id = fid
            self._remaining = list(self._adapters[fid]._dates_iter)  # type: ignore[attr-defined]
            # Reset the adapter's internal iterator so its __call__ pops the
            # same dates we now own.
            self._adapters[fid]._dates_iter = iter(self._remaining)  # type: ignore[attr-defined]
        # Pop our local copy in lockstep so we know when this fold finishes.
        if self._remaining:
            self._remaining.pop(0)
        return self._adapters[self._current_fold_id](bar, train_window)


def main() -> int:
    print("=" * 78)
    print("CYCLE-7 E2E: Ridge × WalkForwardDriver on real ArcticDB bars (Alpaca)")
    print("=" * 78)

    panel = load_panel()
    # Restrict to ≥2019 — gives 5 years of training history (2019-2023) before
    # the first test fold lands in 2024, which is within the SEC §31 + FINRA
    # TAF fee tables (2024-2026). Older years would explode the fee lookup
    # with KeyError on missing year rates (the fee module deliberately
    # refuses to silently default).
    panel = panel.filter(pl.col("date") >= pl.datetime(2019, 1, 1))
    print(f"Panel: {panel.height:,} rows, {panel['symbol'].n_unique()} symbols, "
          f"{panel['date'].min()} → {panel['date'].max()}")

    # Compute the feature panel ONCE on the full universe (SPY/QQQ/IWM needed
    # for vix_proxy + rel_strength_vs_spy broadcasts).
    print("\nComputing features (vix_proxy needs SPY/QQQ/IWM in panel)...")
    feat_pl = compute_features(panel)
    feat_pl = drop_feature_failures(feat_pl)
    full_features_pd = feat_pl.to_pandas()
    full_features_pd["date"] = pd.to_datetime(full_features_pd["date"])
    print(f"  feature rows (all-finite): {len(full_features_pd):,}")

    feature_cols = list(FEATURE_NAMES)

    # Build the bars frame the driver expects (only tradable + benchmarks for
    # consistent date axis; weights on non-tradable are zero, but we keep them
    # in the bars panel so the splitter's date axis is the full one).
    bars = build_bars_for_driver(panel)
    # Restrict bars to tradable universe — the driver charges fees on whatever
    # we hand it; we don't want to accidentally trade SPY/QQQ/IWM if the
    # adapter ever leaks weight to them.
    bars = bars[bars.index.get_level_values(1).isin(TRADABLE_SYMS)]
    print(f"  driver bars rows (tradable only): {len(bars):,}")

    print("\nPre-fitting Ridge per fold + materializing date-array FoldSpecs...")
    fold_specs, adapters = _prepare_folds_and_adapters(
        bars, full_features_pd, feature_cols
    )
    print(f"  prepared {len(fold_specs)} folds")
    if not fold_specs:
        print("ERROR: no folds materialized — insufficient history.")
        return 1

    splitter = _PrefitFoldsSplitter(fold_specs)
    dispatcher = _PerFoldSignalDispatcher(adapters, fold_specs)

    print("\nRunning WalkForwardDriver with broker='alpaca'...")
    driver = WalkForwardDriver(
        bars=bars,
        signal_generator=dispatcher,
        splitter=splitter,
        asset_class="equity",
        initial_capital_usd=100_000.0,
        slippage_k=0.1,
        broker="alpaca",
    )
    result = driver.run()

    # Per-fold report.
    print("\n" + "-" * 78)
    print(f"{'fold':>4} {'n':>4} {'gross':>7} {'net':>7} {'mdd':>7} {'PSR':>6} "
          f"{'fees$':>8} {'slip$':>8} {'turn':>6}")
    print("-" * 78)
    for fm in result.fold_metrics:
        print(
            f"{fm.fold_id:>4} {fm.n_obs:>4d} "
            f"{fm.sharpe_gross:>+7.2f} {fm.sharpe_net:>+7.2f} "
            f"{fm.max_drawdown:>+7.2%} {fm.psr_net:>6.3f} "
            f"{fm.total_fees_usd:>8.0f} {fm.total_slippage_usd:>8.0f} "
            f"{fm.turnover:>6.1f}"
        )

    # Aggregate report.
    print("\n" + "=" * 78)
    print("AGGREGATE — full OOS concatenation across all folds")
    print("=" * 78)
    print(f"  Sharpe gross (training-only proxy, avg-of-folds) : "
          f"{result.aggregate_sharpe_gross:+.3f}")
    print(f"  Sharpe net   (after Alpaca fees + slippage)      : "
          f"{result.aggregate_sharpe_net:+.3f}")
    fee_drag = result.aggregate_sharpe_gross - result.aggregate_sharpe_net
    print(f"  → fee+slippage drag (gross - net)                : "
          f"{fee_drag:+.3f} Sharpe")
    print(f"  Max drawdown (full OOS curve)                    : "
          f"{result.aggregate_max_drawdown:+.2%}")
    print(f"  DSR generic-broken (n_trials={len(result.fold_metrics)} folds) "
          f"  : {result.aggregate_dsr:.4f}")
    # ---- Cycle-8: replace the broken DSR gate. -----------------------------
    # Cycle-7 set n_trials = len(folds), which conflates *sequential walk-
    # forward evaluations of one fitted model* with *independent hyperparam
    # trials*. They are not the same thing — the deflation correction is
    # designed for the latter. The honest number of trials for this Ridge
    # baseline is n_alphas × n_seeds × n_feature_specs from the actual
    # search performed before settling on `alpha=1.0`. For this script:
    # we used 1 alpha, 1 seed (Ridge is deterministic), 1 feature spec →
    # n_trials = 1 → DSR collapses to PSR vs zero, which is the truthful
    # answer for "we did not actually search hyperparameters".
    RIDGE_N_ALPHAS_TRIED = 1   # only alpha=1.0
    RIDGE_N_SEEDS_TRIED = 1    # Ridge is deterministic — no seed sweep
    RIDGE_N_FEATURE_SPECS = 1  # one FEATURE_NAMES tuple, no ablations yet
    honest_n_trials = (
        RIDGE_N_ALPHAS_TRIED * RIDGE_N_SEEDS_TRIED * RIDGE_N_FEATURE_SPECS
    )
    oos_returns = np.diff(np.log(result.equity_curve))
    fold_sharpes = np.asarray(
        [fm.sharpe_net for fm in result.fold_metrics if np.isfinite(fm.sharpe_net)]
    )
    sigma_folds = float(np.std(fold_sharpes, ddof=1)) if fold_sharpes.size > 1 else 1.0
    dsr_honest = bt_metrics.dsr(
        sr_observed=result.aggregate_sharpe_net,
        n_trials=honest_n_trials,
        returns=oos_returns,
    )
    print(
        f"  DSR cycle-8 (n_trials = n_alphas×n_seeds×n_specs = "
        f"{RIDGE_N_ALPHAS_TRIED}×{RIDGE_N_SEEDS_TRIED}×{RIDGE_N_FEATURE_SPECS} = "
        f"{honest_n_trials}) : {dsr_honest:.4f}"
    )
    # Reference: variance-calibrated benchmark held over from cycle-7 for the
    # report only — not the gate any more.
    bench_sr_cal = bt_metrics.dsr_expected_max_sr(
        n_trials=max(2, len(result.fold_metrics)), sr_trials_std=sigma_folds
    )
    dsr_cal = bt_metrics.psr(oos_returns, benchmark_sr=bench_sr_cal)
    print(
        f"  DSR variance-calibrated (reference, sigma_trials="
        f"{sigma_folds:.3f}): {dsr_cal:.4f}"
    )
    print(f"  PBO (IS=gross / OOS=net rank inversion)          : "
          f"{result.aggregate_pbo:.4f}")

    # ---- Cycle-8: bootstrap Sharpe CI (Politis-Romano stationary) ---------
    print("\n" + "-" * 78)
    print("BOOTSTRAP SHARPE 95% CI — Politis-Romano stationary (B=10,000)")
    print("-" * 78)
    boot = bt_metrics.bootstrap_sharpe_ci(
        oos_returns,
        B=10_000,
        block_length="auto",
        alpha=0.05,
        # CONTEXT.md cycle-4 RNG contract: bootstrap CI lives at SeedSequence
        # spawn[0]; here we seed deterministically off the root so a replay
        # reproduces byte-identical bounds.
        seed=int.from_bytes(b"\xB1\x0C\xC0\xDE", "big"),
    )
    print(f"  Point estimate (sample Sharpe, net) : {boot.point_estimate:+.3f}")
    print(f"  95% CI lower bound                  : {boot.ci_lower:+.3f}")
    print(f"  95% CI upper bound                  : {boot.ci_upper:+.3f}")
    print(f"  P(true Sharpe > 0)                  : {boot.p_positive:.4f}")
    print(f"  Auto block-length (Politis-White)   : {boot.block_length:.2f} days")
    print(f"  Resamples                           : {boot.n_resamples:,}")
    total_fees = sum(fm.total_fees_usd for fm in result.fold_metrics)
    total_slip = sum(fm.total_slippage_usd for fm in result.fold_metrics)
    print(f"  Total fees across all folds      : ${total_fees:,.0f}")
    print(f"  Total slippage across all folds  : ${total_slip:,.0f}")

    # Promotion-gate verdict.
    print("\n" + "=" * 78)
    print("PROMOTION GATES (CONTEXT.md §Metrics & gates)")
    print("=" * 78)
    # DSR cycle-8 honest gate.
    if np.isnan(dsr_honest):
        verdict_dsr = (
            "INDETERMINATE (DSR is NaN — sample too small or zero-vol)"
        )
    elif dsr_honest >= 0.95:
        verdict_dsr = (
            f"PASS  (DSR_honest={dsr_honest:.4f} ≥ 0.95, "
            f"n_trials={honest_n_trials})"
        )
    else:
        verdict_dsr = (
            f"FAIL  (DSR_honest={dsr_honest:.4f} < 0.95, "
            f"n_trials={honest_n_trials}) — REJECT"
        )
    # Bootstrap CI lower bound > 0 gate (cycle-8 enforcement).
    if np.isnan(boot.ci_lower):
        verdict_ci = "INDETERMINATE (bootstrap returned NaN)"
    elif boot.ci_lower > 0.0:
        verdict_ci = (
            f"PASS  (95% CI lower = {boot.ci_lower:+.3f} > 0)"
        )
    else:
        verdict_ci = (
            f"FAIL  (95% CI lower = {boot.ci_lower:+.3f} ≤ 0) — REJECT"
        )
    print(f"  DSR honest (cycle-8 fix) : {verdict_dsr}")
    print(f"  Bootstrap CI > 0         : {verdict_ci}")

    # Real expected Sharpe — net is the honest number for tomorrow's paper trading.
    print(f"\n  Honest expected Sharpe for tomorrow's paper trading: "
          f"{result.aggregate_sharpe_net:+.3f}")
    print(f"  (gross was {result.aggregate_sharpe_gross:+.3f}; "
          f"fees+slippage cost us {fee_drag:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
