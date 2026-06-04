"""Cycle-12: long-only top-decile in vix_high (long-short elsewhere).

Cycle 11 ml-research diagnostic uncovered an *inverted decile spread* in the
vix_high bucket: top-decile predicted symbols *under*-perform the bottom-
decile predicted symbols by ~-6.8 bps/day even though the model's rank-IC
is the highest of all three regimes. Mechanically, dollar-neutral long-
short construction shorts the bottom-decile (a +6.8 bps winning leg) and
longs the top-decile (a -6.8 bps losing leg), so the trade goes the wrong
way exactly when the model is most confident.

Hypothesis: in vix_high, the cross-section's *information content* lives
in WHICH names rally rather than in their relative ranking — both decile-1
and decile-10 may have positive forward returns, but decile-10's positives
are smaller than decile-1's. A **long-only top-decile** trade keeps the
informative leg, drops the perverse leg, and (with no shorts) cuts fees
in half. The long-short remains optimal in vix_low / vix_mid (where the
spread sign is correct), so we only swap the construction *conditionally
on regime*.

Methodology — we cannot simply zero out days (that's cycle-10's hard
gate). We must change the *weights* themselves on vix_high days. So we
wrap ``_PerFoldSignalDispatcher`` with a regime-aware adapter that:

  * Calls the underlying dispatcher (gets the cycle-7 long-short
    {sym: weight} dict).
  * Looks up tomorrow's regime label using the same causal RegimeTagger
    cycle-10 used (vix_proxy series via shift(1) quintile cutoffs).
  * In vix_low / vix_mid: returns the cycle-7 dict unchanged.
  * In vix_high: re-ranks the input weights (signed predictions, since
    dollar-neutral demean preserves rank), keeps the TOP-DECILE symbols,
    and emits an EQUAL-WEIGHT long-only book with ``sum(|w|) = max_gross``
    (same gross as the long-short for fair Sharpe comparison; the fee
    saving comes from cutting trade count, not from cutting size).

Top-decile cardinality on an 8-symbol tradable universe is
``max(1, round(0.1 × 8)) = 1`` — i.e. concentrate on the single most-bullish
name in vix_high. This is aggressive but principled: with only 8 names
the granularity choice is "top 1" vs "top 2"; we use the spec literal
0.1 quantile and round to ≥1.

Reports:
  * Aggregate Sharpe vs cycle-10 hard gate (the +2.144 benchmark).
  * Per-regime Sharpe breakdown (the vix_high bucket is the verdict cell).
  * Politis-Romano stationary bootstrap 95% CI on the new stream so the
    point-Sharpe lift carries an honest lower bound.

If the new stream beats cycle-10's +2.144 on point AND on CI lower
bound, the long-only-in-vix_high construction supersedes the hard gate
and we wire it into the live signal generator. If it merely matches the
gate, the engineering complexity is not justified. If it loses, we keep
the cycle-10 hard gate (zero exposure in vix_high) as the live variant.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Re-use cycle-7 wiring.
_spec = _ilu.spec_from_file_location(
    "ridge_e2e", PROJECT_ROOT / "scripts" / "12_backtest_ridge_e2e.py"
)
ridge_e2e = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(ridge_e2e)  # type: ignore[union-attr]

from backtesting import metrics as bt_metrics  # noqa: E402
from backtesting.driver import WalkForwardDriver  # noqa: E402
from backtesting.regime import REGIME_LABELS, RegimeTagger  # noqa: E402
from ml_research.features import (  # noqa: E402
    FEATURE_NAMES,
    compute_features,
    drop_feature_failures,
)


TOP_DECILE_Q = 0.10


def _extract_vix_proxy_series(full_features_pd: pd.DataFrame) -> pd.Series:
    if "vix_proxy" not in full_features_pd.columns:
        raise RuntimeError("vix_proxy column missing from feature panel")
    s = (
        full_features_pd[["date", "vix_proxy"]]
        .drop_duplicates(subset=["date"])
        .set_index("date")["vix_proxy"]
        .sort_index()
    )
    return s.astype(float)


def _per_regime_sharpe(
    returns: np.ndarray, labels: np.ndarray
) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {}
    for regime in REGIME_LABELS:
        mask = labels == regime
        n = int(mask.sum())
        if n < 2:
            out[regime] = (n, float("nan"))
        else:
            out[regime] = (n, float(bt_metrics.sharpe(returns[mask])))
    return out


class _RegimeAwareLongOnlyWrapper:
    """Wrap cycle-7 dispatcher with vix_high → long-only-top-decile rewrite.

    Stateful only via the underlying dispatcher; this object's __call__ is
    purely transformational on the (date-implicit) weights it returns.
    The active date is recovered from the train_window's last index (the
    same trick the dispatcher itself uses).
    """

    def __init__(
        self,
        inner_dispatcher,
        tagger: RegimeTagger,
        vix_series: pd.Series,
        fold_specs,
        max_gross: float = 1.0,
        top_q: float = TOP_DECILE_Q,
    ) -> None:
        self._inner = inner_dispatcher
        self._tagger = tagger
        self._vix_map: dict[pd.Timestamp, float] = {
            pd.Timestamp(k): float(v) for k, v in vix_series.to_dict().items()
        }
        self._max_gross = float(max_gross)
        self._top_q = float(top_q)
        # Replay the dispatcher's date-walk order so we know the active date
        # at each __call__ invocation. The dispatcher walks the test_idx of
        # each FoldSpec in the order folds are yielded.
        self._date_queue: list[pd.Timestamp] = []
        for f in fold_specs:
            for d in f.test_idx:
                self._date_queue.append(pd.Timestamp(d))
        self._cursor = 0

    def __call__(self, bar: pd.Series, train_window: pd.DataFrame) -> dict[str, float]:
        # Get the cycle-7 long-short weights first.
        weights = self._inner(bar, train_window)
        if not weights:
            self._cursor += 1
            return weights

        # Active date — pop the next queued OOS date.
        if self._cursor >= len(self._date_queue):
            return weights
        date = self._date_queue[self._cursor]
        self._cursor += 1

        # Causal regime tag (same as cycle-10 / cycle-11).
        vix_val = self._vix_map.get(date, float("nan"))
        regime = self._tagger.tag(date, vix_val)
        if regime != "vix_high":
            return weights

        # Long-only top-decile rewrite.
        # Cycle-7 weights are dollar-neutral (demeaned preds / |preds|.sum()).
        # The SIGN+RANK of weights still recovers the original pred ranking
        # (positive = above-mean → bullish; we want the top decile of
        # ABSOLUTE predicted-bullishness, which is the largest positive
        # weights only).
        syms = list(weights.keys())
        wvals = np.asarray([weights[s] for s in syms], dtype=np.float64)
        n_syms = len(syms)
        k_top = max(1, int(round(self._top_q * n_syms)))
        # Take the k_top largest weights (most-positive predicted return).
        order = np.argsort(wvals)  # ascending
        top_idx = order[-k_top:]
        per_name = self._max_gross / k_top
        new_weights = {s: 0.0 for s in syms}
        for i in top_idx:
            new_weights[syms[i]] = per_name
        return new_weights


def main() -> int:
    print("=" * 78)
    print("CYCLE-12: long-only top-decile in vix_high (long-short elsewhere)")
    print("=" * 78)

    # ---- replay cycle-7 (identical bars / features / folds) ----------------
    panel = ridge_e2e.load_panel()
    panel = panel.filter(pl.col("date") >= pl.datetime(2019, 1, 1))
    print(f"Panel: {panel.height:,} rows, "
          f"{panel['symbol'].n_unique()} symbols, "
          f"{panel['date'].min()} -> {panel['date'].max()}")

    print("\nComputing features (need vix_proxy)...")
    feat_pl = compute_features(panel)
    feat_pl = drop_feature_failures(feat_pl)
    full_features_pd = feat_pl.to_pandas()
    full_features_pd["date"] = pd.to_datetime(full_features_pd["date"])
    vix_series = _extract_vix_proxy_series(full_features_pd)
    tagger = RegimeTagger.from_series(vix_series)
    print(f"  vix_proxy: {len(vix_series):,} daily values")

    feature_cols = list(FEATURE_NAMES)
    bars = ridge_e2e.build_bars_for_driver(panel)
    bars = bars[bars.index.get_level_values(1).isin(ridge_e2e.TRADABLE_SYMS)]

    print("\nPre-fitting Ridge per fold...")
    fold_specs, adapters = ridge_e2e._prepare_folds_and_adapters(
        bars, full_features_pd, feature_cols
    )
    print(f"  prepared {len(fold_specs)} folds")
    if not fold_specs:
        print("ERROR: no folds materialized.")
        return 1

    splitter = ridge_e2e._PrefitFoldsSplitter(fold_specs)

    # ----- BASELINE run (cycle-7 long-short, no regime intervention) --------
    print("\n[baseline] Running cycle-7 long-short backtest...")
    dispatcher_base = ridge_e2e._PerFoldSignalDispatcher(adapters, fold_specs)
    driver_base = WalkForwardDriver(
        bars=bars,
        signal_generator=dispatcher_base,
        splitter=splitter,
        asset_class="equity",
        initial_capital_usd=100_000.0,
        slippage_k=0.1,
        broker="alpaca",
        regime_tagger=tagger,
        vix_proxy=vix_series,
    )
    result_base = driver_base.run()

    # ----- TREATMENT run (long-only top-decile on vix_high days) ------------
    # Re-prep dispatcher + adapters: the dispatcher mutates adapter state
    # (the date-iterator). Re-build a fresh dispatcher; adapters can be
    # reused only if we also reset their internal _dates_iter — but
    # _prepare_folds_and_adapters fits Ridge per fold which is the
    # expensive step. Cheaper: just re-instantiate the dispatcher; it
    # resets the adapter iterators via the dispatcher's first-call branch.
    print("\n[treatment] Re-prepping fold adapters for clean iterator state...")
    fold_specs_t, adapters_t = ridge_e2e._prepare_folds_and_adapters(
        bars, full_features_pd, feature_cols
    )
    splitter_t = ridge_e2e._PrefitFoldsSplitter(fold_specs_t)
    dispatcher_t = ridge_e2e._PerFoldSignalDispatcher(adapters_t, fold_specs_t)
    wrapped = _RegimeAwareLongOnlyWrapper(
        inner_dispatcher=dispatcher_t,
        tagger=tagger,
        vix_series=vix_series,
        fold_specs=fold_specs_t,
        max_gross=1.0,
        top_q=TOP_DECILE_Q,
    )
    print("[treatment] Running long-only-vix_high backtest...")
    driver_t = WalkForwardDriver(
        bars=bars,
        signal_generator=wrapped,
        splitter=splitter_t,
        asset_class="equity",
        initial_capital_usd=100_000.0,
        slippage_k=0.1,
        broker="alpaca",
        regime_tagger=tagger,
        vix_proxy=vix_series,
    )
    result_t = driver_t.run()

    # ---- recover daily-return streams + dates ------------------------------
    def _streams(result, fspecs) -> tuple[np.ndarray, np.ndarray]:
        eq = np.asarray(result.equity_curve, dtype=np.float64)
        daily = np.empty_like(eq)
        daily[0] = eq[0] - 1.0
        daily[1:] = eq[1:] / eq[:-1] - 1.0
        dates: list[pd.Timestamp] = []
        for f in fspecs:
            dates.extend(pd.Timestamp(d) for d in f.test_idx)
        if len(dates) != eq.size:
            n = min(len(dates), eq.size)
            dates = dates[-n:]
            daily = daily[-n:]
        return daily, np.asarray(dates, dtype="datetime64[ns]")

    daily_base, date_arr = _streams(result_base, fold_specs)
    daily_t, _ = _streams(result_t, fold_specs_t)

    # Regime labels (same date axis for both — folds identical).
    vix_map = vix_series.to_dict()
    labels = np.asarray(
        [tagger.tag(pd.Timestamp(d), vix_map.get(pd.Timestamp(d), float("nan")))
         for d in date_arr]
    )

    # ---- cycle-10 hard-gate stream (post-process baseline) -----------------
    hard = daily_base.copy()
    hard_mask = labels == "vix_high"
    hard[hard_mask] = 0.0
    n_hard = int(hard_mask.sum())

    # ---- aggregate Sharpes -------------------------------------------------
    s_base = float(bt_metrics.sharpe(daily_base))
    s_hard = float(bt_metrics.sharpe(hard))
    s_t = float(bt_metrics.sharpe(daily_t))

    br_base = _per_regime_sharpe(daily_base, labels)
    br_hard = _per_regime_sharpe(hard, labels)
    br_t = _per_regime_sharpe(daily_t, labels)

    # ---- bootstrap CIs -----------------------------------------------------
    seed = int.from_bytes(b"\xB1\x0C\xC0\xDE", "big")
    print("\nBootstrap (Politis-Romano, B=10,000)...")
    boot_base = bt_metrics.bootstrap_sharpe_ci(
        daily_base, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )
    boot_hard = bt_metrics.bootstrap_sharpe_ci(
        hard, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )
    boot_t = bt_metrics.bootstrap_sharpe_ci(
        daily_t, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VARIANT COMPARISON (per-quintile Sharpe of each stream)")
    print("=" * 78)
    hdr = (f"{'Variant':<40} {'Aggregate':>10} {'vix_low':>10} "
           f"{'vix_mid':>10} {'vix_high':>10}")
    print(hdr)
    print("-" * len(hdr))

    def _row(name: str, agg: float,
             br: dict[str, tuple[int, float]]) -> str:
        lo = br["vix_low"][1]
        mi = br["vix_mid"][1]
        hi = br["vix_high"][1]
        return (f"{name:<40} {agg:>+10.3f} {lo:>+10.3f} {mi:>+10.3f} "
                f"{hi:>+10.3f}")

    print(_row("Baseline (cycle-7, long-short)", s_base, br_base))
    print(_row("Hard gate (cycle-10, vix_high=0)", s_hard, br_hard))
    print(_row("Long-only top-decile in vix_high", s_t, br_t))
    n_lo = br_base["vix_low"][0]
    n_mi = br_base["vix_mid"][0]
    n_hi = br_base["vix_high"][0]
    print(f"\n  quintile n_obs : vix_low={n_lo}, vix_mid={n_mi}, "
          f"vix_high={n_hi}; total OOS days={daily_base.size}")
    print(f"  hard-gate flattened days : {n_hard}")
    n_top = max(1, int(round(TOP_DECILE_Q * len(ridge_e2e.TRADABLE_SYMS))))
    print(f"  top-decile size on 8-symbol universe : {n_top} symbol(s)")

    print("\n" + "=" * 78)
    print("BOOTSTRAP 95% CI (Politis-Romano stationary, B=10,000)")
    print("=" * 78)
    print(f"{'Variant':<40} {'point':>8} {'CI_low':>8} {'CI_high':>8} "
          f"{'p(>0)':>8}")
    print("-" * 76)
    for name, b in [
        ("Baseline (cycle-7)", boot_base),
        ("Hard gate (cycle-10)", boot_hard),
        ("Long-only top-decile vix_high (cycle-12)", boot_t),
    ]:
        print(f"{name:<40} {b.point_estimate:>+8.3f} "
              f"{b.ci_lower:>+8.3f} {b.ci_upper:>+8.3f} "
              f"{b.p_positive:>8.4f}")

    print("\n" + "-" * 78)
    print(f"  cycle-12 - hard-gate (point Sharpe) : {s_t - s_hard:+.3f}")
    print(f"  cycle-12 - hard-gate (CI lower)     : "
          f"{boot_t.ci_lower - boot_hard.ci_lower:+.3f}")
    print(f"  cycle-12 - baseline  (point Sharpe) : {s_t - s_base:+.3f}")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    point_beats_hard = s_t > s_hard
    ci_beats_hard = boot_t.ci_lower > boot_hard.ci_lower
    point_beats_base = s_t > s_base
    if point_beats_hard and ci_beats_hard:
        verdict = (
            "CONFIRMED — long-only top-decile in vix_high beats the cycle-10 "
            "hard gate on BOTH point Sharpe and 95% lower bound. The "
            "vix_high cross-section *does* carry exploitable bullish "
            "information; flattening was throwing away alpha. Wire the "
            "regime-conditional construction into the live signal generator."
        )
    elif point_beats_hard and not ci_beats_hard:
        verdict = (
            "PARTIAL — point Sharpe beats cycle-10 but the CI lower bound "
            "does NOT improve. Marginal case; the long-only book is more "
            "concentrated (single name on 8-symbol universe) and the higher "
            "variance shows up in the bootstrap tails. Keep the cycle-10 "
            "hard gate unless concentration risk is later managed by a "
            "broader universe."
        )
    elif (not point_beats_hard) and point_beats_base:
        verdict = (
            "WEAK WIN — long-only top-decile beats the un-gated baseline but "
            "loses to the cycle-10 hard gate. The decile-spread inversion is "
            "real, but the top-decile leg alone is too noisy to beat simple "
            "flattening. Keep the cycle-10 hard gate."
        )
    else:
        verdict = (
            "REJECTED — long-only top-decile in vix_high does NOT beat the "
            "cycle-10 hard gate (and may not beat baseline either). The "
            "inverted decile spread does not translate into a usable "
            "long-only edge on this universe. Keep the cycle-10 hard gate."
        )
    print(f"  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
