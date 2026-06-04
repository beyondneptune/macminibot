"""Cycle-13: counterfactual "vix_mid ONLY" gate on cycle-7 Ridge backtest.

Cycle-12 (`18_decile_sharpe_per_regime.py`) decomposed the per-regime
decile-Sharpe curves and found:

  * vix_low  -> U-shape (extremes lose money, middle wins).  Decile-rank
    correlation is weak / negative; trading the model's confident
    extremes here is a drag.
  * vix_mid  -> the only cleanly *monotonic_up* regime.  D1-Sharpe
    negative, D10-Sharpe positive, D10-D1 ≈ +0.87.  Standard momentum.
  * vix_high -> *inverted-monotonic* (the model predicts the wrong sign;
    D10 underperforms D1 by 0.5+ Sharpe).  Already gated in cycle-10.

Cycle-10 (`15_vix_gated_ridge.py`) zeroed only vix_high and reported the
lift.  That left the vix_low U-shape drag untouched.  This cycle pushes
the logical conclusion: trade ONLY on vix_mid days, flatten on both
vix_low AND vix_high.  Hypothesis: aggregate Sharpe lifts substantially
above the cycle-10 gated number because we strip BOTH drags (the
vix_low U-shape and the vix_high inversion).

Mechanism is the same post-processing filter as cycle-10 — replay the
cycle-7 OOS daily-net stream, tag every date by VIX regime, zero out
every day where ``regime != "vix_mid"``.  Re-bootstrap CI on the
filtered stream and compare against (a) the un-gated cycle-7 baseline
and (b) the cycle-10 vix_high-only gate.

Honest caveat (same as cycle-10): the regime label uses the vix_proxy
value AT day T to decide whether to trade T.  A deployable version
would use T-1's vix_proxy.  This script reports the *upper-bound*
counterfactual — if the lift is small even here, the deployable lift
is smaller and we should drop the idea.  If the lift is large, the
engineering to swap the gate into the live signal generator is
justified.
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

# Re-use cycle-7 wiring (universe, panel loader, fold prep, dispatcher).
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


def main() -> int:
    print("=" * 78)
    print("CYCLE-13: 'vix_mid ONLY' gate vs cycle-10 vix_high-only gate")
    print("=" * 78)

    # ---- replay the cycle-7 backtest with the regime tagger attached -----
    panel = ridge_e2e.load_panel()
    panel = panel.filter(pl.col("date") >= pl.datetime(2019, 1, 1))
    print(f"Panel: {panel.height:,} rows, "
          f"{panel['symbol'].n_unique()} symbols, "
          f"{panel['date'].min()} → {panel['date'].max()}")

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
    dispatcher = ridge_e2e._PerFoldSignalDispatcher(adapters, fold_specs)

    print("\nRunning WalkForwardDriver (broker='alpaca', regime tagger on)...")
    driver = WalkForwardDriver(
        bars=bars,
        signal_generator=dispatcher,
        splitter=splitter,
        asset_class="equity",
        initial_capital_usd=100_000.0,
        slippage_k=0.1,
        broker="alpaca",
        regime_tagger=tagger,
        vix_proxy=vix_series,
    )
    result = driver.run()

    # ---- recover the per-day OOS net-return stream + dates -----------------
    eq = np.asarray(result.equity_curve, dtype=np.float64)
    if eq.size < 2:
        print("ERROR: equity curve too short to recover returns.")
        return 1
    daily_net = np.empty_like(eq)
    daily_net[0] = eq[0] - 1.0
    daily_net[1:] = eq[1:] / eq[:-1] - 1.0

    oos_dates: list[pd.Timestamp] = []
    for f in fold_specs:
        oos_dates.extend(pd.Timestamp(d) for d in f.test_idx)
    if len(oos_dates) != eq.size:
        n = min(len(oos_dates), eq.size)
        oos_dates = oos_dates[-n:]
        daily_net = daily_net[-n:]
    date_arr = np.asarray(oos_dates, dtype="datetime64[ns]")

    vix_map = vix_series.to_dict()
    labels = np.asarray(
        [tagger.tag(pd.Timestamp(d), vix_map.get(pd.Timestamp(d), float("nan")))
         for d in date_arr]
    )

    # ---- baseline (un-gated) numbers --------------------------------------
    base_sharpe = float(bt_metrics.sharpe(daily_net))
    base_by_regime = _per_regime_sharpe(daily_net, labels)

    # ---- counterfactual #1: cycle-10 — zero out vix_high days only --------
    gated_hi = daily_net.copy()
    hi_mask = labels == "vix_high"
    n_hi = int(hi_mask.sum())
    gated_hi[hi_mask] = 0.0
    gated_hi_sharpe = float(bt_metrics.sharpe(gated_hi))
    gated_hi_by_regime = _per_regime_sharpe(gated_hi, labels)

    # ---- counterfactual #2: cycle-13 — trade ONLY vix_mid -----------------
    # i.e. zero out both vix_low AND vix_high days.
    gated_mid = daily_net.copy()
    not_mid_mask = labels != "vix_mid"
    n_not_mid = int(not_mid_mask.sum())
    n_lo_gated = int((labels == "vix_low").sum())
    n_hi_gated = int((labels == "vix_high").sum())
    n_mid_kept = int((labels == "vix_mid").sum())
    gated_mid[not_mid_mask] = 0.0
    gated_mid_sharpe = float(bt_metrics.sharpe(gated_mid))
    gated_mid_by_regime = _per_regime_sharpe(gated_mid, labels)

    # ---- bootstrap CI on all three streams --------------------------------
    seed = int.from_bytes(b"\xB1\x0C\xC0\xDE", "big")
    print("\nBootstrap (Politis-Romano, B=10,000) on baseline + both gates...")
    boot_base = bt_metrics.bootstrap_sharpe_ci(
        daily_net, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )
    boot_hi = bt_metrics.bootstrap_sharpe_ci(
        gated_hi, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )
    boot_mid = bt_metrics.bootstrap_sharpe_ci(
        gated_mid, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VARIANT COMPARISON — aggregate + per-regime Sharpe")
    print("=" * 78)
    hdr = (f"{'Variant':<32} {'Aggregate':>10} {'vix_low':>10} "
           f"{'vix_mid':>10} {'vix_high':>10}")
    print(hdr)
    print("-" * len(hdr))

    def _row(name: str, agg: float,
             br: dict[str, tuple[int, float]]) -> str:
        lo = br["vix_low"][1]
        mi = br["vix_mid"][1]
        hi = br["vix_high"][1]
        return (f"{name:<32} {agg:>+10.3f} {lo:>+10.3f} {mi:>+10.3f} "
                f"{hi:>+10.3f}")

    print(_row("Baseline (cycle-7, un-gated)", base_sharpe, base_by_regime))
    print(_row("Cycle-10 (flat on vix_high)", gated_hi_sharpe,
               gated_hi_by_regime))
    print(_row("Cycle-13 (vix_mid ONLY)", gated_mid_sharpe,
               gated_mid_by_regime))

    n_lo_total = base_by_regime["vix_low"][0]
    n_mi_total = base_by_regime["vix_mid"][0]
    n_hi_total = base_by_regime["vix_high"][0]
    print(f"\n  (n_obs per bucket: vix_low={n_lo_total}, "
          f"vix_mid={n_mi_total}, vix_high={n_hi_total}; "
          f"total OOS days={daily_net.size})")
    print(f"  Cycle-10 zero'd: {n_hi} vix_high days")
    print(f"  Cycle-13 zero'd: {n_not_mid} days "
          f"({n_lo_gated} vix_low + {n_hi_gated} vix_high); "
          f"kept {n_mid_kept} vix_mid days")

    print("\n" + "=" * 78)
    print("DELTA TABLE — vs un-gated baseline")
    print("=" * 78)
    d_hi = gated_hi_sharpe - base_sharpe
    d_mid = gated_mid_sharpe - base_sharpe
    d_incr = gated_mid_sharpe - gated_hi_sharpe
    print(f"  Cycle-10 lift  (gated_hi  - baseline)   : {d_hi:+.3f}")
    print(f"  Cycle-13 lift  (gated_mid - baseline)   : {d_mid:+.3f}")
    print(f"  Cycle-13 incr  (gated_mid - gated_hi)   : {d_incr:+.3f}")

    print("\n" + "=" * 78)
    print("BOOTSTRAP 95% CI")
    print("=" * 78)
    print(f"{'Variant':<32} {'point':>8} {'CI_low':>8} {'CI_high':>8} "
          f"{'p(>0)':>8}")
    print("-" * 68)
    print(f"{'Baseline':<32} {boot_base.point_estimate:>+8.3f} "
          f"{boot_base.ci_lower:>+8.3f} {boot_base.ci_upper:>+8.3f} "
          f"{boot_base.p_positive:>8.4f}")
    print(f"{'Cycle-10 (vix_high -> flat)':<32} "
          f"{boot_hi.point_estimate:>+8.3f} "
          f"{boot_hi.ci_lower:>+8.3f} {boot_hi.ci_upper:>+8.3f} "
          f"{boot_hi.p_positive:>8.4f}")
    print(f"{'Cycle-13 (vix_mid ONLY)':<32} "
          f"{boot_mid.point_estimate:>+8.3f} "
          f"{boot_mid.ci_lower:>+8.3f} {boot_mid.ci_upper:>+8.3f} "
          f"{boot_mid.p_positive:>8.4f}")
    ci_d_hi = boot_hi.ci_lower - boot_base.ci_lower
    ci_d_mid = boot_mid.ci_lower - boot_base.ci_lower
    ci_d_incr = boot_mid.ci_lower - boot_hi.ci_lower
    print(f"\n  Cycle-10 CI_low lift   : {ci_d_hi:+.3f}")
    print(f"  Cycle-13 CI_low lift   : {ci_d_mid:+.3f}")
    print(f"  Cycle-13 CI_low incr   : {ci_d_incr:+.3f}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("HYPOTHESIS VERDICT")
    print("=" * 78)
    if d_incr > 0 and ci_d_incr > 0:
        verdict = (
            "CONFIRMED — gating BOTH vix_low and vix_high (trade vix_mid "
            "only) lifts point Sharpe AND CI lower bound above the cycle-10 "
            "vix_high-only gate.  The vix_low U-shape really was a drag; "
            "ship the symmetric gate."
        )
    elif d_incr > 0:
        verdict = (
            "PARTIAL — point Sharpe rises vs cycle-10 but CI lower bound "
            "does NOT.  Filtering vix_low improves the average without "
            "tightening robustness — marginal benefit, probably noise on "
            "the gain side."
        )
    elif d_mid > 0:
        verdict = (
            "REJECTED — vix_mid-only is WORSE than the cycle-10 vix_high-"
            "only gate.  vix_low contributes net-positive Sharpe in "
            "aggregate even though its decile curve is U-shaped (the "
            "middle deciles' positive PnL outweighs the extremes' losses)."
        )
    else:
        verdict = (
            "REJECTED — vix_mid-only fails to beat the un-gated baseline. "
            "Throwing away ~2/3 of trading days costs more alpha than the "
            "drag-removal saves.  Do NOT ship this gate."
        )
    print(f"  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
