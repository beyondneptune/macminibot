"""Cycle-10: counterfactual VIX-high gate impact on cycle-7 Ridge backtest.

Cycle 9 (`14_regime_stratified_ridge.py`) revealed that the vix_high bucket
contributes a -0.252 Sharpe drag inside the +2.03 aggregate. Hypothesis:
*if positions had been flattened on every vix_high session (zero exposure,
zero P&L, zero fee, zero slippage)*, the aggregate Sharpe should LIFT
above +2.03 and the bootstrap 95% lower bound should improve from +0.66.

This script implements the gate as a **post-processing filter** on the
cycle-7 OOS daily returns: re-tag every OOS session date by VIX regime,
then zero out the per-day net return wherever ``regime == "vix_high"``.
Re-compute aggregate Sharpe + per-regime Sharpe + Politis-Romano
bootstrap CI on the gated stream.

Honest caveat: this is the *idealised* counterfactual — it assumes we can
perfectly forecast tomorrow's VIX regime at the prior session's close
using only causally-available data. The cycle-9 tagger already uses
``shift(1)`` quintile cutoffs so the bucket label *itself* is causal, but
the gate decision (flatten or trade) is made at the close of T-1 using
the vix_proxy value AT T, which is a one-day look-ahead. The fully
causal version (use T-1's vix_proxy value to decide T's exposure) is
the deployment path; this script reports the upper-bound counterfactual
so we know the maximum Sharpe lift the gate could deliver. If even the
idealised lift is small, the deployable lift will be smaller — kill
the gate before wiring it. If the lift is large, justify the engineering
to swap the gate into the live signal generator.
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
        raise RuntimeError(
            "vix_proxy column missing from feature panel"
        )
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
    print("CYCLE-10: counterfactual VIX-high gate on cycle-7 Ridge backtest")
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
    # Driver: equity = cumprod(1 + oos_arr), length == len(full_oos_returns).
    # Recover r: r[0] = eq[0] - 1; r[i] = eq[i]/eq[i-1] - 1 for i >= 1.
    eq = np.asarray(result.equity_curve, dtype=np.float64)
    if eq.size < 2:
        print("ERROR: equity curve too short to recover returns.")
        return 1
    daily_net = np.empty_like(eq)
    daily_net[0] = eq[0] - 1.0
    daily_net[1:] = eq[1:] / eq[:-1] - 1.0

    # Re-derive the OOS session date list (same order the driver walks).
    oos_dates: list[pd.Timestamp] = []
    for f in fold_specs:
        oos_dates.extend(pd.Timestamp(d) for d in f.test_idx)
    # Length should match equity_curve. Defensive trim if not.
    if len(oos_dates) != eq.size:
        n = min(len(oos_dates), eq.size)
        oos_dates = oos_dates[-n:]
        daily_net = daily_net[-n:]
    date_arr = np.asarray(oos_dates, dtype="datetime64[ns]")

    # Tag each return date by regime.
    vix_map = vix_series.to_dict()
    labels = np.asarray(
        [tagger.tag(pd.Timestamp(d), vix_map.get(pd.Timestamp(d), float("nan")))
         for d in date_arr]
    )

    # ---- baseline (un-gated) numbers --------------------------------------
    base_sharpe = float(bt_metrics.sharpe(daily_net))
    base_by_regime = _per_regime_sharpe(daily_net, labels)

    # ---- counterfactual gated stream: zero out vix_high days --------------
    gated = daily_net.copy()
    gate_mask = labels == "vix_high"
    n_gated = int(gate_mask.sum())
    gated[gate_mask] = 0.0
    gated_sharpe = float(bt_metrics.sharpe(gated))
    gated_by_regime = _per_regime_sharpe(gated, labels)

    # ---- bootstrap CI on both streams -------------------------------------
    seed = int.from_bytes(b"\xB1\x0C\xC0\xDE", "big")
    print("\nBootstrap (Politis-Romano, B=10,000) on baseline + gated...")
    boot_base = bt_metrics.bootstrap_sharpe_ci(
        daily_net, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )
    boot_gated = bt_metrics.bootstrap_sharpe_ci(
        gated, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VARIANT COMPARISON")
    print("=" * 78)
    hdr = (f"{'Variant':<28} {'Aggregate':>10} {'vix_low':>10} "
           f"{'vix_mid':>10} {'vix_high':>10}")
    print(hdr)
    print("-" * len(hdr))

    def _row(name: str, agg: float,
             br: dict[str, tuple[int, float]]) -> str:
        lo = br["vix_low"][1]
        mi = br["vix_mid"][1]
        hi = br["vix_high"][1]
        return (f"{name:<28} {agg:>+10.3f} {lo:>+10.3f} {mi:>+10.3f} "
                f"{hi:>+10.3f}")

    print(_row("Baseline (cycle-7)", base_sharpe, base_by_regime))
    print(_row("Gated (flat on vix_high)", gated_sharpe, gated_by_regime))
    n_lo = base_by_regime["vix_low"][0]
    n_mi = base_by_regime["vix_mid"][0]
    n_hi = base_by_regime["vix_high"][0]
    print(f"  (n_obs per bucket: vix_low={n_lo}, vix_mid={n_mi}, "
          f"vix_high={n_hi}; total OOS days={daily_net.size}; "
          f"gated days={n_gated})")

    delta = gated_sharpe - base_sharpe
    print(f"\n  Gate Δ Sharpe (gated - baseline) : {delta:+.3f}")

    print("\n" + "=" * 78)
    print("BOOTSTRAP 95% CI")
    print("=" * 78)
    print(f"{'Variant':<28} {'point':>8} {'CI_low':>8} {'CI_high':>8} "
          f"{'p(>0)':>8}")
    print("-" * 64)
    print(f"{'Baseline':<28} {boot_base.point_estimate:>+8.3f} "
          f"{boot_base.ci_lower:>+8.3f} {boot_base.ci_upper:>+8.3f} "
          f"{boot_base.p_positive:>8.4f}")
    print(f"{'Gated (vix_high → flat)':<28} "
          f"{boot_gated.point_estimate:>+8.3f} "
          f"{boot_gated.ci_lower:>+8.3f} {boot_gated.ci_upper:>+8.3f} "
          f"{boot_gated.p_positive:>8.4f}")
    ci_delta = boot_gated.ci_lower - boot_base.ci_lower
    print(f"\n  Lower-bound improvement (gated - baseline) : {ci_delta:+.3f}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("HYPOTHESIS VERDICT")
    print("=" * 78)
    if delta > 0 and ci_delta > 0:
        verdict = (
            "CONFIRMED — gating vix_high lifts both the point Sharpe and the "
            "95% lower bound. Engineering a causal version of the gate is "
            "justified."
        )
    elif delta > 0:
        verdict = (
            "PARTIAL — point Sharpe lifts but the CI lower bound does NOT "
            "improve. The gate buys nicer averages without statistical-"
            "robustness gain. Marginal case."
        )
    else:
        verdict = (
            "REJECTED — gating vix_high does not lift the aggregate Sharpe. "
            "The -0.252 vix_high drag is offset by alpha lost on those days "
            "that helps the overall ratio. Do NOT wire the gate."
        )
    print(f"  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
