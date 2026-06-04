"""Cycle-9 Epic 17: regime-stratified DSR breakdown on the cycle-7 Ridge backtest.

Wires the new ``backtesting.regime.RegimeTagger`` into ``WalkForwardDriver``
and re-runs the same Ridge baseline that produced the +2.03 net Sharpe in
cycle 7. Reports per-regime (vix_low / vix_mid / vix_high) Sharpe + DSR +
fee_drag so we can see whether the alpha is structurally present across
vol regimes or concentrated in low-vol bull windows.

Promotion-flag rule (per CONTEXT improvement #14): any regime where DSR
collapses by >0.3 vs the aggregate is flagged regardless of whether the
aggregate gate passes.
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

# Reuse all the cycle-7 wiring (universe, fold prep, dispatcher).
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "ridge_e2e", PROJECT_ROOT / "scripts" / "12_backtest_ridge_e2e.py"
)
ridge_e2e = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(ridge_e2e)  # type: ignore[union-attr]

from backtesting.driver import WalkForwardDriver  # noqa: E402
from backtesting.regime import REGIME_LABELS, RegimeTagger  # noqa: E402
from ml_research.features import FEATURE_NAMES  # noqa: E402


def _extract_vix_proxy_series(full_features_pd: pd.DataFrame) -> pd.Series:
    """Pull the per-date vix_proxy series from the already-computed feature
    panel. vix_proxy is broadcast across symbols on each date (it's a
    market-wide measure), so `first()` per date is the canonical value.
    """
    if "vix_proxy" not in full_features_pd.columns:
        raise RuntimeError(
            "vix_proxy column missing from feature panel — "
            "compute_features should always include it."
        )
    s = (
        full_features_pd[["date", "vix_proxy"]]
        .drop_duplicates(subset=["date"])
        .set_index("date")["vix_proxy"]
        .sort_index()
    )
    return s.astype(float)


def main() -> int:
    print("=" * 78)
    print("CYCLE-9 EPIC 17: Regime-stratified DSR on cycle-7 Ridge backtest")
    print("=" * 78)

    panel = ridge_e2e.load_panel()
    panel = panel.filter(pl.col("date") >= pl.datetime(2019, 1, 1))
    print(f"Panel: {panel.height:,} rows, "
          f"{panel['symbol'].n_unique()} symbols, "
          f"{panel['date'].min()} → {panel['date'].max()}")

    print("\nComputing features (need vix_proxy for the tagger)...")
    from ml_research.features import compute_features, drop_feature_failures
    feat_pl = compute_features(panel)
    feat_pl = drop_feature_failures(feat_pl)
    full_features_pd = feat_pl.to_pandas()
    full_features_pd["date"] = pd.to_datetime(full_features_pd["date"])

    vix_series = _extract_vix_proxy_series(full_features_pd)
    print(f"  vix_proxy: {len(vix_series):,} daily values, "
          f"{vix_series.index.min()} → {vix_series.index.max()}")
    print(f"  vix_proxy summary: mean={vix_series.mean():.5f}, "
          f"std={vix_series.std():.5f}, "
          f"min={vix_series.min():.5f}, max={vix_series.max():.5f}")

    tagger = RegimeTagger.from_series(vix_series)

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

    print("\nRunning WalkForwardDriver with regime tagger...")
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

    # Aggregate baseline (matches cycle-7 numbers).
    print("\n" + "=" * 78)
    print("AGGREGATE OOS (baseline cycle-7 numbers, re-confirmed)")
    print("=" * 78)
    print(f"  Sharpe net (aggregate)      : {result.aggregate_sharpe_net:+.3f}")
    print(f"  Sharpe gross (avg of folds) : {result.aggregate_sharpe_gross:+.3f}")
    print(f"  Aggregate DSR (n_folds)     : {result.aggregate_dsr:.4f}")
    print(f"  Max drawdown                : {result.aggregate_max_drawdown:+.2%}")

    # Regime breakdown.
    print("\n" + "=" * 78)
    print("REGIME-STRATIFIED BREAKDOWN (Epic 17 cycle-9)")
    print("=" * 78)
    print(f"{'regime':>10}  {'n':>5}  {'Sharpe_net':>11}  {'DSR':>7}  "
          f"{'fee_drag':>9}")
    print("-" * 50)
    by_regime = {rm.regime: rm for rm in result.regime_metrics}
    for label in REGIME_LABELS:
        rm = by_regime.get(label)
        if rm is None:
            print(f"{label:>10}  (missing)")
            continue
        sr_str = (f"{rm.sharpe_net:>+11.3f}"
                  if np.isfinite(rm.sharpe_net) else f"{'nan':>11}")
        dsr_str = (f"{rm.dsr:>7.4f}" if np.isfinite(rm.dsr)
                   else f"{'nan':>7}")
        fd_str = (f"{rm.fee_drag_sharpe:>+9.3f}"
                  if np.isfinite(rm.fee_drag_sharpe) else f"{'nan':>9}")
        print(f"{label:>10}  {rm.n_obs:>5d}  {sr_str}  {dsr_str}  {fd_str}")

    # >0.3 DSR-collapse flag (CONTEXT improvement #14 rule).
    print("\n" + "-" * 78)
    print("DSR-COLLAPSE FLAGS (per CONTEXT #14 — collapse > 0.3 vs aggregate)")
    print("-" * 78)
    agg_dsr = result.aggregate_dsr
    if not np.isfinite(agg_dsr):
        print("  aggregate DSR is NaN — cannot evaluate collapse rule.")
    else:
        flagged_any = False
        for rm in result.regime_metrics:
            if not np.isfinite(rm.dsr):
                continue
            delta = agg_dsr - rm.dsr
            if delta > 0.3:
                print(f"  FLAG  {rm.regime}: DSR={rm.dsr:.4f} "
                      f"(agg={agg_dsr:.4f}, Δ={delta:+.3f})")
                flagged_any = True
        if not flagged_any:
            print("  no regime exhibits >0.3 DSR collapse vs aggregate.")

    # Headline verdict on the cycle-7 +2.03 Sharpe.
    print("\n" + "=" * 78)
    print("HEADLINE — does +2.03 net Sharpe hold up in vix_high?")
    print("=" * 78)
    high = by_regime.get("vix_high")
    if high is None or not np.isfinite(high.sharpe_net):
        print("  vix_high bucket empty or Sharpe NaN — verdict INDETERMINATE.")
    else:
        if high.sharpe_net >= 1.0:
            verdict = "HOLDS  (vix_high Sharpe still strongly positive)"
        elif high.sharpe_net > 0.0:
            verdict = "PARTIAL  (vix_high Sharpe positive but materially diminished)"
        else:
            verdict = ("CONCENTRATED IN LOW-VOL  (vix_high Sharpe ≤ 0 — "
                       "aggregate +2.03 is bull-window driven)")
        print(f"  vix_high Sharpe = {high.sharpe_net:+.3f}  →  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
