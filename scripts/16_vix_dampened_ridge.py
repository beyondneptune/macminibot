"""Cycle-11: counterfactual VIX *soft-dampening* impact on cycle-7 Ridge backtest.

Cycle 10 (``15_vix_gated_ridge.py``) confirmed that flattening the strategy
on every vix_high session (top quintile, 80th-pct cutoff) lifts the aggregate
Sharpe from +2.027 → +2.144 and improves the 95% lower bound. That is the
*hard gate* (approach 1 in strategy_engine/sizing.py).

Cycle 11 evaluates the *soft dampening* companion
(``vol_target_with_regime_dampening`` — approach 2): instead of a binary
flatten-on-vix_high, scale exposure smoothly:

    pct_vix < 60   → full size  (1.0×)
    60 ≤ pct < 80  → half size  (0.5×)
    pct ≥ 80       → zero size  (0.0×)  (same as hard gate at this band)

The expectation is the soft variant should land *between* baseline and
hard-gate: dampening 60-80 trades some upside for variance reduction, while
zeroing 80+ retains the hard-gate's drag elimination.

Same post-processing methodology as cycle-10: we replay cycle-7, recover the
per-OOS-day net return stream, and apply the dampening multiplier on each
day. The percentile threshold is computed causally — for date T we use the
trailing 252d window of vix_proxy ending at T-1 (shift(1)), exactly matching
the RegimeTagger's causality contract. This is the idealised counterfactual
(the deployable version multiplies *weights* upstream of the fee engine,
which would non-linearly reduce fees + slippage on dampened days — so the
true lift is at least what we measure here, plausibly slightly higher).
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
from strategy_engine.sizing import (  # noqa: E402
    VIX_DAMPEN_INTERMEDIATE_MULT,
    VIX_DAMPEN_PCT_HALT,
    VIX_DAMPEN_PCT_INTERMEDIATE,
)

# Same trailing-window length as RegimeTagger so cycle-10 (quintile gate)
# and cycle-11 (60/80-pct soft) live on the same causal window.
_LOOKBACK_DAYS = 252


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


def _build_soft_dampen_multipliers(
    dates: list[pd.Timestamp],
    vix_series: pd.Series,
    *,
    pct_intermediate: float = VIX_DAMPEN_PCT_INTERMEDIATE,
    pct_halt: float = VIX_DAMPEN_PCT_HALT,
    intermediate_mult: float = VIX_DAMPEN_INTERMEDIATE_MULT,
    lookback_days: int = _LOOKBACK_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each OOS date, compute (multiplier, percentile, regime-string).

    Causal: for date T, the percentile rank of vix_proxy[T] is computed
    against the trailing ``lookback_days`` window ending at T-1 (matches
    ``RegimeTagger`` shift(1) semantics — no peek at T or future).

    Returns
    -------
    mults : np.ndarray of float in {1.0, 0.5, 0.0}
    pcts  : np.ndarray of float in [0, 1] (NaN where window not warm)
    labels: np.ndarray of str in {"normal", "intermediate", "halt"}
    """
    s = vix_series.sort_index().astype(float)
    n = len(dates)
    mults = np.ones(n, dtype=np.float64)
    pcts = np.full(n, np.nan, dtype=np.float64)
    labels = np.array(["normal"] * n, dtype=object)
    # Build a numpy-indexed view for window slicing.
    idx = s.index
    vals = s.to_numpy()
    # Pre-map each test date to its position in the full series.
    pos_map: dict[pd.Timestamp, int] = {pd.Timestamp(d): i for i, d in enumerate(idx)}
    for k, d in enumerate(dates):
        ts = pd.Timestamp(d)
        pos = pos_map.get(ts)
        if pos is None:
            continue
        v_today = vals[pos]
        if not np.isfinite(v_today):
            continue
        # Trailing window: positions [pos - lookback, pos - 1], inclusive.
        lo = max(0, pos - lookback_days)
        window = vals[lo:pos]  # excludes pos (shift(1) semantics)
        window = window[np.isfinite(window)]
        # Require the same warm-up gate as the RegimeTagger (≥ 20 obs OR
        # half-the-window, whichever is higher). Until warm, leave mult=1.0
        # (fail-safe to baseline — no dampening rather than over-restrict).
        min_warm = max(20, lookback_days // 2)
        if window.size < min_warm:
            continue
        pct = float(np.mean(window < v_today))
        pcts[k] = pct
        if pct >= pct_halt:
            mults[k] = 0.0
            labels[k] = "halt"
        elif pct >= pct_intermediate:
            mults[k] = intermediate_mult
            labels[k] = "intermediate"
        # else: stays at 1.0 / "normal"
    return mults, pcts, labels


def main() -> int:
    print("=" * 78)
    print("CYCLE-11: counterfactual VIX soft-dampening on cycle-7 Ridge backtest")
    print("=" * 78)

    # ---- replay cycle-7 (identical to cycle-10) ---------------------------
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

    # ---- recover per-OOS-day net-return stream + dates --------------------
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

    # Quintile regime labels (cycle-10 / cycle-9 buckets).
    vix_map = vix_series.to_dict()
    quintile_labels = np.asarray(
        [tagger.tag(pd.Timestamp(d), vix_map.get(pd.Timestamp(d), float("nan")))
         for d in date_arr]
    )

    # ---- baseline (cycle-7, un-gated) -------------------------------------
    base_sharpe = float(bt_metrics.sharpe(daily_net))
    base_by_regime = _per_regime_sharpe(daily_net, quintile_labels)

    # ---- hard gate (cycle-10): zero vix_high days -------------------------
    hard_gated = daily_net.copy()
    hard_mask = quintile_labels == "vix_high"
    hard_gated[hard_mask] = 0.0
    hard_sharpe = float(bt_metrics.sharpe(hard_gated))
    hard_by_regime = _per_regime_sharpe(hard_gated, quintile_labels)
    n_hard_gated = int(hard_mask.sum())

    # ---- NEW cycle-11: soft dampening (60/80-pct, 0.5×/0.0×) --------------
    mults, pcts, soft_labels = _build_soft_dampen_multipliers(
        oos_dates, vix_series
    )
    soft = daily_net * mults
    soft_sharpe = float(bt_metrics.sharpe(soft))
    soft_by_regime = _per_regime_sharpe(soft, quintile_labels)
    n_intermediate = int((soft_labels == "intermediate").sum())
    n_halt = int((soft_labels == "halt").sum())
    n_normal = int((soft_labels == "normal").sum())

    # ---- bootstrap CIs on all three streams -------------------------------
    seed = int.from_bytes(b"\xB1\x0C\xC0\xDE", "big")
    print("\nBootstrap (Politis-Romano, B=10,000) on baseline + hard-gate + soft...")
    boot_base = bt_metrics.bootstrap_sharpe_ci(
        daily_net, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )
    boot_hard = bt_metrics.bootstrap_sharpe_ci(
        hard_gated, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )
    boot_soft = bt_metrics.bootstrap_sharpe_ci(
        soft, B=10_000, block_length="auto", alpha=0.05, seed=seed,
    )

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VARIANT COMPARISON (per-quintile Sharpe of the gated stream)")
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

    print(_row("Baseline (cycle-7)", base_sharpe, base_by_regime))
    print(_row("Hard gate (cycle-10, vix_high=0)",
               hard_sharpe, hard_by_regime))
    print(_row("Soft dampen (cycle-11, 60/80)",
               soft_sharpe, soft_by_regime))

    n_lo = base_by_regime["vix_low"][0]
    n_mi = base_by_regime["vix_mid"][0]
    n_hi = base_by_regime["vix_high"][0]
    print(f"\n  quintile n_obs : vix_low={n_lo}, vix_mid={n_mi}, "
          f"vix_high={n_hi}; total OOS days={daily_net.size}")
    print(f"  hard-gate flattened days : {n_hard_gated}")
    print(f"  soft-dampen breakdown    : normal={n_normal}, "
          f"intermediate(0.5x)={n_intermediate}, halt(0.0x)={n_halt}")

    print("\n" + "=" * 78)
    print("BOOTSTRAP 95% CI (Politis-Romano stationary, B=10,000)")
    print("=" * 78)
    print(f"{'Variant':<32} {'point':>8} {'CI_low':>8} {'CI_high':>8} "
          f"{'p(>0)':>8}")
    print("-" * 68)
    print(f"{'Baseline (cycle-7)':<32} {boot_base.point_estimate:>+8.3f} "
          f"{boot_base.ci_lower:>+8.3f} {boot_base.ci_upper:>+8.3f} "
          f"{boot_base.p_positive:>8.4f}")
    print(f"{'Hard gate (cycle-10)':<32} {boot_hard.point_estimate:>+8.3f} "
          f"{boot_hard.ci_lower:>+8.3f} {boot_hard.ci_upper:>+8.3f} "
          f"{boot_hard.p_positive:>8.4f}")
    print(f"{'Soft dampen (cycle-11)':<32} {boot_soft.point_estimate:>+8.3f} "
          f"{boot_soft.ci_lower:>+8.3f} {boot_soft.ci_upper:>+8.3f} "
          f"{boot_soft.p_positive:>8.4f}")

    print("\n" + "-" * 78)
    print(f"  hard-gate delta (vs baseline) : "
          f"point={hard_sharpe - base_sharpe:+.3f}, "
          f"CI_low={boot_hard.ci_lower - boot_base.ci_lower:+.3f}")
    print(f"  soft-dampen delta (vs baseline) : "
          f"point={soft_sharpe - base_sharpe:+.3f}, "
          f"CI_low={boot_soft.ci_lower - boot_base.ci_lower:+.3f}")
    print(f"  soft - hard (point)             : "
          f"{soft_sharpe - hard_sharpe:+.3f} "
          f"(negative => soft is *worse* than hard; that's the cost of "
          f"keeping half-size in 60-80 band where Ridge has positive edge)")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    intermediate = (
        base_sharpe < soft_sharpe < hard_sharpe
        or hard_sharpe < soft_sharpe < base_sharpe
    )
    if intermediate:
        bullet = (
            "INTERMEDIATE as predicted — soft dampening lands between the "
            "baseline and the hard gate. The hard gate's full-flatten in the "
            "top quintile remains the dominant variant on point Sharpe."
        )
    elif soft_sharpe >= hard_sharpe and soft_sharpe > base_sharpe:
        bullet = (
            "SURPRISE — soft dampening MATCHES OR BEATS the hard gate. The "
            "vix_mid → 60-80 band Ridge has enough drag that half-sizing it "
            "actually helps; consider the soft path for live deployment."
        )
    elif soft_sharpe <= base_sharpe:
        bullet = (
            "SOFT FAILS — soft dampening underperforms baseline. The 60-80 "
            "band Ridge edge is positive enough that half-sizing it costs "
            "more Sharpe than the 80+ zeroing buys. Use the hard gate "
            "(cycle-10) instead."
        )
    else:
        bullet = (
            "UNEXPECTED ordering — investigate per-band Sharpe before "
            "deploying either variant."
        )
    print(f"  {bullet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
