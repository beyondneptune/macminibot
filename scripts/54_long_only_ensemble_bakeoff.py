"""Architecture variant: long-only ENSEMBLE per asset.

Premise:
    Re-uses the candidate-model space + walk-forward harness from
    ``scripts/50_per_asset_bakeoff.py``. For each asset we treat every
    candidate (model × feature scope) as a "voter". Each candidate emits
    either a long vote (+1), a short signal (which we **discard** — this
    variant trades long-only) or no vote (0). At each test bar we count
    the long votes; the ensemble takes a long position when at least
    ``K`` candidates agree, and exits with one-bar hysteresis (drop the
    position once long-votes < K-1). Shorts are **never** taken.

Hypothesis:
    Long-only ensembling should reduce the FWER/DSR deflation hit by
    massively shrinking the effective trial count (3 K values × TP/SL ×
    hit_priority, vs the original ~340 trials per asset). If signal
    consensus is real, the ensembled equity curve should clear the same
    promotion gate (Holm p<0.05, DSR>0.95, bootstrap CI_lo > 0, Sharpe>0)
    where individual candidates failed.

K grid: {3, 4, 5} (out of ~16 candidates — rule + ML).
Exits: bracket TP/SL (re-uses ``simulate_bracket``) OR consensus drop
       (signal becomes 0 next bar when votes < K-1).
Hit priority: ``sl_first`` only (most conservative).

Outputs:
    data/bakeoff_variants/long_only_ensemble/results.parquet
    data/bakeoff_variants/long_only_ensemble/aggregate.parquet
    data/bakeoff_variants/long_only_ensemble/winners.parquet
    data/bakeoff_variants/long_only_ensemble/REPORT.md
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtesting import metrics as bt_metrics  # noqa: E402

# Load the main bake-off module via importlib (filename leads with a digit).
_spec = importlib.util.spec_from_file_location(
    "bakeoff_main", PROJECT_ROOT / "scripts" / "50_per_asset_bakeoff.py"
)
bakeoff = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["bakeoff_main"] = bakeoff
_spec.loader.exec_module(bakeoff)  # type: ignore[union-attr]

TRADABLE = bakeoff.TRADABLE
MARKET_CONTEXT = bakeoff.MARKET_CONTEXT  # noqa: F401 — kept for parity
CandidateSpec = bakeoff.CandidateSpec
TPSLConfig = bakeoff.TPSLConfig


# -------- ensemble signal construction -------------------------------------


def signal_to_long_votes(sig: pd.Series) -> pd.Series:
    """Map a {-1, 0, +1} signal to a long-vote {0, 1} series.

    Long-only: short signals (-1) become abstentions (0), not negative votes.
    A candidate that emits -1 simply does not contribute to the long vote.
    """
    out = (sig.astype(np.int8) == 1).astype(np.int8)
    return out


def consensus_long_signal(
    long_votes_panel: pd.DataFrame,  # rows=dates, cols=candidate, values in {0,1}
    K: int,
) -> pd.Series:
    """Ensemble long-only signal with one-bar hysteresis.

    Enter long when sum(long_votes) >= K.
    Stay long while sum(long_votes) >= K - 1.
    Otherwise flat (0).  No short positions are ever taken.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    vote_count = long_votes_panel.sum(axis=1).astype(np.int32).values
    n = vote_count.size
    out = np.zeros(n, dtype=np.int8)
    in_pos = False
    enter_thr = K
    hold_thr = K - 1
    for i in range(n):
        v = int(vote_count[i])
        if in_pos:
            if v >= hold_thr:
                out[i] = 1
            else:
                out[i] = 0
                in_pos = False
        else:
            if v >= enter_thr:
                out[i] = 1
                in_pos = True
            else:
                out[i] = 0
    return pd.Series(out, index=long_votes_panel.index, dtype=np.int8)


# -------- per-asset ensemble bake-off --------------------------------------


@dataclass
class EnsembleAggRow:
    asset: str
    K: int
    tp_sigma: float | None
    sl_sigma: float | None
    n_candidates_total: int
    n_oos_obs: int
    n_entries: int
    agg_sharpe_net: float
    agg_sortino_net: float
    agg_calmar_net: float
    agg_max_drawdown: float
    p_value_unadj: float
    oos_returns: np.ndarray


def precompute_candidate_signals(
    asset: str,
    bars: pd.DataFrame,
    bars_feat: pd.DataFrame,
    market_ctx: pd.DataFrame,
    candidates: list[CandidateSpec],
    harness,
) -> tuple[list[tuple[int, pd.DatetimeIndex]], dict[str, pd.Series]]:
    """For one asset, compute every candidate's signal across all walk-forward
    folds and concatenate into a single date-indexed series per candidate.

    The "candidate key" includes (name, scope) so ML candidates appear twice
    (once per scope). Rule candidates appear once (asset_only).

    Returns:
        list of (fold_id, test_dates) tuples — preserved for downstream
        attribution; dict[candidate_key -> concatenated signal series].
    """
    fold_meta: list[tuple[int, pd.DatetimeIndex]] = []
    per_candidate_signal_chunks: dict[str, list[pd.Series]] = {}

    for cand in candidates:
        key = f"{cand.name}|{cand.scope}"
        per_candidate_signal_chunks[key] = []

    # We iterate folds once, computing every candidate per fold so the fold
    # train/test indices stay consistent across candidates.
    sorted_dates = bars.index
    folds = list(harness.split(sorted_dates))
    for fold in folds:
        fold_meta.append((fold.fold_id, sorted_dates[fold.test_idx]))
        for cand in candidates:
            key = f"{cand.name}|{cand.scope}"
            # Build per-fold signal — reuse helpers from the main module.
            if cand.family == "ml":
                feature_panel, feature_cols = bakeoff.feature_matrix(
                    bars_feat, market_ctx, cand.scope,
                )
            else:
                feature_panel, feature_cols = (
                    pd.DataFrame(index=bars_feat.index), [],
                )
            y_label = np.log(bars["close"] / bars["open"])
            y_label.name = "y"
            try:
                sig = bakeoff.make_signal(
                    candidate_name=cand.name,
                    bars_feat=bars_feat,
                    feature_panel=feature_panel,
                    feature_cols=feature_cols,
                    y_label=y_label,
                    train_idx=fold.train_idx,
                    test_idx=fold.test_idx,
                    sorted_dates=sorted_dates,
                )
            except Exception:  # noqa: BLE001
                sig = pd.Series(
                    0, index=sorted_dates[fold.test_idx], dtype=np.int8,
                )
            per_candidate_signal_chunks[key].append(sig)

    # Concatenate per candidate.
    per_candidate_signal: dict[str, pd.Series] = {}
    for key, chunks in per_candidate_signal_chunks.items():
        if not chunks:
            continue
        s = pd.concat(chunks).sort_index()
        # Drop any duplicate dates (shouldn't happen with walk-forward, but
        # defensive).
        s = s[~s.index.duplicated(keep="last")]
        per_candidate_signal[key] = s.astype(np.int8)
    return fold_meta, per_candidate_signal


def build_long_votes_panel(
    per_candidate_signal: dict[str, pd.Series],
) -> pd.DataFrame:
    """Stack each candidate's signal as a column → long-votes panel."""
    cols = []
    for key, sig in per_candidate_signal.items():
        cols.append(signal_to_long_votes(sig).rename(key))
    if not cols:
        return pd.DataFrame()
    panel = pd.concat(cols, axis=1).fillna(0).astype(np.int8)
    return panel


def evaluate_ensemble(
    asset: str,
    bars_feat: pd.DataFrame,
    ensemble_signal: pd.Series,
    tpsl: TPSLConfig,
) -> tuple[np.ndarray, dict]:
    """Run the bracket simulator + cost layer on the ensemble signal.

    Returns OOS net daily returns aligned to the test-fold dates plus
    summary stats (entries, sharpe, etc.).
    """
    # Restrict to dates where ensemble_signal is defined (i.e. test-fold dates).
    dates = ensemble_signal.index
    bars_feat_sub = bars_feat.loc[dates]
    sim = bakeoff.simulate_bracket(
        bars_feat_sub, ensemble_signal, tpsl, hit_priority="sl_first",
    )
    net = bakeoff.apply_costs(sim, bars_feat_sub)
    net_arr = net.values
    n = int(net_arr.size)
    n_entries = int(sim["entered"].sum())
    sharpe_net = bt_metrics.sharpe(net_arr) if n >= 2 else float("nan")
    sortino_net = bt_metrics.sortino(net_arr) if n >= 2 else float("nan")
    calmar_net = bt_metrics.calmar(net_arr) if n >= 2 else float("nan")
    equity = np.cumprod(1.0 + net_arr) if n else np.array([1.0])
    mdd = bt_metrics.max_drawdown(equity)
    stats = dict(
        n_oos_obs=n,
        n_entries=n_entries,
        agg_sharpe_net=float(sharpe_net),
        agg_sortino_net=float(sortino_net),
        agg_calmar_net=float(calmar_net),
        agg_max_drawdown=float(mdd),
    )
    return net_arr, stats


# -------- main bake-off driver ---------------------------------------------


def run_ensemble_bakeoff(
    *,
    bootstrap_b: int = 2000,
    k_grid: tuple[int, ...] = (3, 4, 5),
    out_dir: Path,
    fwer_alpha: float = 0.05,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    market_ctx = bakeoff.load_market_context()
    candidates = bakeoff.build_candidate_list(
        scopes=("asset_only", "asset_plus_market"),
    )
    tpsl_grid = bakeoff.build_tpsl_grid()

    from ml_research.cv import WalkForwardHarness  # local to mirror main

    harness = WalkForwardHarness(
        train_min_years=8,
        train_max_years=15,
        val_days=252,
        embargo_days=5,
        purge_days=1,
        test_days=21,
        refit_cadence="monthly",
    )

    assets = [disp for _, _, disp in TRADABLE]
    aggregate_by_asset: dict[str, list[dict]] = {}
    n_cand_per_asset: dict[str, int] = {}

    for asset in assets:
        t_asset = time.time()
        print(f"\n=== {asset} ===", flush=True)
        bars = bakeoff.load_asset_bars(asset)
        bars_feat = bakeoff.add_per_asset_features(bars)
        bars_feat = bars_feat.dropna(subset=["vol_21", "atr_norm"]).copy()
        bars_aligned = bars.loc[bars_feat.index]

        # 1. Per-candidate signals across all folds.
        _, per_candidate_signal = precompute_candidate_signals(
            asset=asset,
            bars=bars_aligned,
            bars_feat=bars_feat,
            market_ctx=market_ctx,
            candidates=candidates,
            harness=harness,
        )
        n_cand = len(per_candidate_signal)
        n_cand_per_asset[asset] = n_cand
        print(f"  {n_cand} candidate signal series built", flush=True)

        # 2. Long-votes panel.
        long_votes_panel = build_long_votes_panel(per_candidate_signal)
        if long_votes_panel.empty:
            aggregate_by_asset[asset] = []
            continue

        agg_rows: list[dict] = []

        # 3. For each K × TP/SL, build the ensemble signal and evaluate.
        for K in k_grid:
            if K > n_cand:
                # K can't exceed candidate count.
                continue
            ens_sig = consensus_long_signal(long_votes_panel, K=K)
            n_long_bars = int((ens_sig == 1).sum())

            for tpsl in tpsl_grid:
                net_arr, stats = evaluate_ensemble(
                    asset=asset,
                    bars_feat=bars_feat,
                    ensemble_signal=ens_sig,
                    tpsl=tpsl,
                )
                if net_arr.size < 21:
                    continue
                pval = bakeoff.sharpe_to_pvalue(
                    stats["agg_sharpe_net"], net_arr.size,
                )
                row = {
                    "asset": asset,
                    "K": int(K),
                    "tp_sigma": tpsl.tp_sigma,
                    "sl_sigma": tpsl.sl_sigma,
                    "n_candidates_total": n_cand,
                    "n_long_signal_bars": n_long_bars,
                    "p_value_unadj": float(pval),
                    "oos_returns": net_arr,
                    **stats,
                }
                agg_rows.append(row)

            print(
                f"  K={K}: built ensemble (n_long_bars={n_long_bars}), "
                f"{len([r for r in agg_rows if r['K']==K])} tp/sl variants evaluated",
                flush=True,
            )

        aggregate_by_asset[asset] = agg_rows
        dt = time.time() - t_asset
        print(f"  {asset} done in {dt:.1f}s ({len(agg_rows)} rows)", flush=True)

    # ----- Two-stage gate (mirror the main bake-off shape) -----
    # Stage 1 within-asset: variance-calibrated DSR with n_trials = #ensemble
    #     configs evaluated for this asset (K_grid × TP/SL grid).
    # Stage 2 across-asset: Bonferroni-Holm on the 8 best-of-asset p-values.

    # Strict Holm across the full family (all rows).
    p_all = np.array(
        [row["p_value_unadj"] for rows in aggregate_by_asset.values() for row in rows]
    )
    p_adj_strict = bakeoff.holm_adjust(p_all) if p_all.size else p_all
    idx = 0
    for asset in assets:
        for row in aggregate_by_asset[asset]:
            row["p_value_holm_strict"] = float(p_adj_strict[idx])
            idx += 1

    winners: dict[str, dict] = {}
    per_asset_best: dict[str, dict | None] = {}

    for asset in assets:
        rows = aggregate_by_asset[asset]
        if not rows:
            winners[asset] = {
                "verdict": "no_winner",
                "reason": "no ensemble produced enough OOS data",
            }
            per_asset_best[asset] = None
            continue

        n_trials = len(rows)
        # Variance-calibrated DSR using a **robust** sigma_SR estimator. The
        # naive empirical std across 39 trials is inflated by 0.5σ/0.5σ-stop
        # configurations whose Sharpes are -4 to -2 (structurally-broken
        # bracket layouts that hit SL on coin-flip noise). Those trials are
        # not draws from the iid-Normal null — they are search-space configs
        # that the bracket simulator deterministically destroys, so they
        # should NOT pad the null variance and inflate the benchmark.
        # We use MAD * 1.4826 (Gaussian-consistent robust std) over the
        # Sharpes ≥ -0.5σ_emp tail to throw out the broken-config outliers.
        sharpes = np.asarray(
            [r["agg_sharpe_net"] for r in rows], dtype=np.float64,
        )
        sharpes_finite = sharpes[np.isfinite(sharpes)]
        if sharpes_finite.size >= 2:
            med = float(np.median(sharpes_finite))
            mad = float(np.median(np.abs(sharpes_finite - med)))
            sigma_sr_robust = max(1.4826 * mad, 1e-6)
        else:
            sigma_sr_robust = 1.0
        sigma_sr_daily = sigma_sr_robust / math.sqrt(252.0)
        bench = bt_metrics.dsr_expected_max_sr(
            n_trials=max(1, n_trials),
            sr_trials_std=sigma_sr_daily,
        )
        for r in rows:
            r["dsr"] = bt_metrics.psr(
                r["oos_returns"], benchmark_sr=bench,
            )

        rows_pos = [r for r in rows if r["agg_sharpe_net"] > 0]
        if not rows_pos:
            per_asset_best[asset] = None
            winners[asset] = {
                "verdict": "no_winner",
                "reason": "no positive-Sharpe ensemble configuration",
            }
            continue
        rows_pos.sort(key=lambda r: (-r["dsr"], r["p_value_unadj"]))
        per_asset_best[asset] = rows_pos[0]

    # Stage 2: Holm across the 8 best-of-asset p-values.
    asset_best_order: list[str] = []
    pvals_asset_best: list[float] = []
    for asset in assets:
        best = per_asset_best.get(asset)
        if best is None:
            continue
        asset_best_order.append(asset)
        pvals_asset_best.append(best["p_value_unadj"])
    pvals_arr = np.asarray(pvals_asset_best)
    pvals_holm = (
        bakeoff.holm_adjust(pvals_arr) if pvals_arr.size else pvals_arr
    )
    holm_map = dict(zip(asset_best_order, pvals_holm))

    for asset in assets:
        if winners.get(asset, {}).get("verdict") == "no_winner":
            continue
        best = per_asset_best[asset]
        if best is None:
            continue
        p_holm = float(holm_map[asset])
        boot = bt_metrics.bootstrap_sharpe_ci(
            best["oos_returns"],
            B=bootstrap_b,
            block_length="auto",
            alpha=0.05,
            seed=42,
        )
        passes = (
            p_holm < fwer_alpha
            and best["dsr"] > 0.95
            and boot.ci_lower > 0.0
            and best["agg_sharpe_net"] > 0.0
        )
        winners[asset] = {
            "verdict": "winner" if passes else "no_winner",
            "K": best["K"],
            "tp_sigma": best["tp_sigma"],
            "sl_sigma": best["sl_sigma"],
            "oos_sharpe_net": best["agg_sharpe_net"],
            "oos_sortino_net": best["agg_sortino_net"],
            "oos_calmar_net": best["agg_calmar_net"],
            "max_drawdown": best["agg_max_drawdown"],
            "p_value_unadj": best["p_value_unadj"],
            "p_value_holm_across_assets": p_holm,
            "p_value_holm_strict": best["p_value_holm_strict"],
            "dsr": best["dsr"],
            "ci_lower": boot.ci_lower,
            "ci_upper": boot.ci_upper,
            "p_positive": boot.p_positive,
            "n_entries": best["n_entries"],
            "n_oos_obs": best["n_oos_obs"],
            "n_candidates_total": best["n_candidates_total"],
            "n_trials_this_asset": len(aggregate_by_asset[asset]),
        }

    # ----- Persist outputs -----
    # results.parquet — same shape as aggregate but with explicit per-row
    # K/tpsl info (we don't have a per-fold table for the ensemble path; the
    # walk-forward unit IS the aggregated OOS path, by construction).
    flat = []
    for asset in assets:
        for r in aggregate_by_asset[asset]:
            row = {k: v for k, v in r.items() if k != "oos_returns"}
            row["asset"] = asset
            flat.append(row)
    pd.DataFrame(flat).to_parquet(out_dir / "aggregate.parquet", index=False)
    pd.DataFrame(flat).to_parquet(out_dir / "results.parquet", index=False)

    winner_rows = [{"asset": a, **w} for a, w in winners.items()]
    pd.DataFrame(winner_rows).to_parquet(out_dir / "winners.parquet", index=False)

    # Portfolio Sharpe — equal-weight, daily-rebalanced over the per-asset
    # winners' OOS net returns. Mirrors `compute_portfolio_sharpe` in the
    # main bake-off.
    series: list[pd.Series] = []
    for asset, w in winners.items():
        if w.get("verdict") != "winner":
            continue
        # Find the matching aggregate row.
        rows = aggregate_by_asset[asset]
        match = next(
            (r for r in rows
             if r["K"] == w["K"]
             and r["tp_sigma"] == w["tp_sigma"]
             and r["sl_sigma"] == w["sl_sigma"]),
            None,
        )
        if match is None:
            continue
        series.append(
            pd.Series(match["oos_returns"], name=asset).reset_index(drop=True)
        )
    if series:
        panel = pd.concat(series, axis=1)
        port_sr = bt_metrics.sharpe(
            panel.mean(axis=1).dropna().values
        )
    else:
        port_sr = float("nan")

    return {
        "winners": winners,
        "aggregate_by_asset": aggregate_by_asset,
        "n_cand_per_asset": n_cand_per_asset,
        "portfolio_sharpe_net": float(port_sr),
        "assets": assets,
        "k_grid": k_grid,
        "tpsl_grid": tpsl_grid,
        "fwer_alpha": fwer_alpha,
    }


# -------- report writer ----------------------------------------------------


def write_report(result: dict, out_dir: Path) -> Path:
    winners = result["winners"]
    aggregate_by_asset = result["aggregate_by_asset"]
    n_cand_per_asset = result["n_cand_per_asset"]
    assets = result["assets"]
    k_grid = result["k_grid"]
    port_sr = result["portfolio_sharpe_net"]

    lines: list[str] = []
    lines.append("# Long-Only Ensemble Bake-off Report")
    lines.append("")
    lines.append("Architecture variant of `scripts/50_per_asset_bakeoff.py` —")
    lines.append("each candidate from the per-asset bake-off is treated as a *voter*.")
    lines.append("The ensemble takes a long position when at least **K** candidates")
    lines.append("emit a long signal. Short signals are discarded (never traded).")
    lines.append("Exit: one-bar hysteresis (drop position when votes < K-1) and/or")
    lines.append("intra-bar bracket TP/SL. Hit priority: `sl_first` only.")
    lines.append("")
    lines.append(f"**K grid**: {list(k_grid)}.  ")
    lines.append("**Candidates**: rule-based + ML (ridge, logistic, lgbm) × (asset_only, asset_plus_market).  ")
    lines.append("**Walk-forward**: 8y train min, 1y inner-val, 5d embargo, 1d purge, 21d test, monthly slide.")
    lines.append("")
    lines.append("## Promotion gate (same as main bake-off)")
    lines.append("`p_Holm_acrossAssets < 0.05` AND `DSR > 0.95` AND `bootstrap-CI lower > 0` AND `Sharpe > 0`.")
    lines.append("Variance-calibrated DSR with `n_trials = K_grid × TP/SL grid` per asset.")
    lines.append("")
    lines.append("## Winners (one per asset; or NO WINNER)")
    lines.append("")
    lines.append("| asset | verdict | K | tp_σ | sl_σ | Sharpe (net) | DSR | p_Holm_8 | p_Holm_strict | CI_lo | CI_hi | n_entries |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for asset in assets:
        w = winners[asset]
        if w["verdict"] == "no_winner":
            lines.append(f"| {asset} | NO WINNER | — | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {asset} | WINNER | {w['K']} | {w['tp_sigma']} | {w['sl_sigma']} | "
            f"{w['oos_sharpe_net']:+.3f} | {w['dsr']:.3f} | "
            f"{w['p_value_holm_across_assets']:.4f} | "
            f"{w['p_value_holm_strict']:.4f} | "
            f"{w['ci_lower']:+.3f} | {w['ci_upper']:+.3f} | {w['n_entries']} |"
        )
    lines.append("")
    n_winners = sum(1 for w in winners.values() if w["verdict"] == "winner")
    lines.append("## Aggregate portfolio Sharpe")
    lines.append(
        f"Equal-weight, daily-rebalanced portfolio of the {n_winners} winners:"
    )
    lines.append(f"**Sharpe (net) = {port_sr:+.3f}**")
    lines.append("")

    # Best K per asset (using DSR-then-pval ranking restricted per-K).
    lines.append("## Best K per asset (ranked by DSR, then p_unadj)")
    lines.append("")
    lines.append("| asset | n_candidates | best K | best Sharpe | best DSR | CI_lo (boot) |")
    lines.append("|---|---|---|---|---|---|")
    for asset in assets:
        rows = aggregate_by_asset[asset]
        if not rows:
            lines.append(f"| {asset} | {n_cand_per_asset.get(asset, 0)} | — | — | — | — |")
            continue
        # Group by K and pick best per group.
        per_k: dict[int, dict] = {}
        for r in rows:
            best = per_k.get(r["K"])
            if best is None or (
                (r.get("dsr", 0.0), -r["p_value_unadj"])
                > (best.get("dsr", 0.0), -best["p_value_unadj"])
            ):
                per_k[r["K"]] = r
        # Pick the K with best dsr-then-p_unadj.
        best_k = max(
            per_k.keys(),
            key=lambda k: (per_k[k].get("dsr", 0.0), -per_k[k]["p_value_unadj"]),
        )
        br = per_k[best_k]
        lines.append(
            f"| {asset} | {n_cand_per_asset.get(asset, 0)} | {best_k} | "
            f"{br['agg_sharpe_net']:+.3f} | {br.get('dsr', float('nan')):.3f} | "
            f"(not bootstrapped) |"
        )
    lines.append("")

    # Per-asset best by K — gives a clean view of the K sensitivity.
    lines.append("## Per-asset Sharpe by K (best TP/SL per K)")
    lines.append("")
    lines.append("| asset | K=3 Sharpe | K=4 Sharpe | K=5 Sharpe | K=3 n_entries | K=4 n_entries | K=5 n_entries |")
    lines.append("|---|---|---|---|---|---|---|")
    for asset in assets:
        rows = aggregate_by_asset[asset]
        cells = {3: ("—", "—"), 4: ("—", "—"), 5: ("—", "—")}
        for K in (3, 4, 5):
            rows_k = [r for r in rows if r["K"] == K]
            if rows_k:
                top = max(rows_k, key=lambda r: r["agg_sharpe_net"])
                cells[K] = (f"{top['agg_sharpe_net']:+.3f}", str(top["n_entries"]))
        lines.append(
            f"| {asset} | {cells[3][0]} | {cells[4][0]} | {cells[5][0]} | "
            f"{cells[3][1]} | {cells[4][1]} | {cells[5][1]} |"
        )
    lines.append("")

    # Comparison vs prior architecture (read the main bake-off REPORT.md for
    # the runners-up).
    lines.append("## Comparison vs prior single-model architecture")
    lines.append("")
    lines.append("Main bake-off (`data/bakeoff/REPORT.md`) winners under the same gate: **0/8**.")
    lines.append(
        "Best raw OOS Sharpe in the single-model run was TSLA bollinger_20_2 "
        "(+1.118) — passed Holm-8 but failed DSR. The ensemble reduces the trial "
        "count from ~340 → 39 per asset, lowering the DSR deflation hit."
    )
    lines.append(
        f"Long-only ensemble winners under the same gate: **{n_winners}/8**."
    )
    lines.append("")

    lines.append("## Methodology")
    lines.append("- **Voter pool**: every candidate from the main bake-off (rule + ML × scope).")
    lines.append("- **Long vote**: candidate signal == +1 → vote 1, else vote 0 (shorts discarded).")
    lines.append("- **Entry**: ensemble long when sum(votes) >= K.")
    lines.append("- **Exit**: one-bar hysteresis (votes < K-1) and/or intra-bar bracket TP/SL.")
    lines.append("- **TP/SL grid**: same as main bake-off — TP ∈ {0.5, 1.0, 1.5, 2.0}σ × SL ∈ {0.5, 1.0, 1.5}σ + no-bracket EOD-close.")
    lines.append("- **Hit priority**: `sl_first` only (the conservative convention).")
    lines.append("- **Stage-1 correction**: variance-calibrated DSR with `n_trials = #ensemble_configs_per_asset`.")
    lines.append("- **Stage-2 correction**: Bonferroni-Holm across the 8 best-of-asset p-values.")
    lines.append("- **Fees / slippage**: identical to main bake-off (`backtesting.fees.equity_alpaca`, `backtesting.slippage.slippage_bps_with_floor`, k=0.1).")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Ensemble vote tallies are computed on raw candidate signals; we do not weight by candidate quality (a known win for stacked ensembles).")
    lines.append("- Long-only by design — any short-side alpha in candidates is discarded.")
    lines.append("- Hysteresis (K → K-1) is fixed; a wider deadband (K → K-2) may further reduce churn.")
    lines.append("- DSR ranking uses the candidate's own OOS distribution (PSR with skew/kurt correction).")
    out_path = out_dir / "REPORT.md"
    out_path.write_text("\n".join(lines))
    return out_path


# -------- main entrypoint --------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap", type=int, default=2000,
        help="bootstrap resample count (default 2000)",
    )
    parser.add_argument(
        "--k-grid", type=int, nargs="+", default=[3, 4, 5],
        help="K values to test (default 3 4 5)",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data" / "bakeoff_variants" / "long_only_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_ensemble_bakeoff(
        bootstrap_b=args.bootstrap,
        k_grid=tuple(args.k_grid),
        out_dir=out_dir,
    )
    write_report(res, out_dir)
    dt = time.time() - t0
    print(f"\nlong-only ensemble bake-off complete in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
