"""
Cross-variant harmonization v2 — uses each variant's aggregate.parquet
(properly computed across-fold Sharpe and unadjusted p-values) and applies
one consistent DSR + gate methodology and a single Romano-Wolf stepdown.

DSR HARMONIZATION CHOICE
========================
Theoretical-null cross-trial SR std (1/sqrt(T_oos), period units) combined
with Marcenko-Pastur effective trial count from the per-asset candidate ×
fold Sharpe correlation matrix (where available; raw count otherwise as a
sensitivity).

DSR formula (Bailey-Lopez de Prado 2014):
    z = (SR_hat - SR_bench) * sqrt(T - 1)
        / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2)
    DSR = Phi(z)
where SR_hat and SR_bench are in *period* (daily) units.

The expected-max-SR benchmark in period units:
    SR_bench = sigma_period_null * ((1-gamma)*z(1-1/N) + gamma*z(1-1/(N*e)))
with sigma_period_null = 1/sqrt(T_oos).

Skew/kurt: when only aggregate stats are available, default to skew=0,
kurt=3 (Normal). When per-fold sharpes are available, compute from the
fold series.

GATE (uniform across variants)
==============================
    DSR_harm > 0.95 AND p_Holm < 0.05 AND CI_lo > 0 AND Sharpe_net > 0

p_Holm = Bonferroni-Holm across the 8 per-asset bests.
CI_lo  = Politis-Romano stationary block bootstrap (B=2000) on the per-fold
         Sharpe series. For long_only_ensemble we have no fold detail in
         the aggregate, so CI is NaN and the cell cannot win the gate.

Romano-Wolf stepdown is applied to all cells that have per-fold Sharpes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/weeoto/macminibot"
ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "GLD"]
ANN = 252.0
RNG_SEED = 20260518

AGG_PATHS = {
    "bracket": f"{ROOT}/data/bakeoff/aggregate.parquet",
    "signal_flip": f"{ROOT}/data/bakeoff_variants/signal_flip/aggregate.parquet",
    "long_only_ensemble": f"{ROOT}/data/bakeoff_variants/long_only_ensemble/aggregate.parquet",
    "trailing_stop": f"{ROOT}/data/bakeoff_variants/trailing_stop/aggregate.parquet",
    "time_decay": f"{ROOT}/data/bakeoff_variants/time_decay/aggregate.parquet",
}
RES_PATHS = {
    "bracket": f"{ROOT}/data/bakeoff/results.parquet",
    "signal_flip": f"{ROOT}/data/bakeoff_variants/signal_flip/results.parquet",
    "trailing_stop": f"{ROOT}/data/bakeoff_variants/trailing_stop/results.parquet",
    "time_decay": f"{ROOT}/data/bakeoff_variants/time_decay/results.parquet",
    # long_only_ensemble has no per-fold results; results.parquet == aggregate.parquet
}

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def expected_max_sr_null(n_trials: float, sr_trials_std: float) -> float:
    n = max(float(n_trials), 1.0)
    if n == 1:
        return 0.0
    emc = 0.5772156649015329
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return sr_trials_std * ((1.0 - emc) * z1 + emc * z2)


def deflated_sharpe_ratio(sharpe_period: float, n_obs: int,
                          benchmark_sr_period: float,
                          skew: float, kurt: float) -> float:
    if n_obs < 5:
        return float("nan")
    denom_sq = 1.0 - skew * sharpe_period + ((kurt - 1.0) / 4.0) * (sharpe_period ** 2)
    denom_sq = max(denom_sq, 1e-10)
    z = ((sharpe_period - benchmark_sr_period) * math.sqrt(n_obs - 1.0)
         / math.sqrt(denom_sq))
    return float(stats.norm.cdf(z))


def mp_effective_trials(sharpe_matrix: np.ndarray) -> float:
    X = np.asarray(sharpe_matrix, dtype=float)
    X = X[~np.all(np.isnan(X), axis=1)]
    if X.shape[0] < 2 or X.shape[1] < 2:
        return float(max(X.shape[0], 1))
    finite_col = ~np.all(np.isnan(X), axis=0)
    X = X[:, finite_col]
    if X.shape[1] < 2:
        return float(max(X.shape[0], 1))
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X - np.mean(X, axis=1, keepdims=True)
    sd = np.std(X, axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs = X / sd
    N, T = Xs.shape
    if T < 2:
        return float(max(N, 1))
    try:
        s = np.linalg.svd(Xs / math.sqrt(T), compute_uv=False)
        eigvals = s ** 2
    except np.linalg.LinAlgError:
        return float(max(N, 1))
    lam_plus = (1.0 + math.sqrt(N / T)) ** 2
    signal_eigs = eigvals[eigvals > lam_plus]
    return float(max(len(signal_eigs), 1))


def holm_stepdown(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        scale = m - rank
        candidate = min(1.0, p[idx] * scale)
        running_max = max(running_max, candidate)
        adj[idx] = running_max
    return adj


def stationary_block_bootstrap_ci(folds: np.ndarray, B: int = 2000,
                                  block_p: float = 0.3,
                                  alpha: float = 0.05,
                                  seed: int = RNG_SEED) -> tuple[float, float, float]:
    x = np.asarray(folds, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return float(np.mean(x)) if n > 0 else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)
    for b in range(B):
        sampled = np.empty(n, dtype=float)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            L = 1
            while rng.random() > block_p and (i + L) < n:
                L += 1
            for k in range(L):
                if i + k >= n:
                    break
                sampled[i + k] = x[(start + k) % n]
            i += L
        means[b] = sampled.mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1.0 - alpha / 2))
    return float(np.mean(x)), lo, hi


def romano_wolf_stepdown(folds_per_test: list[np.ndarray],
                         alpha: float = 0.05,
                         B: int = 3000,
                         seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:
    K = len(folds_per_test)
    if K == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    rng = np.random.default_rng(seed)
    t_obs = np.zeros(K)
    centered: list[np.ndarray] = []
    n_arr = np.zeros(K, dtype=int)
    for k, fa in enumerate(folds_per_test):
        x = np.asarray(fa, dtype=float)
        x = x[~np.isnan(x)]
        n_arr[k] = len(x)
        if len(x) < 5:
            t_obs[k] = -np.inf
            centered.append(np.zeros(0))
            continue
        m = np.mean(x)
        s = np.std(x, ddof=1)
        t_obs[k] = 0.0 if s < 1e-12 else m / (s / math.sqrt(len(x)))
        centered.append(x - m)
    block_p = 0.3
    T_null = np.zeros((B, K))
    for k in range(K):
        x = centered[k]
        n = len(x)
        if n < 5:
            T_null[:, k] = 0.0
            continue
        for b in range(B):
            buf = np.empty(n)
            i = 0
            while i < n:
                start = rng.integers(0, n)
                L = 1
                while rng.random() > block_p and (i + L) < n:
                    L += 1
                for kk in range(L):
                    if i + kk >= n:
                        break
                    buf[i + kk] = x[(start + kk) % n]
                i += L
            m = buf.mean()
            s = buf.std(ddof=1)
            T_null[b, k] = 0.0 if s < 1e-12 else m / (s / math.sqrt(n))
    order = np.argsort(-t_obs)
    rw_p = np.ones(K, dtype=float)
    remaining = list(order)
    prev_p = 0.0
    while remaining:
        idx = remaining[0]
        sub = remaining
        max_null = T_null[:, sub].max(axis=1)
        p_step = float(np.mean(max_null >= t_obs[idx]))
        p_step = max(p_step, prev_p)
        rw_p[idx] = p_step
        if p_step < alpha:
            prev_p = p_step
            remaining = remaining[1:]
        else:
            for j in remaining:
                rw_p[j] = max(rw_p[j], p_step)
            break
    reject = rw_p < alpha
    return reject, rw_p


# ---------------------------------------------------------------------------
# Variant adapters
# ---------------------------------------------------------------------------
def variant_config_cols(variant: str) -> list[str]:
    return {
        "bracket": ["model_name", "feature_scope", "tp_sigma", "sl_sigma", "hit_priority"],
        "signal_flip": ["model_name", "feature_scope"],
        "long_only_ensemble": ["K", "tp_sigma", "sl_sigma"],
        "trailing_stop": ["model_name", "feature_scope", "n_sigma", "hit_priority"],
        "time_decay": ["model_name", "feature_scope", "n_days"],
    }[variant]


def variant_label(variant: str, row) -> tuple[str, str]:
    """Return (model_label, config_label)."""
    if variant == "long_only_ensemble":
        return f"ensemble_K{row['K']}", f"tp={row['tp_sigma']} sl={row['sl_sigma']}"
    model = f"{row['model_name']}/{row['feature_scope']}"
    if variant == "bracket":
        return model, f"tp={row['tp_sigma']} sl={row['sl_sigma']} hp={row['hit_priority']}"
    if variant == "signal_flip":
        return model, "flip"
    if variant == "trailing_stop":
        return model, f"trail={row['n_sigma']}σ hp={row['hit_priority']}"
    if variant == "time_decay":
        return model, f"N={row['n_days']}"
    return model, ""


def fold_sharpes_for(variant: str, row, results: pd.DataFrame) -> np.ndarray:
    """Pull the per-fold sharpe_net series for a given candidate row."""
    if variant == "long_only_ensemble":
        return np.array([])
    cfg = variant_config_cols(variant)
    mask = (results["asset"] == row["asset"])
    for c in cfg:
        v = row[c]
        if pd.isna(v):
            mask &= results[c].isna()
        else:
            mask &= (results[c] == v)
    return np.asarray(results.loc[mask, "sharpe_net"].values, dtype=float)


# ---------------------------------------------------------------------------
# Pre-load all results once for fold lookups
# ---------------------------------------------------------------------------
def load_all():
    aggs = {v: pd.read_parquet(p) for v, p in AGG_PATHS.items()}
    res = {}
    for v, p in RES_PATHS.items():
        df = pd.read_parquet(p)
        # Restrict bracket/trailing_stop to sl_first via aggregate filter later
        res[v] = df
    return aggs, res


# ---------------------------------------------------------------------------
# Per-asset best selection + DSR harmonization
# ---------------------------------------------------------------------------
def per_asset_pool(variant: str, agg: pd.DataFrame) -> pd.DataFrame:
    """Return the candidate pool we search over (this is the trial space the
    DSR n_trials_eff is computed on, AND the pool we pick the best from)."""
    if variant in ("bracket", "trailing_stop"):
        return agg[agg["hit_priority"] == "sl_first"].copy()
    return agg.copy()


def per_fold_matrix(variant: str, pool: pd.DataFrame, results: pd.DataFrame,
                    asset: str) -> np.ndarray:
    """Build candidates × folds Sharpe matrix for one asset by joining the
    candidate pool to the per-fold results."""
    if variant == "long_only_ensemble":
        # No fold detail.
        return np.zeros((0, 0))
    cfg = variant_config_cols(variant)
    sub_pool = pool[pool["asset"] == asset]
    sub_res = results[results["asset"] == asset]
    if variant in ("bracket", "trailing_stop"):
        sub_res = sub_res[sub_res["hit_priority"] == "sl_first"]
    # Group results by candidate signature.
    rows = []
    for _, prow in sub_pool.iterrows():
        m = pd.Series(True, index=sub_res.index)
        for c in cfg:
            v = prow[c]
            col = sub_res[c]
            if pd.isna(v):
                m &= col.isna()
            else:
                m &= (col == v)
        folds = sub_res.loc[m, "sharpe_net"].values.astype(float)
        rows.append(folds)
    if not rows:
        return np.zeros((0, 0))
    max_f = max(len(r) for r in rows)
    M = np.full((len(rows), max_f), np.nan)
    for i, r in enumerate(rows):
        M[i, :len(r)] = r
    return M


@dataclass
class Cell:
    asset: str
    variant: str
    model_label: str
    config_label: str
    sharpe: float
    p_unadj: float
    p_holm: float
    ci_lo: float
    ci_hi: float
    dsr_harm: float
    dsr_raw: float
    n_obs: int
    n_folds: int
    n_trials_raw: int
    n_trials_eff: float
    bench_sr_period_eff: float
    bench_sr_period_raw: float
    folds_obj: object
    verdict: str


def analyse_variant(variant: str, agg_all: dict, res_all: dict) -> list[Cell]:
    agg = agg_all[variant]
    pool = per_asset_pool(variant, agg)
    res = res_all.get(variant)

    cells: list[Cell] = []
    bests = []  # (cell index in cells list, p_unadj) for Holm

    for asset in ASSETS:
        sub = pool[pool["asset"] == asset].copy()
        if sub.empty:
            continue
        # n_trials_raw = pool size; n_trials_eff via MP on folds matrix.
        n_trials_raw = int(len(sub))
        if variant == "long_only_ensemble":
            n_trials_eff = float(n_trials_raw)  # no fold detail -> conservative
            T_oos = int(sub["n_oos_obs"].iloc[0]) if "n_oos_obs" in sub.columns else 1323
        else:
            M = per_fold_matrix(variant, pool, res, asset)
            n_trials_eff = mp_effective_trials(M)
            T_oos = int(sub["n_oos_obs"].iloc[0])

        sigma_period_null = 1.0 / math.sqrt(max(T_oos, 2))
        bench_per_eff = expected_max_sr_null(n_trials_eff, sigma_period_null)
        bench_per_raw = expected_max_sr_null(n_trials_raw, sigma_period_null)

        # Best candidate by agg_sharpe_net.
        sub_sorted = sub.sort_values("agg_sharpe_net", ascending=False).reset_index(drop=True)
        best = sub_sorted.iloc[0]
        sh_ann = float(best["agg_sharpe_net"])
        sh_per = sh_ann / math.sqrt(ANN)

        # Skew/kurt: per-fold Sharpes if available, else Normal.
        if variant != "long_only_ensemble":
            folds = fold_sharpes_for(variant, best, res)
            folds = folds[np.isfinite(folds)]
            if len(folds) >= 5:
                skew = float(stats.skew(folds, bias=False))
                kurt = float(stats.kurtosis(folds, fisher=False, bias=False))
            else:
                skew, kurt = 0.0, 3.0
        else:
            folds = np.array([])
            skew, kurt = 0.0, 3.0

        dsr_harm = deflated_sharpe_ratio(sh_per, T_oos, bench_per_eff, skew, kurt)
        dsr_raw  = deflated_sharpe_ratio(sh_per, T_oos, bench_per_raw, skew, kurt)

        # p_unadj from the aggregate (already computed per variant).
        p_unadj = float(best.get("p_value_unadj", np.nan))

        # Bootstrap CI on per-fold Sharpe series (annualised units).
        if len(folds) >= 5:
            _, ci_lo, ci_hi = stationary_block_bootstrap_ci(folds, B=2000)
        else:
            ci_lo = ci_hi = float("nan")

        model_label, config_label = variant_label(variant, best)
        cell = Cell(
            asset=asset, variant=variant,
            model_label=model_label, config_label=config_label,
            sharpe=sh_ann, p_unadj=p_unadj, p_holm=float("nan"),
            ci_lo=ci_lo, ci_hi=ci_hi,
            dsr_harm=dsr_harm, dsr_raw=dsr_raw,
            n_obs=T_oos, n_folds=len(folds),
            n_trials_raw=n_trials_raw, n_trials_eff=n_trials_eff,
            bench_sr_period_eff=bench_per_eff,
            bench_sr_period_raw=bench_per_raw,
            folds_obj=folds, verdict="",
        )
        cells.append(cell)
        bests.append(p_unadj)

    # Across-asset Holm.
    p_arr = np.array(bests, dtype=float)
    if len(p_arr) > 0 and np.all(np.isfinite(p_arr)):
        adj = holm_stepdown(p_arr)
        for c, pa in zip(cells, adj):
            c.p_holm = float(pa)
    else:
        finite = np.isfinite(p_arr)
        if finite.any():
            adj = holm_stepdown(p_arr[finite])
            j = 0
            for i, c in enumerate(cells):
                if finite[i]:
                    c.p_holm = float(adj[j]); j += 1
    # Verdict
    for c in cells:
        reasons = []
        if c.sharpe <= 0:
            c.verdict = "FAIL: Sharpe<=0"
            continue
        if not np.isfinite(c.dsr_harm) or c.dsr_harm <= 0.95:
            reasons.append(f"DSR={c.dsr_harm:.2f}")
        if not np.isfinite(c.p_holm) or c.p_holm >= 0.05:
            reasons.append(f"p_Holm={c.p_holm:.3f}")
        if not np.isfinite(c.ci_lo) or c.ci_lo <= 0:
            reasons.append(f"CI_lo={c.ci_lo:.3f}" if np.isfinite(c.ci_lo) else "CI_lo=NaN")
        c.verdict = "WINNER" if not reasons else "FAIL: " + ", ".join(reasons)
    return cells


def main():
    aggs, res = load_all()
    all_cells: list[Cell] = []
    for v in AGG_PATHS:
        print(f"[harmonize-v2] variant {v}")
        cells = analyse_variant(v, aggs, res)
        for c in cells:
            print(f"  {c.asset:5s} {c.model_label:35s} {c.config_label:25s}"
                  f" Sh={c.sharpe:+.3f}  DSR={c.dsr_harm:.3f}"
                  f"  pH={c.p_holm:.4f}  CI_lo={c.ci_lo:+.3f}  -> {c.verdict}")
        all_cells.extend(cells)
        print()

    # Romano-Wolf across 40 cells (only those with per-fold data & Sharpe>0).
    rw_cells = [c for c in all_cells if len(c.folds_obj) >= 5 and c.sharpe > 0]
    folds_list = [c.folds_obj for c in rw_cells]
    print(f"[RW] running stepdown on {len(rw_cells)} cells (B=3000)")
    rej, rw_p = romano_wolf_stepdown(folds_list, alpha=0.05, B=3000)
    rw_map = {(c.asset, c.variant, c.model_label, c.config_label): (bool(rej[i]), float(rw_p[i]))
              for i, c in enumerate(rw_cells)}

    # Build full DataFrame.
    rows = []
    for c in all_cells:
        rj, rp = rw_map.get((c.asset, c.variant, c.model_label, c.config_label), (None, None))
        d = {k: getattr(c, k) for k in c.__dataclass_fields__ if k not in ("folds_obj",)}
        d["rw_pvalue"] = rp
        d["rw_reject_H0"] = rj
        rows.append(d)
    full = pd.DataFrame(rows)
    full.to_parquet(f"{ROOT}/data/bakeoff_variants/_harmonize/all_cells.parquet")

    # Survivor summary.
    summary_rows = []
    for v in AGG_PATHS:
        v_df = full[full["variant"] == v]
        gate_win = v_df[v_df["verdict"] == "WINNER"]
        rw_win = v_df[v_df["rw_reject_H0"] == True]
        summary_rows.append({
            "variant": v,
            "n_gate_winners": int(len(gate_win)),
            "n_rw_winners": int(len(rw_win)),
            "avg_sharpe_gate": float(gate_win["sharpe"].mean()) if len(gate_win) else float("nan"),
            "avg_sharpe_rw": float(rw_win["sharpe"].mean()) if len(rw_win) else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(f"{ROOT}/data/bakeoff_variants/_harmonize/variant_summary.parquet")

    print("\n========== HARMONIZED RESULTS (40 cells) ==========")
    cols = ["asset", "variant", "model_label", "config_label", "sharpe",
            "dsr_harm", "dsr_raw", "p_unadj", "p_holm", "ci_lo", "ci_hi",
            "rw_pvalue", "rw_reject_H0", "verdict"]
    print(full[cols].to_string(index=False))
    print("\n========== Variant survivor counts ==========")
    print(summary.to_string(index=False))

    # ---- Deploy-ready portfolio Sharpe if a winner architecture exists ----
    # Pick the architecture with the most gate winners. Tie-break: fewer
    # config knobs (simpler).
    winners_by_variant = (full[full["verdict"] == "WINNER"]
                          .groupby("variant").size().to_dict())
    print("\nGate-winner counts:", winners_by_variant)
    if winners_by_variant:
        best_variant = max(winners_by_variant, key=winners_by_variant.get)
        print(f"\nLeading variant by gate: {best_variant} ({winners_by_variant[best_variant]} winners)")
        win_df = full[(full["variant"] == best_variant) & (full["verdict"] == "WINNER")]
        # Equal-weight portfolio Sharpe across winners' per-fold Sharpes.
        all_folds = []
        for _, r in win_df.iterrows():
            key = (r["asset"], r["variant"], r["model_label"], r["config_label"])
            for c in all_cells:
                if (c.asset, c.variant, c.model_label, c.config_label) == key:
                    all_folds.append(c.folds_obj)
                    break
        if all_folds:
            min_len = min(len(f) for f in all_folds)
            stacked = np.vstack([f[:min_len] for f in all_folds])
            # Per-fold equal-weight portfolio Sharpe series.
            portfolio_per_fold = stacked.mean(axis=0)
            # Bootstrap CI on the per-fold series (this is correct for the
            # uncertainty of the equal-weight portfolio's Sharpe).
            _, ci_lo, ci_hi = stationary_block_bootstrap_ci(portfolio_per_fold, B=3000)
            # NB: per-fold Sharpes are already annualised; mean(per-fold Sharpe)
            # is the across-fold expected Sharpe under iid folds. This is
            # noisier than the global concatenated-returns Sharpe; we use the
            # average of per-winner aggregate Sharpe instead for the headline.
            mean_sh = float(win_df["sharpe"].mean())
            mean_sh_fold = float(portfolio_per_fold.mean())
            print(f"\nDeploy-ready portfolio Sharpe:")
            print(f"  mean(per-asset agg_sharpe_net)  = {mean_sh:+.3f}  "
                  f"(equal-weight across {len(all_folds)} winners)")
            print(f"  mean(per-fold equal-weight Sh) = {mean_sh_fold:+.3f}  "
                  f"CI95=[{ci_lo:+.3f}, {ci_hi:+.3f}]  n_folds={min_len}")
            # Pretty list
            print("\nPer-asset deploy spec:")
            for _, r in win_df.iterrows():
                print(f"  {r['asset']:5s}  {r['model_label']:35s} {r['config_label']:25s}"
                      f"  Sh={r['sharpe']:+.3f}  DSR={r['dsr_harm']:.3f}  RW_p={r['rw_pvalue']:.4f}")
    else:
        print("\nNo gate winners in any variant.")


if __name__ == "__main__":
    main()
