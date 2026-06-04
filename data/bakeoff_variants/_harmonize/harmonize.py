"""
Cross-variant harmonization pipeline.

Goal: apply ONE consistent DSR + gate methodology across all 5 architecture
variants, then run a Romano-Wolf stepdown across the 40 best-of-(asset×variant)
Sharpes to pick the empirical winner (or honestly say none passes).

DSR HARMONIZATION CHOICE
========================
We use the **theoretical-null calibration** of the cross-trial SR std:
    sigma_period_null = 1 / sqrt(T_oos)
plugged into Bailey-Lopez de Prado's expected-max-SR null benchmark with
    n_trials_effective = MP_effective_trial_count(per_fold_sharpe_matrix)
where MP_effective_trial_count is the Marcenko-Pastur upper edge of the
candidate × fold Sharpe correlation matrix, lower-bounded at 1.

Why this choice (vs the four different per-variant fixes):
  - The original `bakeoff` REPORT used variance-calibrated empirical sigma
    with raw n_trials -> DSR collapses to 0 (over-deflates).
  - `trailing_stop` used MP effective trials + empirical sigma -> reasonable
    but couples two corrections; we keep MP but use the theoretical null sigma.
  - `time_decay` used theoretical-null sigma but raw n_trials -> under-corrects
    the trial correlation but uses the correct null sigma.
  - `signal_flip` and `long_only_ensemble` used their own ad-hoc fixes.

The harmonized formula is the *principled* combo: theoretical-null sigma
(genuine no-skill SR noise) AND MP-effective trials (independent search
directions). We report this as the single primary DSR. As a sensitivity, we
also compute DSR with raw n_trials.

GATE (uniform across variants)
==============================
    DSR_harm > 0.95 AND p_Holm < 0.05 AND CI_lo > 0 AND Sharpe_net > 0

p_Holm: Bonferroni-Holm across the 8 per-asset best p-values
        (per-asset best = highest Sharpe candidate that passes Sharpe > 0).
CI_lo:  Politis-Romano stationary block bootstrap, B=2000, block_length=auto,
        on the asset's per-fold-Sharpe daily-aggregated return proxy. We use
        a deterministic seed for reproducibility.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/weeoto/macminibot"
VARIANT_PATHS = {
    "bracket": f"{ROOT}/data/bakeoff/results.parquet",
    "signal_flip": f"{ROOT}/data/bakeoff_variants/signal_flip/results.parquet",
    "long_only_ensemble": f"{ROOT}/data/bakeoff_variants/long_only_ensemble/results.parquet",
    "trailing_stop": f"{ROOT}/data/bakeoff_variants/trailing_stop/results.parquet",
    "time_decay": f"{ROOT}/data/bakeoff_variants/time_decay/results.parquet",
}

ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "GLD"]
ANN = 252.0
RNG_SEED = 20260518


# ---------------------------------------------------------------------------
# DSR helpers
# ---------------------------------------------------------------------------
def expected_max_sr_null(n_trials: float, sr_trials_std: float) -> float:
    """Bailey-Lopez de Prado expected-max-SR null benchmark in period units.

    n_trials: number of (effective) independent trials.
    sr_trials_std: period-units null-hypothesis std of SR estimator.
    """
    n = max(float(n_trials), 1.0)
    emc = 0.5772156649015329
    if n == 1:
        return 0.0
    # E[max of N iid N(0,1)] ~ (1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N*e))
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    e_max_z = (1.0 - emc) * z1 + emc * z2
    return sr_trials_std * e_max_z


def deflated_sharpe_ratio(sharpe_period: float, n_obs: int,
                          benchmark_sr_period: float,
                          skew: float, kurt: float) -> float:
    """PSR-style DSR vs an arbitrary benchmark Sharpe in *period* units."""
    if n_obs < 5:
        return float("nan")
    denom_sq = 1.0 - skew * sharpe_period + ((kurt - 1.0) / 4.0) * (sharpe_period ** 2)
    denom_sq = max(denom_sq, 1e-10)
    z = ((sharpe_period - benchmark_sr_period) * math.sqrt(n_obs - 1.0)
         / math.sqrt(denom_sq))
    return float(stats.norm.cdf(z))


def mp_effective_trials(sharpe_matrix: np.ndarray) -> float:
    """Marcenko-Pastur effective independent trial count.

    sharpe_matrix: rows = candidates (trials), cols = folds.
    """
    X = np.asarray(sharpe_matrix, dtype=float)
    # Drop trials with all NaN folds
    X = X[~np.all(np.isnan(X), axis=1)]
    if X.shape[0] < 2 or X.shape[1] < 2:
        return float(max(X.shape[0], 1))
    # Fill NaNs with column means; if a column is all-NaN, drop it
    finite_col = ~np.all(np.isnan(X), axis=0)
    X = X[:, finite_col]
    if X.shape[1] < 2:
        return float(max(X.shape[0], 1))
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    # Sanitise non-finite
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Standardise per trial (row)
    X = X - np.mean(X, axis=1, keepdims=True)
    sd = np.std(X, axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs = X / sd
    N, T = Xs.shape  # N trials, T folds
    if T < 2:
        return float(max(N, 1))
    # Empirical correlation matrix among trials (N x N) -- use SVD spectrum.
    try:
        # Singular values of Xs/sqrt(T) give sqrt(eigvals of Xs Xs^T / T)
        s = np.linalg.svd(Xs / math.sqrt(T), compute_uv=False)
        eigvals = s ** 2
    except np.linalg.LinAlgError:
        return float(max(N, 1))
    # MP upper edge for noise eigenvalues under iid: lam+ = (1 + sqrt(N/T))^2
    lam_plus = (1.0 + math.sqrt(N / T)) ** 2
    # Signal eigenvalues = those above lam+
    signal_eigs = eigvals[eigvals > lam_plus]
    n_signal = max(len(signal_eigs), 1)
    return float(max(n_signal, 1))


# ---------------------------------------------------------------------------
# Per-variant aggregation
# ---------------------------------------------------------------------------
def aggregate_fold_level(df: pd.DataFrame,
                         candidate_cols: list[str],
                         sharpe_col: str = "sharpe_net") -> pd.DataFrame:
    """Aggregate a fold-level result frame into one row per candidate.

    Returns columns: candidate_cols + ['agg_sharpe','n_folds','folds_arr'].
    """
    # Each (asset, candidate) gets one row aggregating across folds.
    keep = ["asset"] + candidate_cols + ["fold_id", sharpe_col]
    sub = df[keep].dropna(subset=[sharpe_col]).copy()
    grouped = sub.groupby(["asset"] + candidate_cols, dropna=False)
    rows = []
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        sh = g[sharpe_col].values
        rows.append({
            **dict(zip(["asset"] + candidate_cols, keys)),
            "agg_sharpe": float(np.nanmean(sh)),
            "n_folds": int(len(sh)),
            "folds_arr": sh,
        })
    return pd.DataFrame(rows)


def per_asset_fold_matrix(agg: pd.DataFrame, asset: str) -> np.ndarray:
    """Build candidates × folds Sharpe matrix for one asset."""
    sub = agg[agg["asset"] == asset]
    if sub.empty:
        return np.zeros((0, 0))
    max_folds = int(sub["n_folds"].max())
    M = np.full((len(sub), max_folds), np.nan)
    for i, arr in enumerate(sub["folds_arr"].values):
        a = np.asarray(arr, dtype=float)
        L = min(len(a), max_folds)
        M[i, :L] = a[:L]
    return M


# ---------------------------------------------------------------------------
# Bootstrap CI on a candidate's fold-level Sharpe via stationary block-bootstrap
# on the folds (Politis-Romano). Folds here are 21d test windows; treating each
# fold's Sharpe as a single block-resamplable observation.
# ---------------------------------------------------------------------------
def stationary_block_bootstrap_ci(folds: np.ndarray, B: int = 2000,
                                  block_p: float = 0.3,
                                  alpha: float = 0.05,
                                  seed: int = RNG_SEED) -> tuple[float, float, float]:
    """Return (mean_sharpe, ci_lo, ci_hi) on the per-fold Sharpe array.

    block_p = 1 / expected_block_length. With 60-130 folds, block_p=0.3 gives
    avg block length ~3 which is conservative for monthly-slide non-overlap.
    """
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
            # Geometric block length
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


# ---------------------------------------------------------------------------
# Skew / kurt for DSR per candidate
# ---------------------------------------------------------------------------
def per_candidate_stats(folds: np.ndarray) -> tuple[float, float, float, int]:
    x = np.asarray(folds, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return float("nan"), float("nan"), float("nan"), n
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, fisher=False, bias=False))  # raw kurtosis
    sh = float(np.mean(x))
    return sh, skew, kurt, n


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down (across 8 per-asset bests)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Romano-Wolf stepdown via bootstrap (FWER)
# ---------------------------------------------------------------------------
def romano_wolf_stepdown(folds_per_test: list[np.ndarray],
                         alpha: float = 0.05,
                         B: int = 5000,
                         seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Romano-Wolf FWER stepdown for one-sided H0: mean Sharpe <= 0.

    Returns (reject_bool[K], rw_pvals[K]).

    For each test k, t_k = mean(folds_k) / (std(folds_k)/sqrt(n_k)).
    Build a B-sample bootstrap null distribution by recentering each test's
    folds at zero and resampling with the *same* stationary block-bootstrap
    used for the CIs. The Romano-Wolf adjusted p-value for rank r is the
    bootstrap probability that the max-of-remaining null t-stat exceeds the
    r-th order observed t-stat.
    """
    K = len(folds_per_test)
    if K == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    rng = np.random.default_rng(seed)
    # Observed t-stats
    t_obs = np.zeros(K)
    n_arr = np.zeros(K, dtype=int)
    centered = []
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
        if s < 1e-12:
            t_obs[k] = 0.0
        else:
            t_obs[k] = m / (s / math.sqrt(len(x)))
        centered.append(x - m)

    # Bootstrap matrix of null t-stats (B x K)
    block_p = 0.3
    T_null = np.zeros((B, K))
    for k in range(K):
        x = centered[k]
        n = len(x)
        if n < 5:
            T_null[:, k] = 0.0
            continue
        for b in range(B):
            i = 0
            buf = np.empty(n)
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
            if s < 1e-12:
                T_null[b, k] = 0.0
            else:
                T_null[b, k] = m / (s / math.sqrt(n))

    # Romano-Wolf stepdown
    order = np.argsort(-t_obs)  # descending observed t
    reject = np.zeros(K, dtype=bool)
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
            reject[idx] = True
            prev_p = p_step
            remaining = remaining[1:]
        else:
            # Once we fail to reject, all remaining get this p (monotonic).
            for j in remaining:
                rw_p[j] = max(rw_p[j], p_step)
            break
    return reject, rw_p


# ---------------------------------------------------------------------------
# Variant adapters: turn each variant's raw results.parquet into a uniform
# (asset, variant, model_label, config_label, candidate_id, agg_sharpe,
#  folds_arr, n_folds) table.
# ---------------------------------------------------------------------------
def load_variant(variant: str) -> tuple[pd.DataFrame, list[str]]:
    """Return (uniform_candidates_df, candidate_key_cols)."""
    df = pd.read_parquet(VARIANT_PATHS[variant])

    if variant == "bracket":
        # Restrict to sl_first to match the original promotion logic;
        # this is the conservative convention.
        df = df[df["hit_priority"] == "sl_first"].copy()
        keycols = ["model_name", "feature_scope", "tp_sigma", "sl_sigma"]
        agg = aggregate_fold_level(df, keycols)
        agg["model_label"] = agg["model_name"].astype(str) + "/" + agg["feature_scope"].astype(str)
        agg["config_label"] = ("tp=" + agg["tp_sigma"].astype(str)
                               + " sl=" + agg["sl_sigma"].astype(str))
        return agg, keycols

    if variant == "signal_flip":
        keycols = ["model_name", "feature_scope"]
        agg = aggregate_fold_level(df, keycols)
        agg["model_label"] = agg["model_name"].astype(str) + "/" + agg["feature_scope"].astype(str)
        agg["config_label"] = "flip"
        return agg, keycols

    if variant == "long_only_ensemble":
        # Already aggregated. Synthesise a folds_arr from the aggregate sharpe
        # by treating it as a single observation -> we cannot do per-fold
        # stats. So use raw aggregate Sharpe; bootstrap CI is not available,
        # but the variant's own report already noted CI was not bootstrapped.
        # For the harmonized analysis we therefore mark these candidates
        # ineligible for the Romano-Wolf stepdown (need per-fold data).
        keycols = ["K", "tp_sigma", "sl_sigma"]
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "asset": r["asset"],
                "K": r["K"], "tp_sigma": r["tp_sigma"], "sl_sigma": r["sl_sigma"],
                "agg_sharpe": float(r["agg_sharpe_net"]),
                "n_folds": int(r.get("n_oos_obs", 0)),
                "folds_arr": np.array([float(r["agg_sharpe_net"])]),
                "model_label": f"ensemble_K{r['K']}",
                "config_label": f"tp={r['tp_sigma']} sl={r['sl_sigma']}",
            })
        return pd.DataFrame(rows), keycols

    if variant == "trailing_stop":
        df = df[df["hit_priority"] == "sl_first"].copy()
        keycols = ["model_name", "feature_scope", "n_sigma"]
        agg = aggregate_fold_level(df, keycols)
        agg["model_label"] = agg["model_name"].astype(str) + "/" + agg["feature_scope"].astype(str)
        agg["config_label"] = "trail=" + agg["n_sigma"].astype(str) + "σ"
        return agg, keycols

    if variant == "time_decay":
        keycols = ["model_name", "feature_scope", "n_days"]
        agg = aggregate_fold_level(df, keycols)
        agg["model_label"] = agg["model_name"].astype(str) + "/" + agg["feature_scope"].astype(str)
        agg["config_label"] = "N=" + agg["n_days"].astype(str)
        return agg, keycols

    raise ValueError(variant)


# ---------------------------------------------------------------------------
# Main harmonized analysis
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
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
    n_folds: int
    n_trials_raw: int
    n_trials_eff: float
    bench_sr_period: float
    folds_arr_obj: object  # the np.ndarray of folds (for RW)


def analyse_variant(variant: str) -> tuple[pd.DataFrame, dict]:
    """Return (cell_df, diag) where cell_df has one row per asset (best
    candidate) for this variant under the harmonized methodology."""
    agg, keycols = load_variant(variant)
    diag = {"variant": variant, "n_candidates_per_asset": {}, "n_trials_eff": {}}

    cells = []
    # First pass: compute best-per-asset; collect per-asset MP eff trials.
    for asset in ASSETS:
        sub = agg[agg["asset"] == asset].copy()
        if sub.empty:
            continue
        diag["n_candidates_per_asset"][asset] = int(len(sub))

        # Build the per-asset trials × folds matrix for MP.
        M = per_asset_fold_matrix(sub, asset)
        n_trials_raw = int(M.shape[0])
        n_trials_eff = mp_effective_trials(M)
        diag["n_trials_eff"][asset] = float(n_trials_eff)

        # The OOS T (avg n_folds across candidates of this asset)
        T_oos = int(np.nanmean(sub["n_folds"].values))
        sigma_period_null = 1.0 / math.sqrt(max(T_oos, 2))

        bench_period_eff = expected_max_sr_null(n_trials_eff, sigma_period_null)
        bench_period_raw = expected_max_sr_null(n_trials_raw, sigma_period_null)

        # Pick best candidate by Sharpe.
        sub_sorted = sub.sort_values("agg_sharpe", ascending=False).reset_index(drop=True)
        best = sub_sorted.iloc[0]
        folds = np.asarray(best["folds_arr"], dtype=float)
        folds = folds[~np.isnan(folds)]
        n_obs = len(folds)
        if n_obs < 5:
            # Long-only-ensemble case (single aggregate Sharpe over the full
            # OOS window). Approximate DSR using n_oos_obs as the effective
            # sample size and assume normal returns (skew=0, kurt=3).
            sh_ann_lo = float(best["agg_sharpe"])
            sh_period_lo = sh_ann_lo / math.sqrt(ANN)
            n_eff = max(int(best["n_folds"]), 252)
            T_oos_lo = n_eff
            sig_null_lo = 1.0 / math.sqrt(max(T_oos_lo, 2))
            bench_eff_lo = expected_max_sr_null(n_trials_eff, sig_null_lo)
            bench_raw_lo = expected_max_sr_null(n_trials_raw, sig_null_lo)
            dsr_h_lo = deflated_sharpe_ratio(
                sh_period_lo, n_eff, bench_eff_lo, skew=0.0, kurt=3.0)
            dsr_r_lo = deflated_sharpe_ratio(
                sh_period_lo, n_eff, bench_raw_lo, skew=0.0, kurt=3.0)
            # Approximate unadjusted one-sided p from the normal approx
            t = sh_ann_lo * math.sqrt(n_eff / ANN)
            p_un = float(1.0 - stats.norm.cdf(t))
            cells.append(CellResult(
                asset=asset, variant=variant,
                model_label=str(best.get("model_label", "")),
                config_label=str(best.get("config_label", "")),
                sharpe=sh_ann_lo,
                p_unadj=p_un,
                p_holm=float("nan"),
                ci_lo=float("nan"),
                ci_hi=float("nan"),
                dsr_harm=dsr_h_lo,
                dsr_raw=dsr_r_lo,
                n_folds=n_eff,
                n_trials_raw=n_trials_raw,
                n_trials_eff=n_trials_eff,
                bench_sr_period=bench_eff_lo,
                folds_arr_obj=np.array([]),
            ))
            continue

        sh_ann = float(np.mean(folds))
        # Per-fold Sharpes are already annualised (the bake-off framework
        # writes annualised sharpe_net per fold). So sh_ann lives in
        # annualised units. Convert to period units for DSR.
        # Treat each fold's annualised Sharpe as one period observation.
        # DSR uses period-unit Sharpe and period-unit benchmark.
        sh_period = sh_ann / math.sqrt(ANN)
        _, skew, kurt, _ = per_candidate_stats(folds / math.sqrt(ANN))
        # One-sided p-value for H0: mean Sharpe <= 0 across folds.
        mean = sh_ann
        sd = float(np.std(folds, ddof=1))
        if sd < 1e-12:
            t_stat = 0.0
            p_unadj = 1.0
        else:
            t_stat = mean / (sd / math.sqrt(n_obs))
            p_unadj = float(1.0 - stats.norm.cdf(t_stat))

        # Bootstrap CI on fold Sharpe (annualised)
        _, ci_lo, ci_hi = stationary_block_bootstrap_ci(folds, B=2000)

        # DSR (harmonized: theoretical-null sigma + MP-effective trials)
        dsr_harm = deflated_sharpe_ratio(
            sharpe_period=sh_period,
            n_obs=n_obs,
            benchmark_sr_period=bench_period_eff,
            skew=skew, kurt=kurt,
        )
        dsr_raw = deflated_sharpe_ratio(
            sharpe_period=sh_period,
            n_obs=n_obs,
            benchmark_sr_period=bench_period_raw,
            skew=skew, kurt=kurt,
        )

        cells.append(CellResult(
            asset=asset, variant=variant,
            model_label=str(best.get("model_label", "")),
            config_label=str(best.get("config_label", "")),
            sharpe=sh_ann,
            p_unadj=p_unadj,
            p_holm=float("nan"),  # filled after across-asset Holm
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            dsr_harm=dsr_harm,
            dsr_raw=dsr_raw,
            n_folds=n_obs,
            n_trials_raw=n_trials_raw,
            n_trials_eff=n_trials_eff,
            bench_sr_period=bench_period_eff,
            folds_arr_obj=folds,
        ))

    # Across-asset Holm
    pvals = np.array([c.p_unadj for c in cells], dtype=float)
    if len(pvals) > 0 and np.all(np.isfinite(pvals)):
        adj = holm_stepdown(pvals)
        for c, pa in zip(cells, adj):
            c.p_holm = float(pa)
    elif len(pvals) > 0:
        # For long_only_ensemble: leave as NaN
        finite_mask = np.isfinite(pvals)
        if finite_mask.any():
            adj_sub = holm_stepdown(pvals[finite_mask])
            j = 0
            for i, c in enumerate(cells):
                if finite_mask[i]:
                    c.p_holm = float(adj_sub[j]); j += 1

    rows = [c.__dict__ for c in cells]
    return pd.DataFrame(rows), diag


def verdict(row) -> str:
    if not np.isfinite(row.get("dsr_harm", np.nan)):
        return "INELIGIBLE (no per-fold)"
    if row["sharpe"] <= 0:
        return "FAIL: Sharpe<=0"
    if not np.isfinite(row["p_holm"]):
        return "FAIL: p_holm NaN"
    fails = []
    if row["dsr_harm"] <= 0.95: fails.append(f"DSR={row['dsr_harm']:.2f}")
    if row["p_holm"] >= 0.05: fails.append(f"p_Holm={row['p_holm']:.3f}")
    if not np.isfinite(row["ci_lo"]) or row["ci_lo"] <= 0:
        fails.append(f"CI_lo={row['ci_lo']:.3f}" if np.isfinite(row['ci_lo']) else "CI_lo=NaN")
    return "WINNER" if not fails else "FAIL: " + ", ".join(fails)


def main():
    all_cells = []
    all_diag = {}
    for v in VARIANT_PATHS:
        print(f"[harmonize] processing variant {v}")
        df, diag = analyse_variant(v)
        df["verdict"] = df.apply(verdict, axis=1)
        all_cells.append(df)
        all_diag[v] = diag
        out = df.drop(columns=["folds_arr_obj"])
        out.to_parquet(f"/Users/weeoto/macminibot/data/bakeoff_variants/_harmonize/{v}_best.parquet")
        print(out[["asset", "model_label", "config_label", "sharpe",
                   "dsr_harm", "p_holm", "ci_lo", "verdict"]].to_string(index=False))
        print()

    full = pd.concat(all_cells, ignore_index=True)
    # Save folds_arr_obj separately (pickled) for RW stepdown.
    folds_records = []
    for _, r in full.iterrows():
        folds_records.append({
            "asset": r["asset"], "variant": r["variant"],
            "model_label": r["model_label"], "config_label": r["config_label"],
            "folds": np.asarray(r["folds_arr_obj"], dtype=float).tolist(),
            "sharpe": float(r["sharpe"]),
        })
    with open("/Users/weeoto/macminibot/data/bakeoff_variants/_harmonize/folds.json", "w") as f:
        json.dump(folds_records, f)

    # ---------------- Romano-Wolf across 40 cells (those with folds) -------
    rw_inputs = []
    rw_meta = []
    for _, r in full.iterrows():
        folds = np.asarray(r["folds_arr_obj"], dtype=float)
        if len(folds) >= 5 and r["sharpe"] > 0:
            rw_inputs.append(folds)
            rw_meta.append((r["asset"], r["variant"], r["model_label"], r["config_label"]))
    print(f"[RW] running stepdown on {len(rw_inputs)} cells")
    reject, rw_p = romano_wolf_stepdown(rw_inputs, alpha=0.05, B=5000)
    rw_df = pd.DataFrame([
        {"asset": m[0], "variant": m[1], "model_label": m[2],
         "config_label": m[3], "rw_pvalue": float(p), "rw_reject_H0": bool(rj)}
        for m, p, rj in zip(rw_meta, rw_p, reject)
    ])
    rw_df.to_parquet("/Users/weeoto/macminibot/data/bakeoff_variants/_harmonize/rw.parquet")

    # Merge RW results back
    full = full.merge(rw_df, on=["asset", "variant", "model_label", "config_label"],
                      how="left")
    full_out = full.drop(columns=["folds_arr_obj"])
    full_out.to_parquet("/Users/weeoto/macminibot/data/bakeoff_variants/_harmonize/all_cells.parquet")

    # ---------------- Per-variant survivor summary --------------------------
    print("\n========== HARMONIZED RESULTS (40 cells) ==========")
    cols = ["asset", "variant", "model_label", "config_label", "sharpe",
            "dsr_harm", "dsr_raw", "p_unadj", "p_holm", "ci_lo", "ci_hi",
            "rw_pvalue", "rw_reject_H0", "verdict"]
    print(full_out[cols].to_string(index=False))

    print("\n========== Variant-level survivor counts ==========")
    summary_rows = []
    for v in VARIANT_PATHS:
        v_df = full_out[full_out["variant"] == v]
        gate_win = v_df[v_df["verdict"] == "WINNER"]
        rw_win = v_df[v_df["rw_reject_H0"] == True]
        sharpes_gate = gate_win["sharpe"].values
        sharpes_rw = rw_win["sharpe"].values
        # Sub-portfolio equal-weight Sharpe across RW survivors
        if len(sharpes_rw) >= 1:
            ew_sharpe_rw = float(np.mean(sharpes_rw))
        else:
            ew_sharpe_rw = float("nan")
        summary_rows.append({
            "variant": v,
            "n_gate_winners": int(len(gate_win)),
            "n_rw_winners": int(len(rw_win)),
            "avg_sharpe_gate": float(np.mean(sharpes_gate)) if len(sharpes_gate) else float("nan"),
            "avg_sharpe_rw": ew_sharpe_rw,
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_parquet("/Users/weeoto/macminibot/data/bakeoff_variants/_harmonize/variant_summary.parquet")

    # Diagnostic dump
    with open("/Users/weeoto/macminibot/data/bakeoff_variants/_harmonize/diag.json", "w") as f:
        json.dump(all_diag, f, indent=2, default=str)

    return full_out, summary


if __name__ == "__main__":
    main()
