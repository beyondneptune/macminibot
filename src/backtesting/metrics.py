"""Performance metrics: classical (Sharpe/Sortino/Calmar/MaxDD) +
overfitting-aware (PSR, DSR, PBO) + bootstrap CIs.

References:
* PSR — Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier*,
  J. Risk 15(2). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
* DSR — Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*,
  J. Portfolio Management 40(5).
* PBO — Bailey, Borwein, López de Prado, Zhu (2016), *The Probability of
  Backtest Overfitting*, J. Computational Finance 20(4).
* Stationary bootstrap — Politis & Romano (1994), *The Stationary Bootstrap*,
  J. Am. Statistical Assoc. 89(428).
* Automatic block-length — Politis & White (2004), *Automatic Block-Length
  Selection for the Dependent Bootstrap*, Econometric Reviews 23(1) — with
  Patton/Politis/White (2009) correction.

All functions are pure: accept array-likes (np.ndarray / list / pd.Series),
return ``float`` (or dataclass for the bootstrap result). No side effects,
no I/O, no global RNG (CONTEXT.md cycle-4 — bootstrap takes an explicit
``np.random.Generator`` or ``seed``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

__all__ = [
    "BootstrapSharpeResult",
    "bootstrap_sharpe_ci",
    "calmar",
    "dsr",
    "dsr_expected_max_sr",
    "max_drawdown",
    "pbo",
    "politis_white_block_length",
    "psr",
    "sharpe",
    "sortino",
]

ArrayLike = Sequence[float] | np.ndarray


def _as_array(x: ArrayLike) -> np.ndarray:
    """Coerce input to ``np.ndarray[float64]`` for stable arithmetic."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D returns, got shape {arr.shape}")
    return arr


def sharpe(returns: ArrayLike, periods: int = 252) -> float:
    """Annualised Sharpe ratio (no risk-free rate — already-excess assumed).

        SR = mean(returns) / std(returns, ddof=1) * sqrt(periods)

    Returns ``+inf`` for a zero-vol all-positive series, ``-inf`` for all-negative
    zero-vol, ``0.0`` for an all-zero series. ``nan`` if fewer than 2 samples.
    """
    r = _as_array(returns)
    if r.size < 2:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        if mu > 0:
            return math.inf
        if mu < 0:
            return -math.inf
        return 0.0
    return mu / sd * math.sqrt(periods)


def sortino(returns: ArrayLike, target: float = 0.0, periods: int = 252) -> float:
    """Sortino ratio — Sharpe with downside-only deviation.

        downside = sqrt( mean( min(0, r - target)^2 ) )
        sortino  = (mean(r) - target) / downside * sqrt(periods)

    Returns ``+inf`` if there are no downside deviations and excess > 0; ``nan``
    for <2 samples.
    """
    r = _as_array(returns)
    if r.size < 2:
        return float("nan")
    excess = r - target
    downside_sq = np.minimum(0.0, excess) ** 2
    dd = float(math.sqrt(downside_sq.mean()))
    mu = float(excess.mean())
    if dd == 0.0:
        if mu > 0:
            return math.inf
        if mu < 0:
            return -math.inf
        return 0.0
    return mu / dd * math.sqrt(periods)


def max_drawdown(equity_curve: ArrayLike) -> float:
    """Maximum peak-to-trough drawdown as a non-positive fraction.

    ``0.0`` for a monotonically non-decreasing curve; ``-0.30`` for a 30% DD.
    Accepts either an equity series (e.g. NAV) or a cumulative-return series
    (e.g. 1.0-normalised). Empty / single-point curves return ``0.0``.
    """
    eq = _as_array(equity_curve)
    if eq.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    # Guard against zero running_max (e.g. starts at 0) — treat as no DD.
    safe = np.where(running_max == 0, 1.0, running_max)
    dd = (eq - running_max) / safe
    return float(dd.min())


def calmar(returns: ArrayLike, periods: int = 252) -> float:
    """Calmar ratio: CAGR / |max drawdown|.

    CAGR is geometric-annualised over the *observed* sample length, not over
    a calendar-year. For a 1-year sample with daily returns this matches the
    standard textbook value; for shorter samples it extrapolates and the user
    should interpret accordingly. Returns ``nan`` if MaxDD == 0 (no DD →
    Calmar undefined).
    """
    r = _as_array(returns)
    if r.size < 2:
        return float("nan")
    # Cumulative equity curve assuming compounding from 1.0.
    eq = np.cumprod(1.0 + r)
    total_return = float(eq[-1]) - 1.0
    years = r.size / periods
    if years <= 0:
        return float("nan")
    # Handle wipeout (final equity <= 0) — CAGR undefined.
    if eq[-1] <= 0:
        return -math.inf
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    mdd = max_drawdown(eq)
    if mdd == 0.0:
        return float("nan")
    return cagr / abs(mdd)


def psr(returns: ArrayLike, benchmark_sr: float = 0.0, periods: int = 252) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    Probability that the true (annualised) Sharpe exceeds a benchmark,
    accounting for sample length and the higher-moment correction (skew,
    excess kurtosis) of the daily-returns distribution. Returns a value in
    ``[0, 1]``.

    Formula:
        SR_hat                = sample Sharpe (annualised)
        gamma3, gamma4        = skew, excess kurtosis of *period* returns
        SR_period             = SR_hat / sqrt(periods)
        SR_bench_period       = benchmark_sr / sqrt(periods)
        sigma_SR              = sqrt( (1 - gamma3*SR_p + (gamma4/4)*SR_p^2)
                                       / (T - 1) )
        PSR                   = Phi( (SR_p - SR_bench_period) / sigma_SR )
    """
    r = _as_array(returns)
    if r.size < 3:
        return float("nan")
    sr_hat_period = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=True, bias=False))  # excess kurt
    sr_bench_period = benchmark_sr / math.sqrt(periods)
    T = r.size
    denom_inner = 1.0 - g3 * sr_hat_period + (g4 / 4.0) * sr_hat_period**2
    # Guard pathological negative variance from extreme small-sample moments.
    denom_inner = max(denom_inner, 1e-12)
    sigma_sr = math.sqrt(denom_inner / (T - 1))
    if sigma_sr == 0:
        return 1.0 if sr_hat_period > sr_bench_period else 0.0
    z = (sr_hat_period - sr_bench_period) / sigma_sr
    return float(stats.norm.cdf(z))


def dsr_expected_max_sr(
    n_trials: int,
    sr_trials_std: float = 1.0,
    periods: int = 252,
) -> float:
    """Expected maximum annualised Sharpe under the null across ``n_trials``.

    Bailey & López de Prado (2014) eq. (5):

        E[max SR_n] ≈ sigma_SR * ( (1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N·e)) )

    where ``γ`` is Euler-Mascheroni and ``sigma_SR`` is the *cross-trial
    standard deviation* of Sharpe ratios under the null. The :func:`dsr`
    function fixes ``sigma_SR = 1`` (the de-facto generic gate); this helper
    exposes the underlying calculation so callers with **observed fold
    variance** (e.g. the WalkForwardResult's per-fold Sharpes) can compute a
    properly-calibrated deflation benchmark rather than the one-size-fits-all
    DSR ≥ 0.95 threshold the CONTEXT calls out.

    Usage (calibrating against actual fold variance)::

        sigma = float(np.std([fm.sharpe_net for fm in result.fold_metrics],
                             ddof=1))
        bench_sr = dsr_expected_max_sr(n_trials=len(result.fold_metrics),
                                       sr_trials_std=sigma)
        gate_psr = psr(returns, benchmark_sr=bench_sr)
        # gate_psr ≥ 0.95 is the variance-calibrated DSR gate.

    Args:
        n_trials: number of independent strategy trials in the search. ``1``
            collapses the benchmark to zero (no deflation needed).
        sr_trials_std: standard deviation of Sharpe ratios across trials under
            the null. Defaults to 1.0 (CONTEXT's generic gate). Pass the
            observed across-fold std for variance-calibrated gating.
        periods: annualisation factor.

    Returns:
        Annualised expected-max-SR benchmark, ready to pass as
        ``benchmark_sr`` to :func:`psr`.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if sr_trials_std < 0:
        raise ValueError(f"sr_trials_std must be >= 0, got {sr_trials_std}")
    if n_trials == 1:
        return 0.0

    gamma_em = 0.5772156649015329  # Euler-Mascheroni
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    e_max_period = sr_trials_std * ((1.0 - gamma_em) * z1 + gamma_em * z2)
    return float(e_max_period * math.sqrt(periods))


def dsr(sr_observed: float, n_trials: int, returns: ArrayLike, periods: int = 252) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Corrects PSR for the fact that ``n_trials`` strategies were searched, so
    the maximum observed Sharpe is upward-biased even under the null. The
    expected-max-SR under N independent trials of a standard Normal is:

        E[max SR] ≈ (1 - gamma_em) * Phi^{-1}(1 - 1/N)
                  + gamma_em      * Phi^{-1}(1 - 1/(N*e))

    where ``gamma_em`` is the Euler-Mascheroni constant. We compare the
    observed Sharpe to this null benchmark via PSR.

    Args:
        sr_observed: annualised Sharpe of the best-of-N strategy. (We accept
            it as an argument rather than recomputing it from ``returns`` so
            the caller can pass the gross or net Sharpe explicitly.)
        n_trials: number of independent strategy variants searched.
        returns: the return series of the *best* strategy (for skew/kurt
            higher-moment correction inside PSR).
        periods: annualisation factor.

    Returns:
        DSR in ``[0, 1]``; gate is ``>= 0.95`` for promotion (CONTEXT.md
        §Metrics & gates).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        # No deflation needed — DSR collapses to PSR vs zero benchmark.
        return psr(returns, benchmark_sr=0.0, periods=periods)

    # Default cross-trial sigma=1.0 — generic/conservative gate per CONTEXT.
    # Callers wanting variance-calibrated deflation should compute their own
    # benchmark via :func:`dsr_expected_max_sr` and pass it to :func:`psr`.
    e_max_sr_annual = dsr_expected_max_sr(
        n_trials=n_trials, sr_trials_std=1.0, periods=periods,
    )
    return psr(returns, benchmark_sr=e_max_sr_annual, periods=periods)


def pbo(in_sample_sr: ArrayLike, out_of_sample_sr: ArrayLike) -> float:
    """Probability of Backtest Overfitting (Bailey, Borwein, López de Prado,
    Zhu 2016) — *combinatorially-symmetric cross-validation* simplified form.

    Given paired (IS, OOS) Sharpe samples across ``N`` candidate strategies,
    PBO is the probability that the strategy with the highest IS Sharpe ranks
    in the *bottom half* of OOS Sharpes — i.e. that IS performance does not
    predict OOS performance.

        omega = OOS_rank(argmax_IS) / N   in (0, 1]
        logit = log(omega / (1 - omega))
        PBO   = P(logit < 0) estimated empirically over symmetric splits

    For the simplified pairwise form here (no combinatorial splits required
    at the caller; the caller provides one IS/OOS pair per fold), we compute
    PBO as the fraction of folds where the IS-best strategy fell into the
    OOS bottom half. Both vectors must have equal length >= 2.

    The full combinatorial-CV variant lives in the bootstrap CI Epic 8;
    this function is the contract for Epic 4's per-run gate.
    """
    is_sr = _as_array(in_sample_sr)
    oos_sr = _as_array(out_of_sample_sr)
    if is_sr.shape != oos_sr.shape:
        raise ValueError(
            f"in_sample_sr / out_of_sample_sr shape mismatch: "
            f"{is_sr.shape} vs {oos_sr.shape}"
        )
    n = is_sr.size
    if n < 2:
        return float("nan")
    # OOS rank (1 = worst, n = best) of the strategy that was IS-best.
    best_is = int(np.argmax(is_sr))
    # Rank using average-rank for ties → unbiased.
    oos_ranks = stats.rankdata(oos_sr, method="average")
    rank_of_best_is = float(oos_ranks[best_is])
    omega = rank_of_best_is / n
    # Pathological boundary handling.
    if omega <= 0.0:
        return 1.0
    if omega >= 1.0:
        return 0.0
    # Single-trial PBO degenerates to "did the IS-best land below median?"
    # i.e. omega < 0.5 → overfit. Express as 1 - omega so larger PBO = worse.
    return float(1.0 - omega)


# ---- Bootstrap Sharpe CI (Politis-Romano stationary bootstrap) -----------


@dataclass(frozen=True, slots=True)
class BootstrapSharpeResult:
    """Output of :func:`bootstrap_sharpe_ci`.

    Attributes:
        point_estimate: Sharpe of the input series (annualised).
        ci_lower: Lower bound of the ``(1-alpha)`` percentile CI.
        ci_upper: Upper bound of the ``(1-alpha)`` percentile CI.
        p_positive: Bootstrap-estimated ``P(true Sharpe > 0)`` =
            ``mean(SR_b > 0)`` over the resampled distribution.
        block_length: Mean block length actually used (Politis-White auto if
            requested, otherwise the user-supplied value).
        n_resamples: Number of bootstrap resamples (``B``).
        alpha: Tail probability used for the CI (e.g. ``0.05`` → 95% CI).
        bootstrap_distribution: 1-D ndarray of all ``B`` resampled Sharpes —
            kept so callers can compute their own percentiles / plot the
            null density without re-running the bootstrap.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    p_positive: float
    block_length: float
    n_resamples: int
    alpha: float
    bootstrap_distribution: np.ndarray


def politis_white_block_length(
    returns: ArrayLike,
    *,
    max_lag: int | None = None,
) -> float:
    """Politis-White (2004) automatic mean block-length for the stationary
    bootstrap. Uses the Patton-Politis-White (2009) correction.

    Returns the mean of the geometric distribution that draws block lengths
    in the stationary bootstrap. For daily returns the typical answer for
    backtest series is in the range 5–30.

    Algorithm:
        1. Compute the autocorrelation function ρ̂(k) up to ``max_lag``.
        2. Find the smallest ``k*`` such that |ρ̂(k*+j)| < 2 * sqrt(log10(T)/T)
           for ``j = 1..K_N`` (K_N = max(5, log10(T)) per PPW).
        3. Sum the lag-window-weighted autocovariances → bandwidth estimator.
        4. Optimal block length = (2 * G^2 / D)^(1/3) * T^(1/3), where
           G = Σ_{k=-M..M} |k| * γ̂(k) and
           D = (4/3) * Σ_{k=-M..M} γ̂(k)^2  (stationary-bootstrap variant).

    Falls back to ``max(2, sqrt(T))`` if the autocorrelation never decays.
    """
    r = _as_array(returns)
    T = r.size
    if T < 4:
        return 1.0
    if max_lag is None:
        max_lag = min(T - 1, int(math.ceil(min(3.0 * math.sqrt(T), T / 3.0))))

    # Centre the series.
    r_c = r - r.mean()
    var = float(np.dot(r_c, r_c) / T)
    if var <= 0.0:
        return 1.0

    # Autocovariance γ̂(k) at lag k=0..max_lag.
    gamma = np.empty(max_lag + 1, dtype=np.float64)
    gamma[0] = var
    for k in range(1, max_lag + 1):
        gamma[k] = float(np.dot(r_c[:-k], r_c[k:]) / T)
    rho = gamma / gamma[0]

    # Find M: smallest m such that |rho[m+j]| < threshold for j=1..K_N.
    K_N = max(5, int(math.ceil(math.log10(max(T, 10)))))
    threshold = 2.0 * math.sqrt(math.log10(max(T, 10)) / T)
    M = max_lag
    for m in range(1, max_lag - K_N + 1):
        window = np.abs(rho[m : m + K_N])
        if np.all(window < threshold):
            M = m
            break
    # Patton-Politis-White cap: M ≤ floor(sqrt(T)) for robustness.
    M = min(M, max(1, int(math.floor(math.sqrt(T)))))
    if M < 1:
        M = 1

    # Stationary-bootstrap optimal-block-length numerator G and denominator D.
    # Two-sided sum over k=-M..M with flat-top lag window (Politis-Romano).
    # Flat-top w(k/M): 1 if |k/M| <= 1/2; 2(1-|k/M|) if 1/2 < |k/M| <= 1; else 0.
    def _w(x: float) -> float:
        ax = abs(x)
        if ax <= 0.5:
            return 1.0
        if ax <= 1.0:
            return 2.0 * (1.0 - ax)
        return 0.0

    G = 0.0
    D = 0.0
    for k in range(-M, M + 1):
        ak = abs(k)
        if ak == 0:
            g_k = gamma[0]
        else:
            g_k = gamma[ak]
        w = _w(k / M)
        G += w * ak * g_k
        D += w * g_k * g_k
    # PPW stationary-bootstrap variant: D_sb = 2 * D.
    D_sb = 2.0 * D
    if D_sb <= 0:
        return max(2.0, math.sqrt(T))
    b_opt = (2.0 * G * G / D_sb) ** (1.0 / 3.0) * (T ** (1.0 / 3.0))
    # Guard pathological tiny/huge values.
    b_opt = max(1.0, min(b_opt, T / 2.0))
    return float(b_opt)


def _stationary_bootstrap_indices(
    n: int,
    block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap index sampler.

    Draws ``n`` indices in ``[0, n)`` by starting at a uniform random index
    and, at each subsequent step, either advancing by 1 (with probability
    ``1 - 1/L``) or jumping to a new uniform random index (with probability
    ``1/L``), wrapping around mod ``n``. Mean block length is ``L``.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    p = 1.0 / max(block_length, 1.0)
    # Pre-draw all uniforms in one vectorised shot — much faster than a Python loop.
    starts = rng.integers(0, n, size=n)
    jumps = rng.random(n) < p
    idx = np.empty(n, dtype=np.int64)
    cur = int(starts[0])
    idx[0] = cur
    for i in range(1, n):
        if jumps[i]:
            cur = int(starts[i])
        else:
            cur = (cur + 1) % n
        idx[i] = cur
    return idx


def bootstrap_sharpe_ci(
    returns: ArrayLike,
    *,
    B: int = 10_000,
    block_length: float | str = "auto",
    alpha: float = 0.05,
    periods: int = 252,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> BootstrapSharpeResult:
    """Politis-Romano (1994) stationary block-bootstrap CI for the annualised
    Sharpe ratio. Block length defaults to Politis-White (2004) automatic
    selection — the right thing for autocorrelated daily return series.

    This is the cycle-8 deliverable closing the CONTEXT-pinned promotion gate
    ``Sharpe 95% CI lower bound > 0`` (CONTEXT.md §Metrics & gates). The
    output also reports ``p_positive = mean(SR_b > 0)`` — the bootstrap-implied
    posterior probability of positive Sharpe, which is the more interpretable
    number for a single-strategy-single-window backtest.

    Args:
        returns: 1-D series of per-period (typically daily) returns.
        B: number of bootstrap resamples (default 10,000 per CONTEXT cycle-4
           RNG contract).
        block_length: mean block length for the geometric jump distribution.
           Pass ``"auto"`` (default) for Politis-White selection; pass a
           positive float to force a specific value.
        alpha: tail probability for the percentile CI (``alpha=0.05`` →
           95% CI).
        periods: annualisation factor (default 252 trading days/year).
        rng: explicit ``numpy.random.Generator``. Mutually exclusive with
           ``seed`` — pass one or the other (or neither for a fresh
           ``default_rng()`` which IS allowed here since this function is
           called from a sub-seed slot, not from production code, per
           CONTEXT cycle-4).
        seed: convenience int seed → ``np.random.default_rng(seed)``.

    Returns:
        :class:`BootstrapSharpeResult` with point estimate, CI bounds,
        ``p_positive``, the resolved block length, and the full bootstrap
        distribution for downstream plotting.

    Raises:
        ValueError: ``B < 1``, ``alpha`` out of ``(0, 1)``, both ``rng`` and
            ``seed`` provided, or a non-positive ``block_length`` numeric.
    """
    r = _as_array(returns)
    T = r.size
    if T < 2:
        # Degenerate input — return NaN bounds rather than raise so callers
        # can chain this through aggregate reports that may have empty folds.
        nan_dist = np.array([], dtype=np.float64)
        return BootstrapSharpeResult(
            point_estimate=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            p_positive=float("nan"),
            block_length=float("nan"),
            n_resamples=0,
            alpha=alpha,
            bootstrap_distribution=nan_dist,
        )
    if B < 1:
        raise ValueError(f"B must be >= 1, got {B}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    # Resolve block length.
    if isinstance(block_length, str):
        if block_length != "auto":
            raise ValueError(
                f"block_length must be 'auto' or a positive float, got {block_length!r}"
            )
        L = politis_white_block_length(r)
    else:
        L = float(block_length)
        if L <= 0:
            raise ValueError(f"block_length must be > 0, got {L}")

    point = sharpe(r, periods=periods)

    # Vectorise the per-resample Sharpe computation as much as possible.
    # We still need a Python loop for index generation (the stationary bootstrap
    # is inherently sequential), but the sharpe() reduction is numpy.
    sr_boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = _stationary_bootstrap_indices(T, L, rng)
        sample = r[idx]
        # Inline the sharpe computation to avoid the per-call ArrayLike coercion.
        mu = float(sample.mean())
        sd = float(sample.std(ddof=1))
        if sd == 0.0:
            sr_boot[b] = math.inf if mu > 0 else (-math.inf if mu < 0 else 0.0)
        else:
            sr_boot[b] = mu / sd * math.sqrt(periods)

    # Percentile CI (basic — for bias-corrected we'd need a second bootstrap
    # pass; percentile is the contract per CONTEXT.md "B=10,000 resamples on
    # daily returns" and is the default in PPW's reference implementation).
    finite = sr_boot[np.isfinite(sr_boot)]
    if finite.size == 0:
        ci_lo = float("nan")
        ci_hi = float("nan")
        p_pos = float("nan")
    else:
        ci_lo = float(np.quantile(finite, alpha / 2.0))
        ci_hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
        p_pos = float(np.mean(finite > 0.0))

    return BootstrapSharpeResult(
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_positive=p_pos,
        block_length=L,
        n_resamples=B,
        alpha=alpha,
        bootstrap_distribution=sr_boot,
    )
