"""Vol-target scaler + signal winsorization.

Per CONTEXT.md "Optimizer":
    σ_p     = sqrt(w' · Σ · w) · sqrt(252)
    w_final = w · (σ_target / σ_p)            # σ_target = 12% annualized

And edge-case rule "Signal blowup": winsorize to 3σ of the cross-sectional
distribution. Rule "All signals near zero" (< 5 bps) lives in
``edge_cases.py``; this module only handles the per-call clipping primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_TARGET_VOL: float = 0.12  # 12% annualized
DEFAULT_WINSORIZE_NSIGMA: float = 3.0

# VIX-regime dampening thresholds (percentiles of the vix_proxy distribution).
# Cycle 9 found Ridge loses money in ``vix_high`` (top quintile, Sharpe -0.252).
# Approach (1) — the hard gate — lives in risk-management as
# ``VIX_HIGH_REGIME_HALT`` (cycle 10) and is enforced by the scheduler before
# orders are submitted (``mode != HALTED``).  This module owns approach (2) —
# soft degradation — which kicks in at the *intermediate* (60th–80th pct) band
# so that we lean off risk before the hard gate fires.
#
#  pct < 60 → full target vol (12%)
#  60 ≤ pct < 80 → halved target vol (6%) — intermediate dampening
#  pct ≥ 80 → zero target vol (paper-mode equivalent of HALT)
VIX_DAMPEN_PCT_INTERMEDIATE: float = 0.60
VIX_DAMPEN_PCT_HALT: float = 0.80
VIX_DAMPEN_INTERMEDIATE_MULT: float = 0.5  # 12% × 0.5 = 6%

# ---------------------------------------------------------------------------
# vix-mid-only mode (cycle 13, opt-in alternative to the soft-degradation
# pathway above).  Cycle 12 ml-research showed that the *low*-VIX tail
# (<40th pct) is also a money-loser for the Ridge stack — the alpha lives
# almost entirely in the 40–80th pct "mid" band.  This mode trades *only*
# in that band and goes flat everywhere else, instead of the graduated
# 60/80 dampening that ``vol_target_with_regime_dampening`` applies.
#
#  pct < 40 → zero weights (vix_low — flat)
#  40 ≤ pct < 80 → full base target vol, normal weights (vix_mid)
#  pct ≥ 80 → zero weights (vix_high — flat)
#
# These thresholds are intentionally *independent* of the 60/80 dampening
# constants above: this is a separate strategy, not a tuning of the
# existing one.  Callers pick exactly one of the two regime-aware scalers
# at the scheduler/risk-manager layer.
VIX_MID_ONLY_PCT_LOW: float = 0.40
VIX_MID_ONLY_PCT_HIGH: float = 0.80


@dataclass(frozen=True)
class VolTargetResult:
    """Output of the vol-target scaler."""

    w_scaled: np.ndarray
    realized_vol_pre: float
    realized_vol_post: float
    scaling_factor: float
    clamped: bool  # True if scaling was clamped to a safety ceiling


def vol_target_scale(
    w_unscaled: np.ndarray,
    Sigma: np.ndarray,
    *,
    target_vol_annualized: float = DEFAULT_TARGET_VOL,
    max_leverage_multiplier: float = 5.0,
) -> VolTargetResult:
    """Scale weights so ex-ante annualized portfolio vol equals the target.

    Parameters
    ----------
    w_unscaled:
        Optimizer output (risk-units).
    Sigma:
        Daily-return covariance (NOT annualized; we apply ``sqrt(252)``
        ourselves).
    target_vol_annualized:
        e.g. 0.12 = 12% per year.
    max_leverage_multiplier:
        Hard cap on the scaling factor; a degenerate near-zero portfolio
        would otherwise demand infinite leverage. Default 5× — beyond
        that the gross-exposure cap should have caught it already and
        we clamp + flag.
    """
    w = np.asarray(w_unscaled, dtype=np.float64)
    S = np.asarray(Sigma, dtype=np.float64)
    daily_var = float(w @ S @ w)
    if daily_var < 0:
        daily_var = 0.0
    daily_sigma = float(np.sqrt(daily_var))
    realized_pre = daily_sigma * np.sqrt(TRADING_DAYS_PER_YEAR)

    if realized_pre <= 1e-12:
        # Degenerate portfolio (all zero or perfectly hedged): no scaling possible.
        return VolTargetResult(
            w_scaled=w.copy(),
            realized_vol_pre=realized_pre,
            realized_vol_post=realized_pre,
            scaling_factor=1.0,
            clamped=True,
        )

    raw_factor = target_vol_annualized / realized_pre
    clamped = bool(raw_factor > max_leverage_multiplier)
    factor = float(min(raw_factor, max_leverage_multiplier))
    w_scaled = w * factor
    realized_post = float(np.sqrt(max(w_scaled @ S @ w_scaled, 0.0))) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )

    return VolTargetResult(
        w_scaled=w_scaled,
        realized_vol_pre=realized_pre,
        realized_vol_post=realized_post,
        scaling_factor=factor,
        clamped=clamped,
    )


def _vix_regime_target_vol(
    vix_proxy_value: float,
    vix_distribution: np.ndarray,
    *,
    base_target_vol: float = DEFAULT_TARGET_VOL,
    pct_intermediate: float = VIX_DAMPEN_PCT_INTERMEDIATE,
    pct_halt: float = VIX_DAMPEN_PCT_HALT,
    intermediate_mult: float = VIX_DAMPEN_INTERMEDIATE_MULT,
) -> Tuple[float, str]:
    """Pick the effective vol-target given the current VIX-proxy reading.

    Returns ``(effective_target_vol, regime_label)`` where ``regime_label``
    is one of ``"normal"`` / ``"intermediate"`` / ``"halt"``.

    The percentile is computed against ``vix_distribution`` (the rolling
    historical sample of ``vix_proxy`` values, typically the same series the
    backtester's ``RegimeTagger`` was fit on).  If the distribution is empty
    or non-finite, we default to ``normal`` (fail-open is unsafe; we instead
    fail to the *unscaled* base target and let the hard gate catch real
    blowups — the soft path is purely advisory).
    """
    dist = np.asarray(vix_distribution, dtype=np.float64).ravel()
    dist = dist[np.isfinite(dist)]
    if dist.size == 0 or not np.isfinite(vix_proxy_value):
        return base_target_vol, "normal"
    # Fraction of historical observations strictly below the current reading.
    pct = float(np.mean(dist < float(vix_proxy_value)))
    if pct >= pct_halt:
        return 0.0, "halt"
    if pct >= pct_intermediate:
        return base_target_vol * intermediate_mult, "intermediate"
    return base_target_vol, "normal"


@dataclass(frozen=True)
class RegimeDampenedVolTargetResult:
    """Output of :func:`vol_target_with_regime_dampening`.

    Wraps the inner :class:`VolTargetResult` plus the regime decision so
    the caller (scheduler / risk manager) can log *why* the target vol
    was reduced or zeroed without re-deriving it.
    """

    w_scaled: np.ndarray
    realized_vol_pre: float
    realized_vol_post: float
    scaling_factor: float
    clamped: bool
    regime_label: str  # "normal" | "intermediate" | "halt"
    effective_target_vol: float
    vix_percentile: float


def vol_target_with_regime_dampening(
    w_unscaled: np.ndarray,
    Sigma: np.ndarray,
    vix_proxy_value: float,
    vix_distribution: np.ndarray,
    *,
    base_target_vol: float = DEFAULT_TARGET_VOL,
    max_leverage_multiplier: float = 5.0,
    pct_intermediate: float = VIX_DAMPEN_PCT_INTERMEDIATE,
    pct_halt: float = VIX_DAMPEN_PCT_HALT,
    intermediate_mult: float = VIX_DAMPEN_INTERMEDIATE_MULT,
) -> RegimeDampenedVolTargetResult:
    """Vol-target scaler that dampens the target when VIX-proxy is elevated.

    Soft-degradation companion to risk-management's ``VIX_HIGH_REGIME_HALT``
    hard gate.  At/above the 80th percentile of ``vix_distribution`` we
    zero the target (returns all-zero weights — equivalent to paper-mode
    halt at the sizing layer); between the 60th and 80th we halve it
    (12% → 6%); below the 60th we behave exactly like
    :func:`vol_target_scale`.

    Cycle 10 rationale: cycle 9 showed Ridge has Sharpe -0.252 in
    ``vix_high`` (top quintile).  Leaning off risk one tier early lets the
    soft path absorb most of the damage before the hard gate engages.

    Parameters
    ----------
    w_unscaled, Sigma:
        See :func:`vol_target_scale`.
    vix_proxy_value:
        Current daily reading of the canonical ``vix_proxy`` series
        (the cross-sectional dispersion proxy defined in
        ``ml_research.features._vix_proxy``).
    vix_distribution:
        Rolling historical sample of ``vix_proxy`` (same series the
        backtester's ``RegimeTagger`` was fit on).
    """
    effective_target, regime_label = _vix_regime_target_vol(
        vix_proxy_value,
        vix_distribution,
        base_target_vol=base_target_vol,
        pct_intermediate=pct_intermediate,
        pct_halt=pct_halt,
        intermediate_mult=intermediate_mult,
    )

    dist = np.asarray(vix_distribution, dtype=np.float64).ravel()
    dist = dist[np.isfinite(dist)]
    if dist.size == 0 or not np.isfinite(vix_proxy_value):
        vix_pct = float("nan")
    else:
        vix_pct = float(np.mean(dist < float(vix_proxy_value)))

    if regime_label == "halt":
        w = np.asarray(w_unscaled, dtype=np.float64)
        S = np.asarray(Sigma, dtype=np.float64)
        realized_pre = float(np.sqrt(max(w @ S @ w, 0.0))) * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        return RegimeDampenedVolTargetResult(
            w_scaled=np.zeros_like(w),
            realized_vol_pre=realized_pre,
            realized_vol_post=0.0,
            scaling_factor=0.0,
            clamped=False,
            regime_label="halt",
            effective_target_vol=0.0,
            vix_percentile=vix_pct,
        )

    inner = vol_target_scale(
        w_unscaled,
        Sigma,
        target_vol_annualized=effective_target,
        max_leverage_multiplier=max_leverage_multiplier,
    )
    return RegimeDampenedVolTargetResult(
        w_scaled=inner.w_scaled,
        realized_vol_pre=inner.realized_vol_pre,
        realized_vol_post=inner.realized_vol_post,
        scaling_factor=inner.scaling_factor,
        clamped=inner.clamped,
        regime_label=regime_label,
        effective_target_vol=effective_target,
        vix_percentile=vix_pct,
    )


# ---------------------------------------------------------------------------
# vix_mid_only_mode — opt-in alternative to vol_target_with_regime_dampening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MidOnlyVolTargetResult:
    """Output of :func:`vol_target_with_regime_mid_only_mode`.

    Mirrors :class:`RegimeDampenedVolTargetResult` so callers can log the
    regime decision uniformly.  ``regime_label`` is one of
    ``"vix_low"`` / ``"vix_mid"`` / ``"vix_high"``.
    """

    w_scaled: np.ndarray
    realized_vol_pre: float
    realized_vol_post: float
    scaling_factor: float
    clamped: bool
    regime_label: str  # "vix_low" | "vix_mid" | "vix_high"
    effective_target_vol: float
    vix_percentile: float


def vol_target_with_regime_mid_only_mode(
    w_unscaled: np.ndarray,
    Sigma: np.ndarray,
    vix_proxy_value: float,
    vix_distribution: np.ndarray,
    *,
    base_target_vol: float = DEFAULT_TARGET_VOL,
    max_leverage_multiplier: float = 5.0,
    pct_low: float = VIX_MID_ONLY_PCT_LOW,
    pct_high: float = VIX_MID_ONLY_PCT_HIGH,
) -> MidOnlyVolTargetResult:
    """Vol-target scaler that only trades in the mid VIX-proxy band.

    **Opt-in alternative** to :func:`vol_target_with_regime_dampening`.
    Cycle 12 ml-research showed the Ridge stack's alpha is concentrated
    in the 40–80th-percentile band of the ``vix_proxy`` distribution,
    while *both* tails (the calm-market low tail and the stressed-market
    high tail) erode PnL.  Rather than soft-dampen one tail and trade
    the other at full size, this mode goes flat in *both* tails and
    runs the full 12% target only in the mid band:

    - ``vix_low``  (pct < 40)  → zero weights
    - ``vix_mid``  (40 ≤ pct < 80) → full :func:`vol_target_scale` at base target
    - ``vix_high`` (pct ≥ 80)  → zero weights

    Callers pick exactly one regime-aware scaler at the scheduler /
    risk-manager layer; this is not stacked on top of the soft-dampening
    path.  Like the dampening variant, an empty or non-finite
    distribution / reading falls back to the mid band (fail-to-trade is
    a worse outcome than running un-regimed for a single bar).

    Parameters
    ----------
    w_unscaled, Sigma:
        See :func:`vol_target_scale`.
    vix_proxy_value:
        Current daily reading of the canonical ``vix_proxy`` series.
    vix_distribution:
        Rolling historical sample of ``vix_proxy``.
    pct_low, pct_high:
        Lower (inclusive) and upper (exclusive) percentile boundaries
        of the mid band.  Defaults match the cycle-12 finding.
    """
    dist = np.asarray(vix_distribution, dtype=np.float64).ravel()
    dist = dist[np.isfinite(dist)]
    if dist.size == 0 or not np.isfinite(vix_proxy_value):
        # Fail-open to the mid band — same convention as the dampening path.
        inner = vol_target_scale(
            w_unscaled,
            Sigma,
            target_vol_annualized=base_target_vol,
            max_leverage_multiplier=max_leverage_multiplier,
        )
        return MidOnlyVolTargetResult(
            w_scaled=inner.w_scaled,
            realized_vol_pre=inner.realized_vol_pre,
            realized_vol_post=inner.realized_vol_post,
            scaling_factor=inner.scaling_factor,
            clamped=inner.clamped,
            regime_label="vix_mid",
            effective_target_vol=base_target_vol,
            vix_percentile=float("nan"),
        )

    vix_pct = float(np.mean(dist < float(vix_proxy_value)))

    in_mid_band = (vix_pct >= pct_low) and (vix_pct < pct_high)
    if in_mid_band:
        inner = vol_target_scale(
            w_unscaled,
            Sigma,
            target_vol_annualized=base_target_vol,
            max_leverage_multiplier=max_leverage_multiplier,
        )
        return MidOnlyVolTargetResult(
            w_scaled=inner.w_scaled,
            realized_vol_pre=inner.realized_vol_pre,
            realized_vol_post=inner.realized_vol_post,
            scaling_factor=inner.scaling_factor,
            clamped=inner.clamped,
            regime_label="vix_mid",
            effective_target_vol=base_target_vol,
            vix_percentile=vix_pct,
        )

    # Outside the mid band — zero everything, but still report the pre-scale
    # vol so logs can show what would have been deployed.
    w = np.asarray(w_unscaled, dtype=np.float64)
    S = np.asarray(Sigma, dtype=np.float64)
    realized_pre = float(np.sqrt(max(w @ S @ w, 0.0))) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    label = "vix_low" if vix_pct < pct_low else "vix_high"
    return MidOnlyVolTargetResult(
        w_scaled=np.zeros_like(w),
        realized_vol_pre=realized_pre,
        realized_vol_post=0.0,
        scaling_factor=0.0,
        clamped=False,
        regime_label=label,
        effective_target_vol=0.0,
        vix_percentile=vix_pct,
    )


def signal_winsorize(
    r_hat: np.ndarray,
    *,
    n_sigma: float = DEFAULT_WINSORIZE_NSIGMA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip cross-sectional outliers at ±n_sigma from the cross-sectional mean.

    Returns
    -------
    (clipped_r_hat, mask_where_clipped)
        ``mask_where_clipped[i] == True`` iff the i-th signal was modified.
    """
    r = np.asarray(r_hat, dtype=np.float64).copy()
    if r.size == 0:
        return r, np.zeros(0, dtype=bool)
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=0))
    if sd <= 0:
        return r, np.zeros_like(r, dtype=bool)
    lo = mu - n_sigma * sd
    hi = mu + n_sigma * sd
    mask = (r < lo) | (r > hi)
    r_clipped = np.clip(r, lo, hi)
    return r_clipped, mask


__all__ = [
    "DEFAULT_TARGET_VOL",
    "DEFAULT_WINSORIZE_NSIGMA",
    "TRADING_DAYS_PER_YEAR",
    "VIX_DAMPEN_INTERMEDIATE_MULT",
    "VIX_DAMPEN_PCT_HALT",
    "VIX_DAMPEN_PCT_INTERMEDIATE",
    "VIX_MID_ONLY_PCT_HIGH",
    "VIX_MID_ONLY_PCT_LOW",
    "MidOnlyVolTargetResult",
    "RegimeDampenedVolTargetResult",
    "VolTargetResult",
    "signal_winsorize",
    "vol_target_scale",
    "vol_target_with_regime_dampening",
    "vol_target_with_regime_mid_only_mode",
]
