"""calibration — derive drawdown breakers from backtest fold distribution (Epic 5).

CONTEXT.md improvement I2 says the ``-2% daily / -5% weekly / -10% PTT`` numbers
in ``risk/limits.toml`` are *priors*, not derived. Now that cycle-7's Ridge
baseline yields a real Sharpe +2.03 backtest with per-fold return distribution
(``backtesting.FoldMetrics.daily_returns`` via ``aggregate.json``), we can
calibrate the breakers to the strategy's actual variance rather than copying a
paper.

The CONTEXT prescribes a specific target: **the daily threshold should catch
the worst ~1-2% of historical days but NOT the worst ~10%**. Empirically this
maps to roughly the 1st-percentile daily return (with a small skew-aware
adjustment so a strategy with a fat left tail gets a tighter cap, not a looser
one — Sortino & van der Meer 1991's downside-asymmetry framing).

The function is intentionally a pure dataframe-free helper so it can be called
from a script, a notebook, or a CI gate without touching the chain. The
recommended output goes into ``risk/limits.toml`` under ``[account]`` after
operator review — this module does NOT auto-edit the config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence


# ---------------------------------------------------------------------------
# Quantile helpers (no numpy dep — risk-management stays stdlib-only)
# ---------------------------------------------------------------------------


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (same as ``numpy.quantile`` default).

    ``q`` in ``[0, 1]``. Values may be unsorted; we sort a copy.
    """

    if not values:
        raise ValueError("cannot compute quantile of empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0,1], got {q}")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def _skewness(values: Sequence[float]) -> float:
    """Sample skew (Fisher-Pearson). Negative => fat left tail."""

    n = len(values)
    if n < 3:
        return 0.0
    mu = _mean(values)
    sd = _stddev(values)
    if sd == 0.0:
        return 0.0
    m3 = sum((x - mu) ** 3 for x in values) / n
    return m3 / (sd ** 3)


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawdownCalibration:
    """Recommended drawdown breakers + the evidence they were derived from.

    All ``*_pct`` fields are negative (loss conventions match
    ``AccountLimits``). ``rationale`` is human-readable so the operator can
    paste it into the limits.toml commit message.
    """

    dd_daily_pct: Decimal
    dd_weekly_pct: Decimal
    dd_ptt_pct: Decimal
    # Provenance
    n_obs: int
    daily_mean: float
    daily_std: float
    daily_skew: float
    p01_daily: float
    p05_daily: float
    p10_daily: float
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dd_daily_pct": str(self.dd_daily_pct),
            "dd_weekly_pct": str(self.dd_weekly_pct),
            "dd_ptt_pct": str(self.dd_ptt_pct),
            "n_obs": self.n_obs,
            "daily_mean": self.daily_mean,
            "daily_std": self.daily_std,
            "daily_skew": self.daily_skew,
            "p01_daily": self.p01_daily,
            "p05_daily": self.p05_daily,
            "p10_daily": self.p10_daily,
            "rationale": self.rationale,
        }


def _quantize_loss(loss: float) -> Decimal:
    """Round a negative loss fraction to 4 decimal places (1 bp grid).

    The chain compares ``Decimal`` thresholds, and the calibration output
    feeds into ``limits.toml``; pinning to 1 bp keeps configs diff-able
    across calibration runs that move the number by < 1 bp.
    """

    return Decimal(f"{loss:.4f}")


def calibrate_drawdown_thresholds(
    daily_returns: Sequence[float],
    *,
    target_daily_tail_pct: float = 0.015,
    weekly_to_daily_multiplier: float = 2.5,
    ptt_to_daily_multiplier: float = 5.0,
    floor_daily_pct: float = -0.005,
    ceiling_daily_pct: float = -0.05,
) -> DrawdownCalibration:
    """Derive drawdown breakers from a daily-return distribution.

    Parameters
    ----------
    daily_returns
        Per-day fractional returns (e.g. ``0.0042`` for +42 bps). Must be
        finite. Typically ``[r for fm in result.fold_metrics for r in fm.daily_returns]``
        from a ``backtesting`` walk-forward run.
    target_daily_tail_pct
        Quantile probability for the daily breaker. ``0.015`` (the default)
        targets "catch the worst ~1.5% of historical days", matching CONTEXT.md
        I2's "worst 1-2%, not worst 10%" guidance.
    weekly_to_daily_multiplier, ptt_to_daily_multiplier
        Scaling factors mapping the daily quantile to the weekly / peak-to-trough
        thresholds. Defaults assume daily returns are approximately
        independent (``sqrt(5) ~= 2.24`` for weekly; the wider ``5.0`` for
        PTT covers a 60-day high-water-mark drawdown). Override if the
        strategy exhibits strong serial correlation.
    floor_daily_pct, ceiling_daily_pct
        Sanity clamps: refuse to recommend a daily breaker tighter than
        ``floor_daily_pct`` (avoid hairtrigger HALTs from low-variance
        strategies) or looser than ``ceiling_daily_pct`` (don't let a
        backtest with an anomalously fat tail loosen the breaker beyond
        the operator's risk appetite).

    Returns
    -------
    DrawdownCalibration
        Quantized recommended breakers + provenance fields so the operator
        can audit the derivation before updating ``limits.toml``.
    """

    if not daily_returns:
        raise ValueError("daily_returns must be non-empty")

    finite = [float(r) for r in daily_returns if math.isfinite(float(r))]
    if not finite:
        raise ValueError("daily_returns has no finite values")

    # Empirical quantiles (lower tail) — these are the raw evidence.
    p01 = _quantile(finite, 0.01)
    p05 = _quantile(finite, 0.05)
    p10 = _quantile(finite, 0.10)

    target_q = _quantile(finite, max(0.001, min(0.5, target_daily_tail_pct)))

    # Skew adjustment: negative skew (fat left tail) => tighten by a fraction
    # of stddev. Cap the nudge at 1 stddev so a wildly non-normal sample
    # doesn't dominate the empirical quantile.
    sd = _stddev(finite)
    skew = _skewness(finite)
    skew_adj = 0.0
    if skew < 0.0:
        skew_adj = max(-sd, 0.5 * skew * sd)  # negative; further loss

    daily_raw = target_q + skew_adj  # both terms negative on a loss-skewed tail

    # Clamp to operator-supplied band.
    daily_clamped = max(min(daily_raw, floor_daily_pct), ceiling_daily_pct)

    weekly = daily_clamped * weekly_to_daily_multiplier
    ptt = daily_clamped * ptt_to_daily_multiplier

    # Quantize to 1 bp grid; ensure monotonic ordering (daily > weekly > ptt
    # in the loss direction, i.e. ptt is the most negative).
    dd_daily = _quantize_loss(daily_clamped)
    dd_weekly = _quantize_loss(weekly)
    dd_ptt = _quantize_loss(ptt)

    rationale = (
        f"Calibrated from n={len(finite)} daily returns: "
        f"p01={p01:+.4f}, p05={p05:+.4f}, p10={p10:+.4f}, "
        f"mean={_mean(finite):+.4f}, std={sd:.4f}, skew={skew:+.2f}. "
        f"Target tail {target_daily_tail_pct:.1%} -> q={target_q:+.4f} "
        f"(+ skew adj {skew_adj:+.4f}) -> clamped to [{floor_daily_pct:+.4f},"
        f" {ceiling_daily_pct:+.4f}] = {dd_daily}. Weekly = daily x "
        f"{weekly_to_daily_multiplier:.2f}, PTT = daily x "
        f"{ptt_to_daily_multiplier:.2f}."
    )

    return DrawdownCalibration(
        dd_daily_pct=dd_daily,
        dd_weekly_pct=dd_weekly,
        dd_ptt_pct=dd_ptt,
        n_obs=len(finite),
        daily_mean=_mean(finite),
        daily_std=sd,
        daily_skew=skew,
        p01_daily=p01,
        p05_daily=p05,
        p10_daily=p10,
        rationale=rationale,
    )


__all__ = ["DrawdownCalibration", "calibrate_drawdown_thresholds"]
