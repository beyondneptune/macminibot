"""Regime tagging for stratified backtest evaluation (Epic 17).

CONTEXT.md improvement #14 (cycle 5) ratified: a single all-fold DSR can pass
the 0.95 gate while the strategy is structurally broken in (e.g.) high-vol
regimes — the OOS folds happen to be vol-low-weighted. This module provides
the *ex-ante regime label* the report-renderer stratifies by.

The v1 contract (per the cycle-5 spec) bundles three labels:
    * VIX terciles      (this module — ``tag_regime``)
    * yield-curve slope sign       (TODO: bound to ``market-data`` PIT pull)
    * SPY 200-DMA above/below      (TODO: trivial; deferred until needed)

Cycle 9 lands the VIX-tercile path only (the other two are a one-line
extension of the same pattern once the upstream series are available). The
return labels are ``{"vix_low", "vix_mid", "vix_high"}`` matching the
spec; bucketing uses **rolling 252-day quintile cutoffs** with the bottom
quintile = ``vix_low``, the top quintile = ``vix_high``, and the middle
three quintiles = ``vix_mid``. Quintiles (not terciles) widen the mid
bucket so the tails are tighter — the report flag ("DSR collapses >0.3 in
any regime") triggers more usefully when the tails are well-separated from
the typical regime.

**Causality / no-lookahead guarantee**: the quintile cutoffs for date ``t``
are computed using the rolling 252-day window ending at ``t-1`` (i.e. the
value *at* ``t`` is not part of the percentile estimation that buckets it).
This matches the pattern used in ``ml_research.features._vix_tercile_features``
(``.shift(1)`` before ``.rolling_quantile``) and is the same look-ahead
guard ``market-data`` enforces on its bar-read API.

The ``vix_proxy_value`` argument is the cross-sectional dispersion of
21-day realized vol across {SPY, QQQ, IWM} as defined in
``ml_research.features._vix_proxy`` (the canonical series the rest of the
codebase uses as a daily-bar-substrate VIX surrogate; real CBOE VIX has a
license cost we're not paying for v0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

__all__ = [
    "REGIME_LABELS",
    "RegimeTagger",
    "tag_regime",
]


RegimeLabel = Literal["vix_low", "vix_mid", "vix_high"]
REGIME_LABELS: tuple[RegimeLabel, ...] = ("vix_low", "vix_mid", "vix_high")

_LOOKBACK_DAYS = 252
_LOW_QUINTILE = 0.20
_HIGH_QUINTILE = 0.80


@dataclass(frozen=True, slots=True)
class RegimeTagger:
    """Pre-fitted rolling-quintile regime tagger.

    Holds the date-indexed series of low (20th-pct) and high (80th-pct)
    cutoffs computed on the trailing 252-day window ending **strictly
    before** each date (``shift(1)`` semantics). Lookup is O(1) per date.

    Construction is via :meth:`from_series` which takes the full
    ``vix_proxy`` history once; the resulting tagger is then queried
    per-date by the driver during fold replay or in the aggregate-metrics
    breakdown.
    """

    low_cutoff: pd.Series  # date → 20th-pct over [t-252, t-1]
    high_cutoff: pd.Series  # date → 80th-pct over [t-252, t-1]
    vix_series: pd.Series  # date → vix_proxy value (kept for diagnostics)

    @classmethod
    def from_series(
        cls,
        vix_proxy: pd.Series,
        *,
        lookback_days: int = _LOOKBACK_DAYS,
    ) -> "RegimeTagger":
        """Fit cutoffs from a full ``vix_proxy`` daily series.

        Args:
            vix_proxy: ``pd.Series`` indexed by daily ``DatetimeIndex``.
                Values are the cross-sectional std of 21d realized vol
                across the {SPY, QQQ, IWM} basket (the canonical proxy
                in ``ml_research.features``).
            lookback_days: trailing-window length for the quintile fit.
                Default 252 (≈ one trading year). Matches the
                ``ml_research`` tercile-feature convention.

        Returns:
            Fitted :class:`RegimeTagger`. Dates within the first
            ``lookback_days // 2`` of history will have NaN cutoffs and
            those calls will receive ``"vix_mid"`` (the safe-default
            bucket — no signal-conditional gating until cutoffs warm up).
        """
        if not isinstance(vix_proxy.index, pd.DatetimeIndex):
            raise TypeError(
                f"vix_proxy must be DatetimeIndex; got {type(vix_proxy.index).__name__}"
            )
        s = vix_proxy.sort_index().astype(float)
        # Shift(1) → today's bucket uses only history ending YESTERDAY.
        shifted = s.shift(1)
        low_cut = shifted.rolling(
            window=lookback_days, min_periods=max(20, lookback_days // 2)
        ).quantile(_LOW_QUINTILE)
        high_cut = shifted.rolling(
            window=lookback_days, min_periods=max(20, lookback_days // 2)
        ).quantile(_HIGH_QUINTILE)
        return cls(low_cutoff=low_cut, high_cutoff=high_cut, vix_series=s)

    def tag(self, date: pd.Timestamp, vix_proxy_value: float) -> RegimeLabel:
        """Bucket ``vix_proxy_value`` on ``date`` against the fitted cutoffs.

        Args:
            date: the trading day whose regime we want.
            vix_proxy_value: the ``vix_proxy`` reading at ``date`` (the
                value the strategy actually saw at the open of ``date``).

        Returns:
            One of ``{"vix_low", "vix_mid", "vix_high"}``. ``"vix_mid"`` if
            the cutoffs are not yet warm (insufficient history) or the
            input value is NaN.
        """
        ts = pd.Timestamp(date)
        lo = self.low_cutoff.get(ts, np.nan)
        hi = self.high_cutoff.get(ts, np.nan)
        if (
            vix_proxy_value is None
            or not np.isfinite(vix_proxy_value)
            or not np.isfinite(lo)
            or not np.isfinite(hi)
        ):
            return "vix_mid"
        v = float(vix_proxy_value)
        if v <= float(lo):
            return "vix_low"
        if v >= float(hi):
            return "vix_high"
        return "vix_mid"

    def tag_many(
        self,
        dates: Iterable[pd.Timestamp],
        values: Iterable[float],
    ) -> list[RegimeLabel]:
        """Vectorish convenience: zip-tag a sequence of (date, value) pairs."""
        return [self.tag(d, v) for d, v in zip(dates, values)]


def tag_regime(
    date: pd.Timestamp,
    vix_proxy_value: float,
    *,
    tagger: RegimeTagger,
) -> RegimeLabel:
    """Module-level convenience matching the cycle-9 spec signature.

    ``tag_regime(date, vix_proxy_value) -> {"vix_low","vix_mid","vix_high"}``
    requires a fitted :class:`RegimeTagger` (the rolling-quintile cutoffs
    must be pre-computed over the full daily series to satisfy the no-
    lookahead guarantee; tagging a single date in isolation has no
    well-defined "trailing 252 days" anchor without the history).

    Construct the tagger once per backtest run from the full ``vix_proxy``
    series, then call this per-date inside the driver's per-fold loop.
    """
    return tagger.tag(date, vix_proxy_value)
