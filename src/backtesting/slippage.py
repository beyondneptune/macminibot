"""Almgren-Chriss square-root market-impact model + per-asset spread floor.

Reference: Almgren, Thum, Hauptmann, Li (2005) — *Direct Estimation of Equity
Market Impact*, Risk. Square-root form:

    slip_bps = sigma_daily_bps * k * sqrt(Q / ADV)

For retail size (Q/ADV < 1%) the spread half-cost dominates; the
``SPREAD_HALF_FLOOR_BPS`` table below pins the floor by asset class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "SLIPPAGE_MODEL_VERSION",
    "SPREAD_HALF_FLOOR_BPS",
    "AssetClass",
    "SlippageBreakdown",
    "almgren_chriss",
    "slippage_bps_with_floor",
]

SLIPPAGE_MODEL_VERSION = "almgren_sqrt_k0.1.v1"

AssetClass = Literal["equity", "future", "fx"]

# Per-asset half-spread floors. Equity 1 bp is the conservative blend across
# liquid US large-caps + ETFs; ES "0.25 tick" at ~5000 index → ~0.5 bp; EURUSD
# "0.1 pip" at 1.10 → ~0.09 bp. Rounded up to whole-bps for the safety floor.
SPREAD_HALF_FLOOR_BPS: dict[str, float] = {
    "equity": 1.0,
    "future": 0.5,
    "fx": 0.1,
}


@dataclass(frozen=True, slots=True)
class SlippageBreakdown:
    """Modeled vs floor split — useful for the report's bucketed parity hist."""

    impact_bps: float
    floor_bps: float

    @property
    def total_bps(self) -> float:
        return max(self.impact_bps, self.floor_bps)


def almgren_chriss(
    volume_traded: float,
    adv: float,
    daily_vol_bps: float,
    k: float = 0.1,
) -> float:
    """Pure Almgren-Chriss square-root impact, in bps.

        impact_bps = daily_vol_bps * k * sqrt(volume_traded / adv)

    Args:
        volume_traded: Q — shares (equity), contracts (future), or notional units
            (FX). Must be >= 0.
        adv: 21-day rolling average daily volume in the same units as
            ``volume_traded``. Must be > 0 (else the model is undefined; the
            caller should fall back to the spread-half floor via
            :func:`slippage_bps_with_floor`).
        daily_vol_bps: sigma — annualized realized volatility expressed as
            *daily* bps (i.e. sigma_annual_bps / sqrt(252) already applied).
            Conservative default for liquid US large caps is ~100 bps.
        k: scaling constant; 0.1 per Almgren et al. (2005). Override only with
            a citation. Must be > 0.

    Returns:
        Impact in bps as ``float``. **Floor not applied here** — call
        :func:`slippage_bps_with_floor` to combine.
    """
    if volume_traded < 0:
        raise ValueError(f"volume_traded must be >= 0, got {volume_traded}")
    if adv <= 0:
        raise ValueError(f"adv must be > 0, got {adv}")
    if daily_vol_bps < 0:
        raise ValueError(f"daily_vol_bps must be >= 0, got {daily_vol_bps}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if volume_traded == 0:
        return 0.0

    return daily_vol_bps * k * math.sqrt(volume_traded / adv)


def slippage_bps_with_floor(
    volume_traded: float,
    adv: float,
    daily_vol_bps: float,
    asset_class: AssetClass,
    k: float = 0.1,
) -> SlippageBreakdown:
    """Almgren-Chriss impact, floored at the asset-class spread half-cost.

    Returns both components so the journal can record what the model said
    (``impact_bps``) vs what the sim actually charged (``total_bps``). When
    the floor binds (retail size, <1% ADV), ``impact_bps < floor_bps`` and
    ``total_bps == floor_bps``.
    """
    if asset_class not in SPREAD_HALF_FLOOR_BPS:
        raise ValueError(
            f"unknown asset_class {asset_class!r}; expected one of "
            f"{sorted(SPREAD_HALF_FLOOR_BPS)}"
        )
    floor = SPREAD_HALF_FLOOR_BPS[asset_class]
    impact = almgren_chriss(volume_traded, adv, daily_vol_bps, k=k)
    return SlippageBreakdown(impact_bps=impact, floor_bps=floor)
