"""Edge-case rule book — codified rules from CONTEXT.md.

Each rule is a small pure function so the scheduler can wire them together
and observability can log a clean ``edge_case_rules_fired[]`` per session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas_market_calendars as mcal

from strategy_engine.covariance import ILL_CONDITIONED_THRESHOLD, cond_number

# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

_NYSE = mcal.get_calendar("NYSE")


def is_market_holiday(d: date) -> bool:
    """True if NYSE has no session on ``d`` (full holiday, not half-day)."""
    if isinstance(d, datetime):
        d = d.date()
    sched = _NYSE.schedule(start_date=d.isoformat(), end_date=d.isoformat())
    return len(sched) == 0


def is_half_day(d: date) -> bool:
    """True if NYSE early-closes on ``d``."""
    if isinstance(d, datetime):
        d = d.date()
    sched = _NYSE.schedule(start_date=d.isoformat(), end_date=d.isoformat())
    if len(sched) == 0:
        return False
    close = sched.iloc[0]["market_close"]
    # Normal close is 21:00 UTC (16:00 ET). Anything earlier is a half day.
    return close.hour < 20 or (close.hour == 20 and close.minute < 30)


# ---------------------------------------------------------------------------
# Halt / LULD
# ---------------------------------------------------------------------------


class HaltAction(str, Enum):
    FREEZE = "freeze"  # one-session halt → carry w_prev
    FLATTEN_TWAP = "flatten_twap"  # multi-session halt → emit flatten target


@dataclass(frozen=True)
class HaltDecision:
    action: HaltAction
    hold_reason: str
    target_weight: float  # what to put in TargetPosition.target_weight


def handle_halt(
    symbol: str,
    w_prev_i: float,
    *,
    sessions_halted: int = 1,
) -> HaltDecision:
    """One-session halt freezes w_prev; >1 session emits a flatten target."""
    if sessions_halted > 1:
        return HaltDecision(
            action=HaltAction.FLATTEN_TWAP,
            hold_reason=f"halt.flatten_after_{sessions_halted}_sessions",
            target_weight=0.0,
        )
    return HaltDecision(
        action=HaltAction.FREEZE,
        hold_reason="halt.luld",
        target_weight=float(w_prev_i),
    )


# ---------------------------------------------------------------------------
# Signal validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalValidation:
    accepted: bool
    reason: Optional[str]


SIGNAL_BLOWUP_THRESHOLD: float = 0.05  # |r_hat_daily| > 5% → suspect unit error


def validate_signal_scale(r_hat: float | Decimal) -> SignalValidation:
    """Per CONTEXT.md: |r_hat_daily| > 5% almost certainly a unit error."""
    val = float(r_hat)
    if not np.isfinite(val):
        return SignalValidation(accepted=False, reason="signal.nan_or_inf")
    if abs(val) > SIGNAL_BLOWUP_THRESHOLD:
        return SignalValidation(accepted=False, reason="signal.blowup_gt_5pct")
    return SignalValidation(accepted=True, reason=None)


# ---------------------------------------------------------------------------
# Sigma conditioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SigmaConditioning:
    cond_number: float
    ill_conditioned: bool


def check_sigma_conditioning(Sigma: np.ndarray) -> SigmaConditioning:
    cn = cond_number(np.asarray(Sigma, dtype=np.float64))
    return SigmaConditioning(
        cond_number=cn,
        ill_conditioned=(not np.isfinite(cn)) or cn > ILL_CONDITIONED_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Stale bars (tiered)
# ---------------------------------------------------------------------------

STALE_BAR_AGE_NS: int = 48 * 3600 * 1_000_000_000  # 48h in ns


def is_stale_bars(symbol: str, last_bar_ts_ns: int, *, now_ns: int) -> bool:
    """True if the most recent dollar-bar is older than 48h."""
    return (now_ns - int(last_bar_ts_ns)) > STALE_BAR_AGE_NS


class StaleBarsTier(str, Enum):
    OK = "ok"  # ≤10% stale
    HALVED_TARGET = "halved_target"  # 10-30% stale → drop + halve σ_target
    HALT = "halt"  # >30% stale → halt rebalance


def classify_universe_staleness(stale_count: int, universe_size: int) -> StaleBarsTier:
    """CONTEXT.md three-tier staleness response."""
    if universe_size == 0:
        return StaleBarsTier.HALT
    frac = stale_count / universe_size
    if frac <= 0.10:
        return StaleBarsTier.OK
    if frac <= 0.30:
        return StaleBarsTier.HALVED_TARGET
    return StaleBarsTier.HALT


# ---------------------------------------------------------------------------
# NAV jump
# ---------------------------------------------------------------------------

NAV_JUMP_FRACTION: float = 0.10  # 10% intraday jump halts rebalance


def is_nav_jump(nav_now: Decimal, nav_t1_close: Decimal) -> bool:
    """True when |ΔNAV|/NAV ≥ 10% (boundary inclusive: 10% itself trips the rule)."""
    if nav_t1_close <= 0:
        return True  # degenerate; halt to be safe
    delta = abs(Decimal(nav_now) - Decimal(nav_t1_close)) / Decimal(nav_t1_close)
    return delta >= Decimal(str(NAV_JUMP_FRACTION))


# ---------------------------------------------------------------------------
# Zero-signal day
# ---------------------------------------------------------------------------

ZERO_SIGNAL_THRESHOLD: float = 0.0005  # 5 bps
ZERO_SIGNAL_DECAY_RATE: float = 0.2  # 20% / day toward cash


def is_zero_signal_day(r_hat: np.ndarray) -> bool:
    arr = np.asarray(r_hat, dtype=np.float64)
    if arr.size == 0:
        return True
    return float(np.max(np.abs(arr))) < ZERO_SIGNAL_THRESHOLD


def decay_to_cash(w_prev: np.ndarray, *, rate: float = ZERO_SIGNAL_DECAY_RATE) -> np.ndarray:
    """Per CONTEXT.md: decay 20%/day on quiet sessions, not full flatten."""
    return np.asarray(w_prev, dtype=np.float64) * (1.0 - rate)


# ---------------------------------------------------------------------------
# Turnover ceiling + no-trade band
# ---------------------------------------------------------------------------

NO_TRADE_BAND: float = 0.005  # 0.5% NAV
TURNOVER_CEILING_L1: float = 1.0  # 100% gross turnover


def apply_no_trade_band(
    w_new: np.ndarray,
    w_prev: np.ndarray,
    *,
    band: float = NO_TRADE_BAND,
) -> np.ndarray:
    """Freeze names whose |Δw| is below the no-trade band."""
    w_new = np.asarray(w_new, dtype=np.float64).copy()
    w_prev = np.asarray(w_prev, dtype=np.float64)
    dw = np.abs(w_new - w_prev)
    mask = dw < band
    w_new[mask] = w_prev[mask]
    return w_new


def apply_turnover_ceiling(
    w_new: np.ndarray,
    w_prev: np.ndarray,
    *,
    ceiling: float = TURNOVER_CEILING_L1,
) -> Tuple[np.ndarray, bool]:
    """Scale the trade down to the L1 ceiling. Returns (w_capped, was_capped)."""
    w_new = np.asarray(w_new, dtype=np.float64)
    w_prev = np.asarray(w_prev, dtype=np.float64)
    delta = w_new - w_prev
    l1 = float(np.sum(np.abs(delta)))
    if l1 <= ceiling or l1 <= 0:
        return w_new.copy(), False
    scale = ceiling / l1
    return (w_prev + delta * scale), True


__all__ = [
    "HaltAction",
    "HaltDecision",
    "NAV_JUMP_FRACTION",
    "NO_TRADE_BAND",
    "SIGNAL_BLOWUP_THRESHOLD",
    "STALE_BAR_AGE_NS",
    "SigmaConditioning",
    "SignalValidation",
    "StaleBarsTier",
    "TURNOVER_CEILING_L1",
    "ZERO_SIGNAL_DECAY_RATE",
    "ZERO_SIGNAL_THRESHOLD",
    "apply_no_trade_band",
    "apply_turnover_ceiling",
    "check_sigma_conditioning",
    "classify_universe_staleness",
    "decay_to_cash",
    "handle_halt",
    "is_half_day",
    "is_market_holiday",
    "is_nav_jump",
    "is_stale_bars",
    "is_zero_signal_day",
    "validate_signal_scale",
]
