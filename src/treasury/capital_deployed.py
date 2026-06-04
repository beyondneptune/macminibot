"""treasury.capital_deployed — gross/net notional, leverage, cash utilisation.

Helpers consumed by ``observability`` (dashboard tiles) and
``risk-management`` (pre-trade buying-power checks). Pure functions;
all inputs explicit, no I/O.

Input shape
-----------
``positions`` is a mapping ``{ticker: PositionRow}`` where
``PositionRow`` is either:

  * an alpaca-py ``Position`` object (attribute access:
    ``qty``, ``market_value``, ``side``), or
  * a plain ``dict`` with the same keys.

A position's signed market value is what we use throughout: long
positions are positive, shorts negative. The Alpaca SDK reports
``market_value`` as a positive string for both sides plus a
``side='short'`` flag; we normalise the sign here.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

__all__ = [
    "cash_utilization",
    "leverage",
    "signed_market_value",
    "total_gross_notional",
    "total_net_notional",
]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_decimal(value: Any, *, field: str) -> Decimal:
    if value is None:
        raise ValueError(f"missing required field: {field}")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError(f"{field} cannot be bool")
    if isinstance(value, (int, str, float)):
        # Float tolerated here (Alpaca SDK sometimes hands us floats from
        # JSON); we round-trip through str so Decimal precision is
        # preserved relative to the SDK's text representation.
        return Decimal(str(value))
    raise TypeError(
        f"{field} must be Decimal|int|str (got {type(value).__name__})"
    )


def signed_market_value(position: Any) -> Decimal:
    """Return a position's signed market value (long positive, short negative)."""
    raw_mv = _get(position, "market_value")
    mv = _to_decimal(raw_mv, field="market_value")
    side = _get(position, "side", "long")
    side_str = str(getattr(side, "value", side)).lower()
    # If the venue reports a positive market_value for a short, flip the
    # sign. If it already reports a negative (some SDKs do), leave it.
    if side_str == "short" and mv > 0:
        return -mv
    return mv


def total_gross_notional(positions: Mapping[str, Any]) -> Decimal:
    """Sum of absolute market values across all positions.

    Gross notional represents *total dollar exposure*: a $10k long and
    a $10k short add to $20k gross. This is the quantity to compare
    against the firm's gross-exposure cap.
    """
    return sum(
        (abs(signed_market_value(p)) for p in positions.values()),
        start=Decimal("0"),
    )


def total_net_notional(positions: Mapping[str, Any]) -> Decimal:
    """Sum of *signed* market values across all positions.

    Net notional is the directional bias: a $10k long and a $10k short
    net to $0. Compare against the firm's net-exposure cap (a.k.a.
    "max one-sided tilt").
    """
    return sum(
        (signed_market_value(p) for p in positions.values()),
        start=Decimal("0"),
    )


def leverage(
    positions: Mapping[str, Any],
    *,
    equity: Decimal,
) -> Decimal:
    """Return ``gross_notional / equity`` as a Decimal ratio.

    Conventionally reported as a multiplier (1.5x means 150% gross
    exposure). Raises ``ValueError`` on non-positive equity — a blown-up
    account has no meaningful leverage ratio (the denominator is zero
    or negative).
    """
    _to_decimal(equity, field="equity")  # type-check
    if equity <= 0:
        raise ValueError(
            f"equity must be positive to compute leverage (got {equity})"
        )
    gross = total_gross_notional(positions)
    return gross / equity


def cash_utilization(
    *,
    equity: Decimal,
    cash: Decimal,
) -> Decimal:
    """Fraction of equity currently deployed (``1 - cash / equity``).

    Equals 0 when all equity is in cash, approaches 1 as cash is drawn
    down into positions, and can exceed 1 if the broker has extended
    margin (cash goes negative). Raises on non-positive equity.
    """
    _to_decimal(equity, field="equity")
    _to_decimal(cash, field="cash")
    if equity <= 0:
        raise ValueError(
            f"equity must be positive to compute cash_utilization (got {equity})"
        )
    return Decimal("1") - (cash / equity)
