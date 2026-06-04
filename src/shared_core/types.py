"""shared_core.types — time and decimal helpers.

Per ratified KD1 (`Decimal`, never `float` for money) and KD2 (int64-ns
UTC timestamps everywhere). Helpers here live at the I/O boundary;
in-memory domain code only ever sees `int` (ns) and `Decimal`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

__all__ = [
    "NS_PER_SECOND",
    "money",
    "ns_to_datetime_utc",
    "now_ns",
    "parse_ts",
    "quantity",
    "validate_decimal",
]

NS_PER_SECOND: Final[int] = 1_000_000_000


def now_ns() -> int:
    """Wall-clock UTC timestamp in int64 nanoseconds since Unix epoch.

    Uses ``datetime.now(timezone.utc)`` rather than ``time.time_ns()`` so
    the value is unambiguously UTC (no TZ ambiguity from system locale).
    """
    dt = datetime.now(timezone.utc)
    seconds = int(dt.timestamp())
    return seconds * NS_PER_SECOND + dt.microsecond * 1_000


def ns_to_datetime_utc(ns: int) -> datetime:
    """Convert int64-ns UTC to a tz-aware ``datetime`` in UTC.

    Raises ``TypeError`` if ``ns`` is not an ``int`` (rejects ``float``,
    ``numpy.int64`` is accepted via ``int(ns)`` cast at the caller).
    """
    if not isinstance(ns, int) or isinstance(ns, bool):
        raise TypeError(f"ns must be int (got {type(ns).__name__}={ns!r})")
    seconds, remainder_ns = divmod(ns, NS_PER_SECOND)
    micros = remainder_ns // 1_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=micros)


def parse_ts(s: str) -> int:
    """Parse an ISO-8601 timestamp string into int64-ns UTC.

    Accepts the trailing 'Z' shorthand for UTC. Naive datetimes are
    rejected (per KD2: every timestamp must carry a tz).
    """
    if not isinstance(s, str):
        raise TypeError(f"parse_ts expects str (got {type(s).__name__})")
    normalised = s.replace("Z", "+00:00") if s.endswith("Z") else s
    dt = datetime.fromisoformat(normalised)
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime rejected (KD2): {s!r}")
    dt_utc = dt.astimezone(timezone.utc)
    seconds = int(dt_utc.timestamp())
    return seconds * NS_PER_SECOND + dt_utc.microsecond * 1_000


def validate_decimal(value: Decimal, *, field: str = "value") -> Decimal:
    """Runtime guard ensuring ``value`` is a ``Decimal`` instance.

    Construction from ``float`` should already have been rejected by the
    in-contract guard; this helper provides a callable form for ad-hoc
    use (config parsers, JSON readers).
    """
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise TypeError(
            f"{field} must be a Decimal (got {type(value).__name__}={value!r})"
        )
    return value


def money(value: str | int) -> Decimal:
    """Construct a ``Decimal`` for a money amount.

    Only accepts ``str`` or ``int`` (KD1). ``float`` raises ``TypeError``
    rather than silently turning ``0.1`` into a 17-digit approximation.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(
            f"money(...) accepts only str|int (got {type(value).__name__}={value!r})"
        )
    return Decimal(value)


def quantity(value: str | int) -> Decimal:
    """Construct a ``Decimal`` for a quantity (shares, contracts, units).

    Same float-rejection rule as :func:`money` — both are KD1 amounts.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(
            f"quantity(...) accepts only str|int (got {type(value).__name__}={value!r})"
        )
    return Decimal(value)
