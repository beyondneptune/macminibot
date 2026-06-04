"""treasury.fees — per-trade regulatory fee accounting.

Alpaca paper trading is commission-free (Alpaca commissions == 0), but
US equity trades still incur two *regulatory* fees on the **sell** side
that the broker passes through to the customer. We track them here so:

  (a) live-trading accounting is already correct when we switch venues,
  (b) backtests can subtract the same fee model the live tape will see,
  (c) the fee ledger isn't a surprise when monthly statements arrive.

Rates as of 2026-05-18 (source: SIFMA / FINRA notices).
Refresh manually when a new SEC §31 fee order is published (typically
once per fiscal year, mid-February); see the constant docstrings below.

This module is **pure**: no I/O, no clock reads, all inputs explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

__all__ = [
    "FeeBreakdown",
    "SEC_FEE_RATE",
    "TAF_PER_SHARE",
    "TAF_MAX_PER_TRADE",
    "compute_trade_cost",
]


# ---------------------------------------------------------------------------
# Regulatory rate table — refresh manually when the SEC publishes a new
# Section 31 fee rate order, and when FINRA updates the TAF schedule.
# ---------------------------------------------------------------------------

# SEC Section 31 transaction fee on **sells of covered securities**.
# 2026 rate: $0.0000278 per dollar of principal (gross proceeds). The
# fee changes annually via SEC order; verify against the current SIFMA
# rate sheet before bumping a year boundary.
SEC_FEE_RATE: Decimal = Decimal("0.0000278")

# FINRA Trading Activity Fee (TAF) on **sells of covered securities**.
# 2026 rate: $0.000166 per share, with a $8.30 per-trade cap.
# Source: FINRA Schedule of Section 1 (Trading Activity Fee).
TAF_PER_SHARE: Decimal = Decimal("0.000166")
TAF_MAX_PER_TRADE: Decimal = Decimal("8.30")

# Alpaca paper-trade commission. Zero today; promoted to a constant so
# any future live-venue mapping can override it without touching call
# sites.
ALPACA_PAPER_COMMISSION: Decimal = Decimal("0")

# Cent-rounding quantum for the customer-facing fee. The broker rounds
# each component up to the nearest cent on settlement; we mirror that
# so our internal ledger reconciles with statements.
_CENT = Decimal("0.01")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    """Decomposed per-trade cost.

    ``total = commission + sec_fee + taf`` — addition is exact under
    Decimal so callers can rely on the equality without ULP tolerance.
    """

    commission: Decimal
    sec_fee: Decimal
    taf: Decimal
    total: Decimal


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _validate_decimal(value: Decimal, *, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise TypeError(
            f"{name} must be Decimal (got {type(value).__name__}={value!r})"
        )
    return value


def compute_trade_cost(
    *,
    side: str,
    qty: Decimal,
    price: Decimal,
    commission_override: Decimal | None = None,
) -> FeeBreakdown:
    """Compute the all-in regulatory cost of one fill.

    Parameters
    ----------
    side:
        Case-insensitive ``"BUY"`` or ``"SELL"`` (or the ``Side`` enum
        ``str`` form, which is identical). Buys carry zero SEC/TAF; only
        sells of covered securities do.
    qty:
        Number of shares filled. Must be ``Decimal``, positive.
    price:
        Execution price per share. Must be ``Decimal``, non-negative.
    commission_override:
        Optional explicit per-trade commission. If ``None`` we use
        :data:`ALPACA_PAPER_COMMISSION` (0 today).

    All components are rounded up to the nearest cent (ROUND_HALF_UP for
    the SEC fee, then capped for TAF) before summation — matches the way
    broker statements display the line items.
    """

    _validate_decimal(qty, name="qty")
    _validate_decimal(price, name="price")
    if not isinstance(side, str):
        raise TypeError(f"side must be str (got {type(side).__name__})")
    if qty <= 0:
        raise ValueError(f"qty must be positive: {qty}")
    if price < 0:
        raise ValueError(f"price must be non-negative: {price}")

    commission = (
        commission_override
        if commission_override is not None
        else ALPACA_PAPER_COMMISSION
    )
    _validate_decimal(commission, name="commission_override")
    if commission < 0:
        raise ValueError(f"commission must be non-negative: {commission}")

    side_norm = side.upper()
    if side_norm not in {"BUY", "SELL"}:
        raise ValueError(f"side must be 'BUY' or 'SELL' (got {side!r})")

    if side_norm == "SELL":
        principal = qty * price
        # SEC §31 fee, rounded up to the cent (broker practice).
        sec_fee = (principal * SEC_FEE_RATE).quantize(
            _CENT, rounding=ROUND_HALF_UP
        )
        taf_raw = (qty * TAF_PER_SHARE).quantize(
            _CENT, rounding=ROUND_HALF_UP
        )
        taf = min(taf_raw, TAF_MAX_PER_TRADE)
    else:
        sec_fee = Decimal("0.00")
        taf = Decimal("0.00")

    commission_q = commission.quantize(_CENT, rounding=ROUND_HALF_UP)
    sec_fee_q = sec_fee.quantize(_CENT, rounding=ROUND_HALF_UP)
    taf_q = taf.quantize(_CENT, rounding=ROUND_HALF_UP)
    total = commission_q + sec_fee_q + taf_q

    return FeeBreakdown(
        commission=commission_q,
        sec_fee=sec_fee_q,
        taf=taf_q,
        total=total,
    )
