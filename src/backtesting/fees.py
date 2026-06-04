"""IBKR Pro Tiered fee model — ratified Q1 2026 (see CONTEXT.md §Fee model).

Three pure functions, one per asset class on the v0 bootstrap universe:

* :func:`equity_tiered`  — US equities / ETFs (AAPL, GLD, SPY, TLT, ...)
* :func:`future_cme`     — CME futures (ES E-mini, ...)
* :func:`fx_idealpro`    — IDEALPRO spot FX (EUR.USD, ...)

The numbers come from
https://www.interactivebrokers.com/en/pricing/commissions-home.php (US Tiered,
retail <300k shares/mo). Pass-through fees vary ±20% by venue and order-type;
we use the documented conservative blend.

All functions return USD as ``float`` for now. ``shared_core`` will later
ratify ``Decimal`` for money; the journal already pins these columns as
``float64`` (see CONTEXT.md cycle-4 note on bps dtypes — same reasoning
applies for the modeled commission floor).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ALPACA_FEE_MODEL_VERSION",
    "FEE_MODEL_VERSION",
    "EquityFeeBreakdown",
    "FutureFeeBreakdown",
    "FxFeeBreakdown",
    "equity_alpaca",
    "equity_tiered",
    "for_broker",
    "future_cme",
    "fx_idealpro",
]

# Bumped whenever any constant below moves — bust the per-fold cache layer.
FEE_MODEL_VERSION = "ibkr_pro_tiered_2026q1.v1"
ALPACA_FEE_MODEL_VERSION = "alpaca_commission_free_2026q1.v1"

# SEC §31 fee (sell-side only) and FINRA TAF (sell-side only) — annual
# schedule; rates change mid-year per SEC fee-rate advisory. Add new years
# here as they're published. Lookup is by calendar year of the trade.
# Sources:
#   SEC §31: https://www.sec.gov/divisions/marketreg/mrfreqreq.htm
#   FINRA TAF: https://www.finra.org/rules-guidance/key-topics/trading-activity-fee
_SEC_31_RATE_BY_YEAR: dict[int, float] = {
    # rate-per-dollar-of-notional (sell-side); e.g. 27.80e-6 = $27.80 / $1MM
    2024: 27.80e-6,
    2025: 8.00e-6,
    2026: 8.00e-6,  # placeholder until SEC publishes 2026 fee-rate advisory
}
_FINRA_TAF_PER_SHARE_BY_YEAR: dict[int, float] = {
    2024: 0.000166,
    2025: 0.000166,
    2026: 0.000166,
}
_FINRA_TAF_PER_TRADE_CAP = 8.30  # capped per-trade regardless of share count

# ---- equity constants ------------------------------------------------------
_EQUITY_COMMISSION_PER_SHARE = 0.0035  # IBKR tiered commission, retail tier
_EQUITY_COMMISSION_MIN = 0.35  # $0.35 per order floor
_EQUITY_PASSTHROUGH_PER_SHARE = 0.003  # blended NYSE/NSDQ + SEC §31 + TAF + FINRA + NSCC
_EQUITY_TRADE_VALUE_CAP_FRAC = 0.01  # IBKR caps total commission at 1% of trade value

# ---- ES future constants ---------------------------------------------------
_ES_IBKR_PER_SIDE = 0.85
_ES_CME_EXCHANGE_PER_SIDE = 1.38
_ES_REG_PER_SIDE = 0.02  # NFA reg + CFTC §27

# ---- FX (IDEALPRO) constants -----------------------------------------------
_FX_MIN_PER_SIDE = 2.00
_FX_BPS = 0.20  # 0.20 bps of notional, USD-cross


@dataclass(frozen=True, slots=True)
class EquityFeeBreakdown:
    """Itemised IBKR-side commission + venue/reg pass-through for one fill."""

    commission: float
    passthrough: float

    @property
    def total(self) -> float:
        return self.commission + self.passthrough


@dataclass(frozen=True, slots=True)
class FutureFeeBreakdown:
    ibkr: float
    exchange: float
    reg: float

    @property
    def total(self) -> float:
        return self.ibkr + self.exchange + self.reg


@dataclass(frozen=True, slots=True)
class FxFeeBreakdown:
    commission: float

    @property
    def total(self) -> float:
        return self.commission


def equity_tiered(shares: float, price: float) -> EquityFeeBreakdown:
    """All-in IBKR Pro Tiered fee for a US equity / ETF fill.

    Formula (CONTEXT.md §Fee model row 1):
        IBKR commission = max($0.35, $0.0035 × shares)
        pass-through    = ~$0.003/share (NYSE/NSDQ + SEC §31 + TAF + FINRA + NSCC)
        TOTAL           = clamped to ≤ 1% of trade value (IBKR's cap)

    ``shares`` and ``price`` are accepted as ``float`` for the v0 sim. They are
    coerced to ``abs()`` because the journal stores ``qty`` unsigned (sign comes
    from ``side``). Zero shares returns a zero-fee breakdown (no order, no fee).

    Args:
        shares: number of shares filled. Must be >= 0.
        price:  fill price in USD. Must be >= 0.

    Returns:
        :class:`EquityFeeBreakdown` — keep the split so the journal can write
        ``commission`` / ``exchange_fee`` / ``reg_fee`` / ``clearing_fee`` columns
        separately. We bucket all venue+reg+clearing pass-through into
        ``passthrough`` here; the journal write step is allowed to attribute
        proportionally if it cares.
    """
    if shares < 0:
        raise ValueError(f"shares must be >= 0, got {shares}")
    if price < 0:
        raise ValueError(f"price must be >= 0, got {price}")
    if shares == 0:
        return EquityFeeBreakdown(commission=0.0, passthrough=0.0)

    commission = max(_EQUITY_COMMISSION_MIN, _EQUITY_COMMISSION_PER_SHARE * shares)
    passthrough = _EQUITY_PASSTHROUGH_PER_SHARE * shares
    total = commission + passthrough

    trade_value = shares * price
    cap = _EQUITY_TRADE_VALUE_CAP_FRAC * trade_value
    if trade_value > 0 and total > cap:
        # Scale both components proportionally so the breakdown still sums to
        # the cap. Keeps the commission/passthrough ratio meaningful for the
        # parity report.
        scale = cap / total
        commission *= scale
        passthrough *= scale

    return EquityFeeBreakdown(commission=commission, passthrough=passthrough)


def future_cme(contracts: float) -> FutureFeeBreakdown:
    """Flat per-side CME futures fee — ES E-mini (CONTEXT.md §Fee model row 2).

        $0.85 IBKR + $1.38 CME exchange + $0.02 reg = $2.25 / contract / side

    The same numbers apply to NQ / YM / RTY at the retail tier; if we add
    other CME products later, the per-product split moves here.
    """
    if contracts < 0:
        raise ValueError(f"contracts must be >= 0, got {contracts}")
    if contracts == 0:
        return FutureFeeBreakdown(ibkr=0.0, exchange=0.0, reg=0.0)

    return FutureFeeBreakdown(
        ibkr=_ES_IBKR_PER_SIDE * contracts,
        exchange=_ES_CME_EXCHANGE_PER_SIDE * contracts,
        reg=_ES_REG_PER_SIDE * contracts,
    )


def _sec_31_year(year: int) -> float:
    """SEC §31 rate for the given calendar year; raises on unknown year so a
    silently-stale constant can't survive into a fresh year's run."""
    try:
        return _SEC_31_RATE_BY_YEAR[year]
    except KeyError as exc:
        raise KeyError(
            f"no SEC §31 rate pinned for year {year}; add it to "
            f"_SEC_31_RATE_BY_YEAR after consulting the SEC fee-rate advisory"
        ) from exc


def _finra_taf_year(year: int) -> float:
    try:
        return _FINRA_TAF_PER_SHARE_BY_YEAR[year]
    except KeyError as exc:
        raise KeyError(
            f"no FINRA TAF rate pinned for year {year}; add it to "
            f"_FINRA_TAF_PER_SHARE_BY_YEAR"
        ) from exc


def equity_alpaca(
    shares: float,
    price: float,
    side: str,
    year: int,
) -> EquityFeeBreakdown:
    """Alpaca $0-commission US equity fee model (CONTEXT.md cycle-5 Alpaca branch).

    Alpaca absorbs the IBKR-equivalent commission AND venue routing fees; the
    only mandatory pass-throughs are regulatory and sell-side only:

        commission   = 0
        passthrough  = (SEC §31 + FINRA TAF) if side == "SELL" else 0
                       FINRA TAF is capped at $8.30/trade

    Buys are entirely free. The fee model thus produces an *asymmetric* fee
    drag: half the round-trips pay nothing, half pay only the sell-side regs.

    Args:
        shares: number of shares filled (>= 0).
        price:  fill price USD (>= 0).
        side:   ``"BUY"`` or ``"SELL"`` — case-insensitive.
        year:   trade year for SEC §31 + FINRA TAF rate lookup.

    Returns:
        :class:`EquityFeeBreakdown` with ``commission == 0`` and
        ``passthrough == 0`` for buys; sell-side reg total for sells. The
        journal writer treats ``passthrough`` as ``reg_fee`` in this branch
        (no venue/exchange/clearing components).
    """
    if shares < 0:
        raise ValueError(f"shares must be >= 0, got {shares}")
    if price < 0:
        raise ValueError(f"price must be >= 0, got {price}")
    if shares == 0:
        return EquityFeeBreakdown(commission=0.0, passthrough=0.0)

    side_norm = side.upper()
    if side_norm not in {"BUY", "SELL"}:
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    if side_norm == "BUY":
        return EquityFeeBreakdown(commission=0.0, passthrough=0.0)

    notional = shares * price
    sec_fee = _sec_31_year(year) * notional
    taf_uncapped = _finra_taf_year(year) * shares
    taf_fee = min(taf_uncapped, _FINRA_TAF_PER_TRADE_CAP)
    return EquityFeeBreakdown(commission=0.0, passthrough=sec_fee + taf_fee)


def for_broker(broker: str):
    """Select the equity fee callable for the configured broker.

    Returns a callable matching either :func:`equity_tiered` (IBKR) or
    :func:`equity_alpaca` (Alpaca). Promotion runs are expected to invoke
    *both* and record the delta as ``broker_subsidy_bps`` in the report
    (CONTEXT.md cycle-5 improvement #12). The IBKR-conservative number
    governs the DSR ≥ 0.95 promotion gate.
    """
    b = broker.lower()
    if b in {"ibkr", "ibkr_pro", "interactive_brokers"}:
        return equity_tiered
    if b == "alpaca":
        return equity_alpaca
    raise ValueError(
        f"unknown broker {broker!r}; expected one of "
        f"{{'ibkr_pro', 'alpaca'}}"
    )


def fx_idealpro(notional_usd: float) -> FxFeeBreakdown:
    """IDEALPRO spot-FX commission per side (CONTEXT.md §Fee model row 3).

        max($2.00, 0.20 bps × notional_usd)

    ``notional_usd`` is the USD-equivalent notional of the trade (e.g. for
    EUR.USD, ``size_eur × eurusd_rate``). FX is symmetric per side, so call
    once per side.
    """
    if notional_usd < 0:
        raise ValueError(f"notional_usd must be >= 0, got {notional_usd}")
    if notional_usd == 0:
        return FxFeeBreakdown(commission=0.0)

    bps_cost = (_FX_BPS / 1e4) * notional_usd
    return FxFeeBreakdown(commission=max(_FX_MIN_PER_SIDE, bps_cost))
