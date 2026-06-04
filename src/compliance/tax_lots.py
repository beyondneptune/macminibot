"""compliance.tax_lots — FIFO tax-lot accounting.

Implements the FIFO default ratified in CONTEXT.md. Each BUY opens a
lot; each SELL closes lot(s) in arrival order. Realized P&L is split
into short-term (held <= 365 days) and long-term (> 365 days) per US
IRS Schedule D conventions — the common case for individual operators.

Money is ``Decimal`` internally (KD1). Floats from parquet/journal are
converted via ``Decimal(str(...))`` to avoid binary-float drift.

The book is **deterministic** given an ordered fill sequence: re-running
``apply_fill`` over the same journal slice yields bit-identical lots.
That property is what lets the monthly report regenerate from scratch.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque, Iterable

from shared_core.types import NS_PER_SECOND

__all__ = [
    "FifoTaxLotBook",
    "RealizedPnL",
    "TaxLot",
    "apply_fill",
    "realized_pnl_ytd",
]


_SECONDS_PER_DAY: int = 86_400
# IRS "long-term" threshold is "more than one year" — using 366 days
# means: held strictly more than 365 days. Holdings of exactly one year
# count as short-term, matching the IRS rule (the long-term clock starts
# the day *after* acquisition).
_LONG_TERM_NS: int = 366 * _SECONDS_PER_DAY * NS_PER_SECOND


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TaxLot:
    """One open lot — qty shares acquired at price on ts_ns."""

    symbol: str
    qty: int
    price: Decimal
    acquired_ts_ns: int


@dataclass(slots=True)
class RealizedPnL:
    """Realized P&L bucket, split short-term vs long-term."""

    short_term: Decimal = field(default_factory=lambda: Decimal("0"))
    long_term: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def total(self) -> Decimal:
        return self.short_term + self.long_term

    def __add__(self, other: "RealizedPnL") -> "RealizedPnL":
        return RealizedPnL(
            short_term=self.short_term + other.short_term,
            long_term=self.long_term + other.long_term,
        )


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SaleRecord:
    """Internal: a closed-out portion of a lot (one tax-event leg)."""

    symbol: str
    qty: int
    cost_basis: Decimal
    proceeds: Decimal
    acquired_ts_ns: int
    sold_ts_ns: int

    @property
    def pnl(self) -> Decimal:
        return self.proceeds - self.cost_basis

    @property
    def is_long_term(self) -> bool:
        return (self.sold_ts_ns - self.acquired_ts_ns) >= _LONG_TERM_NS


class FifoTaxLotBook:
    """FIFO lot tracker — one queue per symbol.

    SELL on an empty book is permitted (short selling) and creates a
    *negative* lot; subsequent BUY closes the short. The same long/
    short-term split applies symmetrically.
    """

    def __init__(self) -> None:
        # Open long lots per symbol (FIFO queue).
        self._longs: dict[str, Deque[TaxLot]] = defaultdict(deque)
        # Open short lots per symbol (FIFO queue, qty stored as positive).
        self._shorts: dict[str, Deque[TaxLot]] = defaultdict(deque)
        self._sales: list[_SaleRecord] = []

    # -- mutation --------------------------------------------------------

    def apply_fill(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: Decimal | float | str,
        ts_ns: int,
    ) -> None:
        """Apply a single fill to the book.

        ``side`` is ``"BUY"`` or ``"SELL"``. ``qty`` is positive integer.
        ``price`` accepts Decimal/float/str — converted to Decimal via
        ``str(...)`` to dodge binary-float drift.
        """

        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY|SELL (got {side!r})")
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            raise ValueError(f"qty must be positive int (got {qty!r})")
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(f"ts_ns must be int (got {type(ts_ns).__name__})")

        price_d = (
            price
            if isinstance(price, Decimal)
            else Decimal(str(price))
        )
        sym = symbol.upper()

        if side == "BUY":
            self._apply_buy(sym, qty, price_d, ts_ns)
        else:
            self._apply_sell(sym, qty, price_d, ts_ns)

    def _apply_buy(self, sym: str, qty: int, price: Decimal, ts_ns: int) -> None:
        # First, cover any open short positions in this symbol (FIFO).
        remaining = qty
        shorts = self._shorts[sym]
        while remaining > 0 and shorts:
            lot = shorts[0]
            take = min(lot.qty, remaining)
            # Short cover: proceeds = lot.price * take (we sold at lot.price),
            # cost_basis = price * take (we now buy back at price).
            self._sales.append(
                _SaleRecord(
                    symbol=sym,
                    qty=take,
                    cost_basis=price * take,
                    proceeds=lot.price * take,
                    acquired_ts_ns=lot.acquired_ts_ns,
                    sold_ts_ns=ts_ns,
                )
            )
            lot.qty -= take
            remaining -= take
            if lot.qty == 0:
                shorts.popleft()
        if remaining > 0:
            self._longs[sym].append(
                TaxLot(symbol=sym, qty=remaining, price=price, acquired_ts_ns=ts_ns)
            )

    def _apply_sell(self, sym: str, qty: int, price: Decimal, ts_ns: int) -> None:
        # First, close any open long positions in this symbol (FIFO).
        remaining = qty
        longs = self._longs[sym]
        while remaining > 0 and longs:
            lot = longs[0]
            take = min(lot.qty, remaining)
            self._sales.append(
                _SaleRecord(
                    symbol=sym,
                    qty=take,
                    cost_basis=lot.price * take,
                    proceeds=price * take,
                    acquired_ts_ns=lot.acquired_ts_ns,
                    sold_ts_ns=ts_ns,
                )
            )
            lot.qty -= take
            remaining -= take
            if lot.qty == 0:
                longs.popleft()
        if remaining > 0:
            # Open a short position at this price.
            self._shorts[sym].append(
                TaxLot(symbol=sym, qty=remaining, price=price, acquired_ts_ns=ts_ns)
            )

    # -- queries ---------------------------------------------------------

    def realized_pnl_ytd(
        self,
        symbol: str | None = None,
        *,
        year_start_ns: int | None = None,
    ) -> RealizedPnL:
        """Sum realized P&L for sales whose ``sold_ts_ns`` >= ``year_start_ns``.

        If ``year_start_ns`` is ``None``, all sales in the book are
        included (the typical "since the start of the journal" view).
        If ``symbol`` is given, restrict to that ticker.
        """

        sym_filter = symbol.upper() if symbol else None
        result = RealizedPnL()
        for sale in self._sales:
            if sym_filter is not None and sale.symbol != sym_filter:
                continue
            if year_start_ns is not None and sale.sold_ts_ns < year_start_ns:
                continue
            if sale.is_long_term:
                result.long_term += sale.pnl
            else:
                result.short_term += sale.pnl
        return result

    def open_lots(self, symbol: str | None = None) -> list[TaxLot]:
        """Return a snapshot of all open lots (longs positive, shorts negative-qty)."""

        out: list[TaxLot] = []
        items: Iterable[str] = (
            (symbol.upper(),) if symbol else sorted(set(self._longs) | set(self._shorts))
        )
        for sym in items:
            for lot in self._longs.get(sym, ()):
                out.append(TaxLot(sym, lot.qty, lot.price, lot.acquired_ts_ns))
            for lot in self._shorts.get(sym, ()):
                out.append(TaxLot(sym, -lot.qty, lot.price, lot.acquired_ts_ns))
        return out

    def sales(self) -> list[_SaleRecord]:
        """Return the immutable list of realized sale legs (in arrival order)."""

        return list(self._sales)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (for the CONTEXT.md-named API surface).
#
# These operate on a *singleton* book stored at module scope. They exist
# so callers can ``from compliance.tax_lots import apply_fill`` per the
# task spec; production callers that need isolation should instantiate
# :class:`FifoTaxLotBook` directly (replay script does this).
# ---------------------------------------------------------------------------


_default_book = FifoTaxLotBook()


def apply_fill(
    symbol: str,
    side: str,
    qty: int,
    price: Decimal | float | str,
    ts_ns: int,
) -> None:
    """Apply a fill to the module-level default book."""

    _default_book.apply_fill(symbol, side, qty, price, ts_ns)


def realized_pnl_ytd(
    symbol: str | None = None, *, year_start_ns: int | None = None
) -> RealizedPnL:
    """Read realized P&L from the module-level default book."""

    return _default_book.realized_pnl_ytd(symbol, year_start_ns=year_start_ns)


def _reset_default_book_for_tests() -> None:
    """Test-only hook: clear the module-level book between tests."""

    global _default_book
    _default_book = FifoTaxLotBook()
