"""Starter universe for the macminibot data backfill.

40 contracts across the asset classes the ratified market-data architecture covers,
chosen for liquidity so the first backfill produces meaningful dollar bars quickly.
Will be replaced by the ADV-ranked top-5000 universe once the discovery pipeline
in market-data/todo.md lands. Until then this is the bootstrap set.

``ib_async`` is an optional dependency at import time — when it is not
available (e.g. on CI runners without the IBKR client), this module degrades
to a lightweight fallback so ``from market_data.universe import UNIVERSE``
still works. The fallback's ``contract`` field is a ``dict`` rather than a
real ``ib_async.Contract``; the live-stream/backfill scripts will refuse to
use a fallback contract — only the IBKR-aware scripts construct contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - import shape is environment-dependent
    from ib_async import Contract, Forex, Future, Stock  # type: ignore[import-not-found]

    _HAS_IB = True
except ImportError:  # pragma: no cover - import shape is environment-dependent
    Contract = Any  # type: ignore[assignment,misc]
    _HAS_IB = False

    def _fallback_contract(**kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def Stock(symbol: str, exchange: str, currency: str, primaryExchange: str) -> dict[str, Any]:  # type: ignore[no-redef]
        return _fallback_contract(
            secType="STK",
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            primaryExchange=primaryExchange,
        )

    def Future(symbol: str, exchange: str, includeExpired: bool = False) -> dict[str, Any]:  # type: ignore[no-redef]
        return _fallback_contract(
            secType="FUT", symbol=symbol, exchange=exchange, includeExpired=includeExpired
        )

    def Forex(pair: str) -> dict[str, Any]:  # type: ignore[no-redef]
        return _fallback_contract(secType="CASH", symbol=pair, exchange="IDEALPRO")


@dataclass(frozen=True)
class UniverseEntry:
    name: str  # human label, e.g. "AAPL" or "EURUSD"
    asset_class: str  # "equity" | "etf" | "future" | "fx"
    contract: Any  # ib_async.Contract when available, dict fallback otherwise
    what_to_show_trades: str  # "TRADES" for everything but FX (which uses "MIDPOINT")


def _stock(symbol: str, primary: str) -> Any:
    return Stock(symbol, exchange="SMART", currency="USD", primaryExchange=primary)


# 10 mega-cap US equities (NASDAQ + NYSE) — highest tick rates
_EQUITIES = [
    UniverseEntry("AAPL", "equity", _stock("AAPL", "NASDAQ"), "TRADES"),
    UniverseEntry("MSFT", "equity", _stock("MSFT", "NASDAQ"), "TRADES"),
    UniverseEntry("NVDA", "equity", _stock("NVDA", "NASDAQ"), "TRADES"),
    UniverseEntry("GOOGL", "equity", _stock("GOOGL", "NASDAQ"), "TRADES"),
    UniverseEntry("AMZN", "equity", _stock("AMZN", "NASDAQ"), "TRADES"),
    UniverseEntry("META", "equity", _stock("META", "NASDAQ"), "TRADES"),
    UniverseEntry("TSLA", "equity", _stock("TSLA", "NASDAQ"), "TRADES"),
    UniverseEntry("JPM", "equity", _stock("JPM", "NYSE"), "TRADES"),
    UniverseEntry("V", "equity", _stock("V", "NYSE"), "TRADES"),
    UniverseEntry("BRK B", "equity", _stock("BRK B", "NYSE"), "TRADES"),  # IBKR uses space, not dot
]

# 10 ETFs — broad-market + sector + commodity wrappers (gold, silver, oil, energy)
_ETFS = [
    UniverseEntry("SPY", "etf", _stock("SPY", "ARCA"), "TRADES"),
    UniverseEntry("QQQ", "etf", _stock("QQQ", "NASDAQ"), "TRADES"),
    UniverseEntry("IWM", "etf", _stock("IWM", "ARCA"), "TRADES"),
    UniverseEntry("GLD", "etf", _stock("GLD", "ARCA"), "TRADES"),
    UniverseEntry("SLV", "etf", _stock("SLV", "ARCA"), "TRADES"),
    UniverseEntry("USO", "etf", _stock("USO", "ARCA"), "TRADES"),
    UniverseEntry("UNG", "etf", _stock("UNG", "ARCA"), "TRADES"),
    UniverseEntry("DBC", "etf", _stock("DBC", "ARCA"), "TRADES"),
    UniverseEntry("GDX", "etf", _stock("GDX", "ARCA"), "TRADES"),
    UniverseEntry("XLE", "etf", _stock("XLE", "ARCA"), "TRADES"),
]

# 10 major futures — continuous front-month resolved via Future with no expiry
_FUTURES = [
    UniverseEntry("ES", "future", Future("ES", exchange="CME", includeExpired=False), "TRADES"),
    UniverseEntry("NQ", "future", Future("NQ", exchange="CME", includeExpired=False), "TRADES"),
    UniverseEntry("YM", "future", Future("YM", exchange="CBOT", includeExpired=False), "TRADES"),
    UniverseEntry("RTY", "future", Future("RTY", exchange="CME", includeExpired=False), "TRADES"),
    UniverseEntry("CL", "future", Future("CL", exchange="NYMEX", includeExpired=False), "TRADES"),
    UniverseEntry("GC", "future", Future("GC", exchange="COMEX", includeExpired=False), "TRADES"),
    UniverseEntry("SI", "future", Future("SI", exchange="COMEX", includeExpired=False), "TRADES"),
    UniverseEntry("HG", "future", Future("HG", exchange="COMEX", includeExpired=False), "TRADES"),
    UniverseEntry("ZN", "future", Future("ZN", exchange="CBOT", includeExpired=False), "TRADES"),
    UniverseEntry("6E", "future", Future("6E", exchange="CME", includeExpired=False), "TRADES"),
]

# 10 major FX pairs — IDEALPRO. FX uses MIDPOINT (TRADES returns empty for FX).
_FX = [
    UniverseEntry("EURUSD", "fx", Forex("EURUSD"), "MIDPOINT"),
    UniverseEntry("GBPUSD", "fx", Forex("GBPUSD"), "MIDPOINT"),
    UniverseEntry("USDJPY", "fx", Forex("USDJPY"), "MIDPOINT"),
    UniverseEntry("USDCHF", "fx", Forex("USDCHF"), "MIDPOINT"),
    UniverseEntry("AUDUSD", "fx", Forex("AUDUSD"), "MIDPOINT"),
    UniverseEntry("NZDUSD", "fx", Forex("NZDUSD"), "MIDPOINT"),
    UniverseEntry("USDCAD", "fx", Forex("USDCAD"), "MIDPOINT"),
    UniverseEntry("EURGBP", "fx", Forex("EURGBP"), "MIDPOINT"),
    UniverseEntry("EURJPY", "fx", Forex("EURJPY"), "MIDPOINT"),
    UniverseEntry("GBPJPY", "fx", Forex("GBPJPY"), "MIDPOINT"),
]

UNIVERSE: list[UniverseEntry] = _EQUITIES + _ETFS + _FUTURES + _FX


def by_asset_class() -> dict[str, list[UniverseEntry]]:
    out: dict[str, list[UniverseEntry]] = {}
    for u in UNIVERSE:
        out.setdefault(u.asset_class, []).append(u)
    return out


__all__ = ["UNIVERSE", "UniverseEntry", "by_asset_class"]
