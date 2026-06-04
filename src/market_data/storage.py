"""ArcticDB storage layer (LMDB backend) for macminibot bar data.

Library naming follows the convention ratified in market-data/CONTEXT.md:
    <asset_class>.<bar_type>    (library)
    <exchange>.<symbol>         (symbol within the library)

E.g. library ``equity.daily_bars``, symbol ``NASDAQ.AAPL``.

``ARCTIC_PATH`` resolves to ``<project_root>/data/arctic`` regardless of
whether this module is loaded as ``market_data.storage`` (from ``src/``) or
as the legacy ``_storage`` shim under ``scripts/`` — the path math walks
parents to find the ``data/`` sibling.
"""

from __future__ import annotations

from pathlib import Path

from arcticdb import Arctic


def _resolve_arctic_path() -> Path:
    """Resolve ``<project_root>/data/arctic`` from this file's location.

    The package now lives at ``src/market_data/storage.py``; the project
    root is two parents up (``src/market_data`` -> ``src`` -> root).
    """

    return Path(__file__).resolve().parents[2] / "data" / "arctic"


ARCTIC_PATH = _resolve_arctic_path()


def open_store() -> Arctic:
    ARCTIC_PATH.mkdir(parents=True, exist_ok=True)
    return Arctic(f"lmdb://{ARCTIC_PATH}")


def library_name(asset_class: str, bar_type: str) -> str:
    # bar_type one of: daily_bars, dollar_bars, hourly_bars, ...
    return f"{asset_class}.{bar_type}"


def symbol_name(exchange: str, symbol: str) -> str:
    # exchange '' is allowed (FX has no real exchange, use 'IDEALPRO' for FX)
    ex = exchange or "UNK"
    return f"{ex}.{symbol}"


def ensure_library(store: Arctic, name: str):
    if not store.has_library(name):
        store.create_library(name)
    return store[name]


__all__ = [
    "ARCTIC_PATH",
    "ensure_library",
    "library_name",
    "open_store",
    "symbol_name",
]
