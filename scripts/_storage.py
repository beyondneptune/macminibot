"""ArcticDB storage layer (LMDB backend) for macminibot bar data.

Library naming follows the convention ratified in market-data/CONTEXT.md:
    <asset_class>.<bar_type>    (library)
    <exchange>.<symbol>         (symbol within the library)

E.g. library 'equity.daily_bars', symbol 'NASDAQ.AAPL'.
"""

from __future__ import annotations

from pathlib import Path

from arcticdb import Arctic

ARCTIC_PATH = Path(__file__).resolve().parent.parent / "data" / "arctic"


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
