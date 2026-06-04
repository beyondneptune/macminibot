"""shared_core — leaf-dependency tentacle.

Canonical domain types, time + decimal helpers, config loader, logging
primitives, and error types. This package imports from NOTHING inside
the macminibot project; every other tentacle imports from here.

Public surface (re-exported from .contracts):
    Symbol, Venue, Side, Bar, DollarBar, Tick, TickType,
    Signal, TargetPosition, TargetPositionBatch, OrderIntent,
    NavSnapshot, ChainMode, BreachEvent, CorporateAction,
    CorporateActionType, FoldSpec.
"""

from shared_core.contracts import (
    Bar,
    BreachEvent,
    ChainMode,
    CorporateAction,
    CorporateActionType,
    DollarBar,
    FoldSpec,
    NavSnapshot,
    OrderIntent,
    Side,
    Signal,
    Symbol,
    TargetPosition,
    TargetPositionBatch,
    Tick,
    TickType,
    Venue,
)
from shared_core.types import (
    money,
    ns_to_datetime_utc,
    now_ns,
    parse_ts,
    quantity,
    validate_decimal,
)

__all__ = [
    # contracts
    "Bar",
    "BreachEvent",
    "ChainMode",
    "CorporateAction",
    "CorporateActionType",
    "DollarBar",
    "FoldSpec",
    "NavSnapshot",
    "OrderIntent",
    "Side",
    "Signal",
    "Symbol",
    "TargetPosition",
    "TargetPositionBatch",
    "Tick",
    "TickType",
    "Venue",
    # type helpers
    "money",
    "ns_to_datetime_utc",
    "now_ns",
    "parse_ts",
    "quantity",
    "validate_decimal",
]
