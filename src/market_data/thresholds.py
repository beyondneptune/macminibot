"""Dollar-bar thresholds for the starter universe.

Bootstrap values picked to produce ~50-100 bars per active session per contract.
These will be replaced by ADV-calibrated thresholds (one of the epics in
market-data/todo.md) once we have a few weeks of live data to measure actual
notional flow.

For MIDPOINT streams (no native size), threshold is in *ticks* not dollars.
"""

from __future__ import annotations

# Per-asset-class defaults. Tuple is (threshold, is_midpoint).
# For TRADES contracts (NASDAQ equities, futures): threshold is notional $.
# For MIDPOINT (FX, non-NASDAQ stocks/ETFs): threshold is tick count.

ASSET_CLASS_DEFAULT: dict[str, tuple[float, bool]] = {
    "equity": (5_000_000.0, False),   # $5M per bar for NASDAQ mega-caps
    "etf": (10_000_000.0, False),     # $10M per bar (SPY/QQQ are very liquid)
    "future": (50_000_000.0, False),  # $50M notional for ES/NQ etc.
    "fx": (100, True),                # 100 midpoint updates per bar
}

# Per-symbol overrides for known calibration outliers.
SYMBOL_OVERRIDE: dict[str, tuple[float, bool]] = {
    # NASDAQ ETFs with TRADES permission
    "QQQ": (30_000_000.0, False),  # very liquid
    # Smaller commodity ETFs — lower notional
    "UNG": (1_000_000.0, False),
    "USO": (3_000_000.0, False),
    "DBC": (1_000_000.0, False),
    "GDX": (3_000_000.0, False),
    "SLV": (5_000_000.0, False),
    "GLD": (20_000_000.0, False),
    # NYSE equities (no TRADES — fall back to MIDPOINT tick bars)
    "V": (100, True),
    "JPM": (100, True),
    "BRK B": (100, True),
    # ARCA/AMEX ETFs — fall back to MIDPOINT for live (TRADES worked for daily
    # historical, but live tick subscription may behave differently)
    "SPY": (100, True),
    "IWM": (100, True),
    "XLE": (100, True),
    # Front-month futures with lower notional per contract
    "ZN": (10_000_000.0, False),
    "SI": (5_000_000.0, False),
    "HG": (5_000_000.0, False),
    "RTY": (30_000_000.0, False),
    "YM": (20_000_000.0, False),
}


def threshold_for(name: str, asset_class: str) -> tuple[float, bool]:
    """Return (threshold, is_midpoint) for a contract."""
    if name in SYMBOL_OVERRIDE:
        return SYMBOL_OVERRIDE[name]
    return ASSET_CLASS_DEFAULT[asset_class]


__all__ = ["ASSET_CLASS_DEFAULT", "SYMBOL_OVERRIDE", "threshold_for"]
