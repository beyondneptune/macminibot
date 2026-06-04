"""market_data — market-data substrate (ArcticDB storage, universe, bars).

Promoted from the loose ``scripts/_storage.py``, ``scripts/_universe.py``,
``scripts/_thresholds.py``, ``scripts/_calibration.py``, ``scripts/_bars.py``
into a proper importable package. The original scripts now re-export from
this package for backward compatibility — keep the new modules as the
canonical home for new code.

Public surface (re-exported):

    from market_data import (
        # storage
        ARCTIC_PATH, open_store, library_name, symbol_name, ensure_library,
        # universe
        UNIVERSE, UniverseEntry, by_asset_class,
        # thresholds
        ASSET_CLASS_DEFAULT, SYMBOL_OVERRIDE, threshold_for,
        # calibration
        calibrate_threshold, TARGET_BARS_PER_SESSION, LOOKBACK_DAYS,
        # bars
        DollarBarAccumulator,
    )
"""

from __future__ import annotations

from market_data.bars import DollarBarAccumulator
from market_data.calibration import (
    LOOKBACK_DAYS,
    TARGET_BARS_PER_SESSION,
    calibrate_threshold,
)
from market_data.storage import (
    ARCTIC_PATH,
    ensure_library,
    library_name,
    open_store,
    symbol_name,
)
from market_data.thresholds import (
    ASSET_CLASS_DEFAULT,
    SYMBOL_OVERRIDE,
    threshold_for,
)
from market_data.universe import UNIVERSE, UniverseEntry, by_asset_class

__all__ = [
    "ARCTIC_PATH",
    "ASSET_CLASS_DEFAULT",
    "DollarBarAccumulator",
    "LOOKBACK_DAYS",
    "SYMBOL_OVERRIDE",
    "TARGET_BARS_PER_SESSION",
    "UNIVERSE",
    "UniverseEntry",
    "by_asset_class",
    "calibrate_threshold",
    "ensure_library",
    "library_name",
    "open_store",
    "symbol_name",
    "threshold_for",
]
