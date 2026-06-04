"""Dollar-bar threshold calibration from daily bars.

López de Prado *Advances in Financial ML* Ch. 2 heuristic: threshold = trailing
20-day median daily dollar volume / 50, giving ~50 dollar bars per session.

Reads from the `<asset_class>.daily_bars` ArcticDB library populated by
scripts/03_backfill_daily_bars.py. Falls back to per-asset-class defaults from
_thresholds.py if no daily-bar history is available.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _storage import library_name, open_store, symbol_name  # noqa: E402
from _thresholds import threshold_for as _fallback_threshold  # noqa: E402

TARGET_BARS_PER_SESSION = 50
LOOKBACK_DAYS = 20


def calibrate_threshold(
    name: str,
    asset_class: str,
    exchange_for_symbol: str,
    is_midpoint: bool,
) -> tuple[float, str]:
    """Return (threshold, calibration_source).

    For midpoint-mode (FX, non-NASDAQ stocks/ETFs without TRADES perms), the
    threshold is a tick count — fall back to the static _thresholds.py default
    since "dollar volume" doesn't map to midpoint updates.

    For TRADES-mode, compute from daily bars: median(volume × wap) / 50.
    """
    if is_midpoint:
        thr, _ = _fallback_threshold(name, asset_class)
        return thr, "midpoint-default"

    sym = symbol_name(exchange_for_symbol, name)
    lib_name = library_name(asset_class, "daily_bars")
    store = open_store()
    if not store.has_library(lib_name):
        thr, _ = _fallback_threshold(name, asset_class)
        return thr, "no-daily-library-fallback"

    lib = store[lib_name]
    if sym not in lib.list_symbols():
        thr, _ = _fallback_threshold(name, asset_class)
        return thr, "no-symbol-fallback"

    df = lib.read(sym).data.tail(LOOKBACK_DAYS)
    if df.empty:
        thr, _ = _fallback_threshold(name, asset_class)
        return thr, "empty-history-fallback"

    # Daily dollar volume: prefer volume × wap; fall back to volume × close if wap is 0/NaN
    price = df["wap"]
    if (price <= 0).any() or price.isna().any():
        price = df["close"]
    daily_dv = (df["volume"] * price).dropna()
    if daily_dv.empty:
        thr, _ = _fallback_threshold(name, asset_class)
        return thr, "no-dv-data-fallback"

    median_dv = float(daily_dv.median())
    threshold = median_dv / TARGET_BARS_PER_SESSION
    return threshold, f"calibrated-from-{len(daily_dv)}d-median-dv"
