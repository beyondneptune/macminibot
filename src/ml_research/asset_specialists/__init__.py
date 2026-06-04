"""Per-asset specialist strategies — winners of the trailing-stop bake-off.

Source of truth: ``data/bakeoff_variants/FINAL_ANALYSIS.md`` (Section 6).
Seven of the eight Alpaca-tradable universe assets passed the harmonised
within-variant gate (DSR_harm > 0.95, Holm-p < 0.05, bootstrap CI_lo > 0,
Sharpe > 0). AMZN failed promotion on the bootstrap CI gate (CI_lo = -0.49)
and is excluded from the live deployment.

Per-module API:

  WINNER: dict | None
      Keys: ``model``, ``scope``, ``trail_sigma``, ``exit_kind``,
      plus the harmonised statistics from FINAL_ANALYSIS. ``None``
      means the bake-off rejected every candidate for this asset.

  compute_signal(bars_df) -> float
      Latest causal entry signal in ``{-1, 0, +1}``, computed using only
      information dated ``<= T-1`` (no peek-ahead). The deployment runner
      converts this to FLAT / IN_LONG / IN_SHORT transitions and tracks
      the trailing-stop running peak/trough per asset.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

# 2026-06-04: swapped to 1%/day target deployment — 8 leveraged ETFs sized
# at $50k each on $100k seed (4× intraday-margin stack). See
# scripts/107_one_pct_target_sim.py for the sim that informed this choice.
# Highest-Sharpe config among variants hitting +1.00%/day mean:
# buy_and_hold + trail=2.0σ + PT=3.0σ.
SUPPORTED_ASSETS = ("SOXL", "NVDL", "TQQQ", "UPRO", "TNA", "FAS", "LABU", "TSLL")
DEPLOYABLE_ASSETS = SUPPORTED_ASSETS

# Prior v2 universe — retained so the strategy_journal + benchmark_logger
# can still recognise older positions managed by the intraday-exit-monitor
# while they trail-stop out.
LEGACY_V2_ASSETS = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "GLD")


def winner_for(asset: str) -> dict[str, Any] | None:
    """Return the winning config dict for ``asset`` or ``None`` if no winner.

    Lenient: returns ``None`` for unknown assets (rather than raising).
    The adaptive universe rotation can pick any liquid ETF; the caller is
    responsible for supplying a DAILY_OVERRIDES entry for new tickers.
    """
    try:
        mod = import_module(f"ml_research.asset_specialists.{asset.lower()}")
    except ModuleNotFoundError:
        return None
    return getattr(mod, "WINNER", None)


def compute_signal_for(asset: str, bars) -> float:
    """Dispatch ``compute_signal`` to the asset's specialist module.

    Lenient: falls back to ``buy_and_hold`` signal for unknown tickers
    (so the adaptive universe rotation can pick names without a dedicated
    specialist module).
    """
    try:
        mod = import_module(f"ml_research.asset_specialists.{asset.lower()}")
    except ModuleNotFoundError:
        # Generic buy_and_hold fallback for adaptive-picked tickers
        from ml_research.asset_specialists._signals import signal_buy_and_hold
        return float(signal_buy_and_hold(bars))
    fn: Callable[[Any], float] = getattr(mod, "compute_signal")
    return float(fn(bars))


__all__ = [
    "DEPLOYABLE_ASSETS",
    "SUPPORTED_ASSETS",
    "compute_signal_for",
    "winner_for",
]
