"""Bake-off winner for NVDA: buy_and_hold + 2.0σ trailing stop.

Source: ``data/bakeoff_variants/FINAL_ANALYSIS.md`` (harmonised trailing-stop
variant winner; Sharpe +1.799, DSR_harm 0.999, Holm-p 0.0001, CI_lo +1.15).
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals import signal_buy_and_hold

WINNER = {
    "model": "buy_and_hold",
    "scope": "asset_only",
    "trail_sigma": 2.0,
    "exit_kind": "trailing_stop",
    "oos_sharpe_net": 1.799,
    "dsr_harm": 0.999,
    "p_holm": 0.0001,
    "ci_lower": 1.15,
}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in ``{-1, 0, +1}`` (float for contract uniformity)."""
    return float(signal_buy_and_hold(bars))
