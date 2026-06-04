"""FAS — 1%/day target deployment 2026-06-04 (3× US financials leveraged ETF)."""
from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals import signal_buy_and_hold

WINNER = {
    "model": "buy_and_hold",
    "scope": "asset_only",
    "trail_sigma": 2.0,
    "profit_target_sigmas": 3.0,
    "exit_kind": "trailing_stop+pt",
    "source": "one_pct_target_sim_2026_06_04",
}


def compute_signal(bars: pd.DataFrame) -> float:
    return float(signal_buy_and_hold(bars))
