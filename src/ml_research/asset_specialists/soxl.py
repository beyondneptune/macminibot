"""SOXL — 1%/day target deployment 2026-06-04.

Config from scripts/107_one_pct_target_sim.py — Sharpe-leading config among
the variants hitting +1.00%/day mean target on the 8-asset leveraged sleeve:
  signal     = buy_and_hold
  trail σ    = 2.0
  PT σ       = 3.0
  portfolio mean +1.23%/day  Sharpe +0.56  max DD −425% of seed

NOTE: this config is "high-leverage long-the-trend" — the daily mean comes
from the leveraged-ETF beta, not from per-name signal skill. Drawdowns are
multi-hundred-percent of $100k seed. Deploy only on paper or with risk
tolerance for total-account-equity wipeout multiple times along the path.
"""

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
