"""Bake-off winner for TSLA: macd_signal + 0.5σ trailing stop.

Source: ``data/bakeoff_variants/FINAL_ANALYSIS.md`` (harmonised trailing-stop
variant winner; Sharpe +2.604, DSR_harm 1.000, Holm-p 0.0000, CI_lo +1.15).
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals import signal_macd

# DEPLOYMENT OVERRIDE (daily v2, 2026-05-20): added 3.0σ profit target on
# top of the original 0.5σ trailing stop per comprehensive search (script 77).
# TEST net improved $10,209 → $15,407 on 2026-01 shared OOS window.
WINNER = {
    "model": "macd_signal",
    "scope": "asset_only",
    "trail_sigma": 0.5,
    "profit_target_sigmas": 3.0,
    "exit_kind": "trailing_stop+pt",
    "oos_sharpe_net": 2.604,  # original bake-off
    "dsr_harm": 1.000,
    "p_holm": 0.0000,
    "ci_lower": 1.15,
    "source": "daily_v2_comprehensive_search_train_best",
}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in ``{-1, 0, +1}`` (float for contract uniformity)."""
    return float(signal_macd(bars))
