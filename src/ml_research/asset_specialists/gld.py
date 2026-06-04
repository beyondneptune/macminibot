"""Bake-off winner for GLD: buy_and_hold + 3.0σ trailing stop.

Source: ``data/bakeoff_variants/FINAL_ANALYSIS.md`` (harmonised trailing-stop
variant winner; Sharpe +2.122, DSR_harm 1.000, Holm-p 0.0000, CI_lo +7.36).
GLD is the only cell that also survives Romano-Wolf at alpha=0.05.
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals import signal_buy_and_hold

# DEPLOYMENT OVERRIDE (daily v2, 2026-05-20): tightened trail 3.0σ → 2.0σ
# and added 3.0σ profit target per comprehensive search train-best (script 77).
# TEST net improved $12,509 → $12,297 (Sharpe 1.60 → 1.06 — modestly lower
# Sharpe but tighter risk profile via PT lock-in).
WINNER = {
    "model": "buy_and_hold",
    "scope": "asset_only",
    "trail_sigma": 2.0,
    "profit_target_sigmas": 3.0,
    "exit_kind": "trailing_stop+pt",
    "oos_sharpe_net": 2.122,  # original bake-off
    "dsr_harm": 1.000,
    "p_holm": 0.0000,
    "ci_lower": 7.36,
    "source": "daily_v2_comprehensive_search_train_best",
}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in ``{-1, 0, +1}`` (float for contract uniformity)."""
    return float(signal_buy_and_hold(bars))
