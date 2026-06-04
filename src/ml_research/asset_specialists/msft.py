"""Bake-off winner for MSFT: rsi2_mr_10_90 + 2.0σ trailing stop.

Source: ``data/bakeoff_variants/FINAL_ANALYSIS.md`` (harmonised trailing-stop
variant winner; Sharpe +1.372, DSR_harm 0.991, Holm-p 0.0017, CI_lo +0.14).
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals import signal_rsi2_mr_10_90

# DEPLOYMENT OVERRIDE (daily v2, 2026-05-20): tightened trail 2.0σ → 1.0σ
# and added 5.0σ profit target per comprehensive search train-best (script 77).
# TEST Sharpe improved 0.99 → 1.28 on 2026-01 shared OOS window.
WINNER = {
    "model": "rsi2_mr_10_90",
    "scope": "asset_only",
    "trail_sigma": 1.0,
    "profit_target_sigmas": 5.0,
    "exit_kind": "trailing_stop+pt",
    "oos_sharpe_net": 1.372,  # original bake-off
    "dsr_harm": 0.991,
    "p_holm": 0.0017,
    "ci_lower": 0.14,
    "source": "daily_v2_comprehensive_search_train_best",
}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in ``{-1, 0, +1}`` (float for contract uniformity)."""
    return float(signal_rsi2_mr_10_90(bars))
