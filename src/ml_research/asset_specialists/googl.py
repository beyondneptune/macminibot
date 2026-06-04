"""Bake-off winner for GOOGL: buy_and_hold + 2.0σ trailing stop + 3.0σ PT.

OPTIMIZATION UPDATE (2026-05-27): added 3.0σ profit target via
scripts/101_trail_optimization.py constrained sweep. Trail unchanged at
2.0σ. The new cell strictly Pareto-dominates the baseline:
  OOS_RECENT net: $+2,902 → $+3,052 (Δ +$150)
  COVID max DD:   ×1.00 (unchanged)
  RATESHOCK DD:   ×0.86 (14% improvement)

Original (FINAL_ANALYSIS.md): buy_and_hold + 2.0σ trail, no PT.
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
    "oos_sharpe_net": 1.522,
    "dsr_harm": 0.995,
    "p_holm": 0.0012,
    "ci_lower": 1.97,
    "source": "constrained_sweep_2026_05_27",
}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in ``{-1, 0, +1}`` (float for contract uniformity)."""
    return float(signal_buy_and_hold(bars))
