"""Bake-off winner for AAPL: buy_and_hold + 2.5σ trailing stop + 3.0σ PT.

OPTIMIZATION UPDATE (2026-05-27): trail tightened 3.0σ → 2.5σ via
scripts/101_trail_optimization.py constrained sweep. The 2.5σ cell
strictly Pareto-dominates the 3.0σ baseline:
  OOS_RECENT net: $+1,940 → $+2,851 (Δ +$911)
  COVID max DD:   ×0.96 (4% improvement)
  RATESHOCK DD:   ×0.76 (24% improvement)
PT held at 3.0σ.

Earlier (2026-05-20): widened trail 2.0σ → 3.0σ + added 3.0σ PT per
scripts/77_comprehensive_search.py train-best.

Original (FINAL_ANALYSIS.md): buy_and_hold + 2.0σ trail.
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals import signal_buy_and_hold

WINNER = {
    "model": "buy_and_hold",
    "scope": "asset_only",
    "trail_sigma": 2.5,
    "profit_target_sigmas": 3.0,
    "exit_kind": "trailing_stop+pt",
    "oos_sharpe_net": 1.457,  # original bake-off
    "dsr_harm": 0.994,
    "p_holm": 0.0017,
    "ci_lower": 3.24,
    "source": "constrained_sweep_2026_05_27",
}


def compute_signal(bars: pd.DataFrame) -> float:
    """Latest causal entry signal in ``{-1, 0, +1}`` (float for contract uniformity)."""
    return float(signal_buy_and_hold(bars))
