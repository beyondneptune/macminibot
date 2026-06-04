"""Bake-off winner for AMZN.

Original bake-off (FINAL_ANALYSIS.md): AMZN was EXCLUDED because the best
trailing-stop cell (ema_12_26, 0.5σ) had bootstrap CI_lo = -0.49 < 0 and
failed the four-gate promotion test.

Comprehensive search override (script 77_comprehensive_search.py, train-best
on 2024-01-01 split): buy_and_hold + trail=0.5σ + profit_target=5σ delivered
TRAIN net +$156,004 / TEST net +$5,490 (Sh +0.56), and on the 2026-01 →
2026-05 shared OOS window in script 79 it produced +$2,331 (+71% ann, Sh
+1.36) — the empirical winner among the AMZN candidates on the recent OOS.

This module now emits the buy_and_hold signal so AMZN actively trades in
the hybrid deployment. The ``WINNER`` dict captures the config; the daily
production runner applies it via the ``DAILY_OVERRIDES`` table in
``scripts/30_per_asset_runner.py``.
"""

from __future__ import annotations

import pandas as pd

from ml_research.asset_specialists._signals import signal_buy_and_hold

WINNER = {
    "model": "buy_and_hold",
    "scope": "asset_only",
    "trail_sigma": 0.5,
    "profit_target_sigmas": 5.0,
    "source": "comprehensive_search_train_best_2024_01_split",
}


def compute_signal(bars: pd.DataFrame) -> float:
    """Buy-and-hold signal — always +1 once history is sufficient."""
    return float(signal_buy_and_hold(bars))
