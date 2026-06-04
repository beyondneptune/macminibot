"""MSFU — leveraged-ETF rotation candidate (2026-06-04).

Selected dynamically by scripts/110_universe_rotation.py. Per-asset config
overrides come from data/asset_overrides.json (written by
scripts/112_asset_config_rotation.py). Fallback baseline:
buy_and_hold + trail=2.0σ + PT=3.0σ.
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
    "source": "rotation_candidate_baseline",
}


def compute_signal(bars: pd.DataFrame) -> float:
    return float(signal_buy_and_hold(bars))
