"""strategy_engine — daily portfolio constructor.

Per the ratified CONTEXT.md (4 loop cycles):
- vol-targeted risk-parity with mean-variance tilt
- Ledoit-Wolf shrunk Σ (HRP fallback above n=500)
- cvxpy + ECOS for n ≤ 50, OSQP warm-started for 50 < n ≤ 500
- L1 + small L2 turnover penalty (hybrid; Improvement #1)
- ChainMode-aware batch envelope (Improvement #7)
- explicit hold-name targets to avoid risk-mgmt flatten conflict (Improvement #6)

Cross-tentacle types (``Signal``, ``NavSnapshot``, ``ChainMode``, ``Symbol``)
re-export from ``shared_core.contracts``. ``TargetPosition`` and
``TargetPositionBatch`` remain strategy-engine-local envelopes that wrap the
shared_core canonicals and add the reasoning-trace fields (cycle-7
integration; see ``contracts.py``).
"""

from __future__ import annotations

from strategy_engine.contracts import (
    ChainMode,
    NavSnapshot,
    Signal,
    Symbol,
    TargetPosition,
    TargetPositionBatch,
)
from strategy_engine.covariance import (
    cond_number,
    ledoit_wolf_shrunk,
)
from strategy_engine.long_only import long_only_top_n
from strategy_engine.optimizer import mean_variance_optimize
from strategy_engine.scheduler import run_rebalance
from strategy_engine.sizing import signal_winsorize, vol_target_scale

__all__ = [
    "ChainMode",
    "NavSnapshot",
    "Signal",
    "Symbol",
    "TargetPosition",
    "TargetPositionBatch",
    "cond_number",
    "ledoit_wolf_shrunk",
    "long_only_top_n",
    "mean_variance_optimize",
    "run_rebalance",
    "signal_winsorize",
    "vol_target_scale",
]
