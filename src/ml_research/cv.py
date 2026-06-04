"""ml_research.cv — canonical walk-forward splitter (binding on backtesting).

Per CONTEXT.md "Walk-forward harness — canonical splitter contract":
    WalkForwardHarness is the ONLY acceptable time-series splitter in this
    codebase. `backtesting` MUST import it and wrap with a P&L replay layer,
    never re-implement purge/embargo logic.

API contract (CONTEXT line 136):
    WalkForwardHarness(
        train_min_years=5,
        train_max_years=15,
        val_days=252,
        embargo_days=5,
        purge_days=1,
        test_days=21,
        refit_cadence="monthly",
    ).split(index) -> Iterator[FoldSpec]
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Try to import the canonical FoldSpec from shared-core; fall back to a local
# stub that mirrors the spec one-for-one (CONTEXT line 27 mandate: shared-core
# MUST mirror this dataclass verbatim; until shared-core lands, the stub here
# is the single source of truth).
try:  # pragma: no cover - import path dependent on shared-core landing
    from shared_core.contracts import FoldSpec as _SharedFoldSpec  # type: ignore

    FoldSpec = _SharedFoldSpec
except ImportError:  # noqa: BLE001 — explicit fallback
    @dataclass(frozen=True)
    class FoldSpec:  # type: ignore[no-redef]
        """Walk-forward fold descriptor (stub mirroring shared-core spec).

        train_idx / val_idx / test_idx are positional indices into the
        flat sorted-unique date axis the splitter was constructed with.
        fold_meta carries human-readable dates + bounds for logging.
        """

        train_idx: np.ndarray
        val_idx: np.ndarray
        test_idx: np.ndarray
        fold_id: int
        fold_meta: dict[str, Any] = field(default_factory=dict)


_TRADING_DAYS_PER_YEAR = 252


class WalkForwardHarness:
    """Expanding-window walk-forward CV with embargo + purge.

    Each fold:
        train_idx = [start_of_history, train_end]    (expanding, capped at max_years)
        val_idx   = last `val_days` of train_idx     (for hyperparam selection)
        purge     = `purge_days` after train_end     (dropped from train, not in test)
        embargo   = `embargo_days` after purge       (dropped, not in test)
        test_idx  = next `test_days` after embargo
        next fold advances by `test_days` (monthly cadence = 21 trading days)
    """

    def __init__(
        self,
        *,
        train_min_years: int = 5,
        train_max_years: int = 15,
        val_days: int = _TRADING_DAYS_PER_YEAR,
        embargo_days: int = 5,
        purge_days: int = 1,
        test_days: int = 21,
        refit_cadence: str = "monthly",
    ) -> None:
        if train_min_years <= 0 or train_max_years < train_min_years:
            raise ValueError("invalid train window bounds")
        if val_days <= 0 or test_days <= 0:
            raise ValueError("val_days / test_days must be positive")
        if embargo_days < 0 or purge_days < 0:
            raise ValueError("embargo_days / purge_days must be non-negative")
        if refit_cadence not in ("monthly", "annual", "weekly"):
            raise ValueError(f"unknown refit_cadence: {refit_cadence!r}")
        self.train_min_years = train_min_years
        self.train_max_years = train_max_years
        self.val_days = val_days
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.test_days = test_days
        self.refit_cadence = refit_cadence
        # Map cadence → advancement step in trading days. CONTEXT pins monthly
        # cadence at 21 trading days; we honor whatever caller passes as test_days
        # for monthly. Annual/weekly are convenience aliases for `backtesting`.
        self._step_days = {
            "monthly": test_days,
            "annual": _TRADING_DAYS_PER_YEAR,
            "weekly": 5,
        }[refit_cadence]

    def split(self, index: pd.DatetimeIndex | np.ndarray) -> Iterator[FoldSpec]:
        """Yield FoldSpec for each walk-forward iteration.

        `index` is the sorted unique date axis (one entry per trading day).
        Indices in the returned FoldSpec are positional offsets into this axis.
        """
        if isinstance(index, pd.DatetimeIndex):
            sorted_index = pd.DatetimeIndex(np.sort(index.unique()))
        else:
            sorted_index = pd.DatetimeIndex(np.sort(np.unique(np.asarray(index))))
        n = len(sorted_index)
        min_train = self.train_min_years * _TRADING_DAYS_PER_YEAR
        max_train = self.train_max_years * _TRADING_DAYS_PER_YEAR

        if n < min_train + self.purge_days + self.embargo_days + self.test_days:
            return  # not enough history for even one fold

        fold_id = 0
        train_end = min_train  # first train ends at min_train (exclusive upper)

        while True:
            test_start = train_end + self.purge_days + self.embargo_days
            test_end = test_start + self.test_days
            if test_end > n:
                break

            train_start = max(0, train_end - max_train)
            train_idx = np.arange(train_start, train_end, dtype=np.int64)
            # Inner validation: last val_days of train (kept inside train_idx for
            # the harness; the trainer slices them off for early stopping).
            val_start = max(train_start, train_end - self.val_days)
            val_idx = np.arange(val_start, train_end, dtype=np.int64)
            test_idx = np.arange(test_start, test_end, dtype=np.int64)

            meta = {
                "train_start_date": sorted_index[train_start].isoformat(),
                "train_end_date": sorted_index[train_end - 1].isoformat(),
                "val_start_date": sorted_index[val_start].isoformat(),
                "val_end_date": sorted_index[train_end - 1].isoformat(),
                "test_start_date": sorted_index[test_start].isoformat(),
                "test_end_date": sorted_index[test_end - 1].isoformat(),
                "purge_days": self.purge_days,
                "embargo_days": self.embargo_days,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
            }
            yield FoldSpec(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                fold_id=fold_id,
                fold_meta=meta,
            )
            fold_id += 1
            train_end += self._step_days


__all__ = ["FoldSpec", "WalkForwardHarness"]
