"""equity_curve — persistent NetLiq time-series for drawdown analytics.

Append-only Parquet sink at ``data/equity_curve.parquet``. Designed to be
called every 5 minutes during the regular session by the launchd-driven
``scripts/42_equity_recorder.py``, but also safe to call from interactive
notebooks or the EOD summary.

Schema (one row per observation)
--------------------------------
- ``ts_ns``                  int64   nanoseconds UTC (canonical KD2)
- ``ts_unix``                float64 seconds UTC (convenience for plotting)
- ``net_liq``                float64 NetLiquidation
- ``available_funds``        float64 AvailableFunds
- ``gross_position_value``   float64 GrossPositionValue
- ``cushion``                float64 Cushion (margin headroom 0..1+)

All values are floats — Decimal precision is not needed for a drawdown
chart, and float64 round-trips cleanly through pyarrow. Source-of-truth
for accounting remains the IBKR account_summary; this is a derivative
log used for plots + drawdown breach checks.

Concurrency
-----------
The recorder script holds no lock. The atomic-write pattern
(read -> append -> write to ``.tmp`` -> ``os.replace``) means two
overlapping recorders would race-clobber rather than corrupt; this is
acceptable because launchd serialises a single recorder invocation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


__all__ = [
    "DEFAULT_PARQUET_PATH",
    "EQUITY_COLUMNS",
    "load_curve",
    "max_drawdown_pct",
    "recent",
    "record",
]


EQUITY_COLUMNS = (
    "ts_ns",
    "ts_unix",
    "net_liq",
    "available_funds",
    "gross_position_value",
    "cushion",
)


def _project_root() -> Path:
    """Repo root — the file lives at ``src/observability/equity_curve.py``."""

    return Path(__file__).resolve().parents[2]


def _resolve_path(override: str | os.PathLike[str] | None = None) -> Path:
    if override is not None:
        return Path(override)
    env = os.environ.get("MACMINIBOT_EQUITY_CURVE_PATH")
    if env:
        return Path(env)
    return _project_root() / "data" / "equity_curve.parquet"


DEFAULT_PARQUET_PATH: Path = _resolve_path()


def _to_float(value: Any) -> float:
    """Coerce account-summary scalars (Decimal/str/int/float) to float."""

    if value is None or value == "":
        return 0.0
    return float(value)


def _empty_frame() -> pd.DataFrame:
    """Empty frame with the right dtypes so concat preserves schema."""

    return pd.DataFrame(
        {
            "ts_ns": pd.Series([], dtype="int64"),
            "ts_unix": pd.Series([], dtype="float64"),
            "net_liq": pd.Series([], dtype="float64"),
            "available_funds": pd.Series([], dtype="float64"),
            "gross_position_value": pd.Series([], dtype="float64"),
            "cushion": pd.Series([], dtype="float64"),
        }
    )


def load_curve(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load the full equity curve as a DataFrame; empty frame if missing."""

    p = _resolve_path(path)
    if not p.exists():
        return _empty_frame()
    return pd.read_parquet(p)


def record(
    account_summary: Mapping[str, Any],
    timestamp_ns: int | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Append one observation; return the updated frame.

    ``account_summary`` follows the IBKR ``reqAccountSummary`` shape used
    elsewhere (see ``risk_management.snapshot.build_snapshot``). The
    ``NetLiquidation`` field is required; the rest default to 0.

    ``timestamp_ns`` defaults to ``time.time_ns()`` when omitted.
    """

    if "NetLiquidation" not in account_summary:
        raise KeyError("account_summary missing 'NetLiquidation'")

    ts_ns = int(timestamp_ns if timestamp_ns is not None else time.time_ns())
    row = {
        "ts_ns": ts_ns,
        "ts_unix": ts_ns / 1e9,
        "net_liq": _to_float(account_summary.get("NetLiquidation")),
        "available_funds": _to_float(
            account_summary.get("AvailableFunds", account_summary.get("NetLiquidation"))
        ),
        "gross_position_value": _to_float(account_summary.get("GrossPositionValue", 0)),
        "cushion": _to_float(account_summary.get("Cushion", 0)),
    }

    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing = load_curve(p)
    new_row = pd.DataFrame([row], columns=list(EQUITY_COLUMNS))
    updated = pd.concat([existing, new_row], ignore_index=True)
    # Enforce dtypes — concat can widen if existing was empty.
    updated = updated.astype(
        {
            "ts_ns": "int64",
            "ts_unix": "float64",
            "net_liq": "float64",
            "available_funds": "float64",
            "gross_position_value": "float64",
            "cushion": "float64",
        }
    )

    tmp = p.with_suffix(p.suffix + ".tmp")
    updated.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    return updated


def recent(
    days: int = 60,
    *,
    path: str | os.PathLike[str] | None = None,
    now_ns: int | None = None,
) -> pd.DataFrame:
    """Return rows newer than ``days`` ago (sorted by ts_ns ascending)."""

    if days <= 0:
        raise ValueError(f"days must be positive; got {days}")

    df = load_curve(path)
    if df.empty:
        return df

    end_ns = int(now_ns if now_ns is not None else time.time_ns())
    cutoff_ns = end_ns - int(days * 24 * 3600 * 1_000_000_000)
    out = df.loc[df["ts_ns"] >= cutoff_ns].copy()
    out.sort_values("ts_ns", inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out


def max_drawdown_pct(
    days: int = 60,
    *,
    path: str | os.PathLike[str] | None = None,
    now_ns: int | None = None,
) -> float:
    """Peak-to-trough drawdown over the last ``days`` as a negative fraction.

    Returns 0.0 when the curve is empty or strictly increasing.
    Examples: a 10% drawdown returns ``-0.10``.
    """

    df = recent(days=days, path=path, now_ns=now_ns)
    if df.empty:
        return 0.0

    series = df["net_liq"].astype(float)
    running_peak = series.cummax()
    # Guard against zero/negative net_liq (shouldn't happen, but be safe).
    safe_peak = running_peak.where(running_peak > 0, other=1.0)
    drawdowns = (series - running_peak) / safe_peak
    return float(drawdowns.min())
