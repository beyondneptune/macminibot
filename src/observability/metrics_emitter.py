"""observability.metrics_emitter — dashboard-facing aggregator.

The schema-bound JSONL emitter lives in :mod:`risk_management.metrics_emitter`
(see the module docstring there for the 11-metric contract). This module
re-exports the relevant types and adds :func:`combined_dashboard_state`,
which collapses live state into a single JSON blob suitable for
``streamlit.cache_data``.

The dashboard reads from this single entry point rather than reaching into
4 different tentacles, so a UI rewrite never breaks an internal contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

# Re-export the existing risk_management surface so callers can `from
# observability.metrics_emitter import MetricsSampler` and not care that
# the schema lives under risk_management/.
from risk_management.metrics_emitter import (  # noqa: F401
    COMMON_LABELS,
    METRIC_NAMES,
    METRICS_FILENAME,
    SCHEMA,
    MetricSample,
    MetricSpec,
    MetricsSampler,
    MetricsValidationError,
    new_sampler,
)
from risk_management.breaches import risk_dir

from observability.equity_curve import load_curve, max_drawdown_pct


__all__ = [
    "COMMON_LABELS",
    "METRIC_NAMES",
    "METRICS_FILENAME",
    "SCHEMA",
    "MetricSample",
    "MetricSpec",
    "MetricsSampler",
    "MetricsValidationError",
    "new_sampler",
    "combined_dashboard_state",
]


def _breaches_today(
    risk_root: Path,
    *,
    mode: str = "paper",
    today_ns: int | None = None,
) -> int:
    """Count breach JSONL records emitted since UTC midnight today."""

    fname = "breaches.jsonl"
    path = risk_root / "paper" / fname if mode == "paper" else risk_root / fname
    if not path.exists():
        return 0

    import time

    now_ns = int(today_ns if today_ns is not None else time.time_ns())
    # UTC midnight floor in ns
    one_day_ns = 24 * 3600 * 1_000_000_000
    midnight_ns = (now_ns // one_day_ns) * one_day_ns

    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_ns = rec.get("ts_ns")
            if ts_ns is None:
                continue
            try:
                if int(ts_ns) >= midnight_ns:
                    count += 1
            except (TypeError, ValueError):
                continue
    return count


def _day_pnl_from_curve(curve_path: str | Path | None = None) -> float:
    """Today's P&L = today's latest net_liq minus today's first net_liq."""

    df = load_curve(curve_path)
    if df.empty:
        return 0.0
    import time

    now_ns = time.time_ns()
    one_day_ns = 24 * 3600 * 1_000_000_000
    midnight_ns = (now_ns // one_day_ns) * one_day_ns
    today = df.loc[df["ts_ns"] >= midnight_ns]
    if today.empty:
        return 0.0
    today_sorted = today.sort_values("ts_ns")
    return float(today_sorted["net_liq"].iloc[-1] - today_sorted["net_liq"].iloc[0])


def combined_dashboard_state(
    account_summary: Mapping[str, Any] | None = None,
    positions: Mapping[str, Any] | None = None,
    *,
    vix_regime: str = "unknown",
    mode: str = "paper",
    risk_root: Path | None = None,
    equity_curve_path: str | Path | None = None,
) -> dict[str, Any]:
    """One JSON blob describing live bot state.

    Output schema (stable; pad with ``None`` if any input is unavailable)::

        {
          "equity":         float,
          "day_pnl":        float,
          "positions":      {symbol: float, ...},
          "breaches_today": int,
          "vix_regime":     str,
        }

    All values are JSON-serialisable; the dict round-trips through
    ``json.dumps`` so it slots straight into ``streamlit.cache_data``.
    Pass ``account_summary`` and ``positions`` directly from the live
    IBKR snapshot. ``vix_regime`` is the operator-visible label
    (``"vix_high"`` | ``"vix_mid"`` | ``"vix_low"`` | ``"unknown"``).
    """

    equity = 0.0
    if account_summary is not None and "NetLiquidation" in account_summary:
        try:
            equity = float(account_summary["NetLiquidation"])
        except (TypeError, ValueError):
            equity = 0.0

    pos_out: dict[str, float] = {}
    for sym, qty in (positions or {}).items():
        try:
            pos_out[str(sym)] = float(qty)
        except (TypeError, ValueError):
            continue

    root = risk_root if risk_root is not None else risk_dir()
    breaches_today = _breaches_today(root, mode=mode)

    return {
        "equity": equity,
        "day_pnl": _day_pnl_from_curve(equity_curve_path),
        "positions": pos_out,
        "breaches_today": breaches_today,
        "vix_regime": str(vix_regime),
    }
