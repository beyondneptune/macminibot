"""ml_research.inference — stateless `predict(features) -> list[Signal]` adapter.

Per CONTEXT.md Signal contract (line 30):
    @dataclass(frozen=True, slots=True)
    class Signal:
        symbol: Symbol
        r_hat_daily: Decimal
        confidence: float
        model_version: str
        as_of_ns: int

The trained model produces a z-scored prediction. The inference adapter
inverts the z-score using the cross-section's daily mean/std snapshot stored
in the model artifact so `strategy-engine` consumes a decimal return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

import numpy as np
import polars as pl

try:  # pragma: no cover - prefer shared-core if landed
    from shared_core.contracts import Signal as _SharedSignal  # type: ignore

    Signal = _SharedSignal
except ImportError:
    @dataclass(frozen=True, slots=True)
    class Signal:  # type: ignore[no-redef]
        """Stub mirroring CONTEXT line 30 verbatim until shared-core lands."""

        symbol: str  # placeholder for shared_core.Symbol
        r_hat_daily: Decimal
        confidence: float
        model_version: str
        as_of_ns: int


class _ModelLike(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class InferenceContext:
    """Per-fold inference snapshot persisted alongside the model artifact.

    Carries everything needed to invert the z-score transform back to a
    decimal return so strategy-engine never reasons about z-scores.
    """

    model_version: str
    model: _ModelLike
    feature_names: tuple[str, ...]
    feature_clip_bounds: dict[str, tuple[float, float]]
    # Per-day xs-zscore inversion stats — for the *current* day at inference.
    # In live serving this is computed from the live cross-section, not stored.
    xs_zscore_mean: float = 0.0
    xs_zscore_std: float = 1.0
    confidence_default: float = 0.5
    quantile_lo_model: _ModelLike | None = None
    quantile_hi_model: _ModelLike | None = None
    confidence_spread_scale: float = field(default=1.0)


def _invert_zscore(z: float, mean: float, std: float) -> Decimal:
    """Invert daily xs z-score back to decimal log-return."""
    if not np.isfinite(z):
        return Decimal("0")
    raw = z * std + mean
    # Use string construction to avoid float repr drift in Decimal.
    return Decimal(f"{raw:.10f}")


def _compute_confidence(
    y_pred_mean: np.ndarray,
    ctx: InferenceContext,
    X: np.ndarray,
) -> np.ndarray:
    """confidence = 1 / (1 + spread_normalized) per CONTEXT line 123.

    If quantile models are not provided (v1 baseline), return a constant
    `confidence_default` — the contract is preserved, the value is opaque.
    """
    if ctx.quantile_lo_model is None or ctx.quantile_hi_model is None:
        return np.full_like(y_pred_mean, ctx.confidence_default, dtype=np.float64)
    lo = ctx.quantile_lo_model.predict(X)
    hi = ctx.quantile_hi_model.predict(X)
    spread = np.clip(hi - lo, a_min=0.0, a_max=None)
    norm = spread / (ctx.confidence_spread_scale + 1e-9)
    return 1.0 / (1.0 + norm)


def predict(
    features_df: pl.DataFrame, ctx: InferenceContext, *, as_of_ns: int
) -> list[Signal]:
    """Stateless prediction adapter consumed by strategy-engine.

    features_df schema: symbol (str), date, plus FEATURE_NAMES columns.
    Rows with any NaN/inf feature are dropped (no silent imputation, per
    CONTEXT line 104).
    """
    from ml_research.features import (
        FEATURE_NAMES,
        apply_feature_clip,
        drop_feature_failures,
    )

    df = apply_feature_clip(features_df, ctx.feature_clip_bounds)
    df = drop_feature_failures(df, FEATURE_NAMES)
    if df.height == 0:
        return []

    feature_cols = [c for c in ctx.feature_names if c in df.columns]
    if not feature_cols:
        raise ValueError("inference: no expected feature columns present")
    X = df.select(feature_cols).to_numpy().astype(np.float64, copy=False)
    y_pred_z = ctx.model.predict(X)
    confs = _compute_confidence(y_pred_z, ctx, X)

    symbols = df["symbol"].to_list()
    signals: list[Signal] = []
    for sym, z, conf in zip(symbols, y_pred_z, confs):
        signals.append(
            Signal(
                symbol=sym,
                r_hat_daily=_invert_zscore(float(z), ctx.xs_zscore_mean, ctx.xs_zscore_std),
                confidence=float(np.clip(conf, 0.0, 1.0)),
                model_version=ctx.model_version,
                as_of_ns=int(as_of_ns),
            )
        )
    return signals


__all__ = ["InferenceContext", "Signal", "predict"]
