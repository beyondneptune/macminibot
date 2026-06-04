"""live_divergence — pre-trade sanity gate vs. backtest distribution.

CONTEXT.md "suspiciously high backtest" concern (Sharpe +2.03 baseline): even
with DSR / PSR / bootstrap-CI gates, the real-world failure mode is "live P&L
diverges from backtest distribution" — model decay, regime shift, broker
fee surprise, or a feature pipeline silently rotting. The chain catches
*structural* failures (NaN feature, stale snapshot); it does NOT catch
"the model is still emitting numbers but yesterday's realized P&L was -3σ vs
the distribution it was promoted on."

This module exposes a single pure helper:

    evaluate_live_divergence(recent_live_returns, backtest_returns, ...)
        -> LiveDivergenceVerdict

Verdict.breach is True when the live z-score exceeds ``sigma_threshold``
(default 3.0). Recommended wiring (Epic 7 follow-up): the daily 09:34 ET
preflight calls this on the trailing N live sessions and the published
backtest fold-returns; ``breach=True`` flips HALT with
``rule_id="LIVE_BACKTEST_DIVERGENCE"``. Operator must ack with an explanation
before resume (slot fits I6's WARN/HALT cooldown table — defaults to HALT
since divergence is a "stop and think" event by design).

Pure stdlib; the chain remains numpy-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from risk_management.metrics_emitter import MetricsSampler


DEFAULT_SIGMA_THRESHOLD: float = 3.0
DEFAULT_MIN_LIVE_OBS: int = 5
DEFAULT_MIN_BACKTEST_OBS: int = 60


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveDivergenceVerdict:
    """Result of comparing live-trailing returns against the backtest distribution.

    ``breach`` is the decision input the caller wires to HALT. The numeric
    fields preserve the evidence so the breach event + daily digest can
    explain "why" without recomputing.
    """

    breach: bool
    rule_id: str
    z_score: float          # signed; negative = live underperforms backtest mean
    sigma_threshold: float
    live_mean: float
    live_n: int
    backtest_mean: float
    backtest_std: float
    backtest_n: int
    observed: str           # stringified for BreachEvent.observed
    threshold: str          # stringified for BreachEvent.threshold

    def as_breach_kwargs(
        self,
        *,
        intent_id: str,
        snapshot_hash: str,
        symbol: str = "",
        config_id: str = "",
    ) -> dict[str, object]:
        """Build the kwargs for ``risk_management.breaches.make_breach``.

        Caller still picks ``mode`` and ``ts_ns`` (chain context owns those).
        ``would_halt=True`` because divergence is a HALT-tier rule by default.
        """

        return {
            "rule_id": self.rule_id,
            "observed": self.observed,
            "threshold": self.threshold,
            "intent_id": intent_id,
            "snapshot_hash": snapshot_hash,
            "would_halt": True,
            "config_id": config_id,
            "symbol": symbol,
            "extra": {
                "z_score": self.z_score,
                "live_mean": self.live_mean,
                "live_n": self.live_n,
                "backtest_mean": self.backtest_mean,
                "backtest_std": self.backtest_std,
                "backtest_n": self.backtest_n,
            },
        }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _stddev(values: Sequence[float], *, ddof: int = 1) -> float:
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / (n - ddof)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_live_divergence(
    recent_live_returns: Sequence[float],
    backtest_returns: Sequence[float],
    *,
    sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
    min_live_obs: int = DEFAULT_MIN_LIVE_OBS,
    min_backtest_obs: int = DEFAULT_MIN_BACKTEST_OBS,
    one_sided: bool = True,
    metrics_sampler: "MetricsSampler | None" = None,
) -> LiveDivergenceVerdict:
    """Compare live trailing returns to the backtest distribution.

    Test statistic is the z-score of the live-window mean under the null
    that live is drawn from the same distribution as the backtest:

        z = (mean(live) - mean(backtest)) /
            (std(backtest) / sqrt(len(live)))

    Two-sided ``|z| >= threshold`` => breach. One-sided (default True) only
    flags *under*-performance: ``z <= -threshold`` => breach. The
    over-performance case is suppressed because lucky days don't endanger
    the book — and a lucky day flagging HALT trains the operator to ack
    every divergence, defeating the purpose. (Repeated over-performance is
    a model-regime concern handled by ``backtesting``, not the chain.)

    Parameters
    ----------
    recent_live_returns
        Trailing live daily returns; typically the last 5-20 sessions read
        from ``risk/equity_curve.parquet`` by the 09:34 ET preflight.
    backtest_returns
        Per-day backtest returns from the strategy's promoted manifest
        (concatenated ``fold_metrics.daily_returns``).
    sigma_threshold
        Z-score magnitude that trips ``breach=True`` (default 3.0).
    min_live_obs, min_backtest_obs
        Below either count, the verdict is ``breach=False`` with
        ``rule_id="LIVE_BACKTEST_DIVERGENCE_INSUFFICIENT_DATA"`` — the
        statistic isn't trustworthy yet; emit no HALT but log loudly so the
        operator notices in the daily digest.
    one_sided
        When True (default), only downside divergence triggers. See above.
    metrics_sampler
        Optional :class:`risk_management.metrics_emitter.MetricsSampler`.
        When supplied, the verdict's z-score is published as
        ``bot_risk_drawdown_pct{window="live_divergence_z"}`` and the
        breach itself is published as
        ``bot_risk_check_rejections_total{rule_id="LIVE_BACKTEST_DIVERGENCE",
        severity="halt"}``. Insufficient-data / degenerate verdicts emit
        the rejection counter with their distinct rule_id tag and
        ``severity="warn"`` so dashboards see the "we tried but couldn't
        evaluate" path without false-paging the HALT alert. Mirrors the
        chain's opt-in pattern — emission failures are swallowed so
        observability never tanks a trading decision.

    Returns
    -------
    LiveDivergenceVerdict
    """

    live_n = len(recent_live_returns)
    bt_n = len(backtest_returns)

    if live_n < min_live_obs or bt_n < min_backtest_obs:
        live_mean = _mean(recent_live_returns) if live_n else 0.0
        bt_mean = _mean(backtest_returns) if bt_n else 0.0
        bt_std = _stddev(backtest_returns) if bt_n > 1 else 0.0
        verdict = LiveDivergenceVerdict(
            breach=False,
            rule_id="LIVE_BACKTEST_DIVERGENCE_INSUFFICIENT_DATA",
            z_score=0.0,
            sigma_threshold=sigma_threshold,
            live_mean=live_mean,
            live_n=live_n,
            backtest_mean=bt_mean,
            backtest_std=bt_std,
            backtest_n=bt_n,
            observed=(
                f"live_n={live_n} (min {min_live_obs}), "
                f"backtest_n={bt_n} (min {min_backtest_obs})"
            ),
            threshold=(
                f"live_n>={min_live_obs} and backtest_n>={min_backtest_obs}"
            ),
        )
        _emit_divergence_metrics(metrics_sampler, verdict)
        return verdict

    live_mean = _mean(recent_live_returns)
    bt_mean = _mean(backtest_returns)
    bt_std = _stddev(backtest_returns)

    # Treat near-zero stddev as degenerate too — a constant series stored as
    # IEEE-754 floats often computes to a sub-nanoscale residual std rather
    # than exactly 0.0, which would otherwise blow up the z-score to ±1e17.
    # 1e-12 is far below any plausible daily-return stddev (~1bps = 1e-4).
    if not math.isfinite(bt_std) or bt_std <= 1e-12:
        verdict = LiveDivergenceVerdict(
            breach=False,
            rule_id="LIVE_BACKTEST_DIVERGENCE_DEGENERATE",
            z_score=0.0,
            sigma_threshold=sigma_threshold,
            live_mean=live_mean,
            live_n=live_n,
            backtest_mean=bt_mean,
            backtest_std=bt_std,
            backtest_n=bt_n,
            observed=f"backtest_std={bt_std}",
            threshold="backtest_std>0",
        )
        _emit_divergence_metrics(metrics_sampler, verdict)
        return verdict

    se = bt_std / math.sqrt(live_n)
    z = (live_mean - bt_mean) / se

    if one_sided:
        breach = z <= -sigma_threshold
        threshold_str = f"z>{-sigma_threshold:+.2f}"
    else:
        breach = abs(z) >= sigma_threshold
        threshold_str = f"|z|<{sigma_threshold:.2f}"

    verdict = LiveDivergenceVerdict(
        breach=breach,
        rule_id="LIVE_BACKTEST_DIVERGENCE",
        z_score=z,
        sigma_threshold=sigma_threshold,
        live_mean=live_mean,
        live_n=live_n,
        backtest_mean=bt_mean,
        backtest_std=bt_std,
        backtest_n=bt_n,
        observed=f"z={z:+.3f}",
        threshold=threshold_str,
    )
    _emit_divergence_metrics(metrics_sampler, verdict)
    return verdict


def _emit_divergence_metrics(
    sampler: "MetricsSampler | None", verdict: LiveDivergenceVerdict
) -> None:
    """Best-effort metric emission for a divergence verdict.

    Three signals land on the JSONL sink:
      - ``bot_risk_drawdown_pct{window="live_divergence_z"}`` carries the
        z-score (gauge — operators dashboard the trailing series; the
        sampler accepts any float).
      - ``bot_risk_check_rejections_total{rule_id, symbol="", severity}``
        fires once per verdict that doesn't pass cleanly — ``severity="halt"``
        for an actual breach (matches the chain's HALT_INDUCING heuristic),
        ``severity="warn"`` for INSUFFICIENT_DATA / DEGENERATE so dashboards
        see the "we tried but couldn't evaluate" path without paging.
      - The "clean pass" case (breach=False, rule_id=LIVE_BACKTEST_DIVERGENCE)
        emits no rejection — only the z-score gauge — so the rejection
        counter stays a true rejection signal.

    Mirrors ``chain._emit_chain_metrics``: emission errors are swallowed.
    Observability problems must never tank a real trading decision.
    """

    if sampler is None:
        return
    try:
        sampler.emit(
            "bot_risk_drawdown_pct",
            verdict.z_score,
            labels={"window": "live_divergence_z"},
        )
        if verdict.breach:
            sampler.emit(
                "bot_risk_check_rejections_total",
                1,
                labels={
                    "rule_id": verdict.rule_id,
                    "symbol": "",
                    "severity": "halt",
                },
            )
        elif verdict.rule_id != "LIVE_BACKTEST_DIVERGENCE":
            # INSUFFICIENT_DATA / DEGENERATE — surface as a warn-tier rejection
            # so the daily digest catches "we wanted to evaluate but couldn't".
            sampler.emit(
                "bot_risk_check_rejections_total",
                1,
                labels={
                    "rule_id": verdict.rule_id,
                    "symbol": "",
                    "severity": "warn",
                },
            )
    except Exception:
        # Best-effort — see docstring. A regression test catches schema drift;
        # propagating the exception would let observability break HALT logic.
        return


# ---------------------------------------------------------------------------
# Convenience: Decimal -> float coercion for callers reading equity_curve
# ---------------------------------------------------------------------------


def to_float_returns(values: Sequence[float | Decimal]) -> list[float]:
    """Coerce a mix of float/Decimal returns to floats (drops non-finite)."""

    out: list[float] = []
    for v in values:
        f = float(v)
        if math.isfinite(f):
            out.append(f)
    return out


__all__ = [
    "DEFAULT_MIN_BACKTEST_OBS",
    "DEFAULT_MIN_LIVE_OBS",
    "DEFAULT_SIGMA_THRESHOLD",
    "LiveDivergenceVerdict",
    "evaluate_live_divergence",
    "to_float_returns",
]
