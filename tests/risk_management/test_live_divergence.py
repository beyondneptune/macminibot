"""Tests for risk_management.live_divergence (Epic 7 follow-on / cycle-8 sanity gate).

The verdict is the input the 09:34 ET preflight wires into HALT. These tests
pin the public contract: z-score sign, one-sided default, insufficient-data
gates, breach-kwargs shape.
"""

from __future__ import annotations

import math
import random

import pytest

from risk_management.live_divergence import (
    DEFAULT_MIN_BACKTEST_OBS,
    DEFAULT_MIN_LIVE_OBS,
    DEFAULT_SIGMA_THRESHOLD,
    LiveDivergenceVerdict,
    evaluate_live_divergence,
    to_float_returns,
)


def _backtest_dist(n: int = 200, mean: float = 0.001, std: float = 0.01) -> list[float]:
    random.seed(0)
    return [random.gauss(mean, std) for _ in range(n)]


def test_insufficient_live_obs_does_not_breach():
    bt = _backtest_dist()
    v = evaluate_live_divergence([-0.05] * 2, bt)
    assert v.breach is False
    assert v.rule_id == "LIVE_BACKTEST_DIVERGENCE_INSUFFICIENT_DATA"
    assert v.live_n == 2


def test_insufficient_backtest_obs_does_not_breach():
    v = evaluate_live_divergence([-0.05] * 10, [0.001] * 30)
    assert v.breach is False
    assert v.rule_id == "LIVE_BACKTEST_DIVERGENCE_INSUFFICIENT_DATA"
    assert v.backtest_n == 30


def test_degenerate_backtest_std_does_not_breach():
    """Constant backtest series => degenerate gate, never breach."""

    v = evaluate_live_divergence([-0.05] * 10, [0.001] * 100)
    assert v.breach is False
    assert v.rule_id == "LIVE_BACKTEST_DIVERGENCE_DEGENERATE"
    assert v.backtest_std < 1e-12  # below the degenerate floor


def test_live_in_distribution_does_not_breach():
    bt = _backtest_dist()
    # Live mean ~ backtest mean.
    random.seed(1)
    live = [random.gauss(0.001, 0.01) for _ in range(10)]
    v = evaluate_live_divergence(live, bt)
    assert v.breach is False
    assert v.rule_id == "LIVE_BACKTEST_DIVERGENCE"
    assert abs(v.z_score) < DEFAULT_SIGMA_THRESHOLD


def test_live_severe_underperformance_breaches():
    bt = _backtest_dist()
    # Live persistently 5σ below backtest mean — guaranteed breach.
    live = [-0.05] * 10
    v = evaluate_live_divergence(live, bt)
    assert v.breach is True
    assert v.z_score < -DEFAULT_SIGMA_THRESHOLD


def test_one_sided_default_ignores_overperformance():
    bt = _backtest_dist()
    # Live persistently 5σ ABOVE backtest mean — no breach in one-sided mode.
    live = [0.05] * 10
    v = evaluate_live_divergence(live, bt, one_sided=True)
    assert v.breach is False
    assert v.z_score > DEFAULT_SIGMA_THRESHOLD


def test_two_sided_flags_overperformance():
    bt = _backtest_dist()
    live = [0.05] * 10
    v = evaluate_live_divergence(live, bt, one_sided=False)
    assert v.breach is True
    assert v.z_score > DEFAULT_SIGMA_THRESHOLD


def test_sigma_threshold_param_respected():
    bt = _backtest_dist()
    live = [-0.02] * 10  # roughly -2σ
    v_strict = evaluate_live_divergence(live, bt, sigma_threshold=1.0)
    v_loose = evaluate_live_divergence(live, bt, sigma_threshold=10.0)
    assert v_strict.breach is True
    assert v_loose.breach is False


def test_as_breach_kwargs_shape():
    bt = _backtest_dist()
    live = [-0.05] * 10
    v = evaluate_live_divergence(live, bt)
    kw = v.as_breach_kwargs(
        intent_id="batch:AAPL",
        snapshot_hash="sha256:abc",
        symbol="AAPL",
        config_id="2026-05-17.a",
    )
    assert kw["rule_id"] == "LIVE_BACKTEST_DIVERGENCE"
    assert kw["intent_id"] == "batch:AAPL"
    assert kw["snapshot_hash"] == "sha256:abc"
    assert kw["symbol"] == "AAPL"
    assert kw["config_id"] == "2026-05-17.a"
    assert kw["would_halt"] is True
    extra = kw["extra"]
    assert isinstance(extra, dict)
    assert "z_score" in extra
    assert "live_n" in extra and extra["live_n"] == 10
    assert "backtest_n" in extra and extra["backtest_n"] == len(bt)


def test_z_score_sign_convention():
    """Negative z = live underperforms; positive z = live overperforms."""

    bt = _backtest_dist(mean=0.001)
    v_under = evaluate_live_divergence([-0.01] * 10, bt)
    v_over = evaluate_live_divergence([0.01] * 10, bt)
    assert v_under.z_score < 0.0
    assert v_over.z_score > 0.0


def test_to_float_returns_drops_non_finite():
    from decimal import Decimal as D
    out = to_float_returns([0.01, D("-0.02"), float("nan"), float("inf"), 0.005])
    assert out == [0.01, -0.02, 0.005]


def test_verdict_is_frozen():
    bt = _backtest_dist()
    v = evaluate_live_divergence([-0.05] * 10, bt)
    with pytest.raises(Exception):
        v.breach = False  # type: ignore[misc]


def test_default_thresholds_match_documented_values():
    assert DEFAULT_SIGMA_THRESHOLD == 3.0
    assert DEFAULT_MIN_LIVE_OBS == 5
    assert DEFAULT_MIN_BACKTEST_OBS == 60


# ---------------------------------------------------------------------------
# Metrics emission (cycle-12: parity with chain's optional sampler)
# ---------------------------------------------------------------------------


def _new_sampler():
    from shared_core.contracts import ChainMode

    from risk_management.metrics_emitter import new_sampler
    return new_sampler(ChainMode.WHATIF, "test-cfg")


def test_metrics_sampler_emits_z_score_gauge():
    bt = _backtest_dist()
    live = [random.gauss(0.001, 0.01) for _ in range(10)]
    s = _new_sampler()
    v = evaluate_live_divergence(live, bt, metrics_sampler=s)
    # At least one drawdown_pct sample with the live_divergence_z window tag.
    drawdown = [m for m in s.buffer if m.name == "bot_risk_drawdown_pct"]
    assert len(drawdown) == 1
    assert drawdown[0].labels["window"] == "live_divergence_z"
    assert drawdown[0].value == pytest.approx(v.z_score)


def test_metrics_sampler_emits_halt_rejection_on_breach():
    bt = _backtest_dist()
    live = [-0.05] * 10  # guaranteed breach
    s = _new_sampler()
    v = evaluate_live_divergence(live, bt, metrics_sampler=s)
    assert v.breach is True
    rejs = [m for m in s.buffer if m.name == "bot_risk_check_rejections_total"]
    assert len(rejs) == 1
    assert rejs[0].labels["rule_id"] == "LIVE_BACKTEST_DIVERGENCE"
    assert rejs[0].labels["severity"] == "halt"


def test_metrics_sampler_emits_warn_rejection_on_insufficient_data():
    bt = _backtest_dist()
    s = _new_sampler()
    v = evaluate_live_divergence([-0.05] * 2, bt, metrics_sampler=s)
    assert v.breach is False
    rejs = [m for m in s.buffer if m.name == "bot_risk_check_rejections_total"]
    assert len(rejs) == 1
    assert rejs[0].labels["rule_id"] == "LIVE_BACKTEST_DIVERGENCE_INSUFFICIENT_DATA"
    assert rejs[0].labels["severity"] == "warn"


def test_metrics_sampler_emits_warn_rejection_on_degenerate():
    s = _new_sampler()
    v = evaluate_live_divergence([-0.05] * 10, [0.001] * 100, metrics_sampler=s)
    assert v.rule_id == "LIVE_BACKTEST_DIVERGENCE_DEGENERATE"
    rejs = [m for m in s.buffer if m.name == "bot_risk_check_rejections_total"]
    assert len(rejs) == 1
    assert rejs[0].labels["severity"] == "warn"


def test_metrics_sampler_clean_pass_emits_no_rejection():
    """breach=False AND rule=LIVE_BACKTEST_DIVERGENCE => only z-score gauge."""

    bt = _backtest_dist()
    random.seed(42)
    live = [random.gauss(0.001, 0.01) for _ in range(10)]
    s = _new_sampler()
    v = evaluate_live_divergence(live, bt, metrics_sampler=s)
    assert v.breach is False
    assert v.rule_id == "LIVE_BACKTEST_DIVERGENCE"
    rejs = [m for m in s.buffer if m.name == "bot_risk_check_rejections_total"]
    assert rejs == []
    gauges = [m for m in s.buffer if m.name == "bot_risk_drawdown_pct"]
    assert len(gauges) == 1


def test_metrics_sampler_none_is_default_noop():
    """No sampler supplied => no error, plain verdict returned."""

    bt = _backtest_dist()
    v = evaluate_live_divergence([-0.05] * 10, bt)
    assert v.breach is True  # unchanged behavior


def test_metrics_sampler_errors_are_swallowed():
    """A broken sampler must NOT propagate (observability never tanks chain)."""

    class _Broken:
        def emit(self, *a, **kw):
            raise RuntimeError("sink down")

    bt = _backtest_dist()
    v = evaluate_live_divergence([-0.05] * 10, bt, metrics_sampler=_Broken())  # type: ignore[arg-type]
    assert v.breach is True
