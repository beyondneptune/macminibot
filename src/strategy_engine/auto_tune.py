"""strategy_engine.auto_tune — recalibrate ``lambda_risk`` + ``target_vol_annualized``.

Problem
-------
The Ridge sanity-floor produces tiny predictions (xs-zscore space, ≈ ±0.03).
At the default ``lambda_risk = 5.0`` the MV objective is so risk-dominated
that every optimal weight collapses under the 0.5% NAV no-trade band, and the
scheduler emits no intents. The paper runner currently sidesteps this by
forcing a ranked equal-weight book — that is a *safety net*, not the
production path. We want the optimizer to actually trade on signal.

Approach
--------
Solve, on a representative recent day, for the ``lambda_risk`` that puts
``||w*||_inf`` at a healthy ~10% NAV (the optimizer's ``c_max`` per-name
cap is 5%; ranked-equal-weight short books in the runner use ~5% per name,
so an L_inf target near c_max is *exactly* the right magnitude). Bisect over
log-lambda. Clamp the result.

Then derive ``target_vol_annualized`` from the day's realized portfolio
vol so the vol-target scaler doesn't squash the optimizer's output:

    target_vol_annualized = clip(realized_vol(w*) * boost, 0.01, 0.50)

Persist the (lambda, target_vol) pair to
``data/config/strategy_tuning.json`` keyed by date. The next morning's
``40_daily_morning_routine.py`` picks it up; the paper trade runner reads it
on launch.

Hard rule: lambda_risk ∈ (0.001, 100.0), target_vol_annualized ∈ (0.01, 0.50).
Any out-of-band candidate is clamped + logged WARNING.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from strategy_engine.optimizer import mean_variance_optimize
from strategy_engine.sizing import DEFAULT_TARGET_VOL


# ---------------------------------------------------------------------------
# Clamp bands — hard constraints from the deliverable spec. NEVER widen these
# without an explicit ratification — the optimizer's interior-point solvers
# behave erratically outside.
# ---------------------------------------------------------------------------

LAMBDA_RISK_BOUNDS: tuple[float, float] = (0.001, 100.0)
TARGET_VOL_BOUNDS: tuple[float, float] = (0.01, 0.50)

# Heuristic target for ||w*||_inf when calibrating lambda. The deliverable
# spec gives 0.10 (= 10% NAV per name), which sits above the runtime per-name
# cap ``c_max=0.05`` — so the target is *aspirational*: we drive lambda down
# until either the c_max cap binds (achieved ≈ c_max) or the bisection hits
# the spec target. Either landing is a successful auto-tune. The clamped
# ``lambda_risk`` is the deliverable artifact, not the literal linf number.
DEFAULT_LINF_TARGET: float = 0.10

# Default vol-target safety boost — multiplies realized portfolio vol so the
# vol-target scaler has slack to size up on lower-vol days. 1.25 means we
# advertise 25% more vol budget than the optimizer's raw output asks for.
DEFAULT_VOL_BOOST: float = 1.25


@dataclass(frozen=True)
class AutoTuneResult:
    """The full tuning envelope persisted to ``strategy_tuning.json``."""

    as_of_date: date
    lambda_risk: float
    target_vol_annualized: float
    linf_at_lambda: float  # achieved ||w*||_inf at the chosen lambda
    linf_target: float  # the heuristic we aimed for
    realized_vol_w: float  # portfolio annualized vol on the calibration day
    clamped_lambda: bool
    clamped_target_vol: bool
    n_iter: int  # bisection iterations
    notes: str = ""


# ---------------------------------------------------------------------------
# Bisection
# ---------------------------------------------------------------------------


def _linf_for_lambda(
    *,
    r_hat: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    lambda_risk: float,
    gamma_turnover: float,
    L_gross: float,
    c_max: float,
) -> tuple[float, np.ndarray]:
    """Solve the MV problem at ``lambda_risk`` and return ``(||w*||_inf, w*)``.

    Wrapped so the bisection root-finder doesn't need to know about
    OptimizerResult plumbing.
    """
    res = mean_variance_optimize(
        r_hat=r_hat,
        Sigma=Sigma,
        w_prev=w_prev,
        lambda_risk=float(lambda_risk),
        gamma_turnover=gamma_turnover,
        L_gross=L_gross,
        c_max=c_max,
    )
    w = np.asarray(res.w, dtype=np.float64)
    return float(np.max(np.abs(w))), w


def solve_lambda_for_linf(
    *,
    r_hat: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    linf_target: float = DEFAULT_LINF_TARGET,
    gamma_turnover: float = 0.002,
    L_gross: float = 2.0,
    c_max: float = 0.05,
    lambda_lo: float = LAMBDA_RISK_BOUNDS[0],
    lambda_hi: float = LAMBDA_RISK_BOUNDS[1],
    max_iter: int = 30,
    tol_rel: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> tuple[float, float, np.ndarray, int]:
    """Bisect over **log**-lambda so ``||w*||_inf ≈ linf_target``.

    ``||w*||_inf`` is monotone non-increasing in ``lambda_risk`` (more
    risk-aversion → smaller positions), so bisection on log-lambda is
    well-defined.

    Returns
    -------
    (lambda_star, achieved_linf, w_star, iter_count)
    """
    if logger is None:
        logger = logging.getLogger("auto_tune")

    n = r_hat.shape[0]
    if Sigma.shape != (n, n) or w_prev.shape != (n,):
        raise ValueError("solve_lambda_for_linf: dimension mismatch in inputs")
    if not (LAMBDA_RISK_BOUNDS[0] <= lambda_lo < lambda_hi <= LAMBDA_RISK_BOUNDS[1]):
        raise ValueError(
            f"solve_lambda_for_linf: lambda window {(lambda_lo, lambda_hi)} "
            f"outside hard bounds {LAMBDA_RISK_BOUNDS}"
        )

    log_lo, log_hi = np.log(lambda_lo), np.log(lambda_hi)

    # Sanity check monotonicity at the endpoints.
    linf_lo, _ = _linf_for_lambda(
        r_hat=r_hat, Sigma=Sigma, w_prev=w_prev,
        lambda_risk=lambda_lo, gamma_turnover=gamma_turnover,
        L_gross=L_gross, c_max=c_max,
    )
    linf_hi, _ = _linf_for_lambda(
        r_hat=r_hat, Sigma=Sigma, w_prev=w_prev,
        lambda_risk=lambda_hi, gamma_turnover=gamma_turnover,
        L_gross=L_gross, c_max=c_max,
    )

    # If even the most aggressive lambda can't push ||w||_inf up to the target,
    # the signal is too small (or c_max blocks it) — return the lo endpoint.
    if linf_lo <= linf_target:
        # smallest lambda → biggest positions; if STILL below target, we are
        # signal-bound. Return the lo endpoint.
        logger.warning(
            "auto_tune: linf at lambda_lo=%g is %.4f < target %.4f — signal-bound; "
            "returning lambda_lo",
            lambda_lo, linf_lo, linf_target,
        )
        _, w_star = _linf_for_lambda(
            r_hat=r_hat, Sigma=Sigma, w_prev=w_prev,
            lambda_risk=lambda_lo, gamma_turnover=gamma_turnover,
            L_gross=L_gross, c_max=c_max,
        )
        return lambda_lo, linf_lo, w_star, 0
    if linf_hi >= linf_target:
        # biggest lambda → smallest positions; if still above target, signal is
        # so strong that the c_max cap binds — return lambda_hi.
        logger.warning(
            "auto_tune: linf at lambda_hi=%g is %.4f > target %.4f — c_max bound; "
            "returning lambda_hi",
            lambda_hi, linf_hi, linf_target,
        )
        _, w_star = _linf_for_lambda(
            r_hat=r_hat, Sigma=Sigma, w_prev=w_prev,
            lambda_risk=lambda_hi, gamma_turnover=gamma_turnover,
            L_gross=L_gross, c_max=c_max,
        )
        return lambda_hi, linf_hi, w_star, 0

    iter_count = 0
    achieved = linf_lo
    w_star = np.zeros(n)
    lambda_mid = lambda_lo
    while iter_count < max_iter:
        log_mid = 0.5 * (log_lo + log_hi)
        lambda_mid = float(np.exp(log_mid))
        achieved, w_star = _linf_for_lambda(
            r_hat=r_hat, Sigma=Sigma, w_prev=w_prev,
            lambda_risk=lambda_mid, gamma_turnover=gamma_turnover,
            L_gross=L_gross, c_max=c_max,
        )
        if achieved > linf_target:
            log_lo = log_mid  # too aggressive → push lambda up
        else:
            log_hi = log_mid  # too cautious → pull lambda down
        iter_count += 1
        if achieved == 0.0:
            # Solver degenerate; abort early.
            break
        if abs(achieved - linf_target) / linf_target < tol_rel:
            break

    logger.info(
        "auto_tune: bisect converged in %d iters — lambda=%g linf=%.4f (target=%.4f)",
        iter_count, lambda_mid, achieved, linf_target,
    )
    return lambda_mid, achieved, w_star, iter_count


# ---------------------------------------------------------------------------
# Target-vol derivation
# ---------------------------------------------------------------------------


def _portfolio_annualized_vol(w: np.ndarray, Sigma_daily: np.ndarray) -> float:
    """``sqrt(w' Σ w * 252)``. Assumes Σ is the *daily* covariance."""
    if w.size == 0:
        return 0.0
    var = float(w @ Sigma_daily @ w)
    var = max(var, 0.0)
    return float(np.sqrt(var * 252.0))


def derive_target_vol(
    *,
    w_star: np.ndarray,
    Sigma_daily: np.ndarray,
    boost: float = DEFAULT_VOL_BOOST,
) -> tuple[float, float]:
    """Return ``(target_vol_annualized, realized_vol_w)``.

    Picks the vol-target so the optimizer's calibrated weights *exactly fit
    inside* the vol budget — with a multiplicative ``boost`` for headroom.
    """
    realized = _portfolio_annualized_vol(w_star, Sigma_daily)
    if realized <= 0.0:
        # Degenerate — fall back to the static default.
        return DEFAULT_TARGET_VOL, 0.0
    return float(realized * boost), realized


# ---------------------------------------------------------------------------
# Clamping helpers
# ---------------------------------------------------------------------------


def _clamp_lambda(
    value: float, logger: logging.Logger
) -> tuple[float, bool]:
    lo, hi = LAMBDA_RISK_BOUNDS
    if value < lo:
        logger.warning(
            "auto_tune: lambda_risk %g below floor %g — clamping", value, lo
        )
        return lo, True
    if value > hi:
        logger.warning(
            "auto_tune: lambda_risk %g above ceiling %g — clamping", value, hi
        )
        return hi, True
    return float(value), False


def _clamp_target_vol(
    value: float, logger: logging.Logger
) -> tuple[float, bool]:
    lo, hi = TARGET_VOL_BOUNDS
    if value < lo:
        logger.warning(
            "auto_tune: target_vol %g below floor %g — clamping", value, lo
        )
        return lo, True
    if value > hi:
        logger.warning(
            "auto_tune: target_vol %g above ceiling %g — clamping", value, hi
        )
        return hi, True
    return float(value), False


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def run_auto_tune(
    *,
    r_hat: np.ndarray,
    Sigma_daily: np.ndarray,
    w_prev: np.ndarray,
    as_of_date: date,
    linf_target: float = DEFAULT_LINF_TARGET,
    vol_boost: float = DEFAULT_VOL_BOOST,
    gamma_turnover: float = 0.002,
    L_gross: float = 2.0,
    c_max: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> AutoTuneResult:
    """Compute + clamp the (lambda_risk, target_vol) pair for a single day.

    ``r_hat`` / ``Sigma_daily`` / ``w_prev`` are *representative* recent-day
    inputs (most callers will use yesterday's predictions, yesterday's
    Ledoit-Wolf Σ, and the current book). The result is persisted via
    :func:`persist_tuning_record`.
    """
    if logger is None:
        logger = logging.getLogger("auto_tune")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
            logger.addHandler(h)
            logger.setLevel(logging.INFO)

    lambda_raw, achieved_linf, w_star, n_iter = solve_lambda_for_linf(
        r_hat=r_hat,
        Sigma=Sigma_daily,
        w_prev=w_prev,
        linf_target=linf_target,
        gamma_turnover=gamma_turnover,
        L_gross=L_gross,
        c_max=c_max,
        logger=logger,
    )
    lambda_clamped, lambda_was_clamped = _clamp_lambda(lambda_raw, logger)

    target_vol_raw, realized = derive_target_vol(
        w_star=w_star, Sigma_daily=Sigma_daily, boost=vol_boost
    )
    target_vol_clamped, vol_was_clamped = _clamp_target_vol(target_vol_raw, logger)

    notes = []
    if lambda_was_clamped:
        notes.append("lambda_clamped")
    if vol_was_clamped:
        notes.append("target_vol_clamped")
    if realized <= 0.0:
        notes.append("vol_fallback_default")

    result = AutoTuneResult(
        as_of_date=as_of_date,
        lambda_risk=lambda_clamped,
        target_vol_annualized=target_vol_clamped,
        linf_at_lambda=achieved_linf,
        linf_target=linf_target,
        realized_vol_w=realized,
        clamped_lambda=lambda_was_clamped,
        clamped_target_vol=vol_was_clamped,
        n_iter=n_iter,
        notes=",".join(notes),
    )
    logger.info(
        "auto_tune: as_of=%s lambda=%.4f target_vol=%.4f linf=%.4f realized_vol=%.4f%s",
        as_of_date,
        result.lambda_risk,
        result.target_vol_annualized,
        result.linf_at_lambda,
        result.realized_vol_w,
        f" notes={result.notes}" if result.notes else "",
    )
    return result


# ---------------------------------------------------------------------------
# Persistence (JSON, keyed by date)
# ---------------------------------------------------------------------------


def _config_path(project_root: Path) -> Path:
    return project_root / "data" / "config" / "strategy_tuning.json"


def persist_tuning_record(
    result: AutoTuneResult,
    *,
    project_root: Path | None = None,
) -> Path:
    """Append ``result`` to ``data/config/strategy_tuning.json``.

    The file is a JSON object keyed by ISO date — so running this twice on
    the same calendar day **overwrites** that day's entry (the latest
    invocation wins, which is exactly what the morning routine wants).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    path = _config_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        except json.JSONDecodeError:
            existing = {}

    existing[result.as_of_date.isoformat()] = {
        "lambda_risk": result.lambda_risk,
        "target_vol_annualized": result.target_vol_annualized,
        "linf_at_lambda": result.linf_at_lambda,
        "linf_target": result.linf_target,
        "realized_vol_w": result.realized_vol_w,
        "clamped_lambda": result.clamped_lambda,
        "clamped_target_vol": result.clamped_target_vol,
        "n_iter": result.n_iter,
        "notes": result.notes,
        "written_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
    return path


def load_latest_tuning(project_root: Path | None = None) -> dict | None:
    """Load the most recent tuning record (by date key). Returns ``None`` if
    the file is missing — the paper runner uses this signal to fall back to
    its static lambda_risk=0.1.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    path = _config_path(project_root)
    if not path.exists():
        return None
    with open(path) as f:
        records = json.load(f)
    if not records:
        return None
    latest_key = max(records.keys())  # ISO-date strings sort lexicographically
    record = dict(records[latest_key])
    record["as_of_date"] = latest_key
    return record
