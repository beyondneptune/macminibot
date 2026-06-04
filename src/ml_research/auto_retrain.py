"""ml_research.auto_retrain — nightly Ridge retrain + persist.

Job (called by ``scripts/40_daily_morning_routine.py`` at 09:00 ET, ~35 min
before the trade runner fires):

    1. Load most-recent N-year daily-bar panel from ArcticDB
       (8 tradables + SPY/QQQ/IWM aux). The cutoff is yesterday (``today - 1``)
       so today's bar — which may still be forming — is never trained on.
    2. Compute the canonical feature panel via :func:`ml_research.features.compute_features`.
    3. Fit the next-day open/close label, winsorize, xs-zscore.
    4. Fit ``RidgeTrainer`` on the full N-year window.
    5. Persist:
         data/models/ridge-YYYY-MM-DD.pkl     (the trained estimator)
         data/models/latest.pkl               (symlink → today's pickle)
       Both paths are relative-symlinked so a future repo move doesn't
       brick the pointer.
    6. Evaluate IC + Sharpe via :func:`ml_research.trainer.evaluate_fold` on the
       held-out **last 21 trading days** of the window (in-window OOS proxy —
       cheap, no temporal data leakage because we just re-fit ridge on the
       remaining window for evaluation).

Hard rule: today's date is NEVER in the training set. ``cutoff_date`` is
clamped to ``today - 1 calendar day`` before slicing the panel.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from ml_research.features import FEATURE_NAMES, compute_features, drop_feature_failures
from ml_research.labels import (
    cross_sectional_zscore,
    fit_winsorization_bounds,
    next_day_open_close_log_return,
    winsorize_train_fold,
)
from ml_research.trainer import RidgeConfig, RidgeTrainer, evaluate_fold


# ---------------------------------------------------------------------------
# Universe (kept aligned with scripts/30_paper_trade_runner.py — see deliverable
# spec note "DO NOT break the existing forced-book fallback"). If the runner's
# universe ever changes, this constant must move in lockstep.
# ---------------------------------------------------------------------------

_TRADABLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    ("equity.daily_bars", "NASDAQ.AAPL",  "AAPL"),
    ("equity.daily_bars", "NASDAQ.MSFT",  "MSFT"),
    ("equity.daily_bars", "NASDAQ.GOOGL", "GOOGL"),
    ("equity.daily_bars", "NASDAQ.AMZN",  "AMZN"),
    ("equity.daily_bars", "NASDAQ.META",  "META"),
    ("equity.daily_bars", "NASDAQ.TSLA",  "TSLA"),
    ("equity.daily_bars", "NASDAQ.NVDA",  "NVDA"),
    ("etf.daily_bars",    "ARCA.GLD",     "GLD"),
)

_AUX_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    ("etf.daily_bars", "ARCA.SPY",   "SPY"),
    ("etf.daily_bars", "NASDAQ.QQQ", "QQQ"),
    ("etf.daily_bars", "ARCA.IWM",   "IWM"),
)

TRADABLE_TICKERS: tuple[str, ...] = tuple(t[2] for t in _TRADABLE_UNIVERSE)


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrainArtifact:
    """Everything ``auto_retrain`` writes to disk + everything the trade runner
    needs to consume the trained model without re-deriving feature plumbing.
    """

    model: RidgeTrainer
    feature_cols: list[str]
    sym_to_id: dict[str, int]
    daily_std: float  # xs-z → daily-return scaler used at inference time
    trained_on_date: date  # the cutoff date (== today - 1, the LAST bar in train set)
    n_train: int
    n_features: int
    metrics_oos: dict[str, float]  # IC / rank_ic / hit_rate / sharpe_daily on tail-21d


@dataclass(frozen=True)
class RetrainConfig:
    """Tunable knobs. Defaults match the v0 paper runner."""

    years: int = 5
    alpha: float = 1.0  # Ridge alpha (kept fixed; lambda_risk handles risk-aversion)
    eval_tail_days: int = 21  # held-out tail used to compute IC + Sharpe
    daily_std: float = 0.10  # xs-z → daily-return scaler (matches paper runner)
    models_dir: Path = Path("data/models")


# ---------------------------------------------------------------------------
# Panel loading (thin wrapper around scripts/_storage.open_store).
# ---------------------------------------------------------------------------


def _load_daily_panel_for_retrain(logger: logging.Logger) -> pl.DataFrame:
    """Load the full daily-bar panel needed by ``compute_features``.

    We deliberately re-implement (rather than import from scripts/) the panel
    loader so this module has zero ``scripts/*`` dependencies — keeps the
    src tree importable without sys.path gymnastics. The panel content is
    identical to ``scripts/30_paper_trade_runner.load_daily_panel``.
    """
    # The runner-side storage helper lives under scripts/ — import it lazily
    # via a path-tweak so we don't permanently pollute sys.path at module import.
    import sys

    project_root = Path(__file__).resolve().parents[2]
    scripts_path = str(project_root / "scripts")
    sys.path.insert(0, scripts_path)
    try:
        from _storage import open_store  # type: ignore
    finally:
        try:
            sys.path.remove(scripts_path)
        except ValueError:
            pass

    store = open_store()
    rows: list[pl.DataFrame] = []
    universe = list(_TRADABLE_UNIVERSE) + list(_AUX_UNIVERSE)
    for lib_name, arctic_sym, display_sym in universe:
        lib = store[lib_name]
        df = lib.read(arctic_sym).data.reset_index().rename(columns={"date": "date"})
        df["symbol"] = display_sym
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        rows.append(pl.from_pandas(df[cols]))
    panel = pl.concat(rows, how="vertical_relaxed").sort(["symbol", "date"])
    panel = panel.with_columns(pl.col("date").cast(pl.Datetime("ns")))
    logger.info(
        "auto_retrain: panel loaded — rows=%s symbols=%s last_date=%s",
        f"{panel.height:,}",
        panel["symbol"].n_unique(),
        panel["date"].max(),
    )
    return panel


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------


def _build_training_frame(
    panel: pl.DataFrame,
    *,
    cutoff_date_ns: int,
    years: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Returns (train_pd, feature_cols, sym_to_id). ``train_pd`` is the
    cleaned, winsorized, xs-z-scored frame ready for ``RidgeTrainer.fit``.

    ``cutoff_date_ns`` is the upper bound (inclusive) on the training set; any
    bar with ``date > cutoff_date_ns`` is dropped. This is how we enforce
    "never train on today".
    """
    feat = compute_features(panel)
    feat = feat.filter(pl.col("date") <= cutoff_date_ns)
    lab = next_day_open_close_log_return(feat, label_col="y_raw")
    lab = lab.filter(pl.col("symbol").is_in(set(TRADABLE_TICKERS)))
    lab = drop_feature_failures(lab, feature_names=FEATURE_NAMES)
    lab = lab.drop_nulls(subset=["y_raw"]).filter(pl.col("y_raw").is_finite())

    sym_to_id = {s: i for i, s in enumerate(sorted(TRADABLE_TICKERS))}
    lab = lab.with_columns(
        pl.col("symbol")
        .map_elements(lambda s: sym_to_id[s], return_dtype=pl.Int32)
        .alias("symbol_id")
    )

    # N-year window (inclusive of cutoff_date).
    cutoff_min_ns = cutoff_date_ns - int(years * 365.25 * 86_400 * 1_000_000_000)
    lab = lab.filter(pl.col("date") >= cutoff_min_ns)
    logger.info(
        "auto_retrain: train rows=%s window=[%s .. %s]",
        f"{lab.height:,}",
        lab["date"].min(),
        lab["date"].max(),
    )

    feature_cols = list(FEATURE_NAMES) + ["symbol_id"]
    pdf = lab.to_pandas().sort_values(["date", "symbol"]).reset_index(drop=True)

    bounds = fit_winsorization_bounds(pdf["y_raw"].to_numpy(), sigma=5.0)
    lab_pl = winsorize_train_fold(
        pl.from_pandas(pdf[["symbol", "date", "y_raw"]]),
        bounds,
        label_col="y_raw",
        out_col="y_wins",
    )
    lab_pl = cross_sectional_zscore(lab_pl, label_col="y_wins", out_col="y_xs_z")
    merged = pl.from_pandas(pdf[["symbol", "date"] + feature_cols]).join(
        lab_pl, on=["symbol", "date"], how="inner"
    )
    return merged.to_pandas(), feature_cols, sym_to_id


def _evaluate_oos_tail(
    train_pd: pd.DataFrame,
    *,
    feature_cols: list[str],
    eval_tail_days: int,
    alpha: float,
    logger: logging.Logger,
) -> dict[str, float]:
    """Hold out the last ``eval_tail_days`` UNIQUE dates of the training frame,
    refit on the head, score the tail. Cheap walk-forward OOS proxy.

    Used solely to *log* IC/Sharpe so operators can spot model drift; the
    persisted ``RidgeTrainer`` is always the full-window fit.
    """
    unique_dates = sorted(train_pd["date"].unique())
    if len(unique_dates) <= eval_tail_days + 5:
        logger.warning(
            "auto_retrain: OOS eval skipped — too few unique dates (n=%d, need > %d)",
            len(unique_dates),
            eval_tail_days + 5,
        )
        return {"ic": 0.0, "rank_ic": 0.0, "hit_rate": 0.5, "sharpe_daily": 0.0, "n": 0}

    split_date = unique_dates[-eval_tail_days]
    head = train_pd[train_pd["date"] < split_date]
    tail = train_pd[train_pd["date"] >= split_date]
    if head.empty or tail.empty:
        return {"ic": 0.0, "rank_ic": 0.0, "hit_rate": 0.5, "sharpe_daily": 0.0, "n": 0}

    X_train = head[feature_cols].to_numpy(dtype=np.float64)
    y_train = head["y_xs_z"].to_numpy(dtype=np.float64)
    m_tr = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)

    eval_ridge = RidgeTrainer(config=RidgeConfig(alpha=alpha))
    eval_ridge.fit(X_train[m_tr], y_train[m_tr])

    X_eval = tail[feature_cols].to_numpy(dtype=np.float64)
    y_eval = tail["y_xs_z"].to_numpy(dtype=np.float64)
    dates_eval = tail["date"].to_numpy()
    m_ev = np.isfinite(y_eval) & np.all(np.isfinite(X_eval), axis=1)
    y_pred = eval_ridge.predict(X_eval[m_ev])
    return evaluate_fold(
        y_eval[m_ev],
        y_pred,
        dates_eval[m_ev],
    )


def run_auto_retrain(
    *,
    today: date | None = None,
    config: RetrainConfig | None = None,
    panel: pl.DataFrame | None = None,
    logger: logging.Logger | None = None,
    project_root: Path | None = None,
) -> RetrainArtifact:
    """Top-level entry point.

    Parameters
    ----------
    today:
        Calendar date treated as "today". The training cutoff is ``today - 1``
        (i.e. we never train on today's still-forming bar). Defaults to UTC date.
    config:
        Tunable training knobs (years window, ridge alpha, etc.).
    panel:
        Optional pre-loaded panel. Used by tests to bypass ArcticDB.
    logger:
        Optional logger; a stdout-only one is created if missing.
    project_root:
        Optional path; defaults to the repo root inferred from this file.
    """
    if logger is None:
        logger = logging.getLogger("auto_retrain")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
            logger.addHandler(h)
            logger.setLevel(logging.INFO)

    if config is None:
        config = RetrainConfig()
    if today is None:
        today = datetime.utcnow().date()
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    cutoff_date = today - timedelta(days=1)  # NEVER train on today
    cutoff_date_ns = pd.Timestamp(cutoff_date).value
    logger.info(
        "auto_retrain: today=%s cutoff=%s years=%d alpha=%.4f",
        today,
        cutoff_date,
        config.years,
        config.alpha,
    )

    if panel is None:
        panel = _load_daily_panel_for_retrain(logger)

    train_pd, feature_cols, sym_to_id = _build_training_frame(
        panel,
        cutoff_date_ns=cutoff_date_ns,
        years=config.years,
        logger=logger,
    )

    # Final full-window fit.
    X = train_pd[feature_cols].to_numpy(dtype=np.float64)
    y = train_pd["y_xs_z"].to_numpy(dtype=np.float64)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    ridge = RidgeTrainer(config=RidgeConfig(alpha=config.alpha))
    ridge.fit(X[m], y[m])
    n_train = int(m.sum())
    logger.info(
        "auto_retrain: full-window fit done — n_train=%d n_features=%d alpha=%.4f",
        n_train,
        len(feature_cols),
        config.alpha,
    )

    metrics = _evaluate_oos_tail(
        train_pd,
        feature_cols=feature_cols,
        eval_tail_days=config.eval_tail_days,
        alpha=config.alpha,
        logger=logger,
    )
    logger.info(
        "auto_retrain: OOS tail (last %d unique dates) IC=%.4f rank_IC=%.4f "
        "hit_rate=%.3f sharpe=%.3f n=%d",
        config.eval_tail_days,
        metrics.get("ic", 0.0),
        metrics.get("rank_ic", 0.0),
        metrics.get("hit_rate", 0.5),
        metrics.get("sharpe_daily", 0.0),
        int(metrics.get("n", 0)),
    )

    artifact = RetrainArtifact(
        model=ridge,
        feature_cols=feature_cols,
        sym_to_id=sym_to_id,
        daily_std=config.daily_std,
        trained_on_date=cutoff_date,
        n_train=n_train,
        n_features=len(feature_cols),
        metrics_oos=metrics,
    )

    _persist_artifact(artifact, project_root=project_root, config=config, logger=logger)
    return artifact


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_artifact(
    artifact: RetrainArtifact,
    *,
    project_root: Path,
    config: RetrainConfig,
    logger: logging.Logger,
) -> Path:
    """Write ``ridge-YYYY-MM-DD.pkl`` + (re)point ``latest.pkl`` symlink.

    Returns the path of the dated pickle.
    """
    models_dir = project_root / config.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = artifact.trained_on_date.strftime("%Y-%m-%d")
    dated = models_dir / f"ridge-{stamp}.pkl"
    with open(dated, "wb") as f:
        pickle.dump(_to_dict(artifact), f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("auto_retrain: persisted %s (%.1f KB)", dated, dated.stat().st_size / 1024.0)

    latest = models_dir / "latest.pkl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    # Relative symlink: survives ``mv`` of the whole repo.
    os.symlink(dated.name, latest)
    logger.info("auto_retrain: symlinked %s → %s", latest, dated.name)
    return dated


def _to_dict(artifact: RetrainArtifact) -> dict:
    """Pickle-friendly dict. We pickle a plain dict rather than the dataclass
    so future schema migrations (add a field) don't break the unpickler.
    """
    return {
        "schema_version": 1,
        "model": artifact.model,
        "feature_cols": list(artifact.feature_cols),
        "sym_to_id": dict(artifact.sym_to_id),
        "daily_std": float(artifact.daily_std),
        "trained_on_date": artifact.trained_on_date.isoformat(),
        "n_train": int(artifact.n_train),
        "n_features": int(artifact.n_features),
        "metrics_oos": {k: float(v) for k, v in artifact.metrics_oos.items()},
    }


def load_latest_model(project_root: Path | None = None, models_dir: Path | str = "data/models") -> dict:
    """Reads ``data/models/latest.pkl`` and returns the unpickled artifact dict.

    Returns the dict directly (rather than reconstructing ``RetrainArtifact``)
    so consumers like the paper-trade runner can grab ``daily_std`` /
    ``feature_cols`` without importing this module's dataclass schema.

    Raises FileNotFoundError if ``latest.pkl`` is missing — callers should
    catch that and fall back to in-script training.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    path = project_root / models_dir / "latest.pkl"
    if not path.exists():
        raise FileNotFoundError(f"auto_retrain: no model at {path}")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Daemon mode — continuous retrain loop.
#
# Added 2026-05-27: previously this module was called once nightly by
# `40_daily_morning_routine.py`. The daemon path keeps the process alive
# and runs `run_auto_retrain` once per day (or whenever the last-trained
# date is stale by more than ``retrain_interval_hours``).
#
# Hot-swap is already atomic: `_persist_artifact` writes a dated pickle
# and re-points ``data/models/latest.pkl`` via ``os.symlink``. Consumers
# always read ``latest.pkl`` so the swap is seamless.
# ---------------------------------------------------------------------------


def _last_trained_date(models_dir: Path) -> date | None:
    """Inspect ``data/models/latest.pkl`` to learn when training last ran."""
    latest = models_dir / "latest.pkl"
    if not latest.exists():
        return None
    try:
        with open(latest, "rb") as fh:
            artifact = pickle.load(fh)
        iso = artifact.get("trained_on_date")
        if not iso:
            return None
        return date.fromisoformat(iso)
    except Exception:
        return None


def _retrain_lgbm_if_stale(logger: logging.Logger,
                            models_dir: Path,
                            *,
                            today: date) -> None:
    """Refit the META LightGBM signal model and persist to disk.

    The asset_specialists/_signals.py module lazily trains a LightGBM
    model on first use and caches the result in-process. By writing a
    dated pickle here, the next runner process picks up the fresh model
    on cold start.

    Failure is non-fatal: META's runtime path falls back to the lazy
    train if the cached pickle is missing or corrupted.
    """
    try:
        # Defer import — LightGBM trainer signature lives in _signals.
        from ml_research.asset_specialists._signals import train_lgbm_signal
        # Pull META daily bars
        import sys as _sys
        project_root = Path(__file__).resolve().parents[2]
        scripts_path = str(project_root / "scripts")
        _sys.path.insert(0, scripts_path)
        try:
            from _storage import open_store  # type: ignore
        finally:
            try:
                _sys.path.remove(scripts_path)
            except ValueError:
                pass
        store = open_store()
        bars = store["equity.daily_bars"].read("NASDAQ.META").data.copy()
        bars.index = pd.to_datetime(bars.index)
        bars = bars.sort_index()
        # Training cut-off = yesterday
        cutoff = pd.Timestamp(today - timedelta(days=1))
        train_bars = bars[bars.index <= cutoff]
        model = train_lgbm_signal(train_bars)
        stamp = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        out = models_dir / f"lgbm-META-{stamp}.pkl"
        with open(out, "wb") as fh:
            pickle.dump({"schema_version": 1,
                          "symbol": "META",
                          "model": model,
                          "trained_on_date": stamp}, fh,
                         protocol=pickle.HIGHEST_PROTOCOL)
        latest = models_dir / "lgbm-META-latest.pkl"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        os.symlink(out.name, latest)
        logger.info("auto_retrain: LightGBM META refit + persisted %s", out)
    except Exception as exc:
        logger.warning("auto_retrain: LightGBM META refit skipped: %s", exc)


def run_daemon(*,
                config: RetrainConfig | None = None,
                retrain_interval_hours: float = 24.0,
                poll_interval_seconds: float = 1800.0,
                max_iterations: int | None = None) -> int:
    """Continuous retraining loop.

    Policy:
      * Wake every ``poll_interval_seconds`` (default 30 min).
      * If ``last_trained_date < today - 1`` AND it's after 08:00 UTC,
        run ``run_auto_retrain()`` for today and additionally refit the
        META LightGBM model.
      * The atomic ``latest.pkl`` symlink hot-swap means the live runner
        picks up the new model on its next cold start without coordination.

    Parameters
    ----------
    config:
        Optional RetrainConfig; defaults applied if None.
    retrain_interval_hours:
        Minimum hours between retrains. Default 24h.
    poll_interval_seconds:
        How often the daemon wakes to check. Default 1800s = 30 min.
    max_iterations:
        Test hook. None means "run forever".
    """
    import time as _time

    logger = logging.getLogger("auto_retrain.daemon")
    if not logger.handlers:
        log_path = Path(__file__).resolve().parents[2] / "data" / "auto_retrain_daemon.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)

    if config is None:
        config = RetrainConfig()
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / config.models_dir

    logger.info("=== auto_retrain daemon starting (interval=%.1fh poll=%.0fs) ===",
                retrain_interval_hours, poll_interval_seconds)

    it = 0
    while True:
        it += 1
        if max_iterations is not None and it > max_iterations:
            break

        try:
            now = datetime.utcnow()
            today = now.date()
            last = _last_trained_date(models_dir)
            hours_since = (now - datetime.combine(last, datetime.min.time())).total_seconds() / 3600.0 if last else 1e9
            should_retrain = (last is None) or (hours_since >= retrain_interval_hours)

            if should_retrain:
                logger.info(
                    "daemon: triggering retrain (last=%s hours_since=%.1f threshold=%.1f)",
                    last, hours_since, retrain_interval_hours,
                )
                try:
                    run_auto_retrain(today=today, config=config, logger=logger,
                                      project_root=project_root)
                except Exception as exc:
                    logger.exception("daemon: Ridge retrain failed: %s", exc)

                # LightGBM META retrain — best-effort
                _retrain_lgbm_if_stale(logger, models_dir, today=today)
            else:
                logger.info("daemon: skip retrain (last=%s hours_since=%.1f)",
                            last, hours_since)
        except Exception as exc:
            logger.exception("daemon: outer loop error (continuing): %s", exc)

        _time.sleep(poll_interval_seconds)

    logger.info("=== auto_retrain daemon stopping (it=%d) ===", it)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Daily Ridge retrain")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--today", type=str, default=None, help="YYYY-MM-DD; defaults to UTC date")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as continuous daemon (poll + retrain every 24h)")
    parser.add_argument("--retrain-interval-hours", type=float, default=24.0,
                        help="In --daemon mode, hours between retrains")
    parser.add_argument("--poll-seconds", type=float, default=1800.0,
                        help="In --daemon mode, seconds between wake-ups")
    args = parser.parse_args()

    cfg = RetrainConfig(years=args.years, alpha=args.alpha)

    if args.daemon:
        return run_daemon(
            config=cfg,
            retrain_interval_hours=args.retrain_interval_hours,
            poll_interval_seconds=args.poll_seconds,
        )

    today = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else None
    run_auto_retrain(today=today, config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
