"""Unit tests for ``ml_research.auto_retrain``.

We build a synthetic daily-bar panel (all 8 tradables + 3 aux benchmarks)
covering ~3 years so ``compute_features`` has enough history for the 252d
features. No ArcticDB I/O. ``run_auto_retrain`` is then invoked with
``panel=<synthetic>`` to bypass the LMDB loader.
"""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml_research.auto_retrain import (
    RetrainConfig,
    TRADABLE_TICKERS,
    _AUX_UNIVERSE,
    load_latest_model,
    run_auto_retrain,
)


def _synthetic_panel(n_days: int = 900, seed: int = 7) -> pl.DataFrame:
    """Build a polars panel with (symbol, date, open, high, low, close, volume)
    rows for every (tradable + aux) symbol over ``n_days`` business days.

    Prices follow a simple GBM so realized-vol features are well-defined.
    """
    rng = np.random.default_rng(seed)
    end = date(2026, 5, 17)  # one day before our test "today" of 2026-05-18
    dates = []
    cursor = end
    while len(dates) < n_days:
        if cursor.weekday() < 5:  # M-F only
            dates.append(cursor)
        cursor -= timedelta(days=1)
    dates = list(reversed(dates))

    all_symbols = list(TRADABLE_TICKERS) + [t[2] for t in _AUX_UNIVERSE]
    rows = []
    for sym in all_symbols:
        mu, sigma = 0.0003, 0.015
        log_returns = rng.normal(mu, sigma, size=n_days)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        # Use close as a base; open/high/low jitter around it. Stay deterministic.
        opens = close * (1.0 + rng.normal(0.0, 0.002, size=n_days))
        highs = np.maximum(close, opens) * (1.0 + np.abs(rng.normal(0.0, 0.003, size=n_days)))
        lows = np.minimum(close, opens) * (1.0 - np.abs(rng.normal(0.0, 0.003, size=n_days)))
        vols = rng.integers(1_000_000, 50_000_000, size=n_days).astype(np.int64)
        for i, d in enumerate(dates):
            rows.append({
                "symbol": sym,
                "date": d,
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(close[i]),
                "volume": int(vols[i]),
            })
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("date").cast(pl.Datetime("ns")))
    return df.sort(["symbol", "date"])


def test_run_auto_retrain_writes_pickle_and_symlink(tmp_path: Path):
    panel = _synthetic_panel(n_days=900)
    today = date(2026, 5, 18)
    cfg = RetrainConfig(years=2, eval_tail_days=21)

    artifact = run_auto_retrain(
        today=today,
        config=cfg,
        panel=panel,
        project_root=tmp_path,
    )

    # cutoff_date is today - 1
    assert artifact.trained_on_date == today - timedelta(days=1)
    assert artifact.n_train > 0
    assert artifact.n_features > 0

    dated = tmp_path / "data" / "models" / f"ridge-{artifact.trained_on_date.isoformat()}.pkl"
    latest = tmp_path / "data" / "models" / "latest.pkl"
    assert dated.exists()
    assert latest.exists()
    assert latest.is_symlink()
    # Symlink target is relative (filename only) — survives moving the repo.
    assert Path("ridge-" + artifact.trained_on_date.isoformat() + ".pkl") == Path(latest.readlink())


def test_pickled_artifact_is_loadable(tmp_path: Path):
    panel = _synthetic_panel(n_days=700)
    today = date(2026, 5, 18)
    cfg = RetrainConfig(years=2, eval_tail_days=21)
    run_auto_retrain(today=today, config=cfg, panel=panel, project_root=tmp_path)

    loaded = load_latest_model(project_root=tmp_path)
    assert "model" in loaded
    assert "feature_cols" in loaded
    assert "sym_to_id" in loaded
    assert "daily_std" in loaded
    assert "trained_on_date" in loaded
    assert loaded["schema_version"] == 1
    assert isinstance(loaded["metrics_oos"], dict)
    assert {"ic", "sharpe_daily"}.issubset(loaded["metrics_oos"].keys())


def test_pickle_predict_round_trip(tmp_path: Path):
    """A pickled RidgeTrainer must produce identical predictions to the live one."""
    panel = _synthetic_panel(n_days=700)
    today = date(2026, 5, 18)
    cfg = RetrainConfig(years=2, eval_tail_days=21)
    artifact = run_auto_retrain(today=today, config=cfg, panel=panel, project_root=tmp_path)

    loaded = load_latest_model(project_root=tmp_path)
    n_feat = len(artifact.feature_cols)
    X = np.random.default_rng(0).standard_normal((5, n_feat))
    y_live = artifact.model.predict(X)
    y_pickled = loaded["model"].predict(X)
    np.testing.assert_allclose(y_live, y_pickled, rtol=1e-10)


def test_never_trains_on_today(tmp_path: Path):
    """The training cutoff must be today - 1 — no row at today's date can leak in."""
    panel = _synthetic_panel(n_days=600)
    today = date(2026, 5, 18)
    cfg = RetrainConfig(years=2, eval_tail_days=21)
    artifact = run_auto_retrain(today=today, config=cfg, panel=panel, project_root=tmp_path)
    assert artifact.trained_on_date < today
    assert artifact.trained_on_date == today - timedelta(days=1)


def test_oos_metrics_are_finite(tmp_path: Path):
    panel = _synthetic_panel(n_days=700)
    cfg = RetrainConfig(years=2, eval_tail_days=21)
    artifact = run_auto_retrain(today=date(2026, 5, 18), config=cfg, panel=panel, project_root=tmp_path)
    for k in ("ic", "rank_ic", "hit_rate", "sharpe_daily"):
        assert np.isfinite(artifact.metrics_oos[k]), f"{k} not finite: {artifact.metrics_oos}"


def test_symlink_repointed_on_second_run(tmp_path: Path):
    """Running on two consecutive days should leave both dated pickles + the
    symlink pointing at the newest one.
    """
    panel = _synthetic_panel(n_days=800)
    cfg = RetrainConfig(years=2, eval_tail_days=21)
    run_auto_retrain(today=date(2026, 5, 17), config=cfg, panel=panel, project_root=tmp_path)
    run_auto_retrain(today=date(2026, 5, 18), config=cfg, panel=panel, project_root=tmp_path)

    models = tmp_path / "data" / "models"
    assert (models / "ridge-2026-05-16.pkl").exists()
    assert (models / "ridge-2026-05-17.pkl").exists()
    latest = models / "latest.pkl"
    assert Path(latest.readlink()).name == "ridge-2026-05-17.pkl"


def test_artifact_schema_is_picklable_with_only_stdlib(tmp_path: Path):
    """A consumer that only has ``ml_research.trainer`` (not auto_retrain)
    must still be able to unpickle the artifact dict.
    """
    panel = _synthetic_panel(n_days=600)
    cfg = RetrainConfig(years=2, eval_tail_days=21)
    run_auto_retrain(today=date(2026, 5, 18), config=cfg, panel=panel, project_root=tmp_path)

    raw = (tmp_path / "data" / "models" / "latest.pkl").read_bytes()
    obj = pickle.loads(raw)
    assert isinstance(obj, dict)
    assert obj["schema_version"] == 1
