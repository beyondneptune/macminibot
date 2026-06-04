"""40_daily_morning_routine — pre-trade orchestrator (09:00 ET / 14:00 BST).

Sequence:
    1. ``auto_retrain``  → fresh Ridge model on the most recent N-year window
                           (training cutoff = today - 1; we NEVER train on today).
                           Pickled to data/models/ridge-YYYY-MM-DD.pkl;
                           data/models/latest.pkl symlinked.
    2. ``auto_tune``     → recalibrate (lambda_risk, target_vol_annualized) so
                           the MV optimizer actually trades on signal.
                           Persisted to data/config/strategy_tuning.json.
    3. Write a one-line readiness report to data/morning_routine.log.
    4. Exit.

Then 35 minutes later (09:35 ET) ``scripts/30_paper_trade_runner.py`` picks
up the freshly-pickled model + tuning record.

Run manually:
    .venv/bin/python scripts/40_daily_morning_routine.py
Schedule via launchd or cron at 09:00 ET (14:00 BST) on trading days.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("morning_routine")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _calibration_inputs_from_panel(
    panel, sym_to_id: dict[str, int], feature_cols: list[str], ridge, daily_std: float, logger
):
    """Build (r_hat, Sigma_daily, w_prev) on the calibration day = the latest
    bar in ``panel`` (which by construction is ``today - 1`` because auto_retrain
    already enforces the cutoff upstream).
    """
    import numpy as np
    import polars as pl
    import pandas as pd

    from ml_research.features import FEATURE_NAMES, compute_features, drop_feature_failures
    from strategy_engine.covariance import ledoit_wolf_shrunk

    # Use the same tradable list as auto_retrain (single source of truth).
    from ml_research.auto_retrain import TRADABLE_TICKERS

    feat = compute_features(panel)
    feat = drop_feature_failures(feat, feature_names=FEATURE_NAMES)
    feat = feat.filter(pl.col("symbol").is_in(set(TRADABLE_TICKERS)))
    feat = feat.with_columns(
        pl.col("symbol")
        .map_elements(lambda s: sym_to_id[s], return_dtype=pl.Int32)
        .alias("symbol_id")
    )
    latest = feat.sort(["symbol", "date"]).group_by("symbol").tail(1).sort("symbol")
    latest_pd = latest.to_pandas().set_index("symbol").loc[list(sorted(TRADABLE_TICKERS))]
    X = latest_pd[feature_cols].to_numpy(dtype=np.float64)
    preds_z = ridge.predict(X)
    r_hat = preds_z * daily_std

    # Daily-bar returns matrix → Σ via Ledoit-Wolf shrink.
    df = panel.filter(pl.col("symbol").is_in(set(TRADABLE_TICKERS)))
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol")).log().alias("ret")
    ).drop_nulls(subset=["ret"])
    pdf = df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    wide = pdf.pivot(index="date", columns="symbol", values="ret").dropna()
    wide = wide.tail(252)
    # Reorder columns to match r_hat ordering (sorted tradable tickers).
    cols = list(sorted(TRADABLE_TICKERS))
    missing = [c for c in cols if c not in wide.columns]
    if missing:
        logger.warning("morning_routine: missing return cols for %s — dropping", missing)
        cols = [c for c in cols if c in wide.columns]
    wide = wide[cols]
    shrink = ledoit_wolf_shrunk(wide)
    Sigma = np.asarray(shrink.sigma, dtype=np.float64)

    # Restrict r_hat to symbols present in Sigma.
    if len(cols) != len(latest_pd):
        idx = [list(sorted(TRADABLE_TICKERS)).index(c) for c in cols]
        r_hat = r_hat[idx]

    # Flat book on calibration day (the runner's prev_positions on a fresh
    # paper account is empty; the calibration is "what would today's
    # optimizer do given a flat book?"). This is the right proxy because the
    # tuning is forward-looking.
    w_prev = np.zeros_like(r_hat)
    return r_hat, Sigma, w_prev


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily pre-trade morning routine")
    parser.add_argument("--today", type=str, default=None,
                        help="YYYY-MM-DD; defaults to UTC date. The cutoff for "
                             "training is always today - 1.")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--skip-tune", action="store_true",
                        help="Run retrain only; useful for backfills.")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Run auto_tune only; reuses data/models/latest.pkl.")
    args = parser.parse_args()

    log_path = PROJECT_ROOT / "data" / "morning_routine.log"
    logger = _setup_logger(log_path)

    today = (
        datetime.strptime(args.today, "%Y-%m-%d").date()
        if args.today
        else datetime.now(timezone.utc).date()
    )
    logger.info("=" * 72)
    logger.info("morning_routine: start (today=%s years=%d)", today, args.years)
    logger.info("=" * 72)

    # Late imports — keep the script import-light so --help is snappy.
    from ml_research.auto_retrain import (
        RetrainConfig,
        load_latest_model,
        run_auto_retrain,
    )
    from strategy_engine.auto_tune import (
        persist_tuning_record,
        run_auto_tune,
    )

    try:
        # ---- Step 1: retrain ----
        if not args.skip_retrain:
            artifact = run_auto_retrain(
                today=today,
                config=RetrainConfig(years=args.years),
                logger=logger,
                project_root=PROJECT_ROOT,
            )
            ridge = artifact.model
            feature_cols = artifact.feature_cols
            sym_to_id = artifact.sym_to_id
            daily_std = artifact.daily_std
            metrics_oos = artifact.metrics_oos
            trained_on = artifact.trained_on_date
        else:
            logger.info("morning_routine: --skip-retrain set, loading latest model")
            d = load_latest_model(project_root=PROJECT_ROOT)
            ridge = d["model"]
            feature_cols = d["feature_cols"]
            sym_to_id = d["sym_to_id"]
            daily_std = d["daily_std"]
            metrics_oos = d.get("metrics_oos", {})
            trained_on = date.fromisoformat(d["trained_on_date"])

        # ---- Step 2: auto_tune ----
        tune_summary: dict[str, float | bool | str] = {}
        if not args.skip_tune:
            # We need the same panel auto_retrain used. Reload it (cheap — LMDB).
            from ml_research.auto_retrain import _load_daily_panel_for_retrain
            panel = _load_daily_panel_for_retrain(logger)
            r_hat, Sigma, w_prev = _calibration_inputs_from_panel(
                panel, sym_to_id, feature_cols, ridge, daily_std, logger
            )
            tune_result = run_auto_tune(
                r_hat=r_hat,
                Sigma_daily=Sigma,
                w_prev=w_prev,
                as_of_date=today,
                logger=logger,
            )
            persist_tuning_record(tune_result, project_root=PROJECT_ROOT)
            tune_summary = {
                "lambda_risk": tune_result.lambda_risk,
                "target_vol_annualized": tune_result.target_vol_annualized,
                "linf_at_lambda": tune_result.linf_at_lambda,
                "clamped_lambda": tune_result.clamped_lambda,
                "clamped_target_vol": tune_result.clamped_target_vol,
                "notes": tune_result.notes,
            }
        else:
            logger.info("morning_routine: --skip-tune set, skipping auto_tune")

        # ---- Step 3: one-line readiness report ----
        ic = float(metrics_oos.get("ic", 0.0)) if metrics_oos else 0.0
        sharpe = float(metrics_oos.get("sharpe_daily", 0.0)) if metrics_oos else 0.0
        lam = tune_summary.get("lambda_risk", "n/a")
        tv = tune_summary.get("target_vol_annualized", "n/a")
        logger.info(
            "READINESS: trained_on=%s ic=%.4f sharpe=%.3f lambda_risk=%s target_vol=%s",
            trained_on, ic, sharpe,
            f"{lam:.4f}" if isinstance(lam, float) else lam,
            f"{tv:.4f}" if isinstance(tv, float) else tv,
        )
        logger.info("morning_routine: done — exit 0")
        return 0
    except Exception:
        logger.error("morning_routine: FAILED")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
