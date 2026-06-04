"""ml_research.trainer — LightGBM (primary) + Ridge (sanity floor).

Per CONTEXT.md Key Decisions:
    - Primary: LightGBM, pooled cross-sectionally with contract-ID one-hot,
      objective=regression_l1 (heavy-tailed daily return residuals).
    - Sanity floor: Ridge regression. Every LightGBM model must beat Ridge
      OOS Sharpe per fold or it's rejected as overfit.
    - LightGBM defaults: num_leaves=31, lr=0.05, n_estimators=500.

Both trainers expose a uniform `fit(X, y, sample_weight, categorical_idx) →
predict(X) → np.ndarray` surface so they're swappable from the train loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lightgbm is required: .venv/bin/pip install lightgbm"
    ) from e

from sklearn.linear_model import Ridge


@dataclass
class LightGBMConfig:
    """LightGBM hyperparameters. Defaults from CONTEXT line in deliverable spec."""

    objective: str = "regression_l1"  # CONTEXT line 74 — load-bearing
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    lambda_l2: float = 0.0
    verbose: int = -1
    n_jobs: int = -1  # M4: 10 cores available; let LGBM use them all
    early_stopping_rounds: int | None = 50

    def to_lgb_params(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "metric": "mae",
            "force_col_wise": True,
        }


@dataclass
class LightGBMTrainer:
    """LightGBM regressor with per-fold fit/predict."""

    config: LightGBMConfig = field(default_factory=LightGBMConfig)
    model: lgb.Booster | None = None
    feature_names: list[str] | None = None
    categorical_feature: list[int] | str = "auto"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        categorical_feature: list[int] | str | None = None,
    ) -> LightGBMTrainer:
        params = self.config.to_lgb_params()
        if feature_names is not None:
            self.feature_names = feature_names
        if categorical_feature is not None:
            self.categorical_feature = categorical_feature

        train_set = lgb.Dataset(
            X,
            label=y,
            weight=sample_weight,
            feature_name=self.feature_names if self.feature_names else "auto",
            categorical_feature=self.categorical_feature,
            free_raw_data=False,
        )
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=0)]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_set,
                feature_name=self.feature_names if self.feature_names else "auto",
                categorical_feature=self.categorical_feature,
                free_raw_data=False,
            )
            valid_sets.append(val_set)
            valid_names.append("val")
            if self.config.early_stopping_rounds:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=self.config.early_stopping_rounds,
                        verbose=False,
                    )
                )

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBMTrainer: predict() called before fit()")
        return np.asarray(self.model.predict(X), dtype=np.float64)


@dataclass
class RidgeConfig:
    alpha: float = 1.0
    fit_intercept: bool = True


@dataclass
class RidgeTrainer:
    """Sanity-floor Ridge regressor. LightGBM must beat this OOS per fold."""

    config: RidgeConfig = field(default_factory=RidgeConfig)
    model: Ridge | None = None
    feature_means: np.ndarray | None = None
    feature_stds: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        **_ignored: Any,
    ) -> RidgeTrainer:
        # Standardize features so the L2 penalty has equal pull across scales.
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0, ddof=0)
        self.feature_stds[self.feature_stds == 0.0] = 1.0
        Xs = (X - self.feature_means) / self.feature_stds
        self.model = Ridge(alpha=self.config.alpha, fit_intercept=self.config.fit_intercept)
        self.model.fit(Xs, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.feature_means is None:
            raise RuntimeError("RidgeTrainer: predict() called before fit()")
        Xs = (X - self.feature_means) / self.feature_stds
        return np.asarray(self.model.predict(Xs), dtype=np.float64)


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    *,
    bps_per_unit: float = 1.0,
) -> dict[str, float]:
    """Compute per-fold OOS metrics.

    Returns dict with:
        ic              — Pearson corr(y_pred, y_true), pooled across all rows
        rank_ic         — Spearman rank corr, pooled
        hit_rate        — P(sign(y_pred) == sign(y_true))
        sharpe_daily    — Sharpe of a daily long-short portfolio sized by
                          sign(prediction) (annualized assuming 252 days)
    """
    # Strip non-finite.
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"ic": 0.0, "rank_ic": 0.0, "hit_rate": 0.5, "sharpe_daily": 0.0, "n": int(mask.sum())}
    yt = y_true[mask]
    yp = y_pred[mask]
    d = dates[mask]

    # Pooled IC.
    ic = float(np.corrcoef(yp, yt)[0, 1]) if yt.std() > 0 and yp.std() > 0 else 0.0
    # Rank IC (Spearman) — rank then Pearson.
    yt_rank = np.argsort(np.argsort(yt))
    yp_rank = np.argsort(np.argsort(yp))
    rank_ic = float(np.corrcoef(yp_rank, yt_rank)[0, 1]) if yt.size > 1 else 0.0
    # Hit rate.
    hit_rate = float(np.mean(np.sign(yp) == np.sign(yt)))

    # Daily long-short Sharpe: per-day, take sign(yp) * y_true, average.
    df_daily = {}
    for date, true_v, pred_v in zip(d, yt, yp):
        df_daily.setdefault(date, []).append(np.sign(pred_v) * true_v)
    daily_pnl = np.array([np.mean(v) for v in df_daily.values()])
    if daily_pnl.std(ddof=0) > 0:
        sharpe = float(np.mean(daily_pnl) / np.std(daily_pnl, ddof=0) * np.sqrt(252.0))
    else:
        sharpe = 0.0

    return {
        "ic": ic,
        "rank_ic": rank_ic,
        "hit_rate": hit_rate,
        "sharpe_daily": sharpe,
        "n": int(mask.sum()),
        "n_days": len(df_daily),
    }
