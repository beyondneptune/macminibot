"""75_ensemble_optimizer — try every plausible enhancement empirically.

Enhancements considered (each independent, can be combined):
  A. PER-ASSET FILTER       — fit LightGBM on TRAIN trades per asset, gate
                                TEST entries on classifier score. Threshold
                                picked PER ASSET by TRAIN P&L (no leakage).
  B. REGIME GATE            — only allow entries when SPY > 200d MA.
  C. PROFIT TARGET OVERLAY  — close position when MTM gain ≥ +X*sigma_log
                                (a sigma-anchored take-profit alongside the
                                trailing stop).
  D. VOL-TARGET SIZING      — scale per-name notional by 1 / sigma_log_21
                                (lower-vol names get more capital).

Run a 2x2x2x2 = 16 combo matrix, train on bars[:split], measure each combo's
TRAIN net (no look-ahead — TEST data never informs combo selection), then
report TEST net for each combo + the combo TRAIN said was best.

Run:
    .venv/bin/python scripts/75_ensemble_optimizer.py
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

import lightgbm as lgb  # noqa: E402

from ml_research.asset_specialists import (  # noqa: E402
    DEPLOYABLE_ASSETS, compute_signal_for, winner_for,
)
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from strategy_engine.per_asset_runner import default_bar_loader  # noqa: E402
from scripts._storage import open_store  # noqa: E402

WARMUP = 252
FEATURE_COLS = [
    "ret_1", "ret_5", "ret_21", "ret_63",
    "vol_21", "vol_63", "atr_14", "atr_norm",
    "rsi_2", "zscore_21",
    "macd", "macd_sig", "macd_hist",
]
THRESHOLDS_TO_TRY = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7]
PROFIT_TARGET_SIGMAS = [None, 3.0, 5.0]  # None = no PT; X = +X*entry_sigma


# -------------------- shared helpers ---------------------------------------


def _latest_sigma(feat_row: pd.Series) -> float | None:
    v = feat_row.get("vol_21", float("nan"))
    return float(v) if math.isfinite(v) and v > 0 else None


def load_spy_regime() -> pd.Series:
    """Return a Boolean series indexed by date: True iff SPY close > SPY 200d MA."""
    store = open_store()
    spy = store.get_library("etf.daily_bars").read("ARCA.SPY").data.copy()
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    ma200 = spy["close"].rolling(200, min_periods=200).mean()
    regime = (spy["close"] > ma200).fillna(False)
    return regime


# -------------------- replay engine ---------------------------------------


def precompute_signals(asset: str, bars: pd.DataFrame) -> np.ndarray:
    """Compute the signal once per bar; returns a numpy array of sig_dir
    aligned with bars.index. WARMUP entries are filled with 0."""
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    for i in range(WARMUP, n):
        try:
            s = compute_signal_for(asset, bars.iloc[: i + 1])
            out[i] = int(round(float(s)))
        except Exception:
            out[i] = 0
    return out


def replay(
    asset: str,
    bars: pd.DataFrame,
    feat_all: pd.DataFrame,
    sig_array: np.ndarray,
    start_idx: int,
    end_idx: int,
    *,
    base_notional: float,
    trail: float,
    filter_score_fn=None,
    filter_threshold: float = 0.0,
    regime_series: pd.Series | None = None,
    profit_target_sigmas: float | None = None,
    vol_target: bool = False,
    vol_target_anchor: float = 0.015,  # 1.5% daily vol = "median liquid stock"
) -> dict:
    """One-pass replay with optional enhancements. Uses precomputed sig_array."""
    if start_idx < WARMUP:
        start_idx = WARMUP
    state = "FLAT"
    entry_price = entry_sigma = entry_ts = None
    entry_qty = 0
    entry_features = None
    peak = None
    round_trips: list[dict] = []
    daily_pnl: list[tuple[pd.Timestamp, float]] = []
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    rejected_by_filter = 0
    rejected_by_regime = 0
    closes = bars["close"].astype(float).values
    idxs = bars.index

    for i in range(start_idx, end_idx):
        if i >= len(feat_all):
            break
        row = feat_all.iloc[i]
        sigma = _latest_sigma(row)
        if sigma is None:
            continue
        sig_dir = int(sig_array[i])
        close = float(closes[i])
        ts = idxs[i]

        if prev_close is not None and prev_state == "IN_LONG" and prev_qty > 0:
            daily_pnl.append((ts, (close - prev_close) * prev_qty))
        else:
            daily_pnl.append((ts, 0.0))

        # Maintain peak + check exits.
        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop_price = peak - band
            exit_reason = None
            # Trailing-stop hit?
            if close <= stop_price:
                exit_reason = "trailing_stop"
            # Profit target hit?
            elif profit_target_sigmas is not None:
                pt_threshold = entry_price * (1.0 + entry_sigma * profit_target_sigmas)
                if close >= pt_threshold:
                    exit_reason = "profit_target"
            # Signal flip?
            if exit_reason is None and sig_dir < 0:
                exit_reason = "signal_flip"
            if exit_reason is not None:
                pnl = (close - entry_price) * entry_qty
                round_trips.append({
                    "entry_ts": entry_ts, "exit_ts": ts, "pnl": pnl,
                    "hold_days": (ts - entry_ts).days,
                    "reason": exit_reason,
                    "features": entry_features,
                    "win": int(pnl > 0),
                })
                state = "FLAT"
                entry_price = entry_sigma = peak = entry_ts = None
                entry_qty = 0
                entry_features = None
                prev_state = "FLAT"
                prev_qty = 0
                prev_close = close
                continue

        # Consider new entry.
        if state == "FLAT" and sig_dir > 0:
            # Regime gate (SPY > 200d MA).
            if regime_series is not None:
                rg = regime_series.get(ts, None)
                if rg is None:
                    # try nearest prior date
                    prior = regime_series[regime_series.index <= ts]
                    rg = bool(prior.iloc[-1]) if not prior.empty else False
                if not bool(rg):
                    rejected_by_regime += 1
                    prev_close = close
                    prev_state = state
                    prev_qty = entry_qty
                    continue
            feats = {c: float(row.get(c, np.nan)) for c in FEATURE_COLS}
            # Filter gate.
            if filter_score_fn is not None and filter_threshold > 0:
                p_win = filter_score_fn(feats)
                if p_win < filter_threshold:
                    rejected_by_filter += 1
                    prev_close = close
                    prev_state = state
                    prev_qty = entry_qty
                    continue
            # Sizing.
            if vol_target and sigma > 0:
                size_mult = min(2.5, vol_target_anchor / sigma)
            else:
                size_mult = 1.0
            position_notional = base_notional * size_mult
            qty = int(position_notional // close) if close > 0 else 0
            if qty > 0:
                state = "IN_LONG"
                entry_price = close
                entry_sigma = sigma
                entry_ts = ts
                entry_qty = qty
                entry_features = feats
                peak = close

        prev_close = close
        prev_state = state
        prev_qty = entry_qty

    open_unrz = 0.0
    if state == "IN_LONG" and entry_price is not None:
        last_close = float(bars["close"].iloc[end_idx - 1])
        open_unrz = (last_close - entry_price) * entry_qty

    return {
        "round_trips": round_trips,
        "open_unrealized": open_unrz,
        "n_bars": end_idx - start_idx,
        "daily_pnl": daily_pnl,
        "rejected_by_filter": rejected_by_filter,
        "rejected_by_regime": rejected_by_regime,
    }


def stats(window: dict, notional_anchor: float) -> dict:
    rts = window["round_trips"]
    realized = sum(rt["pnl"] for rt in rts)
    net = realized + window["open_unrealized"]
    wins = [rt for rt in rts if rt["pnl"] > 0]
    win_rate = (len(wins) / len(rts) * 100.0) if rts else 0.0
    daily = window["daily_pnl"]
    sharpe = float("nan")
    max_dd = 0.0
    if daily and notional_anchor > 0:
        rets = np.asarray([pnl / notional_anchor for _, pnl in daily], dtype=float)
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum([pnl for _, pnl in daily])
        peak = np.maximum.accumulate(cum)
        max_dd = float((cum - peak).min())
    n_bars = window["n_bars"]
    years = n_bars / 252.0 if n_bars else 0.0
    ann_ret = float("nan")
    if years > 0 and notional_anchor + net > 0:
        ann_ret = ((1 + net / notional_anchor) ** (1 / years) - 1) * 100.0
    return {
        "n_trades": len(rts), "win_rate": win_rate,
        "net": net, "ann_return_pct": ann_ret, "sharpe": sharpe,
        "max_dd": max_dd, "years": years,
    }


def fit_classifier(trades: list[dict]) -> lgb.LGBMClassifier | None:
    X = pd.DataFrame([t["features"] for t in trades if t["features"] is not None])
    if X.empty:
        return None
    X = X[FEATURE_COLS]
    y = np.array([t["win"] for t in trades if t["features"] is not None], dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 50:
        return None
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=4,
        num_leaves=15, min_child_samples=20, subsample=0.85,
        colsample_bytree=0.85, random_state=42, verbose=-1,
    )
    model.fit(X, y)
    return model


# -------------------- main matrix run --------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2024-01-01")
    ap.add_argument("--net-liq", type=float, default=100000.0)
    args = ap.parse_args()

    notional = args.net_liq / 7.0
    split_ts = pd.Timestamp(args.split_date)
    regime = load_spy_regime()

    # Precompute per-asset histories, classifiers, train trade outcomes.
    asset_state: dict[str, dict] = {}
    for asset in DEPLOYABLE_ASSETS:
        winner = winner_for(asset)
        if winner is None:
            continue
        trail = float(winner["trail_sigma"])
        bars = default_bar_loader(asset)
        feat_all = add_per_asset_features(bars)
        split_idx = bars.index.get_indexer([split_ts], method="bfill")[0]
        if split_idx <= WARMUP or split_idx >= len(bars):
            continue

        # Precompute signals once (the bottleneck).
        print(f"  prep {asset}: precomputing signals ({len(bars)} bars)...", flush=True)
        sig_array = precompute_signals(asset, bars)

        # First-pass replay on TRAIN to collect labeled trades (no enhancements).
        train_w = replay(asset, bars, feat_all, sig_array, WARMUP, split_idx,
                         base_notional=notional, trail=trail)
        train_trades = [rt for rt in train_w["round_trips"]
                        if rt.get("features") is not None]
        clf = fit_classifier(train_trades) if len(train_trades) >= 50 else None

        asset_state[asset] = {
            "winner": winner, "trail": trail,
            "bars": bars, "feat": feat_all, "sig": sig_array,
            "split_idx": split_idx, "clf": clf,
        }
        print(f"  prep {asset}: train trades={len(train_trades)} "
              f"clf={'OK' if clf is not None else 'NONE'}", flush=True)

    # ---- Define the 16-cell enhancement matrix --------------------------
    combos = []
    for filt in [False, True]:
        for regime_on in [False, True]:
            for pt in PROFIT_TARGET_SIGMAS:
                for vt in [False, True]:
                    combos.append({
                        "filter": filt, "regime": regime_on,
                        "pt": pt, "vol_target": vt,
                    })

    print(f"\nMatrix: {len(combos)} configs × {len(asset_state)} assets, "
          f"TRAIN + TEST each = {2 * len(combos) * len(asset_state)} replays.\n")

    # Per asset, evaluate each combo on TRAIN, select the best by TRAIN net.
    # Then SEPARATELY evaluate each combo on TEST (we never let TEST inform
    # selection — but we report all combos on TEST for diagnosis).
    per_asset_results: dict[str, dict] = {}

    for asset, st in asset_state.items():
        winner = st["winner"]
        bars = st["bars"]
        feat = st["feat"]
        split_idx = st["split_idx"]
        trail = st["trail"]
        clf = st["clf"]

        # Build a score_fn around the classifier (or None).
        score_fn = None
        if clf is not None:
            def _score(feats, _m=clf):
                x = pd.DataFrame([{c: feats.get(c, np.nan)
                                    for c in FEATURE_COLS}])
                return float(_m.predict_proba(x)[:, 1][0])
            score_fn = _score

        # For configs using filter, also sweep filter thresholds.
        thresholds_for_filter = THRESHOLDS_TO_TRY if clf is not None else [0.0]

        sig_array = st["sig"]
        train_per_combo: list[dict] = []
        test_per_combo: list[dict] = []
        for c in combos:
            f_score_fn = score_fn if c["filter"] else None
            thr_list = thresholds_for_filter if c["filter"] else [0.0]
            for thr in thr_list:
                # TRAIN
                tw = replay(asset, bars, feat, sig_array, WARMUP, split_idx,
                            base_notional=notional, trail=trail,
                            filter_score_fn=f_score_fn,
                            filter_threshold=thr,
                            regime_series=regime if c["regime"] else None,
                            profit_target_sigmas=c["pt"],
                            vol_target=c["vol_target"])
                ts_train = stats(tw, notional)
                train_per_combo.append({**c, "thr": thr, **{f"train_{k}": v
                                                              for k, v in ts_train.items()}})
                # TEST
                tw_test = replay(asset, bars, feat, sig_array, split_idx, len(bars),
                                 base_notional=notional, trail=trail,
                                 filter_score_fn=f_score_fn,
                                 filter_threshold=thr,
                                 regime_series=regime if c["regime"] else None,
                                 profit_target_sigmas=c["pt"],
                                 vol_target=c["vol_target"])
                ts_test = stats(tw_test, notional)
                test_per_combo.append({**c, "thr": thr, **{f"test_{k}": v
                                                            for k, v in ts_test.items()}})

        # Choose the TRAIN-best per-asset combo. Use risk-adjusted train score:
        # net/(|maxDD|+1) so we don't pick a wild gambler.
        def score_train(r):
            return r["train_net"] / (abs(r["train_max_dd"]) + 1.0)

        train_best = max(train_per_combo, key=score_train)
        # Find the matching test entry for the same config.
        test_for_best = None
        for tr, te in zip(train_per_combo, test_per_combo):
            if all(tr[k] == train_best[k] for k in ("filter", "regime", "pt", "vol_target", "thr")):
                test_for_best = te
                break
        per_asset_results[asset] = {
            "winner_model": winner["model"], "trail": trail,
            "train_best_config": train_best,
            "test_at_train_best": test_for_best,
            "test_at_baseline": next(te for tr, te in zip(train_per_combo, test_per_combo)
                                     if not tr["filter"] and not tr["regime"]
                                     and tr["pt"] is None and not tr["vol_target"]
                                     and tr["thr"] == 0.0),
            "all_test": test_per_combo,
            "all_train": train_per_combo,
        }

    # ---- Report per-asset ------------------------------------------------
    print()
    print("=" * 130)
    print("  Per-asset: best config selected on TRAIN (risk-adj. score), evaluated on TEST")
    print("=" * 130)
    total_train_selected_test_net = 0.0
    total_baseline_test_net = 0.0
    for asset, r in per_asset_results.items():
        tb = r["train_best_config"]
        te = r["test_at_train_best"]
        bl = r["test_at_baseline"]
        total_train_selected_test_net += te["test_net"]
        total_baseline_test_net += bl["test_net"]
        cfg = (f"filter={'Y' if tb['filter'] else 'N'}"
               f"@thr={tb['thr']:.2f}" if tb["filter"] else "filter=N")
        cfg2 = f"regime={'Y' if tb['regime'] else 'N'}"
        cfg3 = (f"pt={tb['pt']}σ" if tb["pt"] is not None else "pt=N")
        cfg4 = f"voltgt={'Y' if tb['vol_target'] else 'N'}"
        delta = te["test_net"] - bl["test_net"]
        print(f"  {asset:<6} {r['winner_model']:<18} "
              f"trail={r['trail']}σ  picked=[{cfg}, {cfg2}, {cfg3}, {cfg4}]")
        print(f"        TRAIN: net=${tb['train_net']:>10,.0f}  "
              f"trades={tb['train_n_trades']:<5} "
              f"DD=${tb['train_max_dd']:>9,.0f}  "
              f"Sharpe={tb['train_sharpe']:+.2f}")
        print(f"        TEST : net=${te['test_net']:>10,.0f}  "
              f"trades={te['test_n_trades']:<5} "
              f"DD=${te['test_max_dd']:>9,.0f}  "
              f"Sharpe={te['test_sharpe']:+.2f}  "
              f"(baseline test net=${bl['test_net']:,.0f}; Δ=${delta:+,.0f})")

    print()
    print("=" * 130)
    print("  PORTFOLIO TEST P&L comparison")
    print("=" * 130)
    portfolio_notional = 7 * notional
    print(f"  Baseline (no enhancements):           ${total_baseline_test_net:+,.2f}  "
          f"({total_baseline_test_net/portfolio_notional*100:+.2f}%)")
    print(f"  TRAIN-selected per-asset best:        ${total_train_selected_test_net:+,.2f}  "
          f"({total_train_selected_test_net/portfolio_notional*100:+.2f}%)")
    delta_total = total_train_selected_test_net - total_baseline_test_net
    pct_improve = (delta_total / total_baseline_test_net * 100.0) if total_baseline_test_net else 0.0
    print(f"  IMPROVEMENT:                          ${delta_total:+,.2f}  "
          f"({pct_improve:+.2f}%)")

    # ---- Also show the BEST TEST combo per asset (informational — would be
    # look-ahead in production, but useful to know the ceiling).
    print()
    print("=" * 130)
    print("  Per-asset CEILING (best TEST net — informational; choosing here would be look-ahead)")
    print("=" * 130)
    ceiling = 0.0
    for asset, r in per_asset_results.items():
        best_test = max(r["all_test"], key=lambda x: x["test_net"])
        ceiling += best_test["test_net"]
        cfg = (f"filter={'Y' if best_test['filter'] else 'N'}@thr={best_test['thr']:.2f}"
               if best_test["filter"] else "filter=N")
        cfg2 = f"regime={'Y' if best_test['regime'] else 'N'}"
        cfg3 = (f"pt={best_test['pt']}σ" if best_test["pt"] is not None else "pt=N")
        cfg4 = f"voltgt={'Y' if best_test['vol_target'] else 'N'}"
        print(f"  {asset:<6} best TEST net=${best_test['test_net']:>10,.0f}  "
              f"config=[{cfg}, {cfg2}, {cfg3}, {cfg4}]")
    print(f"  CEILING (sum): ${ceiling:+,.2f}  "
          f"({ceiling/portfolio_notional*100:+.2f}%)  "
          f"— this is the theoretical max if we had a crystal ball.")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
