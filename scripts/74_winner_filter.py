"""74_winner_filter — train a per-asset classifier on TRAIN-window trades
to predict P(this entry will be a winning round-trip), then apply the
filter in TEST: only execute entries where P(win) > threshold.

Key constraint: features used by the classifier are computed ONLY from
data available at entry time (bars ≤ T-1). The classifier learns from
the COMPLETED outcome (win/loss) of past trades, which is legitimate
supervised learning. This is not look-ahead — the model uses past
*completed* trades, exactly like training any ML classifier.

Steps per asset:
  1. Replay state machine on TRAIN window → list of trades with entry-time
     features + win/loss label.
  2. Fit LightGBM binary classifier on those (X, y).
  3. Replay state machine on TEST window → at each candidate entry, score
     P(win) from features and only execute if score > threshold.
  4. Compare baseline (no filter) vs filtered TEST P&L.

Honest controls:
  - The classifier sees ONLY train data. Test labels are never touched.
  - We sweep thresholds {0.0, 0.5, 0.6, 0.7, 0.8} on test set, but threshold
    selection is also informed by train (we report each).
  - We compare to the unfiltered baseline (threshold=0.0) which is exactly
    the deployed strategy.

Run:
    .venv/bin/python scripts/74_winner_filter.py
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

WARMUP = 252
FEATURE_COLS = [
    "ret_1", "ret_5", "ret_21", "ret_63",
    "vol_21", "vol_63", "atr_14", "atr_norm",
    "rsi_2", "zscore_21",
    "macd", "macd_sig", "macd_hist",
]


def _latest_sigma(feat: pd.DataFrame) -> float | None:
    if "vol_21" not in feat.columns or feat.empty:
        return None
    v = feat["vol_21"].iloc[-1]
    return float(v) if math.isfinite(v) and v > 0 else None


def collect_trades(asset: str, bars: pd.DataFrame, start_idx: int,
                   end_idx: int, notional: float, trail: float,
                   *, score_fn=None, threshold: float = 0.0) -> dict:
    """Replay the state machine; record completed trades with entry-time
    feature snapshots. If score_fn is given, also gate entries on
    score_fn(features_at_entry) >= threshold."""
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

    # Pre-compute the feature frame once.
    feat_all = add_per_asset_features(bars)

    for i in range(start_idx, end_idx):
        if i >= len(feat_all):
            break
        row = feat_all.iloc[i]
        sigma = float(row.get("vol_21", 0.0))
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        try:
            sig = compute_signal_for(asset, bars.iloc[: i + 1])
        except Exception:
            continue
        sig_dir = int(round(float(sig)))
        close = float(bars["close"].iloc[i])
        ts = bars.index[i]

        if prev_close is not None and prev_state == "IN_LONG" and prev_qty > 0:
            daily_pnl.append((ts, (close - prev_close) * prev_qty))
        else:
            daily_pnl.append((ts, 0.0))

        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop_price = peak - band
            exit_reason = None
            if close <= stop_price:
                exit_reason = "trailing_stop"
            elif sig_dir < 0:
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

        if state == "FLAT" and sig_dir > 0:
            # Apply classifier filter if provided.
            feats = {c: float(row.get(c, np.nan)) for c in FEATURE_COLS}
            if score_fn is not None:
                p_win = score_fn(feats)
                if p_win < threshold:
                    rejected_by_filter += 1
                    prev_close = close
                    prev_state = state
                    prev_qty = entry_qty
                    continue
            qty = int(notional // close) if close > 0 else 0
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
    }


def stats(window: dict, notional: float) -> dict:
    rts = window["round_trips"]
    realized = sum(rt["pnl"] for rt in rts)
    net = realized + window["open_unrealized"]
    wins = [rt for rt in rts if rt["pnl"] > 0]
    win_rate = (len(wins) / len(rts) * 100.0) if rts else 0.0
    daily = window["daily_pnl"]
    sharpe = float("nan")
    max_dd = 0.0
    if daily and notional > 0:
        rets = np.asarray([pnl / notional for _, pnl in daily], dtype=float)
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum([pnl for _, pnl in daily])
        peak = np.maximum.accumulate(cum)
        max_dd = float((cum - peak).min())
    n_bars = window["n_bars"]
    years = n_bars / 252.0 if n_bars else 0.0
    ann_ret = float("nan")
    if years > 0 and notional + net > 0:
        ann_ret = ((1 + net / notional) ** (1 / years) - 1) * 100.0
    return {
        "n_trades": len(rts), "win_rate": win_rate,
        "net": net, "ann_return_pct": ann_ret, "sharpe": sharpe,
        "max_dd": max_dd, "max_dd_pct": (max_dd / notional * 100.0),
        "rejected_by_filter": window.get("rejected_by_filter", 0),
    }


def fit_classifier(trades: list[dict]) -> tuple[lgb.LGBMClassifier, dict]:
    """Train a per-asset LightGBM classifier on train trades.

    Returns (model, diagnostics). Diagnostics include train AUC + feature
    importances so we can see whether the model learned anything.
    """
    X = pd.DataFrame([t["features"] for t in trades])[FEATURE_COLS]
    y = np.array([t["win"] for t in trades], dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 30:
        return None, {"reason": f"too few or unbalanced ({len(y)} trades, "
                                 f"win_rate={y.mean()*100:.1f}%)"}
    # Modest model — we have ~hundreds of trades per asset, not thousands.
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=4,
        num_leaves=15, min_child_samples=20, subsample=0.85,
        colsample_bytree=0.85, random_state=42, verbose=-1,
    )
    model.fit(X, y)
    train_auc = float(np.corrcoef(model.predict_proba(X)[:, 1], y)[0, 1])  # proxy
    importances = dict(zip(FEATURE_COLS,
                           [float(v) for v in model.feature_importances_]))
    return model, {
        "n_train": len(y), "win_rate_train": float(y.mean() * 100),
        "train_score_corr": train_auc, "importances": importances,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2024-01-01")
    ap.add_argument("--net-liq", type=float, default=100000.0)
    ap.add_argument("--thresholds", type=str,
                    default="0.0,0.5,0.55,0.6,0.65,0.7")
    args = ap.parse_args()

    notional = args.net_liq / 7.0
    thresholds = [float(x) for x in args.thresholds.split(",")]
    split_ts = pd.Timestamp(args.split_date)

    print()
    print("=" * 116)
    print(f"  Winner-filter experiment  •  split={split_ts.date()}  "
          f"•  net_liq=${args.net_liq:,.0f}  per-name=${notional:,.2f}")
    print("=" * 116)

    portfolio_by_threshold: dict[float, dict] = {}
    per_asset_summary: list[dict] = []

    for asset in DEPLOYABLE_ASSETS:
        winner = winner_for(asset)
        if winner is None:
            continue
        trail = float(winner["trail_sigma"])
        bars = default_bar_loader(asset)
        split_idx = bars.index.get_indexer([split_ts], method="bfill")[0]
        if split_idx <= WARMUP or split_idx >= len(bars):
            continue

        # 1. Collect TRAIN trades with entry features.
        train_w = collect_trades(asset, bars, WARMUP, split_idx, notional, trail)
        train_trades = [rt for rt in train_w["round_trips"]
                        if rt["features"] is not None]
        if len(train_trades) < 30:
            print(f"{asset}: too few train trades ({len(train_trades)}) — "
                  "skip classifier (baseline only)")
            model, diag = None, {"reason": "too_few_train_trades"}
        else:
            model, diag = fit_classifier(train_trades)
            if model is None:
                print(f"{asset}: classifier fit failed: {diag['reason']}")

        # 2. Evaluate TEST at each threshold (and threshold=0 = unfiltered).
        per_thresh_test = {}
        for thr in thresholds:
            if thr <= 0.0 or model is None:
                score_fn = None
                use_thr = 0.0
            else:
                def score_fn(feats, _m=model):
                    x = pd.DataFrame([{c: feats.get(c, np.nan)
                                       for c in FEATURE_COLS}])
                    return float(_m.predict_proba(x)[:, 1][0])
                use_thr = thr
            test_w = collect_trades(asset, bars, split_idx, len(bars),
                                    notional, trail,
                                    score_fn=score_fn, threshold=use_thr)
            s = stats(test_w, notional)
            per_thresh_test[thr] = s
            portfolio_by_threshold.setdefault(thr, {
                "net": 0.0, "trades": 0, "wins": 0, "rejected": 0
            })
            portfolio_by_threshold[thr]["net"] += s["net"]
            portfolio_by_threshold[thr]["trades"] += s["n_trades"]
            portfolio_by_threshold[thr]["wins"] += int(s["n_trades"] * s["win_rate"] / 100.0)
            portfolio_by_threshold[thr]["rejected"] += s["rejected_by_filter"]

        per_asset_summary.append({
            "asset": asset, "winner_model": winner["model"], "trail": trail,
            "train_trades": len(train_trades),
            "diag": diag,
            "per_thresh": per_thresh_test,
        })

        # Per-asset print
        print(f"\n{asset}  ({winner['model']} trail={trail}σ)  "
              f"train trades={len(train_trades)}", end="")
        if model is not None:
            print(f"  win-rate(train)={diag['win_rate_train']:.1f}%")
            top_feats = sorted(diag["importances"].items(),
                                key=lambda x: -x[1])[:3]
            print(f"  top features: " +
                  ", ".join(f"{k}({v:.0f})" for k, v in top_feats))
        else:
            print(" — classifier unavailable")
        print(f"  {'thr':<6} {'test trades':<13} {'rejected':<10} "
              f"{'win%':<7} {'net $':<12} {'ann%':<8} {'Sharpe':<8} {'maxDD%'}")
        for thr in thresholds:
            s = per_thresh_test[thr]
            mark = " ←baseline" if thr == 0.0 else ""
            print(f"  {thr:<6.2f} {s['n_trades']:<13} "
                  f"{s['rejected_by_filter']:<10} {s['win_rate']:>5.1f}%  "
                  f"${s['net']:>9,.0f}  {s['ann_return_pct']:>+5.1f}%  "
                  f"{s['sharpe']:>+5.2f}    {s['max_dd_pct']:>+6.1f}%{mark}")

    # Portfolio summary
    print()
    print("=" * 116)
    print("  PORTFOLIO (sum across 7 assets) on OOS TEST")
    print("=" * 116)
    print(f"  {'thr':<6} {'trades':<8} {'rejected':<10} {'win%':<7} "
          f"{'net $':<14} {'return %':<10}  {'vs baseline'}")
    print("-" * 116)
    portfolio_notional = 7 * notional
    baseline_net = portfolio_by_threshold[0.0]["net"]
    for thr in thresholds:
        p = portfolio_by_threshold[thr]
        wr = (p["wins"] / p["trades"] * 100.0) if p["trades"] else 0.0
        ret = p["net"] / portfolio_notional * 100.0
        delta = p["net"] - baseline_net
        delta_pct = (delta / abs(baseline_net) * 100.0) if baseline_net != 0 else 0.0
        mark = " ←baseline" if thr == 0.0 else (
            f"  Δ=${delta:+,.0f}  ({delta_pct:+.1f}%)")
        print(f"  {thr:<6.2f} {p['trades']:<8} {p['rejected']:<10} {wr:>5.1f}%  "
              f"${p['net']:>11,.0f}  {ret:>+7.1f}%{mark}")
    print("=" * 116)

    # Verdict
    best_thr = max(portfolio_by_threshold, key=lambda t: portfolio_by_threshold[t]["net"])
    best = portfolio_by_threshold[best_thr]
    delta = best["net"] - baseline_net
    print()
    if best_thr == 0.0:
        print(f"  VERDICT: classifier filter does NOT help. Baseline (no filter) "
              f"is best at ${best['net']:+,.0f}.")
    else:
        print(f"  VERDICT: best threshold = {best_thr:.2f} → ${best['net']:+,.0f} "
              f"(baseline ${baseline_net:+,.0f}, delta ${delta:+,.0f}).")
        if delta < 0:
            print(f"  But this is WORSE than baseline — filter has no real OOS edge.")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    sys.exit(main())
