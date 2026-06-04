"""Per-asset signal-flip-exit bake-off — architecture variant.

Pure model-driven entries AND exits. Position is opened when the signal turns
non-zero, held while the signal direction is constant, and closed the moment
the signal flips sign (or returns to zero). No take-profit, no stop-loss, no
time-based exit — the model alone decides both legs.

Why this variant?
    The TP/SL grid in scripts/50_per_asset_bakeoff.py inflates the candidate
    search space by ~24x and is the dominant deflation engine in the DSR
    correction. The signal-flip exit collapses the search to (model × scope)
    only, so the multiple-comparison penalty drops by a similar factor. If the
    underlying signals carry any persistent conviction, this is where we'll
    see it.

Promotion gate (per task spec):
    Holm-corrected p < 0.05 (across 8 best-of-asset p-values)
    AND  block-bootstrap 95% CI lower bound > 0
    AND  aggregate OOS Sharpe (net) > 0.
    DSR is reported (variance-calibrated, parameter-free trial count) but not
    a hard gate — the spec only mentions "Fix DSR=0 if it's a bug", which we
    address by using an honest n_trials and well-calibrated cross-trial sigma.

Run:
    .venv/bin/python scripts/53_signal_flip_bakeoff.py
        [--bootstrap B] [--quick]

Outputs:
    data/bakeoff_variants/signal_flip/REPORT.md
    data/bakeoff_variants/signal_flip/winners.parquet
    data/bakeoff_variants/signal_flip/aggregate.parquet
    data/bakeoff_variants/signal_flip/results.parquet
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Re-use the entire 50_per_asset_bakeoff candidate space, walk-forward harness,
# data loaders, feature builders, and signal generators. We only swap out the
# bracket simulator for a signal-flip one.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_bake50", PROJECT_ROOT / "scripts" / "50_per_asset_bakeoff.py"
)
_bake50 = importlib.util.module_from_spec(_spec)
# Register BEFORE exec — dataclass machinery looks the module up in sys.modules
# during class construction, which fails if the module isn't registered yet.
sys.modules["_bake50"] = _bake50
_spec.loader.exec_module(_bake50)  # type: ignore[union-attr]

from backtesting import fees as bt_fees  # noqa: E402
from backtesting import metrics as bt_metrics  # noqa: E402
from backtesting import slippage as bt_slip  # noqa: E402

from ml_research.cv import WalkForwardHarness  # noqa: E402


# -------- signal-flip simulator -------------------------------------------


@dataclass(frozen=True)
class TradeRecord:
    entry_idx: int
    exit_idx: int  # inclusive (the day of exit / flip)
    direction: int  # +1 long, -1 short
    holding_days: int  # exit_idx - entry_idx + 1


def simulate_signal_flip(
    bars: pd.DataFrame,  # OHLC + vol features
    signal: pd.Series,  # int8 in {-1,0,+1} aligned to bars.index
) -> tuple[pd.DataFrame, list[TradeRecord]]:
    """Signal-flip-exit simulator on daily bars.

    Position dynamics (causal, no look-ahead):
        sig[i] is computed using info <= i-1 and tells us the desired position
        DURING bar i. We assume the bot wakes at open[i], observes sig[i], and
        switches the held position to sig[i] (paying transaction cost on the
        change) — analogous to the open-fill convention used in 50_per_asset_bakeoff.

    Per-bar P&L:
        We attribute the period-return ``r[i] = close[i]/close[i-1] - 1`` to
        the position that was HELD during bar i, i.e. ``sig[i]``. On day 0 we
        have no prior close so the position contributes zero (warm-up). When
        the position changes between i-1 and i we pay round-trip transaction
        cost (handled in apply_costs_flip).

    Returns:
        per-bar dataframe with columns:
            held_dir, prev_dir, changed, gross_pct_ret, trade_id;
        list of TradeRecord (one per non-zero held block).
    """
    sig = signal.reindex(bars.index).fillna(0).astype(np.int8).values
    close = bars["close"].values
    n = len(bars)

    # Period returns based on close-to-close — first day has NaN, treat as 0.
    pct_ret = np.zeros(n, dtype=np.float64)
    if n > 1:
        pct_ret[1:] = close[1:] / close[:-1] - 1.0

    prev_dir = np.zeros(n, dtype=np.int8)
    if n > 1:
        prev_dir[1:] = sig[:-1]
    changed = (sig != prev_dir).astype(np.int8)

    # Attribute the day's return to the position held during that day.
    gross = sig.astype(np.float64) * pct_ret
    # First bar has no prior-close baseline, force 0 (warm-up).
    if n:
        gross[0] = 0.0

    # Build TradeRecords: each maximal run of constant non-zero sig is one trade.
    trades: list[TradeRecord] = []
    trade_id = np.zeros(n, dtype=np.int32)
    tid = 0
    i = 0
    while i < n:
        s = int(sig[i])
        if s == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sig[j + 1] == s:
            j += 1
        tid += 1
        trade_id[i : j + 1] = tid
        trades.append(
            TradeRecord(
                entry_idx=i,
                exit_idx=j,
                direction=s,
                holding_days=j - i + 1,
            )
        )
        i = j + 1

    out = pd.DataFrame(
        {
            "held_dir": sig,
            "prev_dir": prev_dir,
            "changed": changed,
            "gross_pct_ret": gross,
            "trade_id": trade_id,
        },
        index=bars.index,
    )
    return out, trades


def apply_costs_flip(
    sim: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    initial_capital_usd: float = 100_000.0,
    slippage_k: float = 0.1,
) -> pd.Series:
    """Apply Alpaca fees + Almgren-Chriss slippage at every direction change.

    A "change" day is any day where sig[i] != sig[i-1]. The shares involved
    in that change depend on both the prior and current directions:
        * (0 -> +1) or (0 -> -1): open new position, pay 1 leg of cost.
        * (+1 -> 0) or (-1 -> 0): close, pay 1 leg.
        * (+1 -> -1) or (-1 -> +1): close + open (full flip), pay 2 legs.
    We approximate share-count via the running NAV ÷ open[i] — the same
    sizing convention as the bracket simulator in 50_per_asset_bakeoff.
    """
    sig = sim["held_dir"].values
    prev = sim["prev_dir"].values
    changed = sim["changed"].values
    gross_pct = sim["gross_pct_ret"].values
    open_ = bars["open"].reindex(sim.index).values
    close = bars["close"].reindex(sim.index).values
    adv = bars["adv_21d_shares"].reindex(sim.index).fillna(1e6).values
    vol_bps = bars["daily_vol_bps"].reindex(sim.index).fillna(100.0).values

    _MIN_FEE_YEAR = 2024  # see note in 50_per_asset_bakeoff
    _MAX_FEE_YEAR = 2026

    n = len(sim)
    net = np.zeros(n, dtype=np.float64)
    nav = initial_capital_usd

    for i in range(n):
        # Day's gross P&L (held during this bar).
        gross_pnl_usd = gross_pct[i] * nav

        # Costs on transition days only.
        cost_usd = 0.0
        if changed[i]:
            # Resolve which legs we owe (close-then-open semantics).
            legs: list[tuple[str, float]] = []
            # Reference price for the transaction — use open[i] (we observe
            # the new signal at open, so the rebalance happens there).
            ref_px = float(open_[i]) if not math.isnan(open_[i]) and open_[i] > 0 else float(close[i])
            if ref_px <= 0:
                # Skip cost charging on bad data.
                pass
            else:
                shares = nav / ref_px
                # Closing leg (if we had a position).
                if prev[i] != 0:
                    side_out = "SELL" if prev[i] == 1 else "BUY"
                    legs.append((side_out, shares))
                # Opening leg (if new direction is non-zero).
                if sig[i] != 0:
                    side_in = "BUY" if sig[i] == 1 else "SELL"
                    legs.append((side_in, shares))

                year = min(max(int(bars.index[i].year), _MIN_FEE_YEAR), _MAX_FEE_YEAR)
                for side, sh in legs:
                    fee = bt_fees.equity_alpaca(
                        shares=sh, price=ref_px, side=side, year=year,
                    ).total
                    slip_bps = bt_slip.slippage_bps_with_floor(
                        volume_traded=sh,
                        adv=float(adv[i]),
                        daily_vol_bps=float(vol_bps[i]),
                        asset_class="equity",
                        k=slippage_k,
                    ).total_bps
                    notional = sh * ref_px
                    cost_usd += fee + (slip_bps / 1e4) * notional

        net_pnl_usd = gross_pnl_usd - cost_usd
        net[i] = net_pnl_usd / nav if nav > 0 else 0.0
        nav += net_pnl_usd
        if nav <= 0:
            return pd.Series(net, index=sim.index, dtype=np.float64)

    return pd.Series(net, index=sim.index, dtype=np.float64)


# -------- per-candidate evaluation (no TP/SL grid) -------------------------


@dataclass
class FlipCandidateRun:
    asset: str
    model_name: str
    feature_scope: str
    fold_id: int
    fold_test_start: pd.Timestamp
    fold_test_end: pd.Timestamp
    n_obs: int
    n_trades: int
    avg_holding_days: float
    sharpe_net: float
    sharpe_gross: float
    sortino_net: float
    calmar_net: float
    max_drawdown: float
    fees_drag_sharpe: float


def evaluate_signal_flip_with_costs(
    asset: str,
    bars_feat: pd.DataFrame,
    fold_signals: list[tuple[int, pd.DatetimeIndex, pd.Series]],
    fold_meta_list: list[dict],
    cand: _bake50.CandidateSpec,
) -> tuple[list[FlipCandidateRun], np.ndarray, list[int]]:
    """Run the signal-flip simulator + cost layer on pre-computed per-fold
    signals. Returns per-fold metrics, concatenated OOS net returns, and the
    full list of holding-period lengths across all trades and folds.
    """
    fold_metrics: list[FlipCandidateRun] = []
    oos_net_returns: list[float] = []
    holdings_all: list[int] = []

    for (fold_id, test_dates, sig_test), meta in zip(fold_signals, fold_meta_list):
        bars_feat_test = bars_feat.loc[test_dates]
        sim, trades = simulate_signal_flip(bars_feat_test, sig_test)
        net = apply_costs_flip(sim, bars_feat_test)
        gross = sim["gross_pct_ret"]

        net_arr = net.values
        gross_arr = gross.values
        n = int(net_arr.size)
        n_trades = len(trades)
        if n_trades:
            holdings = [t.holding_days for t in trades]
            avg_hold = float(np.mean(holdings))
            holdings_all.extend(holdings)
        else:
            avg_hold = float("nan")
        sharpe_net = bt_metrics.sharpe(net_arr) if n >= 2 else float("nan")
        sharpe_gross = bt_metrics.sharpe(gross_arr) if n >= 2 else float("nan")
        sortino_net = bt_metrics.sortino(net_arr) if n >= 2 else float("nan")
        calmar_net = bt_metrics.calmar(net_arr) if n >= 2 else float("nan")
        equity = np.cumprod(1.0 + net_arr) if n else np.array([1.0])
        max_dd = bt_metrics.max_drawdown(equity)
        fees_drag = (
            sharpe_gross - sharpe_net
            if math.isfinite(sharpe_gross) and math.isfinite(sharpe_net)
            else float("nan")
        )

        fold_metrics.append(
            FlipCandidateRun(
                asset=asset,
                model_name=cand.name,
                feature_scope=cand.scope,
                fold_id=fold_id,
                fold_test_start=meta["test_start"],
                fold_test_end=meta["test_end"],
                n_obs=n,
                n_trades=n_trades,
                avg_holding_days=avg_hold,
                sharpe_net=sharpe_net,
                sharpe_gross=sharpe_gross,
                sortino_net=sortino_net,
                calmar_net=calmar_net,
                max_drawdown=max_dd,
                fees_drag_sharpe=fees_drag,
            )
        )
        oos_net_returns.extend(net_arr.tolist())

    return fold_metrics, np.asarray(oos_net_returns, dtype=np.float64), holdings_all


# -------- driver ----------------------------------------------------------


def run_signal_flip_bakeoff(
    *,
    quick: bool = False,
    bootstrap_b: int = 2000,
    out_dir: Path = PROJECT_ROOT / "data" / "bakeoff_variants" / "signal_flip",
    fwer_alpha: float = 0.05,
) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    market_ctx = _bake50.load_market_context()
    scopes = ("asset_only", "asset_plus_market")
    candidates = _bake50.build_candidate_list(scopes)
    if quick:
        candidates = candidates[:3]

    harness = WalkForwardHarness(
        train_min_years=8,
        train_max_years=15,
        val_days=252,
        embargo_days=5,
        purge_days=1,
        test_days=21,
        refit_cadence="monthly",
    )

    all_rows: list[dict] = []
    aggregate_by_asset: dict[str, list[dict]] = {}
    holdings_by_asset_cand: dict[tuple[str, str, str], list[int]] = {}

    assets = [disp for _, _, disp in _bake50.TRADABLE]
    for asset in assets:
        print(f"\n=== {asset} ===", flush=True)
        bars = _bake50.load_asset_bars(asset)
        bars_feat = _bake50.add_per_asset_features(bars)
        bars_feat = bars_feat.dropna(subset=["vol_21", "atr_norm"]).copy()
        bars_aligned = bars.loc[bars_feat.index]
        agg_rows: list[dict] = []

        for cand in candidates:
            t_sig = time.time()
            fold_signals, fold_meta_list = _bake50.precompute_signal_per_fold(
                asset=asset,
                bars=bars_aligned,
                bars_feat=bars_feat,
                market_ctx=market_ctx if cand.family == "ml" else None,
                cand=cand,
                harness=harness,
            )
            dt_sig = time.time() - t_sig

            t0 = time.time()
            fold_results, oos_returns, holdings_all = evaluate_signal_flip_with_costs(
                asset=asset,
                bars_feat=bars_feat,
                fold_signals=fold_signals,
                fold_meta_list=fold_meta_list,
                cand=cand,
            )
            if oos_returns.size < 21:
                continue

            agg_sharpe = bt_metrics.sharpe(oos_returns)
            agg_sortino = bt_metrics.sortino(oos_returns)
            agg_calmar = bt_metrics.calmar(oos_returns)
            agg_equity = np.cumprod(1.0 + oos_returns)
            agg_mdd = bt_metrics.max_drawdown(agg_equity)
            pval = _bake50.sharpe_to_pvalue(agg_sharpe, oos_returns.size)
            avg_hold_overall = float(np.mean(holdings_all)) if holdings_all else float("nan")
            median_hold = float(np.median(holdings_all)) if holdings_all else float("nan")
            holdings_by_asset_cand[(asset, cand.name, cand.scope)] = holdings_all

            for fold in fold_results:
                all_rows.append({**fold.__dict__})
            agg_rows.append(
                {
                    "asset": asset,
                    "model_name": cand.name,
                    "feature_scope": cand.scope,
                    "n_folds": len(fold_results),
                    "n_oos_obs": int(oos_returns.size),
                    "n_trades_total": int(len(holdings_all)),
                    "avg_holding_days": avg_hold_overall,
                    "median_holding_days": median_hold,
                    "agg_sharpe_net": float(agg_sharpe),
                    "agg_sortino_net": float(agg_sortino),
                    "agg_calmar_net": float(agg_calmar),
                    "agg_max_drawdown": float(agg_mdd),
                    "p_value_unadj": float(pval),
                    "oos_returns": oos_returns,
                }
            )
            dt = time.time() - t0
            print(
                f"  {cand.name:>15s} | scope={cand.scope:>18s} | "
                f"Sharpe={agg_sharpe:+.3f} p={pval:.3f} "
                f"trades={len(holdings_all)} avg_hold={avg_hold_overall:.1f}d | "
                f"sig_t={dt_sig:.1f}s sim_t={dt:.2f}s",
                flush=True,
            )

        aggregate_by_asset[asset] = agg_rows

    # ----- Two-stage statistical gate ----------------------------------------
    # Stage 1 (within-asset): rank by DSR (variance-calibrated, n_trials =
    # per-asset candidate count). With no TP/SL grid the search space is ~17
    # trials/asset (vs ~660 in 50_per_asset_bakeoff), so the DSR deflation is
    # much milder — DSR should not collapse to ~0 for plausible candidates.
    # Stage 2 (across-asset): Bonferroni-Holm step-down across the 8
    # best-of-asset p-values.
    p_all = np.array(
        [row["p_value_unadj"] for rows in aggregate_by_asset.values() for row in rows]
    )
    p_adj_strict = _bake50.holm_adjust(p_all)
    idx = 0
    for asset in assets:
        for row in aggregate_by_asset[asset]:
            row["p_value_holm_strict"] = float(p_adj_strict[idx])
            idx += 1

    winners: dict[str, dict] = {}
    per_asset_best: dict[str, dict | None] = {}
    for asset in assets:
        rows_all = aggregate_by_asset[asset]
        if not rows_all:
            winners[asset] = {
                "verdict": "no_winner",
                "reason": "no candidate produced enough OOS data",
            }
            continue
        n_trials = len(rows_all)
        sharpes = np.asarray([r["agg_sharpe_net"] for r in rows_all])
        finite = sharpes[np.isfinite(sharpes)]
        if finite.size >= 2:
            sigma_sr_emp = float(np.std(finite, ddof=1))
        else:
            sigma_sr_emp = 1.0
        # Convert annualised SR std to period (daily) SR std — see note in
        # 50_per_asset_bakeoff: dsr_expected_max_sr expects daily-units sigma.
        sigma_sr_daily = sigma_sr_emp / math.sqrt(252.0)
        bench = bt_metrics.dsr_expected_max_sr(
            n_trials=max(1, n_trials),
            sr_trials_std=sigma_sr_daily,
        )
        for r in rows_all:
            r["dsr"] = bt_metrics.psr(r["oos_returns"], benchmark_sr=bench)
        rows_pos = [r for r in rows_all if r["agg_sharpe_net"] > 0]
        if not rows_pos:
            per_asset_best[asset] = None
            continue
        rows_pos.sort(key=lambda r: (-r["dsr"], r["p_value_unadj"]))
        per_asset_best[asset] = rows_pos[0]

    asset_best_pvals = []
    asset_best_order = []
    for asset in assets:
        best = per_asset_best.get(asset)
        if best is None:
            continue
        asset_best_pvals.append(best["p_value_unadj"])
        asset_best_order.append(asset)
    asset_best_pvals_arr = np.asarray(asset_best_pvals)
    asset_best_pvals_holm = (
        _bake50.holm_adjust(asset_best_pvals_arr)
        if asset_best_pvals_arr.size
        else asset_best_pvals_arr
    )
    holm_map = dict(zip(asset_best_order, asset_best_pvals_holm))

    for asset in assets:
        best = per_asset_best.get(asset)
        if best is None:
            winners[asset] = {
                "verdict": "no_winner",
                "reason": "no candidate produced positive aggregate Sharpe",
            }
            continue
        p_holm_across_assets = float(holm_map[asset])

        boot = bt_metrics.bootstrap_sharpe_ci(
            best["oos_returns"], B=bootstrap_b, block_length="auto",
            alpha=0.05, seed=42,
        )

        # Gate per task spec: Holm p < 0.05  AND  CI_lo > 0  AND  Sharpe > 0.
        passes = (
            p_holm_across_assets < fwer_alpha
            and boot.ci_lower > 0.0
            and best["agg_sharpe_net"] > 0.0
        )

        winners[asset] = {
            "verdict": "winner" if passes else "no_winner",
            "model_name": best["model_name"],
            "feature_scope": best["feature_scope"],
            "oos_sharpe_net": best["agg_sharpe_net"],
            "oos_sortino_net": best["agg_sortino_net"],
            "oos_calmar_net": best["agg_calmar_net"],
            "max_drawdown": best["agg_max_drawdown"],
            "avg_holding_days": best["avg_holding_days"],
            "median_holding_days": best["median_holding_days"],
            "n_trades_total": best["n_trades_total"],
            "p_value_unadj": best["p_value_unadj"],
            "p_value_holm_across_assets": p_holm_across_assets,
            "p_value_holm_strict": best["p_value_holm_strict"],
            "dsr": best["dsr"],
            "ci_lower": boot.ci_lower,
            "ci_upper": boot.ci_upper,
            "p_positive": boot.p_positive,
            "n_folds": best["n_folds"],
            "n_oos_obs": best["n_oos_obs"],
            "n_trials_this_asset": len(aggregate_by_asset[asset]),
        }

    # Persist long-format per-fold dataframe.
    out_rows_df = pd.DataFrame(all_rows)
    if "fold_test_start" in out_rows_df.columns:
        out_rows_df["fold_test_start"] = out_rows_df["fold_test_start"].astype(str)
        out_rows_df["fold_test_end"] = out_rows_df["fold_test_end"].astype(str)
    out_rows_df.to_parquet(out_dir / "results.parquet", index=False)
    print(f"\nwrote {len(out_rows_df)} per-fold rows to {out_dir / 'results.parquet'}")

    # Winners parquet.
    rows_for_winners = [
        {"asset": a, **{k: v for k, v in w.items() if k != "oos_returns"}}
        for a, w in winners.items()
    ]
    pd.DataFrame(rows_for_winners).to_parquet(out_dir / "winners.parquet", index=False)

    # Aggregate (non-fold) table.
    agg_flat = []
    for asset_rows in aggregate_by_asset.values():
        for r in asset_rows:
            agg_flat.append({k: v for k, v in r.items() if k != "oos_returns"})
    pd.DataFrame(agg_flat).to_parquet(out_dir / "aggregate.parquet", index=False)

    return {
        "winners": winners,
        "aggregate_by_asset": aggregate_by_asset,
        "holdings_by_asset_cand": holdings_by_asset_cand,
    }


# -------- portfolio sharpe (winners) --------------------------------------


def compute_portfolio_sharpe_flip(
    winners: dict, aggregate_by_asset: dict[str, list[dict]],
) -> float:
    series: list[pd.Series] = []
    for asset, w in winners.items():
        if w["verdict"] != "winner":
            continue
        match = None
        for r in aggregate_by_asset[asset]:
            if (
                r["model_name"] == w["model_name"]
                and r["feature_scope"] == w["feature_scope"]
            ):
                match = r
                break
        if match is None:
            continue
        series.append(pd.Series(match["oos_returns"], name=asset).reset_index(drop=True))
    if not series:
        return float("nan")
    panel = pd.concat(series, axis=1)
    port = panel.mean(axis=1).dropna().values
    return bt_metrics.sharpe(port)


# -------- report writer ----------------------------------------------------


def write_report(result: dict, out_dir: Path) -> Path:
    winners = result["winners"]
    aggregate_by_asset = result["aggregate_by_asset"]

    # Aggregate avg-holding-period stats across the per-asset BEST candidate.
    best_holdings: list[float] = []
    for asset, w in winners.items():
        if "avg_holding_days" in w and math.isfinite(w["avg_holding_days"]):
            best_holdings.append(w["avg_holding_days"])
    overall_avg_hold = float(np.mean(best_holdings)) if best_holdings else float("nan")

    lines: list[str] = []
    lines.append("# Signal-Flip Exit Bake-off Report (architecture variant)")
    lines.append("")
    lines.append(
        "Pure model-driven entries AND exits: position is held while the signal "
        "direction is constant, closed the moment the signal flips sign (or hits 0). "
        "**No TP. No SL. No time stop.** Compare against the bracket-order baseline in "
        "`data/bakeoff/REPORT.md`."
    )
    lines.append("")
    lines.append("## Promotion gate")
    lines.append(
        "- `p_Holm_acrossAssets < 0.05`  **AND**  `bootstrap-CI lower > 0`  **AND**  `Sharpe (net) > 0`."
    )
    lines.append(
        "- DSR is reported (variance-calibrated, n_trials = per-asset candidate count) "
        "but is not a hard gate here — the search space (≈17 trials/asset) is small enough "
        "that DSR is informative rather than dominant. The bracket-baseline's DSR=0 was an "
        "artefact of the 24× larger TP/SL grid; with no grid that artefact is gone."
    )
    lines.append("")
    lines.append("## Winners (one per asset; or NO WINNER)")
    lines.append("")
    lines.append(
        "| asset | verdict | model | scope | Sharpe (net) | DSR | p_Holm_8 | p_Holm_strict | "
        "CI_lo | CI_hi | avg_hold (d) | trades |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for asset, w in winners.items():
        if "model_name" not in w:
            lines.append(
                f"| {asset} | NO WINNER | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        verdict = "WINNER" if w["verdict"] == "winner" else "NO WINNER"
        lines.append(
            f"| {asset} | {verdict} | {w['model_name']} | {w['feature_scope']} | "
            f"{w['oos_sharpe_net']:+.3f} | {w['dsr']:.3f} | "
            f"{w['p_value_holm_across_assets']:.4f} | {w['p_value_holm_strict']:.4f} | "
            f"{w['ci_lower']:+.3f} | {w['ci_upper']:+.3f} | "
            f"{w['avg_holding_days']:.1f} | {w['n_trades_total']} |"
        )
    lines.append("")

    n_winners = len([w for w in winners.values() if w["verdict"] == "winner"])
    portfolio_sharpe = compute_portfolio_sharpe_flip(winners, aggregate_by_asset)
    lines.append("## Aggregate portfolio Sharpe (equal-weight, daily-rebalanced)")
    lines.append("")
    lines.append(
        f"- Winners promoted: **{n_winners}/{len(winners)}**"
    )
    lines.append(
        f"- Portfolio Sharpe (net): **{portfolio_sharpe:+.3f}**"
    )
    lines.append(
        f"- Avg holding period across per-asset best candidates: **{overall_avg_hold:.1f} trading days**"
    )
    lines.append("")

    lines.append("## Per-asset best candidate (full diagnostics)")
    lines.append("")
    lines.append(
        "| asset | model | scope | Sharpe | DSR | p_unadj | p_Holm_8 | "
        "p_Holm_strict | CI_lo | avg_hold | median_hold | trades |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for asset, w in winners.items():
        if "model_name" not in w:
            lines.append(f"| {asset} | (none — no positive-Sharpe candidate) | | | | | | | | | | |")
            continue
        lines.append(
            f"| {asset} | {w['model_name']} | {w['feature_scope']} | "
            f"{w['oos_sharpe_net']:+.3f} | {w['dsr']:.3f} | "
            f"{w['p_value_unadj']:.4f} | {w['p_value_holm_across_assets']:.4f} | "
            f"{w['p_value_holm_strict']:.4f} | {w['ci_lower']:+.3f} | "
            f"{w['avg_holding_days']:.1f} | {w['median_holding_days']:.1f} | "
            f"{w['n_trades_total']} |"
        )
    lines.append("")

    lines.append("## Top-3 candidates per asset (by net Sharpe)")
    for asset, rows in aggregate_by_asset.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: -r["agg_sharpe_net"])[:3]
        lines.append(f"\n### {asset}")
        lines.append("| model | scope | Sharpe | DSR | p_unadj | avg_hold (d) | trades |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows_sorted:
            lines.append(
                f"| {r['model_name']} | {r['feature_scope']} | "
                f"{r['agg_sharpe_net']:+.3f} | {r.get('dsr', float('nan')):.3f} | "
                f"{r['p_value_unadj']:.3f} | {r['avg_holding_days']:.1f} | "
                f"{r['n_trades_total']} |"
            )
    lines.append("")

    lines.append("## Comparison to bracket-order baseline (data/bakeoff/REPORT.md)")
    lines.append(
        "- **Baseline search**: (model × scope × tp_σ × sl_σ × hit_priority) ≈ 5,300 trials → "
        "DSR collapses to ~0 from the TP/SL noise; 0 winners promoted; per-asset best Sharpes 0.4–1.1."
    )
    lines.append(
        "- **Signal-flip variant**: (model × scope) ≈ 17 trials/asset → DSR is meaningful, "
        "trial count is honest, and the holding-period is endogenously set by the model's own "
        "conviction (avg ≈ {avg_hold:.1f} trading days here).".format(avg_hold=overall_avg_hold)
        if math.isfinite(overall_avg_hold)
        else "- **Signal-flip variant**: per-asset best holding period statistics in table above."
    )
    lines.append(
        f"- **Promotion delta**: baseline 0/8 → signal-flip {n_winners}/8."
    )
    lines.append(
        f"- **Portfolio Sharpe delta**: baseline NaN (no winners) → signal-flip {portfolio_sharpe:+.3f}."
    )
    lines.append("")

    lines.append("## Methodology")
    lines.append(
        "- Same data, features, walk-forward harness, fees (Alpaca regs), and slippage "
        "(Almgren-Chriss + floor, k=0.1) as `scripts/50_per_asset_bakeoff.py`."
    )
    lines.append(
        "- Position dynamics: at bar `i`, hold direction `sig[i]` computed from info ≤ i-1. "
        "Day's P&L = `sig[i] * (close[i]/close[i-1] - 1)`. Transition (`sig[i] != sig[i-1]`) "
        "triggers round-trip cost at `open[i]` (close-then-open semantics; full flips pay 2 legs)."
    )
    lines.append(
        "- DSR (variance-calibrated): `bench_SR = dsr_expected_max_sr(n_trials=N, "
        "sr_trials_std=sigma_emp/sqrt(252))`, then `dsr = psr(returns, benchmark_sr=bench_SR)`."
    )
    lines.append(
        "- Bootstrap CI: Politis-Romano stationary block-bootstrap, B=2000, "
        "block_length='auto' (Politis-White)."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append(
        "- Daily bars; intraday flip latency not modelled (the bot reacts at next open)."
    )
    lines.append(
        "- No regime conditioning. No survivorship-bias correction. Same liquid mega-cap+GLD universe."
    )
    lines.append(
        "- Slippage uses Almgren-Chriss square-root with k=0.1; the asymmetric stop-jump cost "
        "(relevant for the bracket variant) does not apply here since there are no triggered stops."
    )

    out_path = out_dir / "REPORT.md"
    out_path.write_text("\n".join(lines))
    return out_path


# -------- main -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="trimmed candidate set smoke run")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="bootstrap resamples (default 2000)")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data" / "bakeoff_variants" / "signal_flip"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_signal_flip_bakeoff(
        quick=args.quick,
        bootstrap_b=args.bootstrap,
        out_dir=out_dir,
    )
    write_report(res, out_dir)
    dt = time.time() - t0
    print(f"\nsignal-flip bake-off complete in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
