"""91_leveraged_etf_bakeoff — comprehensive search on leveraged ETFs to test
if a higher-vol universe can push toward the 1%/day target.

Universe: TSLL, SOXL, NVDL, TNA, UVXY (3-5x daily std)
Pulls 1 year of DAILY bars via Alpaca → pandas in-memory (no ArcticDB).
Per asset: 6 signals × 4 trail-σ × 3 PTs = 72 cells.
Split: 6-mo train / 6-mo test. Pick train-best (net / (|maxDD|+1), gated
net>0 + sharpe>0). Compute portfolio test P&L.

Per-name notional: $20k. Cost: 3 bp per side.

Run:
    .venv/bin/python scripts/91_leveraged_etf_bakeoff.py
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402

WARMUP = 252  # need 252 bars for warmup; we'll pull ~2y so train window starts post-warmup
LEVERAGED_ETFS = ["TSLL", "SOXL", "NVDL", "TNA", "UVXY"]

SIGNAL_TYPES = ["buy_and_hold", "macd", "rsi2", "bollinger", "donchian", "mom21"]
TRAIL_SIGMAS = [0.5, 1.0, 2.0, 3.0]
PROFIT_TARGETS = [None, 3.0, 5.0]

PER_NAME_NOTIONAL = 20_000.0
COST_BPS_PER_SIDE = 3.0  # 3 bp per side


def pull_alpaca_daily(symbols: list[str], years: float = 2.0) -> dict[str, pd.DataFrame]:
    """Pull daily bars from Alpaca for the given symbols. Returns dict
    sym → DataFrame indexed by date with OHLCV columns."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ["ALPACA_API_KEY"]
    secret = os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"]

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret)
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(years * 365.25))
    print(f"Pulling Alpaca daily bars {start.date()} → {end.date()} for {symbols}...")
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(req)

    out = {}
    for sym in symbols:
        rows = bars.data.get(sym, [])
        if not rows:
            print(f"  {sym}: NO DATA returned")
            continue
        records = []
        for b in rows:
            ts = pd.Timestamp(b.timestamp).tz_convert("UTC").tz_localize(None).normalize()
            records.append({
                "date": ts,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
            })
        df = pd.DataFrame(records).set_index("date").sort_index()
        df.index.name = "date"
        out[sym] = df
        print(f"  {sym}: {len(df)} bars, {df.index.min().date()} → {df.index.max().date()}")
    return out


def precompute_signal(bars: pd.DataFrame, feat: pd.DataFrame, mode: str) -> np.ndarray:
    """Same as scripts/77, copied here to avoid coupling."""
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "buy_and_hold":
        out[WARMUP:] = 1
        return out
    if mode == "macd":
        h = feat["macd_hist"].shift(1)
        out = np.where(h > 0, 1, np.where(h < 0, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "rsi2":
        r = feat["rsi_2"].shift(1)
        out = np.where(r < 10, 1, np.where(r > 90, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "bollinger":
        close = bars["close"]
        ma20 = close.rolling(20).mean().shift(1)
        sd20 = close.rolling(20).std().shift(1)
        prev_close = close.shift(1)
        z = (prev_close - ma20) / sd20
        out = np.where(z < -2, 1, np.where(z > 2, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "donchian":
        high20 = bars["high"].rolling(20).max().shift(1)
        low20 = bars["low"].rolling(20).min().shift(1)
        prev_close = bars["close"].shift(1)
        out = np.where(prev_close >= high20, 1,
                       np.where(prev_close <= low20, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    if mode == "mom21":
        m = feat["ret_21"].shift(1)
        out = np.where(m > 0.05, 1, np.where(m < -0.05, -1, 0)).astype(np.int8)
        out[:WARMUP] = 0
        return out
    raise ValueError(f"unknown signal mode {mode!r}")


def replay(bars: pd.DataFrame, feat: pd.DataFrame, sig_array: np.ndarray,
           start_idx: int, end_idx: int, notional: float,
           trail: float, profit_target_sigmas: float | None,
           cost_bps_per_side: float = COST_BPS_PER_SIDE) -> dict:
    """Replay with trailing-stop + optional PT + signal-flip exits.

    Mirrors scripts/77 logic but adds explicit transaction costs (3 bp/side)
    deducted on every entry and every exit.
    """
    if start_idx < WARMUP:
        start_idx = WARMUP
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    round_trips: list[dict] = []
    daily_pnl: list[float] = []
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0
    cost_per_side = cost_bps_per_side / 10_000.0

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(feat["vol_21"].iloc[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig_array[i])
        close = float(closes[i])

        # MTM on prior position before action this bar
        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            daily_pnl.append((close - prev_close) * prev_qty * prev_side)
        else:
            daily_pnl.append(0.0)

        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail
            stop = peak - band
            reason = None
            if close <= stop:
                reason = "trail"
            elif (profit_target_sigmas is not None and
                  close >= entry_price * (1 + entry_sigma * profit_target_sigmas)):
                reason = "pt"
            elif sig_dir < 0:
                reason = "flip"
            if reason:
                gross = (close - entry_price) * entry_qty
                exit_cost = close * entry_qty * cost_per_side
                round_trips.append({"side": "LONG", "pnl": gross - exit_cost})
                # also charge the per-bar daily MTM line for the exit cost
                daily_pnl[-1] -= exit_cost
                state, entry_price, entry_sigma, peak = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue
        elif state == "IN_SHORT":
            if trough is None or close < trough:
                trough = close
            band = entry_price * entry_sigma * trail
            stop = trough + band
            reason = None
            if close >= stop:
                reason = "trail"
            elif (profit_target_sigmas is not None and
                  close <= entry_price * (1 - entry_sigma * profit_target_sigmas)):
                reason = "pt"
            elif sig_dir > 0:
                reason = "flip"
            if reason:
                gross = (entry_price - close) * entry_qty
                exit_cost = close * entry_qty * cost_per_side
                round_trips.append({"side": "SHORT", "pnl": gross - exit_cost})
                daily_pnl[-1] -= exit_cost
                state, entry_price, entry_sigma, trough = "FLAT", None, None, None
                entry_qty = 0
                prev_state, prev_qty, prev_side = "FLAT", 0, 0
                prev_close = close
                continue

        if state == "FLAT" and sig_dir != 0:
            qty = int(notional // close) if close > 0 else 0
            if qty > 0:
                state = "IN_LONG" if sig_dir > 0 else "IN_SHORT"
                entry_price = close
                entry_sigma = sigma
                entry_qty = qty
                # entry cost
                entry_cost = close * qty * cost_per_side
                daily_pnl[-1] -= entry_cost
                # we'll book this against the FIRST round_trip's pnl when it closes
                # — easier: track entry costs on round-trip side
                # Actually simpler: subtract from pnl at close. Track via attr.
                if sig_dir > 0:
                    peak, trough = close, None
                else:
                    trough, peak = close, None
                # store pending entry cost so we deduct at close
                _pending_entry_cost = entry_cost  # noqa: F841 — kept clear

        prev_close = close
        prev_state = state
        prev_qty = entry_qty
        prev_side = 1 if state == "IN_LONG" else (-1 if state == "IN_SHORT" else 0)

    open_unrz = 0.0
    if state != "FLAT" and entry_price is not None:
        last_close = float(closes[end_idx - 1])
        if state == "IN_LONG":
            open_unrz = (last_close - entry_price) * entry_qty
        else:
            open_unrz = (entry_price - last_close) * entry_qty

    realized = sum(rt["pnl"] for rt in round_trips)
    # net already includes exit cost via round_trips and entry cost via daily_pnl
    # but daily_pnl sums to gross MTM minus entry/exit costs, so use daily_pnl directly:
    daily_total = sum(daily_pnl)
    # daily_total = sum of MTM moves - entry costs - exit costs
    # for closed trades, MTM sum equals (close-entry)*qty = gross; then minus entry cost (already in daily) and exit cost (subtracted twice — once in daily, once in round_trip).
    # To avoid double counting, use net = daily_total + open_unrz.
    net = daily_total + open_unrz

    wins = sum(1 for rt in round_trips if rt["pnl"] > 0)
    losses = sum(1 for rt in round_trips if rt["pnl"] <= 0)
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0

    sharpe = float("nan")
    max_dd = 0.0
    if daily_pnl and notional > 0:
        rets = np.asarray(daily_pnl) / notional
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        cum = np.cumsum(daily_pnl)
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())

    n_bars = max(end_idx - start_idx, 1)
    years = n_bars / 252.0
    ann_ret = float("nan")
    if years > 0 and notional + net > 0:
        ann_ret = ((1 + net / notional) ** (1 / years) - 1) * 100.0

    daily_ret_pct_mean = float(np.mean(daily_pnl) / notional * 100.0) if daily_pnl else 0.0

    return {
        "n_trades": len(round_trips), "wins": wins, "losses": losses,
        "n_long": sum(1 for r in round_trips if r["side"] == "LONG"),
        "n_short": sum(1 for r in round_trips if r["side"] == "SHORT"),
        "win_rate": win_rate, "realized": realized, "open_unrz": open_unrz,
        "net": net, "max_dd": max_dd, "sharpe": sharpe,
        "ann_return_pct": ann_ret,
        "daily_ret_pct_mean": daily_ret_pct_mean,
        "n_days": n_bars,
    }


def main() -> int:
    print()
    print("=" * 130)
    print(f"  Leveraged-ETF bake-off  •  {len(LEVERAGED_ETFS)} assets × "
          f"{len(SIGNAL_TYPES)} signals × {len(TRAIL_SIGMAS)} trails × "
          f"{len(PROFIT_TARGETS)} exits = "
          f"{len(LEVERAGED_ETFS)*len(SIGNAL_TYPES)*len(TRAIL_SIGMAS)*len(PROFIT_TARGETS)} cells")
    print(f"  Per-name notional: ${PER_NAME_NOTIONAL:,.0f}  •  Cost: {COST_BPS_PER_SIDE} bp/side")
    print("=" * 130)

    # Pull ~2y so we have warmup + 6mo train + 6mo test
    bar_dict = pull_alpaca_daily(LEVERAGED_ETFS, years=2.0)

    all_results = []
    for asset in LEVERAGED_ETFS:
        if asset not in bar_dict:
            print(f"{asset}: skipped (no data)")
            continue
        bars = bar_dict[asset]
        if len(bars) < WARMUP + 100:
            print(f"{asset}: only {len(bars)} bars — need at least {WARMUP+100}; skip")
            continue

        feat = add_per_asset_features(bars)

        # Define test = last ~6mo (126 trading bars), train = preceding 6mo (126 bars)
        test_len = 126
        train_len = 126
        if len(bars) < WARMUP + train_len + test_len:
            # fall back: train = WARMUP..(N-test_len), test = last test_len
            train_start = WARMUP
            test_start = len(bars) - test_len
            train_end = test_start
        else:
            test_start = len(bars) - test_len
            train_end = test_start
            train_start = train_end - train_len

        sig_cache = {m: precompute_signal(bars, feat, m) for m in SIGNAL_TYPES}

        cells = []
        for sigm in SIGNAL_TYPES:
            for tr in TRAIL_SIGMAS:
                for pt in PROFIT_TARGETS:
                    train_s = replay(bars, feat, sig_cache[sigm], train_start, train_end,
                                     PER_NAME_NOTIONAL, tr, pt)
                    test_s = replay(bars, feat, sig_cache[sigm], test_start, len(bars),
                                    PER_NAME_NOTIONAL, tr, pt)
                    cells.append({"signal": sigm, "trail": tr, "pt": pt,
                                  "train": train_s, "test": test_s})

        # Train-best: gated net>0 + sharpe>0; key = net / (|maxDD|+1)
        viable = [c for c in cells
                  if c["train"]["net"] > 0 and c["train"]["sharpe"] > 0]
        best = (max(viable,
                    key=lambda c: c["train"]["net"] / (abs(c["train"]["max_dd"]) + 1.0))
                if viable else None)
        ceiling = max(cells, key=lambda c: c["test"]["net"])

        all_results.append({
            "asset": asset, "best_train": best, "ceiling_test": ceiling,
            "n_cells": len(cells), "any_viable": bool(viable),
            "train_window": (bars.index[train_start].date(), bars.index[train_end-1].date()),
            "test_window": (bars.index[test_start].date(), bars.index[-1].date()),
        })

        if best:
            pt_str = f"{best['pt']}σ" if best["pt"] else "—"
            print(f"  {asset:<6} train-best: signal={best['signal']:<13} "
                  f"trail={best['trail']}σ pt={pt_str:<4} | "
                  f"TRAIN net=${best['train']['net']:>+9,.0f} "
                  f"Sh={best['train']['sharpe']:+.2f} DD=${best['train']['max_dd']:>+9,.0f} | "
                  f"TEST net=${best['test']['net']:>+9,.0f} "
                  f"Sh={best['test']['sharpe']:+.2f} DD=${best['test']['max_dd']:>+9,.0f} "
                  f"L/S={best['test']['n_long']}/{best['test']['n_short']}")
        else:
            print(f"  {asset:<6} NO VIABLE config (every train cell either net≤0 or sharpe≤0)")

    # ------------------------------------------------------------------
    # Portfolio summary
    # ------------------------------------------------------------------
    print()
    print("=" * 130)
    print("  PORTFOLIO — equal-weight $20k per asset, deploy only train-viable picks")
    print("=" * 130)
    deployable = [r for r in all_results if r["best_train"] is not None]
    rejected = [r for r in all_results if r["best_train"] is None]
    portfolio_test_net = sum(r["best_train"]["test"]["net"] for r in deployable)
    portfolio_train_net = sum(r["best_train"]["train"]["net"] for r in deployable)
    portfolio_notional = len(deployable) * PER_NAME_NOTIONAL

    print(f"  deployable: {len(deployable)}/{len(all_results)} → "
          f"{sorted(r['asset'] for r in deployable)}")
    print(f"  rejected:   {len(rejected)} → "
          f"{sorted(r['asset'] for r in rejected)}")
    print()
    print(f"  TRAIN net (picked configs): ${portfolio_train_net:>+12,.0f}")
    print(f"  TEST  net (picked configs): ${portfolio_test_net:>+12,.0f}  "
          f"({portfolio_test_net/portfolio_notional*100:+.2f}% on ${portfolio_notional:,.0f} deployed)")

    # Annualized portfolio test return
    if deployable:
        test_days = deployable[0]["best_train"]["test"]["n_days"]
        portfolio_test_pct = portfolio_test_net / portfolio_notional
        years = test_days / 252.0
        port_ann = ((1 + portfolio_test_pct) ** (1/years) - 1) * 100.0 if years > 0 else float("nan")
        # daily mean return % on deployed capital
        # sum each asset's daily MTM divided by total deployed notional
        # equivalent: mean of (per-asset daily_ret_pct_mean × notional_per_asset) / portfolio_notional
        port_daily_pct = sum(r["best_train"]["test"]["daily_ret_pct_mean"] * PER_NAME_NOTIONAL
                             for r in deployable) / portfolio_notional
        print(f"  TEST  days: {test_days}  •  Years: {years:.3f}  •  "
              f"Port annualized: {port_ann:+.2f}%  •  Port mean-daily: {port_daily_pct:+.3f}%/day")

    print()
    print(f"  Reference: current 8-asset daily v2 OOS test ≈ +$19,144")
    print("=" * 130)

    # Per-asset detail table
    print()
    print("  Per-asset detail (train-picked + test ceiling)")
    print("-" * 130)
    print(f"  {'asset':<6} {'sig':<13} {'tr':<5} {'pt':<5} "
          f"{'TRAIN $':<10} {'TEST $':<10} {'TEST Sh':<8} {'TEST DD':<10} "
          f"{'TEST ann%':<10} {'TEST/day%':<10}")
    print("-" * 130)
    for r in all_results:
        a = r["asset"]
        if r["best_train"]:
            b = r["best_train"]
            pt = f"{b['pt']}σ" if b["pt"] else "—"
            print(f"  {a:<6} {b['signal']:<13} {b['trail']}σ{'':<2} {pt:<5} "
                  f"${b['train']['net']:>+8,.0f} ${b['test']['net']:>+8,.0f} "
                  f"{b['test']['sharpe']:>+6.2f}   ${b['test']['max_dd']:>+8,.0f} "
                  f"{b['test']['ann_return_pct']:>+8.2f}%  "
                  f"{b['test']['daily_ret_pct_mean']:>+8.3f}%")
        else:
            print(f"  {a:<6} NO VIABLE")
    print("=" * 130)

    # Cross-asset ceiling info (look-ahead)
    print()
    print("  Test-ceiling per asset (look-ahead — sanity bound, NOT deployable)")
    print("-" * 130)
    for r in all_results:
        c = r["ceiling_test"]
        pt = f"{c['pt']}σ" if c["pt"] else "—"
        print(f"  {r['asset']:<6} ceil sig={c['signal']:<13} tr={c['trail']}σ pt={pt:<4} "
              f"net=${c['test']['net']:>+8,.0f}  Sh={c['test']['sharpe']:+.2f}")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
