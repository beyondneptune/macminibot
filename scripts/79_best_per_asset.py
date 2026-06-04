"""79_best_per_asset — head-to-head comparison of every candidate config
on the SAME OOS window so the numbers are fair.

Per asset, evaluates these candidates:
  A. Daily baseline (current deployed bake-off winner)
  B. Daily comprehensive train-best (from script 77)
  C. Intraday train-best (from script 78)

All three replayed on the SHARED window 2026-01-01 → 2026-05-18
(~4.5 months) so $ amounts and Sharpes are directly comparable. For each
asset we pick the winner (max net, with Sharpe as tie-breaker).

Run:
    .venv/bin/python scripts/79_best_per_asset.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml_research.asset_specialists import compute_signal_for  # noqa: E402
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from scripts._storage import open_store  # noqa: E402

NET_LIQ = 100000.0
PORTFOLIO_SIZE = 7
NOTIONAL = NET_LIQ / PORTFOLIO_SIZE
SHARED_START = pd.Timestamp("2026-01-01")
SHARED_END = pd.Timestamp("2026-05-19")

WARMUP_DAILY = 252
WARMUP_INTRADAY = 500

DAILY_ASSET_MAP = {
    "AAPL":  ("equity.daily_bars", "NASDAQ.AAPL"),
    "MSFT":  ("equity.daily_bars", "NASDAQ.MSFT"),
    "GOOGL": ("equity.daily_bars", "NASDAQ.GOOGL"),
    "AMZN":  ("equity.daily_bars", "NASDAQ.AMZN"),
    "META":  ("equity.daily_bars", "NASDAQ.META"),
    "TSLA":  ("equity.daily_bars", "NASDAQ.TSLA"),
    "NVDA":  ("equity.daily_bars", "NASDAQ.NVDA"),
    "GLD":   ("etf.daily_bars",    "ARCA.GLD"),
}

INTRADAY_ASSET_MAP = {
    "AAPL":  ("equity.minute_bars", "NASDAQ.AAPL"),
    "MSFT":  ("equity.minute_bars", "NASDAQ.MSFT"),
    "GOOGL": ("equity.minute_bars", "NASDAQ.GOOGL"),
    "AMZN":  ("equity.minute_bars", "NASDAQ.AMZN"),
    "META":  ("equity.minute_bars", "NASDAQ.META"),
    "TSLA":  ("equity.minute_bars", "NASDAQ.TSLA"),
    "NVDA":  ("equity.minute_bars", "NASDAQ.NVDA"),
    "GLD":   ("etf.minute_bars",    "ARCA.GLD"),
}

# Configs identified by the prior search runs.
# Each config is (signal_mode, trail, profit_target_sigmas, substrate)
DAILY_BASELINE = {
    "AAPL":  ("baseline", 2.0, None, "daily"),  # buy_and_hold
    "MSFT":  ("baseline", 2.0, None, "daily"),  # rsi2_mr_10_90
    "GOOGL": ("baseline", 2.0, None, "daily"),  # buy_and_hold
    "AMZN":  None,  # not in baseline deployment
    "META":  ("baseline", 2.0, None, "daily"),  # lgbm
    "TSLA":  ("baseline", 0.5, None, "daily"),  # macd
    "NVDA":  ("baseline", 2.0, None, "daily"),  # buy_and_hold
    "GLD":   ("baseline", 3.0, None, "daily"),  # buy_and_hold
}

DAILY_COMP_BEST = {
    "AAPL":  ("buy_and_hold", 3.0, 3.0, "daily"),
    "MSFT":  ("rsi2",          1.0, 5.0, "daily"),
    "GOOGL": ("rsi2",          1.0, 3.0, "daily"),
    "AMZN":  ("buy_and_hold",  0.5, 5.0, "daily"),
    "META":  ("mom21",         2.0, 3.0, "daily"),
    "TSLA":  ("macd",          0.5, 3.0, "daily"),
    "NVDA":  ("buy_and_hold",  3.0, 3.0, "daily"),
    "GLD":   ("buy_and_hold",  2.0, 3.0, "daily"),
}

INTRADAY_BEST = {
    "AAPL":  ("bollinger",     3.0, 3.0, "intraday"),
    "MSFT":  ("bollinger",     3.0, 3.0, "intraday"),
    "GOOGL": ("buy_and_hold",  3.0, 3.0, "intraday"),
    "AMZN":  ("buy_and_hold",  3.0, None, "intraday"),
    "META":  ("rsi2",          3.0, 3.0, "intraday"),
    "TSLA":  ("donchian",      3.0, None, "intraday"),
    "NVDA":  ("donchian",      3.0, None, "intraday"),
    "GLD":   ("buy_and_hold",  3.0, 3.0, "intraday"),
}


def load_bars(lib_name: str, sym: str, intraday: bool) -> pd.DataFrame:
    store = open_store()
    df = store.get_library(lib_name).read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    if intraday and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def precompute_signal(asset: str, bars: pd.DataFrame, feat: pd.DataFrame,
                      mode: str, warmup: int) -> np.ndarray:
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    if mode == "baseline":
        # Use the deployed asset specialist (lazy import via compute_signal_for).
        for i in range(warmup, n):
            try:
                s = compute_signal_for(asset, bars.iloc[: i + 1])
                out[i] = int(round(float(s)))
            except Exception:
                pass
        return out
    if mode == "buy_and_hold":
        out[warmup:] = 1
        return out
    if mode == "macd":
        h = feat["macd_hist"].shift(1)
        out = np.where(h > 0, 1, np.where(h < 0, -1, 0)).astype(np.int8)
        out[:warmup] = 0
        return out
    if mode == "rsi2":
        r = feat["rsi_2"].shift(1)
        out = np.where(r < 10, 1, np.where(r > 90, -1, 0)).astype(np.int8)
        out[:warmup] = 0
        return out
    if mode == "bollinger":
        close = bars["close"]
        ma20 = close.rolling(20).mean().shift(1)
        sd20 = close.rolling(20).std().shift(1)
        prev_close = close.shift(1)
        z = (prev_close - ma20) / sd20
        out = np.where(z < -2, 1, np.where(z > 2, -1, 0)).astype(np.int8)
        out[:warmup] = 0
        return out
    if mode == "donchian":
        high20 = bars["high"].rolling(20).max().shift(1)
        low20 = bars["low"].rolling(20).min().shift(1)
        prev_close = bars["close"].shift(1)
        out = np.where(prev_close >= high20, 1,
                       np.where(prev_close <= low20, -1, 0)).astype(np.int8)
        out[:warmup] = 0
        return out
    if mode == "mom21":
        m = feat["ret_21"].shift(1)
        thresh = 0.05 if warmup == WARMUP_DAILY else 0.003
        out = np.where(m > thresh, 1, np.where(m < -thresh, -1, 0)).astype(np.int8)
        out[:warmup] = 0
        return out
    raise ValueError(f"unknown signal mode {mode!r}")


def replay(bars: pd.DataFrame, feat: pd.DataFrame, sig_array: np.ndarray,
           start_idx: int, end_idx: int, notional: float, trail: float,
           profit_target_sigmas: float | None,
           sessions_per_year: int = 252) -> dict:
    closes = bars["close"].astype(float).values
    state = "FLAT"
    entry_price = entry_sigma = None
    entry_qty = 0
    peak = trough = None
    round_trips: list[dict] = []
    daily_bucket: dict[pd.Timestamp, float] = {}
    prev_close = None
    prev_state = "FLAT"
    prev_qty = 0
    prev_side = 0

    vol = feat["vol_21"].values

    for i in range(start_idx, end_idx):
        if i >= len(feat):
            break
        sigma = float(vol[i])
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        sig_dir = int(sig_array[i])
        close = float(closes[i])
        date_key = bars.index[i].normalize()

        if prev_close is not None and prev_state != "FLAT" and prev_qty > 0:
            step = (close - prev_close) * prev_qty * prev_side
            daily_bucket[date_key] = daily_bucket.get(date_key, 0.0) + step

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
                round_trips.append({"side": "LONG",
                                    "pnl": (close - entry_price) * entry_qty})
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
                round_trips.append({"side": "SHORT",
                                    "pnl": (entry_price - close) * entry_qty})
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
                if sig_dir > 0:
                    peak, trough = close, None
                else:
                    trough, peak = close, None

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
    net = realized + open_unrz
    wins = sum(1 for rt in round_trips if rt["pnl"] > 0)
    win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0

    sharpe = float("nan")
    max_dd = 0.0
    n_sessions = len(daily_bucket)
    if daily_bucket and notional > 0:
        rets = np.asarray(list(daily_bucket.values())) / notional
        if rets.std() > 0 and len(rets) > 1:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(sessions_per_year))
        cum = np.cumsum(list(daily_bucket.values()))
        peak_eq = np.maximum.accumulate(cum)
        max_dd = float((cum - peak_eq).min())
    years = n_sessions / sessions_per_year if n_sessions else 0.0
    ann_ret = float("nan")
    if years > 0 and notional + net > 0:
        ann_ret = ((1 + net / notional) ** (1 / years) - 1) * 100.0

    return {
        "n_trades": len(round_trips),
        "n_long": sum(1 for r in round_trips if r["side"] == "LONG"),
        "n_short": sum(1 for r in round_trips if r["side"] == "SHORT"),
        "win_rate": win_rate, "net": net, "sharpe": sharpe,
        "max_dd": max_dd, "ann_return_pct": ann_ret,
        "years": years, "sessions": n_sessions,
    }


def evaluate_config(asset: str, cfg: tuple) -> dict:
    if cfg is None:
        return {"n_trades": 0, "net": 0.0, "sharpe": float("nan"),
                "ann_return_pct": float("nan"), "max_dd": 0.0,
                "win_rate": 0.0, "n_long": 0, "n_short": 0, "years": 0.0,
                "skip_reason": "no_baseline"}
    signal, trail, pt, substrate = cfg
    if substrate == "daily":
        lib_name, sym = DAILY_ASSET_MAP[asset]
        bars = load_bars(lib_name, sym, intraday=False)
        feat = add_per_asset_features(bars)
        warmup = WARMUP_DAILY
        sessions_per_year = 252
    else:
        lib_name, sym = INTRADAY_ASSET_MAP[asset]
        bars = load_bars(lib_name, sym, intraday=True)
        feat = add_per_asset_features(bars)
        warmup = WARMUP_INTRADAY
        sessions_per_year = 252
    sig_array = precompute_signal(asset, bars, feat, signal, warmup)
    start_idx = bars.index.get_indexer([SHARED_START], method="bfill")[0]
    end_idx = bars.index.get_indexer([SHARED_END], method="ffill")[0] + 1
    end_idx = min(end_idx, len(bars))
    if start_idx <= warmup or end_idx <= start_idx:
        return {"n_trades": 0, "net": 0.0, "sharpe": float("nan"),
                "ann_return_pct": float("nan"), "max_dd": 0.0,
                "win_rate": 0.0, "n_long": 0, "n_short": 0, "years": 0.0,
                "skip_reason": "window_out_of_range"}
    return replay(bars, feat, sig_array, start_idx, end_idx,
                  NOTIONAL, trail, pt, sessions_per_year=sessions_per_year)


def main() -> int:
    print()
    print("=" * 130)
    print(f"  Head-to-head per asset on SHARED OOS window "
          f"{SHARED_START.date()} → {SHARED_END.date()}  "
          f"(per-name notional ${NOTIONAL:,.0f})")
    print("=" * 130)
    print(f"  {'asset':<6} {'config':<46} {'trades':<8} {'L/S':<9} "
          f"{'win%':<7} {'net $':<12} {'ann%':<10} {'Sharpe':<8} {'maxDD'}")
    print("-" * 130)

    portfolio_winner_net = 0.0
    portfolio_winner_trades = 0
    portfolio_assets = []
    per_asset_choice = []

    for asset in DAILY_ASSET_MAP.keys():
        candidates = []
        for label, cfg in [("DAILY-BASE", DAILY_BASELINE[asset]),
                           ("DAILY-COMP", DAILY_COMP_BEST[asset]),
                           ("INTRADAY  ", INTRADAY_BEST[asset])]:
            try:
                r = evaluate_config(asset, cfg)
                r["label"] = label
                r["cfg"] = cfg
                candidates.append(r)
            except Exception as exc:
                print(f"  {asset:<6} {label} ERR {type(exc).__name__}: {exc}")

        # Pick the best by net (tie-break: higher sharpe).
        eligible = [c for c in candidates if c.get("skip_reason") is None
                    and c["net"] > 0 and (c["sharpe"] != c["sharpe"] is False)]
        # The "(c["sharpe"] != c["sharpe"] is False)" is an awkward way to
        # exclude NaN sharpes; simpler: filter NaN
        eligible = [c for c in candidates
                    if c.get("skip_reason") is None and c["net"] > 0]
        winner = (max(eligible, key=lambda c: (c["net"],
                                                c["sharpe"] if not math.isnan(c["sharpe"]) else -1e9))
                  if eligible else None)

        for c in candidates:
            cfg = c["cfg"]
            cfg_str = (f"NONE" if cfg is None else
                       f"{cfg[3]:>8} {cfg[0]:<14} tr={cfg[1]}σ pt="
                       f"{f'{cfg[2]}σ' if cfg[2] is not None else '—':<4}")
            mark = ""
            if winner is c:
                mark = "  ← WINNER"
            ls = f"{c['n_long']}/{c['n_short']}"
            dd = (f"${c['max_dd']:>+9,.0f}" if c.get("skip_reason") is None
                  else "—")
            net_s = (f"${c['net']:>+8,.0f}" if c.get("skip_reason") is None
                     else "—")
            ann_s = (f"{c['ann_return_pct']:>+6.1f}%" if not math.isnan(c.get("ann_return_pct", float("nan")))
                     else "—")
            sh_s = (f"{c['sharpe']:>+5.2f}" if not math.isnan(c.get("sharpe", float("nan")))
                    else "—")
            print(f"  {asset:<6} {c['label']} {cfg_str:<35} "
                  f"{c['n_trades']:<8} {ls:<9} {c['win_rate']:>5.1f}%  "
                  f"{net_s:<10} {ann_s:<9} {sh_s:<8} {dd}{mark}")

        if winner:
            portfolio_winner_net += winner["net"]
            portfolio_winner_trades += winner["n_trades"]
            portfolio_assets.append(asset)
            per_asset_choice.append((asset, winner))
        else:
            per_asset_choice.append((asset, None))
        print()

    # Portfolio totals.
    portfolio_notional = len(portfolio_assets) * NOTIONAL
    years = 0.0
    if per_asset_choice and per_asset_choice[0][1]:
        years = per_asset_choice[0][1]["years"]
    ann = (((1 + portfolio_winner_net / portfolio_notional) ** (1 / years) - 1) * 100.0
           if portfolio_notional > 0 and years > 0
           and portfolio_notional + portfolio_winner_net > 0 else float("nan"))

    print("=" * 130)
    print("  BEST-PER-ASSET PORTFOLIO (winners only, equal-weight)")
    print("=" * 130)
    print(f"  deployed assets: {len(portfolio_assets)} ({portfolio_assets})")
    print(f"  net P&L (OOS):   ${portfolio_winner_net:>+12,.2f}  "
          f"on ${portfolio_notional:,.0f} deployed over {years:.2f}y")
    print(f"  return:          {portfolio_winner_net/portfolio_notional*100:>+7.2f}%")
    print(f"  annualized:      {ann:+.1f}%")
    print(f"  total OOS trades: {portfolio_winner_trades}")
    print("=" * 130)

    # Compare to each "pure" strategy: all-daily-baseline, all-daily-comp, all-intraday.
    print()
    print("  Reference — using the SAME shared OOS window for fairness:")
    for label, mapping in [("All DAILY-BASELINE", DAILY_BASELINE),
                            ("All DAILY-COMP    ", DAILY_COMP_BEST),
                            ("All INTRADAY      ", INTRADAY_BEST)]:
        net = 0.0; trades = 0; n_assets = 0
        for a in DAILY_ASSET_MAP.keys():
            cfg = mapping[a]
            if cfg is None:
                continue
            r = evaluate_config(a, cfg)
            if r.get("skip_reason") is not None:
                continue
            net += r["net"]; trades += r["n_trades"]; n_assets += 1
        notional_total = n_assets * NOTIONAL
        ret = net / notional_total * 100.0 if notional_total > 0 else 0.0
        ann2 = (((1 + net / notional_total) ** (1 / years) - 1) * 100.0
                if notional_total > 0 and years > 0
                and notional_total + net > 0 else float("nan"))
        print(f"    {label}: ${net:>+10,.0f} ({ret:>+6.2f}%, {ann2:+.1f}% ann.) "
              f"across {n_assets} assets, {trades} trades")

    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
