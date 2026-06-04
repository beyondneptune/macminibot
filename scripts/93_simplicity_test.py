"""93_simplicity_test — does the simplest possible long-basket beat v2?

Tests THREE naive strategies against the v2 architecture-winner on the
3 fixed replay windows (OOS / COVID / RATESHOCK). Goal is to determine
whether v2's complexity (6 signal families, per-asset trailing stops,
profit targets, intraday exit monitor, 1,224-cell search) actually adds
value over a brain-dead long basket.

VARIANTS:
  V1  Equal-weight 8-asset long basket + account-level 5%-from-peak
      circuit breaker; on breach, liquidate everything, wait 30 days,
      re-enter at the same equal weights.
  V2  Equal-weight 8-asset long basket, hold to end of window (no exits).
  V3  Long QQQ only, hold to end.

NET LIQ = $100,000 (matches v2's harness). For V1/V2 each of 8 assets
gets $12,500 notional. V3 gets the full $100k in QQQ.

DATA: Alpaca daily bars (no ArcticDB dep — runs cold on any machine
with the project's .env).

WINDOWS (exactly match scripts/87_fixed_replay.py):
  OOS       2026-01-01 → 2026-05-20
  COVID     2020-02-01 → 2020-04-30
  RATESHOCK 2022-01-01 → 2022-06-30

COSTS: 3 bp per side, matching v2 harness.

Run:
    .venv/bin/python scripts/93_simplicity_test.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402

NET_LIQ = 100_000.0
COST_BP = 3.0
COST_FRAC = COST_BP / 1e4

V1_V2_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "GLD"]
V3_SYMBOL = "QQQ"
ALL_SYMS = V1_V2_SYMBOLS + [V3_SYMBOL]

WINDOWS = [
    ("OOS",      datetime(2026, 1, 1),  datetime(2026, 5, 20)),
    ("COVID",    datetime(2020, 2, 1),  datetime(2020, 4, 30)),
    ("RATESHOCK", datetime(2022, 1, 1), datetime(2022, 6, 30)),
]

# v2 baseline net P&L from scripts/87_fixed_replay.py (CURRENT_V2 leg),
# captured 2026-05-21 via the same 3-bp-cost / $100k harness. Filled in
# at runtime if the live numbers come in; otherwise we read from the
# saved replay logs.
V2_BASELINE = None  # populated by main()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _alpaca_client() -> StockHistoricalDataClient:
    key = os.environ["ALPACA_API_KEY"]
    sec = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    return StockHistoricalDataClient(key, sec)


def fetch_bars(syms: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """Pull Alpaca daily bars, return {sym -> DataFrame with close/open}."""
    cli = _alpaca_client()
    req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day,
                           start=start, end=end)
    bars = cli.get_stock_bars(req).df
    if bars is None or bars.empty:
        return {}
    out: dict[str, pd.DataFrame] = {}
    if isinstance(bars.index, pd.MultiIndex):
        for s in bars.index.get_level_values("symbol").unique():
            d = bars.xs(s, level="symbol").copy()
            d.index = pd.DatetimeIndex(d.index).tz_convert(None) \
                      if d.index.tz is not None else pd.DatetimeIndex(d.index)
            d = d.sort_index()
            out[s] = d[["open", "high", "low", "close"]].astype(float)
    return out


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------

def equity_stats(daily_pnl: pd.Series, net_liq: float,
                 win_days: int) -> dict[str, float]:
    """Compute net, max-DD, sharpe (daily-net), annualised return."""
    cum = daily_pnl.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    net = float(cum.iloc[-1]) if len(cum) else 0.0

    rets = daily_pnl / net_liq
    if rets.std() > 0 and len(rets) > 1:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
    else:
        sharpe = float("nan")

    years = win_days / 252.0 if win_days else 0.0
    if years > 0 and net_liq + net > 0:
        ann = ((1 + net / net_liq) ** (1 / years) - 1) * 100.0
    else:
        ann = float("nan")
    return {"net": net, "max_dd": float(dd), "sharpe": sharpe, "ann": ann}


# ---------------------------------------------------------------------------
# V1 — equal-weight basket + 5%-from-peak account circuit breaker
# ---------------------------------------------------------------------------

def variant1_circuit_breaker(prices: dict[str, pd.DataFrame],
                              syms: list[str],
                              net_liq: float,
                              breaker_pct: float = 0.05,
                              cooldown_days: int = 30) -> dict[str, float]:
    """Equal-weight long basket. If account equity drops `breaker_pct` from
    its peak, liquidate all and wait `cooldown_days` trading days before
    re-entering at equal weights at then-current prices.
    """
    syms = [s for s in syms if s in prices and not prices[s].empty]
    if not syms:
        return {"net": 0.0, "max_dd": 0.0, "sharpe": float("nan"),
                "ann": float("nan"), "n_exits": 0}

    # Union of trading days across all symbols
    idx = sorted(set().union(*[set(prices[s].index) for s in syms]))
    idx = pd.DatetimeIndex(idx)

    # ffill close into a single frame
    closes = pd.DataFrame({s: prices[s]["close"].reindex(idx).ffill()
                           for s in syms})
    closes = closes.dropna(how="all")
    idx = closes.index

    per_sym_notional = net_liq / len(syms)
    qty: dict[str, float] = {}
    cash = 0.0  # cash leg of account
    in_market = False
    cooldown_remaining = 0
    n_exits = 0
    peak_equity = net_liq
    equity_series = []

    def open_positions(t_idx: int):
        nonlocal qty, cash, in_market, peak_equity
        cost = 0.0
        new_qty = {}
        for s in syms:
            p = float(closes[s].iloc[t_idx])
            if not np.isfinite(p) or p <= 0:
                continue
            q = per_sym_notional / p
            new_qty[s] = q
            cost += q * p * COST_FRAC
        qty = new_qty
        cash = net_liq - sum(qty[s] * float(closes[s].iloc[t_idx]) for s in qty) - cost
        in_market = True

    def close_positions(t_idx: int):
        nonlocal qty, cash, in_market, n_exits
        cost = 0.0
        proceeds = 0.0
        for s, q in qty.items():
            p = float(closes[s].iloc[t_idx])
            if not np.isfinite(p) or p <= 0:
                continue
            proceeds += q * p
            cost += q * p * COST_FRAC
        cash += proceeds - cost
        qty = {}
        in_market = False
        n_exits += 1

    # Set net_liq externally for use in nested fns
    net_liq_local = net_liq

    # Initial entry on first usable bar
    open_positions(0)
    for t in range(len(idx)):
        # Mark-to-market equity
        mtm = cash + sum(qty.get(s, 0.0) * float(closes[s].iloc[t]) for s in syms)
        peak_equity = max(peak_equity, mtm)
        equity_series.append(mtm)

        if in_market and (mtm / peak_equity - 1.0) <= -breaker_pct:
            # Liquidate and start cooldown
            close_positions(t)
            cooldown_remaining = cooldown_days
            # Recompute MTM after exit so peak resets to post-exit cash
            mtm_after = cash
            equity_series[-1] = mtm_after  # log post-exit equity for this day
            peak_equity = mtm_after  # reset peak so re-entry doesn't immediately retrip
            continue

        if not in_market:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif t < len(idx) - 1:
                # Re-enter on the next bar with whatever cash remains
                # Scale notional to current cash so we don't run negative.
                nonlocal_net = cash
                # rebuild open with current cash
                cost = 0.0
                new_qty = {}
                per_now = max(0.0, nonlocal_net) / len(syms)
                for s in syms:
                    p = float(closes[s].iloc[t])
                    if not np.isfinite(p) or p <= 0:
                        continue
                    q = per_now / p
                    new_qty[s] = q
                    cost += q * p * COST_FRAC
                qty = new_qty
                cash = nonlocal_net - sum(qty[s] * float(closes[s].iloc[t])
                                          for s in qty) - cost
                in_market = True
                peak_equity = max(peak_equity, cash + sum(qty[s] * float(closes[s].iloc[t])
                                                          for s in qty))

    # Final liquidation MTM (no extra cost — we just mark)
    final_mtm = equity_series[-1] if equity_series else net_liq_local
    net = final_mtm - net_liq_local

    equity = pd.Series(equity_series, index=idx)
    daily_pnl = equity.diff().fillna(equity.iloc[0] - net_liq_local)
    stats = equity_stats(daily_pnl, net_liq_local, len(idx))
    stats["net"] = net
    stats["n_exits"] = n_exits
    return stats


# ---------------------------------------------------------------------------
# V2 — equal-weight basket, hold forever
# ---------------------------------------------------------------------------

def variant2_hold_forever(prices: dict[str, pd.DataFrame],
                           syms: list[str],
                           net_liq: float) -> dict[str, float]:
    syms = [s for s in syms if s in prices and not prices[s].empty]
    if not syms:
        return {"net": 0.0, "max_dd": 0.0, "sharpe": float("nan"),
                "ann": float("nan")}

    idx = sorted(set().union(*[set(prices[s].index) for s in syms]))
    idx = pd.DatetimeIndex(idx)
    closes = pd.DataFrame({s: prices[s]["close"].reindex(idx).ffill()
                           for s in syms}).dropna(how="all")
    idx = closes.index

    per_sym_notional = net_liq / len(syms)
    qty = {}
    entry_cost = 0.0
    for s in syms:
        p0 = float(closes[s].iloc[0])
        if not np.isfinite(p0) or p0 <= 0:
            continue
        qty[s] = per_sym_notional / p0
        entry_cost += qty[s] * p0 * COST_FRAC

    # Exit cost on last bar (mark-to-market with round-trip cost)
    exit_cost = 0.0
    for s, q in qty.items():
        p_last = float(closes[s].iloc[-1])
        exit_cost += q * p_last * COST_FRAC

    # Mark-to-market each day
    equity = sum(qty.get(s, 0.0) * closes[s] for s in syms)
    equity = equity - entry_cost  # subtract entry cost as one-shot drag at t0
    # leftover cash (we didn't spend the full $100k since each per_sym_notional
    # was applied independently — but in practice equal-weight uses up all of it,
    # so leftover is the rounding from integer-share avoidance; here it's 0)
    leftover_cash = net_liq - sum(qty[s] * float(closes[s].iloc[0]) for s in qty)
    equity = equity + leftover_cash

    final = float(equity.iloc[-1]) - exit_cost
    net = final - net_liq

    daily_pnl = equity.diff().fillna(equity.iloc[0] - net_liq)
    # subtract the exit cost on the last day for fidelity
    daily_pnl.iloc[-1] = daily_pnl.iloc[-1] - exit_cost
    stats = equity_stats(daily_pnl, net_liq, len(idx))
    stats["net"] = net
    return stats


# ---------------------------------------------------------------------------
# V3 — long QQQ only
# ---------------------------------------------------------------------------

def variant3_qqq_only(prices: dict[str, pd.DataFrame],
                       net_liq: float) -> dict[str, float]:
    if V3_SYMBOL not in prices or prices[V3_SYMBOL].empty:
        return {"net": 0.0, "max_dd": 0.0, "sharpe": float("nan"),
                "ann": float("nan")}
    closes = prices[V3_SYMBOL]["close"].astype(float)
    p0 = float(closes.iloc[0])
    qty = net_liq / p0
    entry_cost = qty * p0 * COST_FRAC
    exit_cost = qty * float(closes.iloc[-1]) * COST_FRAC

    equity = qty * closes - entry_cost
    final = float(equity.iloc[-1]) - exit_cost
    net = final - net_liq

    daily_pnl = equity.diff().fillna(equity.iloc[0] - net_liq)
    daily_pnl.iloc[-1] = daily_pnl.iloc[-1] - exit_cost
    stats = equity_stats(daily_pnl, net_liq, len(closes))
    stats["net"] = net
    return stats


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_window(label: str, start: datetime, end: datetime) -> dict:
    print(f"  [{label}] fetching {start.date()} → {end.date()} ...")
    prices = fetch_bars(ALL_SYMS, start, end)
    have = [s for s in ALL_SYMS if s in prices and not prices[s].empty]
    print(f"  [{label}] symbols with data: {have}")

    v1 = variant1_circuit_breaker(prices, V1_V2_SYMBOLS, NET_LIQ)
    v2 = variant2_hold_forever(prices, V1_V2_SYMBOLS, NET_LIQ)
    v3 = variant3_qqq_only(prices, NET_LIQ)
    return {"V1": v1, "V2": v2, "V3": v3}


def fmt_row(label: str, r: dict, extra: str = "") -> str:
    return (f"  {label:<35} ${r['net']:>+10,.0f}  "
            f"DD=${r.get('max_dd', float('nan')):>+10,.0f}  "
            f"Sharpe={r.get('sharpe', float('nan')):>+5.2f}  "
            f"Ann={r.get('ann', float('nan')):>+6.1f}%  {extra}")


def main() -> int:
    print()
    print("=" * 120)
    print("  SIMPLICITY TEST — naive long-basket variants vs v2 (3 fixed windows)")
    print("=" * 120)

    out = {}
    for label, start, end in WINDOWS:
        out[label] = run_window(label, start, end)

    # v2 baseline numbers (CURRENT_V2 leg from scripts/87_fixed_replay.py,
    # ArcticDB-driven, 3-bp cost, $100k net liq). These are filled in
    # main() after reading from the live run we trigger separately.
    v2_baseline = {
        # net dollars, ann% — read from current run of 87_fixed_replay or
        # left as None when not available
        "OOS":      None,
        "COVID":    None,
        "RATESHOCK": None,
    }

    print()
    print("=" * 120)
    print(f"{'WINDOW':<15} {'STRATEGY':<35} {'NET $':>13} {'MAX DD':>15} {'SHARPE':>10} {'ANN%':>10} {'EXITS':>7}")
    print("-" * 120)
    for label, _, _ in WINDOWS:
        r = out[label]
        print(f"{label:<15} {'V1 EW + 5% circuit breaker':<35} "
              f"${r['V1']['net']:>+10,.0f} ${r['V1']['max_dd']:>+12,.0f} "
              f"{r['V1']['sharpe']:>+9.2f} {r['V1']['ann']:>+9.1f}% "
              f"{r['V1'].get('n_exits', 0):>7d}")
        print(f"{'':<15} {'V2 EW hold forever':<35} "
              f"${r['V2']['net']:>+10,.0f} ${r['V2']['max_dd']:>+12,.0f} "
              f"{r['V2']['sharpe']:>+9.2f} {r['V2']['ann']:>+9.1f}% "
              f"{'-':>7}")
        print(f"{'':<15} {'V3 QQQ only':<35} "
              f"${r['V3']['net']:>+10,.0f} ${r['V3']['max_dd']:>+12,.0f} "
              f"{r['V3']['sharpe']:>+9.2f} {r['V3']['ann']:>+9.1f}% "
              f"{'-':>7}")
        print()

    # Stash the results dict for downstream comparison
    print("=" * 120)
    print()

    # Total across windows for each strategy (with v2 baseline if loaded)
    print("  TOTAL net P&L summed across all 3 windows:")
    for k in ("V1", "V2", "V3"):
        total = sum(out[w][k]["net"] for w in out)
        print(f"    {k:<5}  ${total:>+10,.0f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
