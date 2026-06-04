"""47_signal_preview — predict EXACTLY what 30_per_asset_runner will emit at 14:35 BST.

Pre-fire dry-preview for the per-asset trailing-stop production runner.
Mirrors the state-machine logic in
``strategy_engine.per_asset_runner.PerAssetRunner._step_state_machine`` —
but READS state from ``data/per_asset_state.json`` WITHOUT mutating it,
and uses the CURRENT Alpaca quote rather than yesterday's daily close for
the trail/PT distance calculation only (signal computation still uses
T-1 daily bars exactly as the runner does).

This script is SAFE to run repeatedly any time of day. It never:
  - opens orders,
  - mutates the per-asset state file, or
  - calls into the risk chain (chain results are inferred from current
    positions/marks/account state via a simple net-liq probe).

Usage::

    .venv/bin/python scripts/47_signal_preview.py

Output format
-------------
For each asset:
  - last daily close (T-1) and current Alpaca mid (live)
  - latest specialist signal (-1 / 0 / +1)
  - state today (FLAT / IN_LONG / IN_SHORT) and qty
  - what the runner WOULD do at the next 14:35 BST fire if state machine
    were stepped against the current signal + current Alpaca mid
  - trailing-stop distance (price − stop) in absolute and % terms
  - profit-target distance (PT price − price) in absolute and % terms

Then a portfolio summary: total expected entries, exits, no-ops, and the
total notional turnover (|delta_qty * mark|) summed across all predicted
actions.

Exit codes:
  0  — clean preview (regardless of how many actions are predicted)
  2  — missing Alpaca credentials
  3  — Alpaca quotes unreachable
  5  — internal error
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from ml_research.asset_specialists import (  # noqa: E402
    SUPPORTED_ASSETS,
    compute_signal_for,
    winner_for,
)
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from strategy_engine.per_asset_runner import (  # noqa: E402
    _ASSET_META,
    AssetState,
    load_state_file,
)


# Extra ETF universe entries the production runner does NOT include today,
# but we preview anyway so the operator sees what they WOULD trade if the
# leveraged sleeve were turned on. SOXL + NVDL have specialist modules;
# TSLL does not (no compute_signal). We skip TSLL gracefully.
EXTRA_ETF_META: dict[str, tuple[str, str]] = {
    "SOXL": ("etf.daily_bars", "ARCA.SOXL"),
    "NVDL": ("etf.daily_bars", "ARCA.NVDL"),
    "TSLL": ("etf.daily_bars", "ARCA.TSLL"),
}


# Matches the DAILY_OVERRIDES dict in scripts/30_per_asset_runner.py.
# Duplicated here (not imported) to keep this script independent of the
# runner's import-time side effects (heartbeat write).
DAILY_OVERRIDES: dict[str, dict[str, Any]] = {
    "AAPL": {"trail_sigma": 3.0, "profit_target_sigmas": 3.0,
             "source": "daily_v2_aapl_widen_trail_add_pt"},
    "MSFT": {"trail_sigma": 1.0, "profit_target_sigmas": 5.0,
             "source": "daily_v2_msft_tighten_trail_add_pt"},
    "AMZN": {"model": "buy_and_hold", "scope": "asset_only",
             "trail_sigma": 0.5, "profit_target_sigmas": 5.0,
             "source": "daily_v2_amzn_add"},
    "TSLA": {"profit_target_sigmas": 3.0,
             "source": "daily_v2_tsla_pt"},
    "GLD":  {"trail_sigma": 2.0, "profit_target_sigmas": 3.0,
             "source": "daily_v2_tighten"},
}


# ---------------------------------------------------------------------------
# Bar/feature/signal helpers — identical math to the runner
# ---------------------------------------------------------------------------


def load_daily_bars(library: str, sym: str):
    import pandas as pd
    from scripts._storage import open_store

    store = open_store()
    df = store[library].read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def latest_sigma_log(bars) -> float | None:
    """Replicates ``PerAssetRunner._latest_sigma`` exactly."""
    import math
    feat = add_per_asset_features(bars)
    if "vol_21" not in feat.columns or feat.empty:
        return None
    val = feat["vol_21"].iloc[-1]
    if not math.isfinite(val) or val <= 0:
        return None
    return float(val)


# ---------------------------------------------------------------------------
# Alpaca quote + account fetch
# ---------------------------------------------------------------------------


def fetch_marks(api_key: str, secret: str, tickers: tuple[str, ...]) -> dict[str, Decimal]:
    """Pull live IEX quotes (same path the runner uses)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    data = StockHistoricalDataClient(api_key, secret)
    req = StockLatestQuoteRequest(symbol_or_symbols=list(tickers))
    quotes = data.get_stock_latest_quote(req)
    marks: dict[str, Decimal] = {}
    for t in tickers:
        q = quotes.get(t)
        if q is None:
            continue
        bid = float(getattr(q, "bid_price", 0.0) or 0.0)
        ask = float(getattr(q, "ask_price", 0.0) or 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
        if mid <= 0:
            continue
        marks[t] = Decimal(f"{mid:.4f}")
    return marks


def fetch_net_liq(api_key: str, secret: str) -> Decimal:
    """Pull net-liq from the paper account so we can size hypothetical entries."""
    from alpaca.trading.client import TradingClient

    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    tc = TradingClient(api_key, secret, paper="paper" in base_url.lower())
    acct = tc.get_account()
    return Decimal(str(getattr(acct, "portfolio_value", acct.equity)))


# ---------------------------------------------------------------------------
# Sizing — same equal-weight formula as PerAssetRunner._share_qty
# ---------------------------------------------------------------------------


def share_qty(net_liq: Decimal, last_close: float, portfolio_size: int = 8) -> int:
    if net_liq is None or net_liq <= 0 or last_close <= 0:
        return 0
    notional_per_position = Decimal(net_liq) / Decimal(portfolio_size)
    shares = (notional_per_position / Decimal(str(last_close))).to_integral_value(
        rounding=ROUND_DOWN
    )
    return max(0, int(shares))


# ---------------------------------------------------------------------------
# Prediction record — what the runner WOULD do
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    asset: str
    last_close: float          # T-1 daily close (what the runner trades against)
    live_mark: float | None    # current Alpaca mid (for trail/PT distance display)
    signal: int                # -1, 0, +1
    state: str                 # FLAT / IN_LONG / IN_SHORT
    state_qty: int             # qty held per state file
    entry_price: float | None  # entry_price from state (None if FLAT)
    peak_or_trough: float | None  # running peak (IN_LONG) or trough (IN_SHORT)
    action: str                # HOLD / BUY_ENTRY / SELL_EXIT / SHORT_ENTRY / COVER_EXIT
    action_qty: int            # 0 for HOLD
    notional: float            # action_qty * last_close, 0 for HOLD
    reasoning: str             # human-readable explanation
    trail_stop_price: float | None    # stop price if IN_LONG/IN_SHORT
    trail_distance: float | None      # last_close − stop_price (+ for long, − for short)
    trail_pct: float | None           # trail_distance / last_close * 100
    pt_price: float | None            # profit-target price if PT set
    pt_distance: float | None         # last_close − pt_price (+ when below for long)
    pt_pct: float | None              # |pt_distance / last_close * 100|


# ---------------------------------------------------------------------------
# State-machine simulator — PURE, no I/O, mirrors _step_state_machine
# ---------------------------------------------------------------------------


def predict_action(
    *,
    asset: str,
    state: AssetState,
    signal_dir: int,
    last_close: float,
    sigma_log: float,
    trail_sigma_w: float,
    profit_target_sigmas_w: float | None,
    net_liq: Decimal,
    portfolio_size: int = 8,
    live_mark: float | None = None,
) -> Prediction:
    """One-step prediction of what PerAssetRunner._step_state_machine would emit.

    Pure — does NOT mutate ``state``. Mirrors the priority order in the runner:
    trailing-stop first, profit-target second, signal-flip third, FLAT-entry
    fourth, otherwise HOLD.
    """
    # 1) maintain hypothetical peak/trough using the runner's prior-bar close
    #    convention. We do NOT mutate state.peak/state.trough; we copy them.
    hyp_peak = state.peak
    hyp_trough = state.trough
    if state.state == "IN_LONG":
        if hyp_peak is None or last_close > hyp_peak:
            hyp_peak = last_close
    elif state.state == "IN_SHORT":
        if hyp_trough is None or last_close < hyp_trough:
            hyp_trough = last_close
    pk_or_tr = hyp_peak if state.state == "IN_LONG" else (hyp_trough if state.state == "IN_SHORT" else None)

    # Use frozen-at-entry sigma if available, else live sigma — matches the
    # runner's ``state.entry_sigma_log or sigma_log`` precedence.
    entry_sig = state.entry_sigma_log if state.entry_sigma_log else sigma_log
    trail_w = state.trail_sigma if state.trail_sigma else trail_sigma_w
    pt_w = (state.profit_target_sigmas
            if state.profit_target_sigmas is not None
            else profit_target_sigmas_w)

    trail_stop_price: float | None = None
    trail_distance: float | None = None
    trail_pct: float | None = None
    pt_price: float | None = None
    pt_distance: float | None = None
    pt_pct: float | None = None

    if state.state in ("IN_LONG", "IN_SHORT") and state.entry_price is not None:
        band = state.entry_price * entry_sig * trail_w
        if state.state == "IN_LONG" and hyp_peak is not None:
            trail_stop_price = hyp_peak - band
            trail_distance = last_close - trail_stop_price
            trail_pct = trail_distance / last_close * 100.0 if last_close > 0 else None
        elif state.state == "IN_SHORT" and hyp_trough is not None:
            trail_stop_price = hyp_trough + band
            trail_distance = last_close - trail_stop_price  # negative when above stop
            trail_pct = trail_distance / last_close * 100.0 if last_close > 0 else None

    if (state.state in ("IN_LONG", "IN_SHORT")
            and state.entry_price is not None
            and pt_w is not None):
        if state.state == "IN_LONG":
            pt_price = state.entry_price * (1.0 + entry_sig * pt_w)
            pt_distance = pt_price - last_close
        else:
            pt_price = state.entry_price * (1.0 - entry_sig * pt_w)
            pt_distance = pt_price - last_close
        pt_pct = (pt_distance / last_close * 100.0) if last_close > 0 else None

    # Common kwargs for every Prediction return — keeps the rest of this
    # function concise.
    base_kw: dict[str, Any] = dict(
        asset=asset, last_close=last_close, live_mark=live_mark,
        signal=signal_dir, state=state.state, state_qty=state.qty,
        entry_price=state.entry_price, peak_or_trough=pk_or_tr,
        trail_stop_price=trail_stop_price, trail_distance=trail_distance,
        trail_pct=trail_pct, pt_price=pt_price, pt_distance=pt_distance,
        pt_pct=pt_pct,
    )

    # Decision logic — same priority order as the runner.
    # 2a) Trailing stop check
    if state.state == "IN_LONG" and trail_stop_price is not None and last_close <= trail_stop_price:
        return Prediction(
            **base_kw, action="SELL_EXIT", action_qty=state.qty,
            notional=state.qty * last_close,
            reasoning=f"trailing_stop tripped: close ${last_close:.2f} <= stop ${trail_stop_price:.2f}",
        )
    if state.state == "IN_SHORT" and trail_stop_price is not None and last_close >= trail_stop_price:
        return Prediction(
            **base_kw, action="COVER_EXIT", action_qty=state.qty,
            notional=state.qty * last_close,
            reasoning=f"trailing_stop tripped: close ${last_close:.2f} >= stop ${trail_stop_price:.2f}",
        )

    # 2b) Profit target check
    if state.state == "IN_LONG" and pt_price is not None and last_close >= pt_price:
        return Prediction(
            **base_kw, action="SELL_EXIT", action_qty=state.qty,
            notional=state.qty * last_close,
            reasoning=f"profit_target hit: close ${last_close:.2f} >= PT ${pt_price:.2f}",
        )
    if state.state == "IN_SHORT" and pt_price is not None and last_close <= pt_price:
        return Prediction(
            **base_kw, action="COVER_EXIT", action_qty=state.qty,
            notional=state.qty * last_close,
            reasoning=f"profit_target hit: close ${last_close:.2f} <= PT ${pt_price:.2f}",
        )

    # 3) Signal flip exit
    if state.state == "IN_LONG" and signal_dir < 0:
        return Prediction(
            **base_kw, action="SELL_EXIT", action_qty=state.qty,
            notional=state.qty * last_close,
            reasoning="signal flipped negative — exit long",
        )
    if state.state == "IN_SHORT" and signal_dir > 0:
        return Prediction(
            **base_kw, action="COVER_EXIT", action_qty=state.qty,
            notional=state.qty * last_close,
            reasoning="signal flipped positive — cover short",
        )

    # 4) FLAT + signal → entry
    if state.state == "FLAT" and signal_dir != 0:
        qty = share_qty(net_liq, last_close, portfolio_size)
        if qty <= 0:
            return Prediction(
                **base_kw, action="HOLD", action_qty=0, notional=0.0,
                reasoning=f"signal {signal_dir:+d} but qty=0 (net_liq too small)",
            )
        if signal_dir > 0:
            return Prediction(
                **base_kw, action="BUY_ENTRY", action_qty=qty,
                notional=qty * last_close,
                reasoning=f"FLAT + signal +1 → long entry qty={qty}",
            )
        return Prediction(
            **base_kw, action="SHORT_ENTRY", action_qty=qty,
            notional=qty * last_close,
            reasoning=f"FLAT + signal -1 → short entry qty={qty}",
        )

    # 5) HOLD
    if state.state == "FLAT":
        reason = "FLAT + signal 0 — no entry"
    else:
        reason = f"{state.state} maintained: signal {signal_dir:+d}, no stop/PT trip"
    return Prediction(
        **base_kw, action="HOLD", action_qty=0, notional=0.0, reasoning=reason,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt_opt_price(v: float | None) -> str:
    return f"${v:.2f}" if v is not None else "—"


def _fmt_opt_pct(v: float | None) -> str:
    return f"{v:+.2f}%" if v is not None else "—"


def print_preview(preds: list[Prediction], net_liq: Decimal) -> None:
    print("=" * 110)
    print(f"SIGNAL PREVIEW — what 14:35 BST fire would emit RIGHT NOW (net_liq=${net_liq})")
    print("=" * 110)
    print()
    # Per-asset detail table
    header = (f"{'Asset':6s}  {'T-1 Close':>10s}  {'Live Mid':>10s}  "
              f"{'Sig':>3s}  {'State':9s}  {'Qty':>5s}  "
              f"{'Action':12s}  {'ActQty':>6s}  {'Notional':>11s}")
    print(header)
    print("-" * len(header))
    for p in preds:
        live_str = _fmt_opt_price(p.live_mark)
        notional_str = f"${p.notional:,.0f}" if p.notional > 0 else "—"
        sig_str = "0" if p.signal == 0 else f"{p.signal:+d}"
        print(f"{p.asset:6s}  ${p.last_close:>9.2f}  {live_str:>10s}  "
              f"{sig_str:>3s}  {p.state:9s}  {p.state_qty:>5d}  "
              f"{p.action:12s}  {p.action_qty:>6d}  {notional_str:>11s}")
    print()
    # Trail/PT detail table (only meaningful for IN_LONG/IN_SHORT)
    print("STOP / PROFIT-TARGET DISTANCES (relative to T-1 close):")
    print("-" * 110)
    tphdr = (f"{'Asset':6s}  {'State':9s}  {'Entry':>10s}  {'Peak/Trough':>12s}  "
             f"{'Trail Stop':>11s}  {'Trail Dist':>11s}  {'PT Price':>10s}  {'PT Dist':>10s}")
    print(tphdr)
    print("-" * len(tphdr))
    for p in preds:
        if p.state == "FLAT":
            continue
        entry_str = _fmt_opt_price(p.entry_price)
        peak_str = _fmt_opt_price(p.peak_or_trough)
        stop_str = _fmt_opt_price(p.trail_stop_price)
        td_str = _fmt_opt_pct(p.trail_pct) if p.trail_pct is not None else "—"
        pt_str = _fmt_opt_price(p.pt_price)
        ptd_str = _fmt_opt_pct(p.pt_pct) if p.pt_pct is not None else "—"
        print(f"{p.asset:6s}  {p.state:9s}  {entry_str:>10s}  {peak_str:>12s}  "
              f"{stop_str:>11s}  {td_str:>11s}  {pt_str:>10s}  {ptd_str:>10s}")
    print()
    # Per-asset reasoning
    print("REASONING:")
    for p in preds:
        print(f"  {p.asset:6s} → {p.reasoning}")
    print()
    # Portfolio summary
    n_buy = sum(1 for p in preds if p.action == "BUY_ENTRY")
    n_short = sum(1 for p in preds if p.action == "SHORT_ENTRY")
    n_sell = sum(1 for p in preds if p.action == "SELL_EXIT")
    n_cover = sum(1 for p in preds if p.action == "COVER_EXIT")
    n_hold = sum(1 for p in preds if p.action == "HOLD")
    total_entries = n_buy + n_short
    total_exits = n_sell + n_cover
    total_notional = sum(p.notional for p in preds)
    print("=" * 110)
    print("PORTFOLIO SUMMARY")
    print("=" * 110)
    print(f"  Total assets previewed       : {len(preds)}")
    print(f"  Predicted entries            : {total_entries}  (BUY_ENTRY={n_buy}, SHORT_ENTRY={n_short})")
    print(f"  Predicted exits              : {total_exits}  (SELL_EXIT={n_sell}, COVER_EXIT={n_cover})")
    print(f"  Predicted HOLDs              : {n_hold}")
    print(f"  Total notional turnover      : ${total_notional:,.0f}")
    print(f"  Net-liq (paper account)      : ${net_liq}")
    pct = (total_notional / float(net_liq) * 100.0) if float(net_liq) > 0 else 0.0
    print(f"  Turnover / NAV               : {pct:.2f}%")
    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 110)
    print("47_signal_preview — non-mutating preview of the next 30_per_asset_runner tick")
    print("=" * 110)

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not secret:
        print("ERROR: missing ALPACA_API_KEY / ALPACA_SECRET in .env")
        return 2
    print(".env loaded (paper-trading)")

    # 1. Build asset universe — same logic the runner uses.
    INTRADAY_MANAGED: set[str] = set()  # mirror current production state
    winners: dict[str, dict[str, Any]] = {}
    for ticker in SUPPORTED_ASSETS:
        if ticker in INTRADAY_MANAGED:
            continue
        spec = winner_for(ticker)
        if spec is None and ticker not in DAILY_OVERRIDES:
            continue
        merged = dict(spec) if spec else {}
        if ticker in DAILY_OVERRIDES:
            merged.update(DAILY_OVERRIDES[ticker])
        winners[ticker] = merged

    # 2. Optionally add SOXL / NVDL / TSLL preview (not in production universe).
    extras_available: dict[str, dict[str, Any]] = {}
    for ticker in EXTRA_ETF_META:
        spec = winner_for(ticker) if ticker in SUPPORTED_ASSETS else None
        # SOXL / NVDL have specialist modules even though they're not in
        # SUPPORTED_ASSETS — try to import directly.
        if spec is None:
            try:
                from importlib import import_module
                mod = import_module(f"ml_research.asset_specialists.{ticker.lower()}")
                spec = getattr(mod, "WINNER", None)
            except ModuleNotFoundError:
                spec = None
        if spec is None:
            continue
        extras_available[ticker] = spec

    all_assets = sorted(set(winners.keys()) | set(extras_available.keys()))
    print(f"production universe ({len(winners)}): {sorted(winners)}")
    print(f"extra ETF preview  ({len(extras_available)}): {sorted(extras_available)}")
    print(f"will preview {len(all_assets)} total assets: {all_assets}")
    print()

    # 3. Pull live marks (best-effort — outside RTH the IEX feed may return
    #    stale quotes; we still display whatever it gives us).
    tickers_for_marks = tuple(all_assets)
    try:
        marks = fetch_marks(api_key, secret, tickers_for_marks)
    except Exception as exc:
        print(f"ERROR fetching marks from Alpaca: {exc}")
        return 3
    print(f"live marks pulled: {len(marks)}/{len(tickers_for_marks)} symbols")

    # 4. Pull net-liq for hypothetical sizing.
    try:
        net_liq = fetch_net_liq(api_key, secret)
    except Exception as exc:
        print(f"ERROR fetching net_liq: {exc}")
        return 3
    print(f"net_liq=${net_liq}")

    # 5. Load persistent state (READ-ONLY — we MUST NOT save it back).
    state_path = PROJECT_ROOT / "data" / "per_asset_state.json"
    states = load_state_file(state_path)
    # Make sure every previewed asset has a state record. We use a separate
    # local dict — load_state_file returns a mutable dict, but we copy
    # AssetState objects too so any in-place mutation by predict_action would
    # NOT leak back. (predict_action is designed pure but we belt+braces.)
    state_snapshots: dict[str, AssetState] = {}
    for a in all_assets:
        existing = states.get(a)
        if existing is None:
            state_snapshots[a] = AssetState(asset=a)
        else:
            # Shallow copy via from_dict(to_dict()) ensures isolation.
            state_snapshots[a] = AssetState.from_dict(existing.to_dict())

    # 6. For each asset: load daily bars, compute signal+sigma, predict action.
    preds: list[Prediction] = []
    skipped: list[tuple[str, str]] = []
    for asset in all_assets:
        winner = winners.get(asset) or extras_available.get(asset)
        if winner is None:
            skipped.append((asset, "no winner spec"))
            continue
        # Pick the library + symbol.
        if asset in _ASSET_META:
            library, sym, _v, _c = _ASSET_META[asset]
        elif asset in EXTRA_ETF_META:
            library, sym = EXTRA_ETF_META[asset]
        else:
            skipped.append((asset, "no bar metadata"))
            continue
        try:
            bars = load_daily_bars(library, sym)
        except Exception as exc:
            skipped.append((asset, f"bar load failed: {exc}"))
            continue
        if bars is None or len(bars) < 30:
            skipped.append((asset, f"insufficient bars ({0 if bars is None else len(bars)})"))
            continue
        sigma_log = latest_sigma_log(bars)
        if sigma_log is None:
            skipped.append((asset, "sigma_unavailable"))
            continue
        try:
            if asset in SUPPORTED_ASSETS:
                sig = compute_signal_for(asset, bars)
            else:
                # Extra ETF preview (SOXL/NVDL/TSLL) — import the specialist
                # module directly since these aren't in SUPPORTED_ASSETS yet.
                from importlib import import_module
                mod = import_module(f"ml_research.asset_specialists.{asset.lower()}")
                sig = float(mod.compute_signal(bars))
        except Exception as exc:
            skipped.append((asset, f"signal_error: {exc}"))
            continue

        last_close = float(bars["close"].iloc[-1])
        live_mark = float(marks[asset]) if asset in marks else None
        state = state_snapshots[asset]
        trail_w = float(winner["trail_sigma"])
        pt_w_raw = winner.get("profit_target_sigmas")
        pt_w = float(pt_w_raw) if pt_w_raw is not None else None
        pred = predict_action(
            asset=asset,
            state=state,
            signal_dir=int(round(sig)),
            last_close=last_close,
            sigma_log=sigma_log,
            trail_sigma_w=trail_w,
            profit_target_sigmas_w=pt_w,
            net_liq=net_liq,
            portfolio_size=8,
            live_mark=live_mark,
        )
        preds.append(pred)

    # 7. Output table + summary
    print_preview(preds, net_liq)

    # Skipped assets
    if skipped:
        print()
        print("SKIPPED (no prediction):")
        for asset, reason in skipped:
            print(f"  {asset:6s} → {reason}")

    # Tail: show that we did NOT mutate state.
    print()
    print(f"state file UNCHANGED: {state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
