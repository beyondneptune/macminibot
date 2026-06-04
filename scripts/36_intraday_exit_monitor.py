"""36_intraday_exit_monitor — runs every 5 minutes during US trading hours,
checks open positions in `data/per_asset_state.json` against current
Alpaca quotes for trailing-stop / profit-target hits, and submits exit
market orders the moment any exit condition is met.

This is the "why hold all day?" fix. Daily runner still owns ENTRY decisions
at 14:35 BST. This monitor owns EXIT decisions at 5-min granularity through
the session.

Per cycle:
  1. Trading-window guard (else exit 6)
  2. Load state file → find positions in IN_LONG / IN_SHORT
  3. Pull current quotes for those tickers from Alpaca IEX
  4. For each open position:
       a. Update running peak/trough using current price
       b. Check trailing stop (peak − entry_price × entry_sigma_log × trail_sigma)
       c. Check profit target if configured (entry × (1 ± entry_sigma × pt_sigmas))
       d. If exit fires → submit market order in the closing direction
  5. Persist updated state + write journal row for any fills

Does NOT compute new entry signals (those stay in the daily runner).

Exit codes:
  0 — completed normally
  2 — missing credentials
  6 — outside US trading window (clean skip)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from execution.alpaca_client import AlpacaClient  # noqa: E402
from shared_core.contracts import OrderIntent, Side, Symbol, Venue  # noqa: E402
from strategy_engine.per_asset_runner import (  # noqa: E402
    AssetState, _ASSET_META, _make_symbol, _quantize_price,
    load_state_file, save_state_file,
)

STRATEGY_SHA = "intraday_exit_v1"
STATE_PATH = PROJECT_ROOT / "data" / "per_asset_state.json"


def setup_logger() -> logging.Logger:
    log_path = PROJECT_ROOT / "data" / f"36_intraday_exit_{date.today().isoformat()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("intraday_exit")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def is_within_us_trading_window(now_utc: datetime) -> bool:
    from zoneinfo import ZoneInfo
    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    minute_of_day = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= minute_of_day < 16 * 60


def fetch_marks(api_key: str, secret: str, tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    data = StockHistoricalDataClient(api_key, secret)
    req = StockLatestQuoteRequest(symbol_or_symbols=tickers)
    quotes = data.get_stock_latest_quote(req)
    marks: dict[str, float] = {}
    for t in tickers:
        q = quotes.get(t)
        if q is None:
            continue
        bid = float(getattr(q, "bid_price", 0.0) or 0.0)
        ask = float(getattr(q, "ask_price", 0.0) or 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
        if mid > 0:
            marks[t] = mid
    return marks


def check_exit(state: AssetState, current_price: float) -> str | None:
    """Returns exit reason ('trailing_stop' | 'profit_target') or None."""
    if state.state == "FLAT" or state.entry_price is None:
        return None
    if state.entry_sigma_log is None or state.trail_sigma is None:
        return None
    band = state.entry_price * state.entry_sigma_log * state.trail_sigma

    if state.state == "IN_LONG":
        # update peak
        if state.peak is None or current_price > state.peak:
            state.peak = current_price
        stop_price = state.peak - band
        if current_price <= stop_price:
            return "trailing_stop"
        if state.profit_target_sigmas is not None:
            pt_price = state.entry_price * (
                1.0 + state.entry_sigma_log * state.profit_target_sigmas
            )
            if current_price >= pt_price:
                return "profit_target"

    elif state.state == "IN_SHORT":
        if state.trough is None or current_price < state.trough:
            state.trough = current_price
        stop_price = state.trough + band
        if current_price >= stop_price:
            return "trailing_stop"
        if state.profit_target_sigmas is not None:
            pt_price = state.entry_price * (
                1.0 - state.entry_sigma_log * state.profit_target_sigmas
            )
            if current_price <= pt_price:
                return "profit_target"

    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-window", action="store_true",
                        help="Bypass trading-window guard. SAFETY: auto-enables "
                             "--whatif unless --i-mean-it is ALSO passed.")
    parser.add_argument("--i-mean-it", action="store_true",
                        help="Authorize real submissions with --force-window.")
    parser.add_argument("--whatif", action="store_true",
                        help="Run logic but skip Alpaca submissions.")
    args = parser.parse_args()

    # SAFETY: --force-window without --i-mean-it = implicit --whatif
    if args.force_window and not args.i_mean_it:
        args.whatif = True

    now_utc = datetime.now(timezone.utc)
    if not args.force_window and not is_within_us_trading_window(now_utc):
        return 6

    logger = setup_logger()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not secret:
        logger.error("missing ALPACA_API_KEY / ALPACA_SECRET")
        return 2

    states = load_state_file(STATE_PATH)
    if not states:
        return 0
    open_positions = {a: s for a, s in states.items() if s.state != "FLAT"}
    if not open_positions:
        return 0

    tickers = sorted(open_positions.keys())
    try:
        marks = fetch_marks(api_key, secret, tickers)
    except Exception as exc:
        logger.exception("fetch_marks failed: %s", exc)
        return 0

    alpaca = AlpacaClient.from_env(PROJECT_ROOT / ".env")
    exit_fills: list[dict] = []
    now_ns = int(time.time() * 1e9)

    # TIME-OF-DAY EXIT FILTER (deployed 2026-05-20 per script 89 empirical test):
    # Block trail-stop exits during 09:30-10:00 ET (first 30 min of session).
    # Empirical finding: 58% of premature exits clustered in this window;
    # vol there is 3.6× daily mean. Skipping exit checks (but still updating
    # peak/trough) here improved 2-year intraday OOS by +$7,995 (+8.0%).
    from zoneinfo import ZoneInfo
    et_now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    et_min = et_now.hour * 60 + et_now.minute
    EXIT_BLOCK_START = 9 * 60 + 30   # 09:30 ET
    EXIT_BLOCK_END = 10 * 60          # 10:00 ET
    in_morning_block = EXIT_BLOCK_START <= et_min < EXIT_BLOCK_END

    if in_morning_block:
        logger.info(f"MORNING BLOCK active (ET {et_now.strftime('%H:%M')}, "
                    f"09:30-10:00 filter) — peak/trough update only, "
                    f"no exits possible. Positions: {sorted(open_positions)}")
    for asset, state in open_positions.items():
        if asset not in marks:
            continue
        current_price = marks[asset]
        if in_morning_block:
            # Update peak/trough so trail bookkeeping stays accurate,
            # but skip exit decision entirely.
            if state.state == "IN_LONG":
                if state.peak is None or current_price > state.peak:
                    state.peak = current_price
            elif state.state == "IN_SHORT":
                if state.trough is None or current_price < state.trough:
                    state.trough = current_price
            continue
        reason = check_exit(state, current_price)
        # Always persist updated peak/trough even if no exit
        if reason is None:
            continue
        # Build exit intent
        was_long = state.state == "IN_LONG"
        exit_side = Side.SELL if was_long else Side.BUY
        qty = state.qty
        if qty <= 0:
            state.state = "FLAT"
            continue
        sym = _make_symbol(asset)
        intent = OrderIntent(
            intent_id=f"intra-exit-{int(time.time())}-{asset}",
            symbol=sym, side=exit_side,
            delta_qty=Decimal(qty),
            mark=_quantize_price(current_price),
            batch_id=f"intra-exit-{date.today().isoformat()}",
            decision_id=f"intra-exit-{int(time.time())}-{asset}",
        )
        try:
            if args.whatif:
                logger.warning(
                    f"WHATIF: would EXIT {asset} {exit_side.value} qty={qty} "
                    f"@ ${current_price:.2f} reason={reason} — SKIPPED"
                )
                continue
            coid = alpaca.submit_order(intent, account="PAPER",
                                       attempt=0, strategy_sha=STRATEGY_SHA)
            pnl = ((current_price - state.entry_price) * qty
                   if was_long else (state.entry_price - current_price) * qty)
            pnl_pct = ((current_price / state.entry_price - 1.0) * 100.0
                       if was_long else (state.entry_price / current_price - 1.0) * 100.0)
            logger.info(
                f"EXIT {asset} {exit_side.value} qty={qty} @ ${current_price:.2f} "
                f"reason={reason} entry=${state.entry_price:.2f} "
                f"P&L=${pnl:+,.2f} ({pnl_pct:+.2f}%) coid={coid}"
            )
            exit_fills.append({
                "ts_unix": time.time(),
                "intent_id": intent.intent_id,
                "client_order_id": coid,
                "symbol": asset,
                "side": exit_side.value,
                "qty": qty, "fill_qty": qty,
                "mark": current_price, "status": "FILLED",
                "strategy_sha": STRATEGY_SHA,
                "batch_id": intent.batch_id,
                "decision_id": intent.decision_id,
                "exit_reason": reason,
            })
            # Reset state to FLAT
            state.state = "FLAT"
            state.entry_price = None
            state.entry_sigma_log = None
            state.profit_target_sigmas = None
            state.peak = None
            state.trough = None
            state.qty = 0
            state.entry_ts_ns = None
        except Exception as exc:
            logger.exception("submit_order failed for %s: %s", asset, exc)

    # WHATIF state-isolation: skip persistence so smoke tests don't corrupt state.
    if args.whatif:
        logger.info("WHATIF: skipping state persistence — no mutations")
    else:
        save_state_file(STATE_PATH, states)

    if exit_fills:
        dt = date.today().isoformat()
        journal_dir = PROJECT_ROOT / "data" / "paper" / f"dt={dt}"
        journal_dir.mkdir(parents=True, exist_ok=True)
        out = journal_dir / f"intraday_exits_{int(time.time())}.parquet"
        try:
            pd.DataFrame(exit_fills).to_parquet(out, index=False)
            logger.info("journal: wrote %d exits → %s", len(exit_fills), out)
        except Exception as exc:
            logger.exception("journal write failed: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
