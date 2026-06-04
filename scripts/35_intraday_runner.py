"""35_intraday_runner — PRODUCTION 5-min intraday runner for AAPL + MSFT.

Fires every 5 minutes during US trading hours via launchd. Pulls the latest
5-min bars from Alpaca, refreshes the ArcticDB cache, runs the Bollinger
3σ/3σ state machine, validates intents through the 10-rule risk chain
(mode=PAPER), and submits market orders.

Empirically justified by scripts/79_best_per_asset.py — intraday Bollinger
3σ/3σ on AAPL/MSFT delivers Sharpe 1.65/2.29 vs daily baseline 0.87/0.99
on the shared OOS window. Hybrid portfolio with these two intraday plus 6
daily assets shows +66.8% annualized vs +50.7% daily-only.

Exit codes:
  0 — completed normally
  2 — missing credentials
  3 — Alpaca data unreachable
  5 — runner raised
  6 — outside US trading window (clean skip)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _write_runner_heartbeat() -> None:
    import json
    state_dir = Path(os.environ.get("MACMINIBOT_STATE_DIR",
                                    str(PROJECT_ROOT / "state")))
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "heartbeat.json").write_text(json.dumps({
        "pid": os.getpid(),
        "ts_unix": time.time(),
        "version": "intraday-runner-v1",
    }))


_write_runner_heartbeat()

from dotenv import load_dotenv  # noqa: E402

from execution.alpaca_client import AlpacaClient  # noqa: E402
from risk_management.chain import (  # noqa: E402
    ChainContext, run_check_chain,
)
from risk_management.limits import load_limits  # noqa: E402
from risk_management.snapshot import build_snapshot  # noqa: E402
from shared_core.contracts import (  # noqa: E402
    ChainMode, OrderIntent, Side,
)
from strategy_engine.intraday_runner import (  # noqa: E402
    INTRADAY_ASSETS, IntradayRunner, IntradayRunnerConfig,
)

STRATEGY_SHA = "intraday_bollinger_v1"
PER_NAME_NOTIONAL = Decimal("12500")  # $100k / 8 (hybrid book sizing)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("intraday_runner")
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


def refresh_arctic_for_intraday(api_key: str, secret: str,
                                 tickers: tuple[str, ...],
                                 logger: logging.Logger) -> None:
    """Pull last ~6 hours of 5-min bars and upsert into ArcticDB so the
    runner's state machine sees the latest closed bar."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    import pandas as pd

    from scripts._storage import open_store

    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(hours=8)
    client = StockHistoricalDataClient(api_key, secret)
    store = open_store()

    asset_map = {
        "AAPL": ("equity.minute_bars", "NASDAQ.AAPL"),
        "MSFT": ("equity.minute_bars", "NASDAQ.MSFT"),
    }
    req = StockBarsRequest(
        symbol_or_symbols=list(tickers),
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_utc, end=end_utc,
    )
    bars = client.get_stock_bars(req)
    for t in tickers:
        rows = bars.data.get(t, [])
        if not rows:
            logger.warning("refresh: no bars returned for %s", t)
            continue
        records = []
        for b in rows:
            ts = pd.Timestamp(b.timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert("UTC").tz_localize(None)
            records.append({
                "timestamp": ts,
                "open": float(b.open), "high": float(b.high), "low": float(b.low),
                "close": float(b.close), "volume": float(b.volume),
                "bar_count": int(b.trade_count or 0), "wap": float(b.vwap),
            })
        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        df.index.name = "timestamp"
        lib_name, sym = asset_map[t]
        lib = store.get_library(lib_name)
        try:
            lib.update(sym, df, upsert=True)
        except Exception as exc:
            logger.warning("refresh: ArcticDB upsert failed for %s: %s", t, exc)
    logger.info("refresh: pulled %d hours of 5-min bars for %s", 8, list(tickers))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-window", action="store_true",
                        help="Skip the US-trading-window guard (debug only).")
    parser.add_argument("--skip-refresh", action="store_true",
                        help="Skip Alpaca refresh; use ArcticDB as-is.")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = PROJECT_ROOT / "data" / f"35_intraday_run_{today}.log"
    logger = setup_logger(log_path)

    now_utc = datetime.now(timezone.utc)
    if not args.force_window and not is_within_us_trading_window(now_utc):
        return 6

    logger.info("=" * 72)
    logger.info("INTRADAY RUNNER fire @ %s UTC", now_utc.isoformat(timespec="seconds"))
    logger.info("=" * 72)

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not secret:
        logger.error("missing ALPACA_API_KEY / ALPACA_SECRET")
        return 2

    if not args.skip_refresh:
        try:
            refresh_arctic_for_intraday(api_key, secret, INTRADAY_ASSETS, logger)
        except Exception as exc:
            logger.exception("ArcticDB refresh failed: %s", exc)
            return 3

    # Open Alpaca + limits.
    alpaca = AlpacaClient.from_env(PROJECT_ROOT / ".env")
    limits = load_limits(PROJECT_ROOT / "data" / "limits.toml")

    # Snapshot (positions, net_liq).
    acct = alpaca.client.get_account()
    net_liq = Decimal(str(getattr(acct, "portfolio_value", acct.equity)))
    buying_power = Decimal(str(acct.buying_power))
    positions: dict[str, Decimal] = {}
    for pos in alpaca.client.get_all_positions():
        ticker = getattr(pos, "symbol", None)
        qty_raw = getattr(pos, "qty", None)
        if ticker is None or qty_raw is None:
            continue
        qty = Decimal(str(qty_raw))
        side_str = str(getattr(getattr(pos, "side", "long"), "value", "long")).lower()
        if side_str == "short" and qty > 0:
            qty = -qty
        positions[str(ticker)] = qty

    # Build prev_close marks for snapshot — use the asset's last close.
    from strategy_engine.intraday_runner import default_intraday_bar_loader
    prev_close: dict[str, Decimal] = {}
    for t in INTRADAY_ASSETS:
        try:
            df = default_intraday_bar_loader(t)
            prev_close[t] = Decimal(f"{float(df['close'].iloc[-1]):.4f}")
        except Exception:
            prev_close[t] = Decimal("100")

    as_of_ns = int(time.time() * 1e9)
    risk_snap = build_snapshot(
        account_summary={
            "NetLiquidation": str(net_liq),
            "AvailableFunds": str(buying_power),
            "GrossPositionValue": str(net_liq),
        },
        as_of_ns=as_of_ns,
        positions={k: str(v) for k, v in positions.items()},
        broker_positions={k: str(v) for k, v in positions.items()},
        prev_close={k: str(v) for k, v in prev_close.items()},
        feature_ok={t: True for t in INTRADAY_ASSETS},
        limits=limits,
    )

    cfg = IntradayRunnerConfig(
        state_path=PROJECT_ROOT / "data" / "intraday_state.json",
        per_position_size=PER_NAME_NOTIONAL,
        strategy_sha=STRATEGY_SHA,
    )
    runner = IntradayRunner(cfg)
    try:
        result = runner.run()
    except Exception as exc:
        logger.exception("intraday_runner.run failed: %s", exc)
        return 5

    if not result.intents:
        # Silent fire (HOLD on both assets) — don't log spam every 5 min.
        return 0

    logger.info("intraday: emitted %d intents (decision_id=%s batch_id=%s)",
                len(result.intents), result.decision_id, result.batch_id)

    # Risk chain.
    from risk_management.diff import IntentContext
    ctx = ChainContext(limits=limits, mode=ChainMode.PAPER)
    for it in result.intents:
        ticker = it.symbol.ticker
        current = positions.get(ticker, Decimal("0"))
        signed_delta = it.delta_qty if it.side is Side.BUY else -it.delta_qty
        ctx.intent_contexts[it.intent_id] = IntentContext(
            ticker=ticker, current_qty=current, target_qty=current + signed_delta,
        )
    marks_dec = {t: prev_close.get(t, Decimal("0")) for t in INTRADAY_ASSETS}
    ctx.planned_gross_notional = sum(
        (abs(ic.target_qty) * marks_dec.get(ic.ticker, Decimal("0"))
         for ic in ctx.intent_contexts.values()),
        start=Decimal("0"),
    )
    ctx.planned_turnover_notional = sum(
        (abs(ic.target_qty - ic.current_qty) * marks_dec.get(ic.ticker, Decimal("0"))
         for ic in ctx.intent_contexts.values()),
        start=Decimal("0"),
    )

    approved: list[OrderIntent] = []
    for it in result.intents:
        try:
            dec = run_check_chain(it, risk_snap, it.mark, mode=ChainMode.PAPER, ctx=ctx)
        except Exception as exc:
            logger.exception("chain raised on %s: %s", it.symbol.ticker, exc)
            continue
        if dec.rule_id == "APPROVED":
            approved.append(it)
            logger.info("APPROVED %s %s qty=%s @ $%s",
                        it.symbol.ticker, it.side.value, it.delta_qty, it.mark)
        else:
            logger.warning("REJECTED %s rule=%s obs=%s thr=%s",
                           it.symbol.ticker, dec.rule_id, dec.observed, dec.threshold)

    if not approved:
        return 0

    errors = 0
    for it in approved:
        try:
            coid = alpaca.submit_order(it, account="PAPER", attempt=0,
                                        strategy_sha=STRATEGY_SHA)
            logger.info("SUBMITTED %s %s qty=%s coid=%s",
                        it.symbol.ticker, it.side.value, it.delta_qty, coid)
        except Exception as exc:
            errors += 1
            logger.exception("submit_order failed for %s: %s", it.symbol.ticker, exc)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
