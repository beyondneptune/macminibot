"""37_leveraged_sleeve_runner — PRODUCTION runner for the v3 leveraged ETF
sleeve (SOXL, NVDL, TSLL). NOT YET LIVE — sits as deployment-ready code
until intraday bake-off (script 94) validates the OOS signal.

Architecture: identical to scripts/30_per_asset_runner.py but operates on
its own state file (data/leveraged_sleeve_state.json) and universe.

Sizing: per-name notional = $20k (sleeve = $60k total, alongside the
$100k v2 daily book ⇒ ~1.6× effective gross at full allocation, requires
2× margin to be safe).

Activation steps (when ready):
1. Validate intraday bake-off — script 94 must show OOS net > B&H.
2. Sanity-check live paper for ≥10 days at $5k notional (¼ size).
3. Bootstrap com.macminibot.leveraged-sleeve-runner.plist.
4. Update memory project_daily_v2_deployment.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

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
        "version": "leveraged-sleeve-runner-v1",
    }))


_write_runner_heartbeat()

from dotenv import load_dotenv  # noqa: E402

from execution.alpaca_client import AlpacaClient  # noqa: E402
from ml_research.asset_specialists import winner_for  # noqa: E402
from risk_management.chain import ChainContext, run_check_chain  # noqa: E402
from risk_management.limits import load_limits  # noqa: E402
from risk_management.snapshot import build_snapshot  # noqa: E402
from shared_core.contracts import ChainMode, OrderIntent, Side  # noqa: E402
from strategy_engine.per_asset_runner import (  # noqa: E402
    PerAssetRunner, PerAssetRunnerConfig,
)

LEVERAGED_UNIVERSE = ("SOXL", "NVDL", "TSLL")
PER_NAME_NOTIONAL = Decimal("20000")  # $20k per name = $60k sleeve total
PORTFOLIO_SIZE = 5  # leverage-safe denominator ($100k / 5 = $20k)
STRATEGY_SHA = "leveraged_sleeve_v1"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("leveraged_sleeve_runner")
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


def fetch_marks(api_key: str, secret: str,
                tickers: tuple[str, ...]) -> dict[str, Decimal]:
    if not tickers:
        return {}
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


def build_snapshots(
    alpaca: AlpacaClient,
    marks: dict[str, Decimal],
    as_of_ns: int,
    limits,
    tickers: tuple[str, ...],
):
    """Return ``(risk_snap, net_liq, positions)`` for the sleeve tick."""
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

    prev_close = {t: marks.get(t, Decimal("100")) for t in tickers}
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
        feature_ok={t: True for t in tickers},
        limits=limits,
    )
    return risk_snap, net_liq, positions


def derive_intent_contexts(intents: list[OrderIntent],
                           positions: dict[str, Decimal]):
    from risk_management.diff import IntentContext

    ctxs: dict[str, IntentContext] = {}
    for it in intents:
        ticker = it.symbol.ticker
        current = positions.get(ticker, Decimal("0"))
        signed_delta = it.delta_qty if it.side is Side.BUY else -it.delta_qty
        target = current + signed_delta
        ctxs[it.intent_id] = IntentContext(
            ticker=ticker, current_qty=current, target_qty=target,
        )
    return ctxs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-window", action="store_true",
                        help="Bypass trading-window guard. SAFETY: auto-enables --whatif unless --i-mean-it is passed.")
    parser.add_argument("--i-mean-it", action="store_true",
                        help="Authorize real submissions when --force-window is set.")
    parser.add_argument("--whatif", action="store_true",
                        help="Run end-to-end but use ChainMode.WHATIF (no real Alpaca orders).")
    args = parser.parse_args()
    if args.force_window and not args.i_mean_it:
        args.whatif = True

    now_utc = datetime.now(timezone.utc)
    if not args.force_window and not is_within_us_trading_window(now_utc):
        return 6

    today = now_utc.strftime("%Y-%m-%d")
    log_path = PROJECT_ROOT / "data" / f"37_leveraged_sleeve_run_{today}.log"
    logger = setup_logger(log_path)
    logger.info("=" * 72)
    logger.info(f"LEVERAGED SLEEVE RUNNER — mode={'WHATIF' if args.whatif else 'PAPER'}")
    logger.info("=" * 72)

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not secret:
        logger.error("missing ALPACA_API_KEY/SECRET")
        return 2

    # Sleeve assets are not (yet) in ml_research.asset_specialists registry;
    # winner_for() raises ValueError on unknown asset. Until script 94 bake-off
    # validates and registers them, treat missing winners as a clean no-op.
    winners: dict[str, Any] = {}
    for t in LEVERAGED_UNIVERSE:
        try:
            spec = winner_for(t)
        except ValueError:
            spec = None
        if spec is not None:
            winners[t] = spec
    if not winners:
        logger.warning("no leveraged-sleeve winners registered — exit clean "
                       "(sleeve activation pending bake-off script 94)")
        return 0
    tickers = tuple(sorted(winners.keys()))
    logger.info(f"sleeve universe: {tickers}")

    # 4. Marks
    try:
        marks = fetch_marks(api_key, secret, tickers)
    except Exception as exc:
        logger.exception("fetch_marks failed: %s", exc)
        return 3
    logger.info("marks: %s", {t: f"${v}" for t, v in sorted(marks.items())})

    # 5. Alpaca client + snapshot
    alpaca = AlpacaClient.from_env(PROJECT_ROOT / ".env")
    limits = load_limits(PROJECT_ROOT / "data" / "limits.toml")
    as_of_ns = int(time.time() * 1e9)
    try:
        risk_snap, net_liq, positions = build_snapshots(
            alpaca, marks, as_of_ns, limits, tickers
        )
    except Exception as exc:
        logger.exception("build_snapshots failed: %s", exc)
        return 4
    logger.info("snapshot: net_liq=$%s open_positions=%d", net_liq, len(positions))

    # 6. PerAssetRunner.run on the sleeve's own state file.
    cfg = PerAssetRunnerConfig(
        state_path=PROJECT_ROOT / "data" / "leveraged_sleeve_state.json",
        assets=tickers,
        portfolio_size=PORTFOLIO_SIZE,
        strategy_sha=STRATEGY_SHA,
    )
    runner = PerAssetRunner(cfg)
    # WHATIF state-isolation: PerAssetRunner.run() persists state unconditionally.
    # Snapshot the sleeve state file BEFORE the run and restore it AFTER so
    # smoke tests can't corrupt the live state.
    state_backup = None
    if args.whatif and cfg.state_path.exists():
        state_backup = cfg.state_path.read_text()
        logger.info("WHATIF: snapshotted state file (%d bytes) for restore", len(state_backup))
    try:
        result = runner.run(net_liq=net_liq)
    except Exception as exc:
        logger.exception("runner failed: %s", exc)
        if state_backup is not None:
            cfg.state_path.write_text(state_backup)
            logger.info("WHATIF: state restored after exception")
        return 5
    if state_backup is not None:
        cfg.state_path.write_text(state_backup)
        logger.info("WHATIF: state file restored — smoke test had no side effects")

    logger.info("sleeve runner: emitted %d intents (decision_id=%s batch_id=%s)",
                len(result.intents), result.decision_id, result.batch_id)
    for skipped_asset, reason in result.skipped:
        logger.info("  skipped %s: %s", skipped_asset, reason)

    if not result.intents:
        logger.info("no intents — nothing to submit.")
        return 0

    # 7. Risk chain — same wiring as script 30.
    chain_mode = ChainMode.WHATIF if args.whatif else ChainMode.PAPER
    if args.whatif:
        logger.warning("WHATIF MODE — no real orders will be submitted")
    ctx = ChainContext(limits=limits, mode=chain_mode)
    ctx.intent_contexts.update(
        derive_intent_contexts(list(result.intents), positions)
    )
    marks_dec = {t: marks.get(t, Decimal("0")) for t in tickers}
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
            dec = run_check_chain(
                it, risk_snap, it.mark, mode=chain_mode, ctx=ctx
            )
        except Exception as exc:
            logger.exception("chain raised on %s: %s", it.symbol.ticker, exc)
            continue
        if dec.rule_id == "APPROVED":
            approved.append(it)
            logger.info("APPROVED %s %s qty=%s @ $%s (intent_id=%s)",
                        it.symbol.ticker, it.side.value, it.delta_qty,
                        it.mark, it.intent_id)
        else:
            logger.warning("REJECTED %s rule=%s observed=%s threshold=%s",
                           it.symbol.ticker, dec.rule_id,
                           dec.observed, dec.threshold)

    logger.info("chain results: %d approved / %d total", len(approved), len(result.intents))

    if not approved:
        return 0

    # 8. Submit (market orders — trailing-stop logic is internal).
    if args.whatif:
        logger.info("WHATIF: skipping all order submissions (%d would-be intents)",
                    len(approved))
        for it in approved:
            logger.info("WOULD-SUBMIT %s %s qty=%s @ $%s",
                        it.symbol.ticker, it.side.value, it.delta_qty, it.mark)
        logger.info("=" * 72)
        logger.info("LEVERAGED SLEEVE RUNNER — done (WHATIF, no orders submitted)")
        logger.info("=" * 72)
        return 0

    submit_errors = 0
    submitted_fills: list[dict[str, Any]] = []
    for it in approved:
        try:
            coid = alpaca.submit_order(
                it, account="PAPER", attempt=0, strategy_sha=STRATEGY_SHA
            )
            logger.info("SUBMITTED %s %s qty=%s coid=%s",
                        it.symbol.ticker, it.side.value, it.delta_qty, coid)
            submitted_fills.append({
                "ts_unix": time.time(),
                "intent_id": it.intent_id,
                "client_order_id": coid,
                "symbol": it.symbol.ticker,
                "side": it.side.value,
                "qty": int(it.delta_qty),
                "fill_qty": int(it.delta_qty),
                "mark": float(it.mark),
                "status": "FILLED",
                "strategy_sha": STRATEGY_SHA,
                "batch_id": it.batch_id,
                "decision_id": it.decision_id,
            })
        except Exception as exc:
            submit_errors += 1
            logger.exception("submit_order failed for %s: %s",
                             it.symbol.ticker, exc)

    # 8b. Write fills to the paper journal so auto-halt's POSITION_RECONCILE
    #     rule reconciles broker positions against the bot's intended fills.
    if submitted_fills:
        import pandas as pd
        from datetime import date as _date
        dt = _date.today().isoformat()
        journal_dir = PROJECT_ROOT / "data" / "paper" / f"dt={dt}"
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal_path = journal_dir / f"leveraged_sleeve_{int(time.time())}.parquet"
        try:
            pd.DataFrame(submitted_fills).to_parquet(journal_path, index=False)
            logger.info("journal: wrote %d fills → %s",
                        len(submitted_fills), journal_path)
        except Exception as exc:
            logger.exception("journal write failed: %s", exc)

    logger.info("submission: %d ok / %d errors", len(approved) - submit_errors, submit_errors)
    logger.info("=" * 72)
    logger.info("LEVERAGED SLEEVE RUNNER — done")
    logger.info("=" * 72)
    return 1 if submit_errors else 0


if __name__ == "__main__":
    sys.exit(main())
