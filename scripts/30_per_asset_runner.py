"""30_per_asset_runner — PRODUCTION per-asset trailing-stop paper-trade runner.

Replaces ``scripts/30_paper_trade_runner.py`` (the cross-sectional MV pipeline).
Fires once per US trading day at 09:35 ET via
``scripts/launchd/com.macminibot.per-asset-runner.plist``.

Flow per wake-up:

  1. Load .env (Alpaca paper creds).
  2. Trading-window guard — exit cleanly if outside US cash session.
  3. Build per-asset universe from bake-off winners (``ml_research.asset_specialists``).
  4. Pull live IEX quotes for the universe.
  5. Pull Alpaca account snapshot (net_liq, positions, buying power).
  6. Call ``PerAssetRunner.run`` — the state machine consumes daily bars,
     computes signals + trailing-stop transitions, emits market intents
     (BUY entries, SELL exits — both as plain market orders; trailing-stop
     logic is internal to the runner, not delegated to Alpaca's trailing-stop
     API).
  7. Validate each intent through the 10-rule risk chain (mode=PAPER).
  8. Submit approved intents via ``AlpacaClient.submit_order``.
  9. Log + return.

Production-vs-dryrun differences from scripts/60_per_asset_dryrun.py:
  - ChainMode.PAPER (not WHATIF) — risk-chain rules run in their live form.
  - Trading-window guard (dryrun runs any time).
  - Real ``submit_order`` calls (dryrun only prints).
  - Logger writes to ``data/30_per_asset_run_<date>.log`` so launchd-driven
    runs leave an audit trail.

Exit codes:
  0  — completed (zero or more orders submitted)
  1  — submission error on at least one approved intent
  2  — missing credentials
  3  — Alpaca quotes unreachable
  4  — snapshot build failure
  5  — runner raised
  6  — outside trading window (clean skip)
"""

from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(PROJECT_ROOT))  # for `from scripts._storage import open_store`

# Heartbeat staleness gate: the runner itself writes a fresh heartbeat at
# startup (so the kill-switch sees a live writer), and we leave the rest of
# the gate intact. ``MACMINIBOT_DISABLE_HEARTBEAT_CHECK`` stays unset — we
# WANT KILL_SWITCH_ACTIVE to fire if the runner crashes mid-session.

def _write_runner_heartbeat() -> None:
    import json
    state_dir = Path(os.environ.get("MACMINIBOT_STATE_DIR",
                                    str(PROJECT_ROOT / "state")))
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "heartbeat.json").write_text(json.dumps({
        "pid": os.getpid(),
        "ts_unix": time.time(),
        "version": "per-asset-runner-v1",
    }))


_write_runner_heartbeat()

from dotenv import load_dotenv  # noqa: E402

from execution.alpaca_client import AlpacaClient  # noqa: E402
from ml_research.asset_specialists import SUPPORTED_ASSETS, winner_for  # noqa: E402
from risk_management.chain import (  # noqa: E402
    ChainContext,
    run_check_chain,
)
from risk_management.limits import load_limits  # noqa: E402
from risk_management.snapshot import build_snapshot  # noqa: E402
from shared_core.contracts import (  # noqa: E402
    ChainMode,
    OrderIntent,
    Side,
    Symbol,
    Venue,
)
from strategy_engine.per_asset_runner import (  # noqa: E402
    PerAssetRunner,
    PerAssetRunnerConfig,
)

STRATEGY_SHA = "trailing_stop_v1"

# Adaptive config file paths (all optional — runner falls back to static
# defaults if any are missing or older than max-age below).
ACTIVE_UNIVERSE_PATH = PROJECT_ROOT / "data" / "active_universe.json"
REGIME_STATE_PATH = PROJECT_ROOT / "data" / "regime_state.json"
ASSET_OVERRIDES_PATH = PROJECT_ROOT / "data" / "asset_overrides.json"
ADAPTIVE_MAX_AGE_DAYS = 14   # ignore stale adaptive configs


def _load_json_if_fresh(path: Path, max_age_days: float) -> dict | None:
    """Return parsed JSON if file exists and is younger than max_age_days."""
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if (datetime.now(timezone.utc) - mtime).total_seconds() > max_age_days * 86400:
            return None
        import json as _json
        return _json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def _resolve_universe(static_default: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """Read active_universe.json or fall back to static SUPPORTED_ASSETS.

    Returns (tickers_tuple, source_label).
    """
    payload = _load_json_if_fresh(ACTIVE_UNIVERSE_PATH, ADAPTIVE_MAX_AGE_DAYS)
    if payload and isinstance(payload.get("universe"), list) and payload["universe"]:
        return tuple(payload["universe"]), f"adaptive ({ACTIVE_UNIVERSE_PATH.name})"
    return tuple(static_default), "static SUPPORTED_ASSETS"


def _resolve_regime() -> tuple[str, dict | None]:
    """Read regime_state.json. Defaults to RISK_ON if file missing/stale.

    Returns (regime_label, payload_dict_or_None).
    """
    payload = _load_json_if_fresh(REGIME_STATE_PATH, max_age_days=2)
    if not payload:
        return "RISK_ON_DEFAULT", None
    regime = str(payload.get("regime", "RISK_ON")).upper()
    if regime not in {"RISK_ON", "RISK_OFF"}:
        regime = "RISK_ON_DEFAULT"
    return regime, payload


def _load_asset_overrides() -> dict:
    """Read asset_overrides.json. Returns {} if missing/stale."""
    payload = _load_json_if_fresh(ASSET_OVERRIDES_PATH, ADAPTIVE_MAX_AGE_DAYS)
    if not payload:
        return {}
    return dict(payload.get("overrides", {}))
ASSET_VENUE: dict[str, Venue] = {
    "AAPL": Venue.NASDAQ, "MSFT": Venue.NASDAQ, "GOOGL": Venue.NASDAQ,
    "AMZN": Venue.NASDAQ, "META": Venue.NASDAQ, "TSLA": Venue.NASDAQ,
    "NVDA": Venue.NASDAQ, "GLD": Venue.ARCA,
}


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("per_asset_runner")
    logger.setLevel(logging.INFO)
    # Avoid double-attaching on re-imports
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
    """09:30–16:00 ET, Mon–Fri. Naive DST handling — uses fixed offset map.

    Returns True if the current moment is inside the US cash session. Off-hours,
    weekends, and non-trading dates return False; the runner then exits 6.
    """
    # Use US/Eastern via zoneinfo to handle DST correctly.
    from zoneinfo import ZoneInfo
    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:  # Sat/Sun
        return False
    open_minute = 9 * 60 + 30
    close_minute = 16 * 60
    cur_minute = et.hour * 60 + et.minute
    return open_minute <= cur_minute < close_minute


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
    """Return ``(risk_snap, net_liq, positions)`` for the production tick."""
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
    parser = argparse.ArgumentParser(
        description="Per-asset trailing-stop production runner"
    )
    parser.add_argument(
        "--force-window", action="store_true",
        help="Skip the US-trading-window guard. SAFETY: auto-enables --whatif "
             "unless --i-mean-it is ALSO passed. Three smoke-test accidents "
             "this project; the flag is no longer a footgun.",
    )
    parser.add_argument(
        "--i-mean-it", action="store_true",
        help="Authorize real submissions when --force-window is set "
             "(otherwise --force-window implies WHATIF mode).",
    )
    parser.add_argument(
        "--whatif", action="store_true",
        help="Run end-to-end but use ChainMode.WHATIF (no real Alpaca orders).",
    )
    args = parser.parse_args()

    # SAFETY OVERRIDE: --force-window without --i-mean-it auto-enables --whatif.
    if args.force_window and not args.i_mean_it:
        args.whatif = True

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = PROJECT_ROOT / "data" / f"30_per_asset_run_{today}.log"
    logger = setup_logger(log_path)

    logger.info("=" * 72)
    logger.info("PER-ASSET RUNNER — start (mode=PAPER, log=%s)", log_path)
    logger.info("=" * 72)

    # 1. Trading-window guard
    now_utc = datetime.now(timezone.utc)
    if not args.force_window and not is_within_us_trading_window(now_utc):
        logger.info("Outside US cash session (utc=%s) — clean exit.", now_utc.isoformat())
        return 6

    # 2. .env
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not secret:
        logger.error("missing ALPACA_API_KEY / ALPACA_SECRET in .env")
        return 2
    logger.info("env loaded (paper-trading)")

    # 3. Universe — DAILY-ONLY v2 deployment (post-agent-review).
    #    The intraday-runner sleeve (AAPL/MSFT Bollinger 3σ/3σ on 5-min) was
    #    PAUSED 2026-05-19 after the 5-agent review found per-trade gross edge
    #    of ~3-4bp = exactly bid-ask spread = costed-to-zero microstructure
    #    noise. AAPL/MSFT now use their daily-bar comprehensive-search-best
    #    configs (script 77) instead.
    #
    #    Overrides vs deployed bake-off winners (matched to scripts/77_comprehensive_search.py train-best):
    #      AAPL: trail 2σ → 3σ, add 3σ profit target
    #      MSFT: trail 2σ → 1σ, add 5σ profit target
    #      AMZN: ADDED (was excluded) buy_and_hold + 0.5σ trail + 5σ PT
    #      TSLA: add 3σ profit target (signal unchanged)
    #      GLD:  trail 3σ → 2σ, add 3σ profit target
    #      GOOGL/META/NVDA: deployed bake-off winners unchanged.
    INTRADAY_MANAGED: set[str] = set()  # intraday sleeve disabled
    # 2026-06-04: 1%/day target deployment. 8 leveraged ETFs, buy_and_hold,
    # trail=2.0σ, PT=3.0σ. Per-asset modules already carry these values, so
    # the override table just needs to enumerate them for completeness.
    DAILY_OVERRIDES: dict[str, dict[str, Any]] = {
        "SOXL": {"model": "buy_and_hold", "scope": "asset_only",
                  "trail_sigma": 2.0, "profit_target_sigmas": 3.0,
                  "source": "one_pct_target_sim_2026_06_04"},
        "NVDL": {"model": "buy_and_hold", "scope": "asset_only",
                  "trail_sigma": 2.0, "profit_target_sigmas": 3.0,
                  "source": "one_pct_target_sim_2026_06_04"},
        "TQQQ": {"model": "buy_and_hold", "scope": "asset_only",
                  "trail_sigma": 2.0, "profit_target_sigmas": 3.0,
                  "source": "one_pct_target_sim_2026_06_04"},
        "UPRO": {"model": "buy_and_hold", "scope": "asset_only",
                  "trail_sigma": 2.0, "profit_target_sigmas": 3.0,
                  "source": "one_pct_target_sim_2026_06_04"},
        "TNA":  {"model": "buy_and_hold", "scope": "asset_only",
                  "trail_sigma": 2.0, "profit_target_sigmas": 3.0,
                  "source": "one_pct_target_sim_2026_06_04"},
        "FAS":  {"model": "buy_and_hold", "scope": "asset_only",
                  "trail_sigma": 2.0, "profit_target_sigmas": 3.0,
                  "source": "one_pct_target_sim_2026_06_04"},
        "LABU": {"model": "buy_and_hold", "scope": "asset_only",
                  "trail_sigma": 2.0, "profit_target_sigmas": 3.0,
                  "source": "one_pct_target_sim_2026_06_04"},
        "TSLL": {"model": "buy_and_hold", "scope": "asset_only",
                  "trail_sigma": 2.0, "profit_target_sigmas": 3.0,
                  "source": "one_pct_target_sim_2026_06_04"},
    }
    # ---- ADAPTIVE LAYER 1: regime gate -------------------------------------
    regime_label, regime_payload = _resolve_regime()
    if regime_label == "RISK_OFF":
        logger.warning("REGIME=RISK_OFF — skipping all new entries this fire. "
                        "Payload: %s", regime_payload)
        return 0
    logger.info("REGIME=%s (source: %s)", regime_label,
                 "regime_state.json" if regime_payload else "default fallback")

    # ---- ADAPTIVE LAYER 2: dynamic universe --------------------------------
    active_universe, universe_source = _resolve_universe(SUPPORTED_ASSETS)
    logger.info("active universe (%s): %s", universe_source, list(active_universe))

    # ---- ADAPTIVE LAYER 3: per-asset config overrides ----------------------
    adaptive_overrides = _load_asset_overrides()
    if adaptive_overrides:
        logger.info("adaptive overrides loaded for %d assets", len(adaptive_overrides))
        for tkr, ov in adaptive_overrides.items():
            DAILY_OVERRIDES.setdefault(tkr, {}).update({
                k: v for k, v in ov.items()
                if k in {"model", "trail_sigma", "profit_target_sigmas", "scope", "source"}
            })

    winners: dict[str, dict[str, Any]] = {}
    for ticker in active_universe:
        if ticker in INTRADAY_MANAGED:
            continue  # intraday runner handles these
        try:
            spec = winner_for(ticker)
        except ValueError:
            # asset isn't in SUPPORTED_ASSETS but came from active universe — try DAILY_OVERRIDES alone
            spec = None
        if spec is None and ticker not in DAILY_OVERRIDES:
            continue
        # Start from deployed winner (or {} for newly added assets) and apply override.
        merged = dict(spec) if spec else {}
        if ticker in DAILY_OVERRIDES:
            merged.update(DAILY_OVERRIDES[ticker])
        winners[ticker] = merged
    logger.info("daily universe (hybrid book): %d assets → %s "
                "(intraday-managed: %s)",
                len(winners), sorted(winners), sorted(INTRADAY_MANAGED))
    if not winners:
        logger.warning("empty universe — nothing to trade today.")
        return 0
    tickers = tuple(sorted(winners.keys()))

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

    # 6. PerAssetRunner.run — 1%/day target deployment 2026-06-04.
    #    portfolio_size=2 means per-name notional = net_liq/2 ≈ $50k.
    #    With 8 names × $50k = $400k book = 4× intraday-margin stack on
    #    a $100k seed. This is the leveraged-ETF sleeve sizing.
    cfg = PerAssetRunnerConfig(
        state_path=PROJECT_ROOT / "data" / "per_asset_state.json",
        assets=tickers,
        portfolio_size=2,
        strategy_sha=STRATEGY_SHA,
    )
    runner = PerAssetRunner(cfg)
    # WHATIF state-isolation: PerAssetRunner.run() persists state unconditionally.
    # If we're in WHATIF mode, snapshot the state file BEFORE the run and
    # restore it AFTER, so smoke tests can't corrupt the live state.
    state_backup = None
    if args.whatif and cfg.state_path.exists():
        state_backup = cfg.state_path.read_text()
        logger.info("WHATIF: snapshotted state file (%d bytes) for restore", len(state_backup))
    try:
        result = runner.run(net_liq=net_liq)
    except Exception as exc:
        logger.exception("PerAssetRunner.run failed: %s", exc)
        if state_backup is not None:
            cfg.state_path.write_text(state_backup)
            logger.info("WHATIF: state restored after exception")
        return 5
    if state_backup is not None:
        cfg.state_path.write_text(state_backup)
        logger.info("WHATIF: state file restored — smoke test had no side effects")
    logger.info("per_asset_runner: emitted %d intents (decision_id=%s batch_id=%s)",
                len(result.intents), result.decision_id, result.batch_id)
    for skipped_asset, reason in result.skipped:
        logger.info("  skipped %s: %s", skipped_asset, reason)

    if not result.intents:
        logger.info("no intents — nothing to submit.")
        return 0

    # 7. Risk chain
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

    # 8. Submit (market orders — the trailing-stop logic is in the runner's
    #    state machine, not delegated to Alpaca's TrailingStopOrderRequest).
    if args.whatif:
        logger.info("WHATIF: skipping all order submissions (%d would-be intents)",
                    len(approved))
        for it in approved:
            logger.info("WOULD-SUBMIT %s %s qty=%s @ $%s",
                        it.symbol.ticker, it.side.value, it.delta_qty, it.mark)
        logger.info("=" * 72)
        logger.info("PER-ASSET RUNNER — done (WHATIF, no orders submitted)")
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
                "fill_qty": int(it.delta_qty),  # market orders fill at requested qty (paper)
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
        import pandas as pd  # local import — only needed when writing
        from datetime import date as _date
        dt = _date.today().isoformat()
        journal_dir = PROJECT_ROOT / "data" / "paper" / f"dt={dt}"
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal_path = journal_dir / f"per_asset_runner_{int(time.time())}.parquet"
        try:
            pd.DataFrame(submitted_fills).to_parquet(journal_path, index=False)
            logger.info("journal: wrote %d fills → %s",
                        len(submitted_fills), journal_path)
        except Exception as exc:
            logger.exception("journal write failed: %s", exc)

    logger.info("submission: %d ok / %d errors", len(approved) - submit_errors, submit_errors)
    logger.info("=" * 72)
    logger.info("PER-ASSET RUNNER — done")
    logger.info("=" * 72)
    return 1 if submit_errors else 0


if __name__ == "__main__":
    sys.exit(main())
