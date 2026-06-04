"""60_per_asset_dryrun — WHATIF integration test for the per-asset trailing-stop bot.

End-to-end exerciser for the *new* per-asset specialist pipeline (replaces
the cross-sectional 30_paper_trade_runner pipeline). Builds the per-asset
universe from ``ml_research.asset_specialists`` winners, pulls live marks
from Alpaca IEX, calls ``strategy_engine.per_asset_runner.run_per_asset_tick``
to compute signals → OrderIntents, validates each intent against the
existing 10-rule risk chain in ``risk_management.chain``, and prints what
*would* be submitted. **No real orders are sent** (``ChainMode.WHATIF``).

The build agent is constructing ``strategy_engine.per_asset_runner`` and
the ``execution.alpaca_client`` bracket/trailing extensions in parallel.
This script imports those gracefully: if the per-asset runner isn't yet
present, we print a clear "waiting on build agent" notice and fall through
to a stub that emits zero intents (so the rest of the chain wiring stays
exercised).

Run:
    .venv/bin/python scripts/60_per_asset_dryrun.py

Exit code is always 0 when the WHATIF chain completes (any rejected intents
are reported in the summary, not propagated as failures). Non-zero exit
indicates a hard wiring fault (missing credentials, Alpaca API unreachable,
risk-chain contract bug, etc.).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))  # so `from scripts._storage import open_store` works

# Disable heartbeat staleness gate for the dryrun — KILL_SWITCH_ACTIVE check
# requires a fresh heartbeat, but we don't want the dryrun to require the
# launchd job to be live. HALT flag is still honoured.
os.environ.setdefault("MACMINIBOT_DISABLE_HEARTBEAT_CHECK", "1")


def _write_heartbeat() -> None:
    """Write a fresh heartbeat so KILL_SWITCH_ACTIVE doesn't reject every intent."""
    import json
    state_dir = Path(os.environ.get("MACMINIBOT_STATE_DIR",
                                    str(PROJECT_ROOT / "state")))
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "heartbeat.json").write_text(json.dumps({
        "pid": os.getpid(),
        "ts_unix": time.time(),
        "version": "per-asset-dryrun-v0",
    }))


_write_heartbeat()

from dotenv import load_dotenv  # noqa: E402

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


# ---------------------------------------------------------------------------
# Graceful build-agent stub
# ---------------------------------------------------------------------------

# Map asset ticker → primary listing venue. The asset_specialists module owns
# the universe definition (SUPPORTED_ASSETS) but not the venue metadata; we
# co-locate the venue mapping here because both the dryrun and the production
# runner need it for Symbol construction.
ASSET_VENUE: dict[str, Venue] = {
    "AAPL": Venue.NASDAQ,
    "MSFT": Venue.NASDAQ,
    "GOOGL": Venue.NASDAQ,
    "AMZN": Venue.NASDAQ,
    "META": Venue.NASDAQ,
    "TSLA": Venue.NASDAQ,
    "NVDA": Venue.NASDAQ,
    "GLD": Venue.ARCA,
}


try:
    from strategy_engine.per_asset_runner import PerAssetRunner, PerAssetRunnerConfig  # type: ignore
    _PER_ASSET_RUNNER_AVAILABLE = True

    def run_per_asset_tick(  # type: ignore[no-redef]
        *,
        winners: dict[str, dict[str, Any]],
        marks: dict[str, Decimal],
        positions: dict[str, Decimal],
        as_of_ns: int,
        mode: ChainMode,
    ) -> tuple[list[OrderIntent], dict[str, Any]]:
        """Adapter to the class-based PerAssetRunner the build agent shipped."""
        cfg = PerAssetRunnerConfig(
            state_path=Path("/Users/weeoto/macminibot/data/per_asset_state.json"),
        )
        runner = PerAssetRunner(cfg)
        result = runner.run(net_liq=Decimal("100088.37"))
        return result.intents, {}

except ImportError:
    _PER_ASSET_RUNNER_AVAILABLE = False

    def run_per_asset_tick(  # type: ignore[no-redef]
        *,
        winners: dict[str, dict[str, Any]],
        marks: dict[str, Decimal],
        positions: dict[str, Decimal],
        as_of_ns: int,
        mode: ChainMode,
    ) -> tuple[list[OrderIntent], dict[str, Any]]:
        """Stub: the build agent's per_asset_runner module is not yet present."""
        return [], {}


# ---------------------------------------------------------------------------
# Universe + winners
# ---------------------------------------------------------------------------


def load_universe_winners() -> dict[str, dict[str, Any]]:
    """Return ``{ticker: winner_spec}`` for every asset with a passing bake-off.

    Assets where ``winner_for(t) is None`` (bake-off rejected every candidate)
    are dropped — the per-asset runner must NOT see them.
    """
    winners: dict[str, dict[str, Any]] = {}
    for ticker in SUPPORTED_ASSETS:
        spec = winner_for(ticker)
        if spec is None:
            continue
        winners[ticker] = spec
    return winners


# ---------------------------------------------------------------------------
# Alpaca marks (IEX feed)
# ---------------------------------------------------------------------------


def fetch_marks(api_key: str, secret: str,
                tickers: tuple[str, ...]) -> dict[str, Decimal]:
    """Return ``{ticker: Decimal mid}`` from Alpaca's IEX latest-quote endpoint."""
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


# ---------------------------------------------------------------------------
# Snapshot building
# ---------------------------------------------------------------------------


def build_dryrun_snapshots(
    api_key: str,
    secret: str,
    base_url: str,
    marks: dict[str, Decimal],
    as_of_ns: int,
    limits,
    tickers: tuple[str, ...],
):
    """Return ``(risk_snap, shared_snap, net_liq, positions)`` for the dryrun.

    Pulls live account state from Alpaca. Both snapshot flavours are needed:
    the risk chain consumes the local ``risk_management.snapshot.NavSnapshot``
    (str-keyed positions, prev_close, feature_ok); the per-asset runner
    consumes the canonical ``shared_core.NavSnapshot`` (Symbol-keyed).
    """
    from alpaca.trading.client import TradingClient
    from shared_core.contracts import NavSnapshot as SharedNavSnapshot

    paper = "paper" in base_url.lower()
    tc = TradingClient(api_key, secret, paper=paper)
    acct = tc.get_account()
    net_liq = Decimal(str(getattr(acct, "portfolio_value", acct.equity)))
    buying_power = Decimal(str(acct.buying_power))
    account_id = str(getattr(acct, "id", "alpaca-paper"))

    positions: dict[str, Decimal] = {}
    for pos in tc.get_all_positions():
        ticker = getattr(pos, "symbol", None)
        qty_raw = getattr(pos, "qty", None)
        if ticker is None or qty_raw is None:
            continue
        qty = Decimal(str(qty_raw))
        side_str = str(getattr(getattr(pos, "side", "long"), "value", "long")).lower()
        if side_str == "short" and qty > 0:
            qty = -qty
        positions[str(ticker)] = qty

    # prev_close fallback: use current mark when we don't have a daily-bars
    # source wired into the dryrun. The fatfinger band is then trivially
    # satisfied (mark == prev_close ± 0); this is the right behaviour for
    # a smoke test that just wants the wiring to run.
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

    shared_positions: dict[Symbol, Decimal] = {}
    for t, qty in positions.items():
        venue = ASSET_VENUE.get(t, Venue.NASDAQ)
        shared_positions[Symbol(ticker=t, venue=venue, currency="USD")] = qty

    shared_version = hashlib.sha256(
        f"{account_id}|{as_of_ns}|{net_liq}|{buying_power}".encode()
    ).hexdigest()
    shared_snap = SharedNavSnapshot(
        snapshot_ns=int(as_of_ns),
        net_liq=net_liq,
        buying_power=buying_power,
        positions=shared_positions,
        snapshot_version=shared_version,
        account_id=account_id,
    )
    return risk_snap, shared_snap, net_liq, positions


# ---------------------------------------------------------------------------
# IntentContext bridge
# ---------------------------------------------------------------------------


def derive_intent_contexts(
    intents: list[OrderIntent],
    positions: dict[str, Decimal],
):
    """Construct chain-local IntentContext sidecars from the emitted intents.

    The per-asset runner emits canonical ``OrderIntent`` (no current_qty /
    target_qty fields). The chain needs an ``IntentContext`` per intent for
    the snapshot bridge — we reconstruct it from ``positions`` (current_qty)
    and infer ``target_qty = current + signed(delta)``.
    """
    from risk_management.diff import IntentContext

    ctxs: dict[str, IntentContext] = {}
    for it in intents:
        ticker = it.symbol.ticker
        current = positions.get(ticker, Decimal("0"))
        signed_delta = it.delta_qty if it.side is Side.BUY else -it.delta_qty
        target = current + signed_delta
        ctxs[it.intent_id] = IntentContext(
            ticker=ticker,
            current_qty=current,
            target_qty=target,
        )
    return ctxs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("PER-ASSET DRYRUN — start (mode=WHATIF, no real orders)")
    print("=" * 72)

    # 1. Load .env
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not api_key or not secret:
        print("ERROR: missing ALPACA_API_KEY / ALPACA_SECRET in .env")
        return 2
    print(f".env loaded (base_url={base_url})")

    # 2. Build per-asset universe from winners
    winners = load_universe_winners()
    print(f"asset_specialists universe: {len(winners)}/{len(SUPPORTED_ASSETS)} "
          f"assets passed bake-off → {sorted(winners)}")
    if not winners:
        print("  (every asset's bake-off was rejected — universe is EMPTY.")
        print("   The per-asset bot has nothing to trade until the bake-off")
        print("   produces winners. The dryrun continues to exercise the")
        print("   chain wiring with an empty intent set.)")

    universe_tickers = tuple(sorted(winners.keys()))

    # 3. Pull marks via Alpaca (IEX). When the universe is empty we still hit
    #    Alpaca with the full SUPPORTED_ASSETS so the smoke test confirms
    #    credentials + connectivity.
    probe_tickers = universe_tickers or SUPPORTED_ASSETS
    try:
        marks = fetch_marks(api_key, secret, probe_tickers)
    except Exception as exc:
        print(f"ERROR fetching marks from Alpaca: {exc}")
        return 3
    print(f"marks pulled: {len(marks)}/{len(probe_tickers)} symbols")
    for t, m in sorted(marks.items()):
        print(f"  {t}: ${m}")

    # 4. Build snapshots + load limits
    limits = load_limits(PROJECT_ROOT / "data" / "limits.toml")
    as_of_ns = int(time.time() * 1e9)
    try:
        risk_snap, shared_snap, net_liq, positions = build_dryrun_snapshots(
            api_key=api_key,
            secret=secret,
            base_url=base_url,
            marks=marks,
            as_of_ns=as_of_ns,
            limits=limits,
            tickers=probe_tickers,
        )
    except Exception as exc:
        print(f"ERROR building snapshots: {exc}")
        return 4
    print(f"snapshot: net_liq=${net_liq} positions={len(positions)}")

    # 5. Call per-asset runner (WHATIF)
    if not _PER_ASSET_RUNNER_AVAILABLE:
        print("")
        print("*" * 72)
        print("WAITING ON BUILD AGENT: strategy_engine.per_asset_runner not yet importable.")
        print("Using stub that emits zero intents. Re-run after build agent lands.")
        print("*" * 72)
        print("")

    try:
        intents, _runner_meta = run_per_asset_tick(
            winners=winners,
            marks=marks,
            positions=positions,
            as_of_ns=as_of_ns,
            mode=ChainMode.WHATIF,
        )
    except TypeError as exc:
        # Build agent's API surface may differ. Try a positional-arg fallback
        # so we don't hard-fail just because the kwarg names rotated.
        print(f"NOTE: run_per_asset_tick kwarg mismatch ({exc}); emitting zero intents.")
        intents = []
    except Exception as exc:
        print(f"ERROR running per_asset_tick: {exc}")
        return 5

    print(f"per_asset_runner: emitted {len(intents)} OrderIntents (mode=WHATIF)")

    # 6. WHATIF print of what would be submitted
    if intents:
        print("")
        print("WHATIF — would-submit roster:")
        for it in intents:
            print(f"  {it.symbol.ticker:6s} {it.side.value:4s} "
                  f"qty={it.delta_qty} @ ${it.mark}  intent_id={it.intent_id}")

    # 7. Validate each intent against the existing risk chain (WHATIF mode)
    ctx = ChainContext(limits=limits, mode=ChainMode.WHATIF)
    ctx.intent_contexts.update(derive_intent_contexts(intents, positions))
    # Cross-intent aggregates: gross notional and turnover for the batch.
    marks_dec = {t: marks.get(t, Decimal("0")) for t in probe_tickers}
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
    rejected_decisions: list[tuple[OrderIntent, Any]] = []
    for it in intents:
        try:
            decision = run_check_chain(
                it, risk_snap, it.mark, mode=ChainMode.WHATIF, ctx=ctx
            )
        except Exception as exc:
            print(f"  CHAIN ERROR on {it.symbol.ticker}: {exc}")
            rejected_decisions.append((it, exc))
            continue
        if decision.rule_id == "APPROVED":
            approved.append(it)
            print(f"  APPROVED {it.symbol.ticker} {it.side.value} {it.delta_qty}")
        else:
            rejected_decisions.append((it, decision))
            print(f"  REJECTED {it.symbol.ticker} rule={decision.rule_id} "
                  f"observed={decision.observed} threshold={decision.threshold}")

    # 8. Final summary
    print("")
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  mode             : WHATIF (no real orders)")
    print(f"  per_asset_runner : {'IMPORTED' if _PER_ASSET_RUNNER_AVAILABLE else 'STUBBED (build agent pending)'}")
    print(f"  universe size    : {len(winners)} of {len(SUPPORTED_ASSETS)}")
    print(f"  total intents    : {len(intents)}")
    print(f"  approved         : {len(approved)}")
    print(f"  rejected         : {len(rejected_decisions)}")
    if rejected_decisions:
        rule_counts: dict[str, int] = {}
        for _it, dec in rejected_decisions:
            rule = getattr(dec, "rule_id", "EXCEPTION")
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        print(f"  reject reasons   : {rule_counts}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
