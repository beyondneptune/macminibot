"""Shared fixtures for risk_management tests.

Isolates state/risk filesystem writes into a temp dir per test so
KILL_SWITCH and breach JSONL tests never touch the operator's real
~/macminibot/state or risk dirs.

Epic 26 (cycle-7): cross-tentacle types come from ``shared_core.contracts``.
``make_intent`` constructs a shared_core ``OrderIntent`` (Symbol-typed,
delta_qty/mark/decision_id required) AND the corresponding
``IntentContext`` sidecar. Most call sites need both — the helper returns
a 2-tuple, with a convenience fixture that auto-installs the sidecar on
``ChainContext.intent_contexts``.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest

from shared_core.contracts import Symbol, Venue

from risk_management.chain import ChainContext, ChainMode
from risk_management.diff import (
    IntentContext,
    OrderIntent,
    Side,
    TargetPosition,
    TargetPositionBatch,
)
from risk_management.kill_switch import write_heartbeat
from risk_management.limits import (
    AccountLimits,
    Limits,
    SymbolLimits,
)
from risk_management.snapshot import NavSnapshot, build_snapshot


@pytest.fixture(autouse=True)
def _isolate_fs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect state + risk dirs to a per-test temp directory."""

    state_dir = tmp_path / "state"
    risk_dir = tmp_path / "risk"
    state_dir.mkdir(parents=True, exist_ok=True)
    risk_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MACMINIBOT_STATE_DIR", str(state_dir))
    monkeypatch.setenv("MACMINIBOT_RISK_DIR", str(risk_dir))
    # Default: write a fresh heartbeat so KILL_SWITCH passes unless a test
    # explicitly halts or deletes it.
    write_heartbeat(pid=os.getpid(), version="test")
    return tmp_path


def _account(
    *,
    max_single_name_pct: str = "0.15",
    max_gross_exposure_pct: str = "1.00",
    max_daily_turnover_pct: str = "0.50",
    kelly_fraction: str = "0.25",
    fatfinger_band_pct: str = "0.15",
    min_intent_notional_usd: str = "50.00",
    dd_daily_pct: str = "-0.02",
    dd_weekly_pct: str = "-0.05",
    dd_ptt_pct: str = "-0.10",
) -> AccountLimits:
    return AccountLimits(
        max_single_name_pct=Decimal(max_single_name_pct),
        max_gross_exposure_pct=Decimal(max_gross_exposure_pct),
        max_daily_turnover_pct=Decimal(max_daily_turnover_pct),
        kelly_fraction=Decimal(kelly_fraction),
        fatfinger_band_pct=Decimal(fatfinger_band_pct),
        min_intent_notional_usd=Decimal(min_intent_notional_usd),
        dd_daily_pct=Decimal(dd_daily_pct),
        dd_weekly_pct=Decimal(dd_weekly_pct),
        dd_ptt_pct=Decimal(dd_ptt_pct),
    )


def _defaults_symbol(
    *,
    max_notional_pct: str = "0.15",
    kelly_fraction: str = "0.25",
    fatfinger_band_pct: str = "0.15",
) -> SymbolLimits:
    return SymbolLimits(
        max_notional_pct=Decimal(max_notional_pct),
        kelly_fraction=Decimal(kelly_fraction),
        fatfinger_band_pct=Decimal(fatfinger_band_pct),
    )


@pytest.fixture
def default_limits() -> Limits:
    return Limits(
        schema_version=2,
        config_id="test.001",
        account=_account(),
        defaults_symbol=_defaults_symbol(),
        symbol_overrides={},
        limits_hash="sha256:test",
        raw_text="",
    )


@pytest.fixture
def snapshot(default_limits: Limits) -> NavSnapshot:
    """100k NetLiq, AAPL=80 shares already long, prev_close=150."""

    return build_snapshot(
        {
            "NetLiquidation": "100000",
            "AvailableFunds": "100000",
            "GrossPositionValue": "12000",
        },
        as_of_ns=1_700_000_000_000_000_000,
        positions={"AAPL": "80"},
        broker_positions={"AAPL": "80"},
        prev_close={"AAPL": "150", "MSFT": "300", "TSLA": "200"},
        feature_ok={"AAPL": True, "MSFT": True, "TSLA": True},
        limits=default_limits,
    )


@pytest.fixture
def ctx_live(default_limits: Limits) -> ChainContext:
    return ChainContext(limits=default_limits, mode=ChainMode.LIVE)


@pytest.fixture
def ctx_whatif(default_limits: Limits) -> ChainContext:
    return ChainContext(limits=default_limits, mode=ChainMode.WHATIF)


def make_symbol(ticker: str = "AAPL") -> Symbol:
    """Default Symbol — NASDAQ/USD is the canonical test exchange/currency."""

    return Symbol(ticker=ticker, venue=Venue.NASDAQ, currency="USD")


def make_intent(
    *,
    symbol: str = "AAPL",
    target_qty: str | int = "100",
    current_qty: str | int = "80",
    batch_id: str = "test-batch",
    mark: str | int = "150",
    decision_id: str | None = None,
    ctx: ChainContext | None = None,
) -> OrderIntent:
    """Construct a shared_core ``OrderIntent`` + register the IntentContext.

    When ``ctx`` is supplied, the IntentContext sidecar is auto-installed
    on ``ctx.intent_contexts``. Otherwise the caller must install it
    manually (the diff layer does this automatically in normal use).

    ``mark`` defaults to 150 to match the AAPL prev_close in ``snapshot``;
    the chain check ``PRICE_BAND_FATFINGER`` reads ``mark`` from the
    chain-call argument, NOT from ``intent.mark`` (those are kept in sync
    by ``run_check_chain``'s caller).

    Note on zero-delta intents: shared_core ``OrderIntent`` requires
    ``delta_qty > 0``. The chain's INTENT_NONZERO check is for the
    *notional* threshold ($50), not for literal zero qty — so we
    synthesize a 1-share minimum delta when callers ask for a zero delta,
    keeping the prior test semantics ("delta=0 → INTENT_NONZERO reject"
    by way of the $50 threshold check). Tests that want exact-zero
    behavior should construct delta of e.g. 0.1 shares directly.
    """

    sym = make_symbol(symbol)
    tq = Decimal(str(target_qty))
    cq = Decimal(str(current_qty))
    delta = tq - cq

    # shared_core OrderIntent requires delta_qty > 0; floor at a tiny
    # positive number for the chain-local "delta=0 should reject under
    # INTENT_NONZERO" test case (caller's intent is to fail the $50
    # notional check, not to violate the contract).
    if delta == 0:
        contract_delta = Decimal("0.000001")
    else:
        contract_delta = abs(delta)
    side = Side.BUY if delta >= 0 else Side.SELL

    intent_id = f"{batch_id}:{symbol}"
    decision = decision_id or batch_id
    intent = OrderIntent(
        intent_id=intent_id,
        symbol=sym,
        side=side,
        delta_qty=contract_delta,
        mark=Decimal(str(mark)),
        batch_id=batch_id,
        decision_id=decision,
    )

    if ctx is not None:
        ctx.intent_contexts[intent_id] = IntentContext(
            ticker=symbol, current_qty=cq, target_qty=tq
        )

    return intent


def install_intent_context(
    ctx: ChainContext,
    intent: OrderIntent,
    *,
    ticker: str,
    current_qty: str | int,
    target_qty: str | int,
) -> None:
    """Install an IntentContext on the chain context (used by tests that
    constructed the intent without passing ctx to ``make_intent``)."""

    ctx.intent_contexts[intent.intent_id] = IntentContext(
        ticker=ticker,
        current_qty=Decimal(str(current_qty)),
        target_qty=Decimal(str(target_qty)),
    )


@pytest.fixture
def make_intent_fn():
    return make_intent
