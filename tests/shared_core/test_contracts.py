"""Tests for shared_core.contracts.

Verifies the four invariants the ratified contract relies on:
1. All dataclasses are frozen (mutation raises FrozenInstanceError).
2. Decimal-from-float is rejected at the contract boundary.
3. Valid construction works for every type.
4. Enum string values match the spec.
5. Timestamp helpers round-trip.
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from shared_core.contracts import (
    Bar,
    BreachEvent,
    ChainMode,
    CorporateAction,
    CorporateActionType,
    DollarBar,
    FoldSpec,
    NavSnapshot,
    OrderIntent,
    Side,
    Signal,
    Symbol,
    TargetPosition,
    TargetPositionBatch,
    Tick,
    TickType,
    Venue,
)
from shared_core.types import (
    NS_PER_SECOND,
    money,
    ns_to_datetime_utc,
    now_ns,
    parse_ts,
    quantity,
    validate_decimal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def aapl() -> Symbol:
    return Symbol(ticker="AAPL", venue=Venue.NASDAQ, currency="USD")


@pytest.fixture()
def spy() -> Symbol:
    return Symbol(ticker="SPY", venue=Venue.ARCA, currency="USD")


@pytest.fixture()
def sha256_hash() -> str:
    return hashlib.sha256(b"snapshot-fixture").hexdigest()


# ---------------------------------------------------------------------------
# 1. Frozen invariant — every dataclass rejects mutation
# ---------------------------------------------------------------------------


def test_symbol_frozen(aapl: Symbol) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        aapl.ticker = "MSFT"  # type: ignore[misc]


def test_bar_frozen(aapl: Symbol) -> None:
    b = Bar(
        symbol=aapl,
        ts_ns_utc=1_700_000_000_000_000_000,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=Decimal("1000"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.close = Decimal("999")  # type: ignore[misc]


def test_signal_frozen(aapl: Symbol) -> None:
    s = Signal(
        symbol=aapl,
        r_hat_daily=Decimal("0.0042"),
        confidence=0.7,
        model_version="lgbm-v1",
        as_of_ns=1_700_000_000_000_000_000,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.confidence = 0.99  # type: ignore[misc]


def test_target_position_frozen(aapl: Symbol) -> None:
    tp = TargetPosition(
        symbol=aapl, target_qty=Decimal("10"), hold_reason=None, reason_trace={}
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        tp.target_qty = Decimal("0")  # type: ignore[misc]


def test_order_intent_frozen(aapl: Symbol) -> None:
    oi = OrderIntent(
        intent_id="01HABCDEF",
        symbol=aapl,
        side=Side.BUY,
        delta_qty=Decimal("10"),
        mark=Decimal("150.25"),
        batch_id="batch-1",
        decision_id="batch-1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        oi.delta_qty = Decimal("100")  # type: ignore[misc]


def test_nav_snapshot_frozen(aapl: Symbol, sha256_hash: str) -> None:
    ns = NavSnapshot(
        snapshot_ns=1_700_000_000_000_000_000,
        net_liq=Decimal("100000"),
        buying_power=Decimal("100000"),
        positions={aapl: Decimal("10")},
        snapshot_version=sha256_hash,
        account_id="U1234567",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ns.net_liq = Decimal("0")  # type: ignore[misc]


def test_breach_event_frozen(sha256_hash: str) -> None:
    be = BreachEvent(
        breach_id="01HBE0001",
        rule_id="KILL_SWITCH_ACTIVE",
        observed="HALT",
        threshold="NOT_HALT",
        mode=ChainMode.LIVE,
        ts_ns=1_700_000_000_000_000_000,
        intent_id=None,
        snapshot_hash=sha256_hash,
        would_halt=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        be.would_halt = False  # type: ignore[misc]


def test_corporate_action_frozen(aapl: Symbol) -> None:
    ca = CorporateAction(
        ca_id="01HCA0001",
        symbol=aapl,
        action_type=CorporateActionType.SPLIT,
        effective_date_ns=1_700_000_000_000_000_000,
        ratio_numerator=2,
        ratio_denominator=1,
        cash_amount=Decimal("0"),
        multiplier=Decimal(2) / Decimal(1),
        new_symbol=None,
        payable_date_ns=None,
        source="ibkr.reqContractDetails",
        as_of_ns=1_700_000_000_000_000_000,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ca.ratio_numerator = 3  # type: ignore[misc]


def test_fold_spec_frozen() -> None:
    fs = FoldSpec(
        fold_id=0,
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3, 4]),
        test_idx=np.array([5, 6]),
        fold_meta={"train_start_ns": 0},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        fs.fold_id = 1  # type: ignore[misc]


def test_slots_blocks_typo_attributes(aapl: Symbol) -> None:
    """Frozen + slots (KD3) prevents accidentally setting attributes that
    don't exist — typos become errors at write time, not silent.

    The exact exception type depends on CPython internals: bare frozen
    raises ``FrozenInstanceError``, slots alone raises ``AttributeError``,
    and the slots+frozen combo on 3.11 can raise ``TypeError`` from the
    generated ``super().__setattr__`` (gh-90562). Either way: not silent.
    """
    s = Signal(
        symbol=aapl,
        r_hat_daily=Decimal("0.01"),
        confidence=0.5,
        model_version="m1",
        as_of_ns=0,
    )
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError, TypeError)):
        s.confidance = 0.5  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2. Decimal-from-float is rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"open": 100.0, "high": Decimal("101"), "low": Decimal("99"), "close": Decimal("100"), "volume": Decimal("1")},
        {"open": Decimal("100"), "high": 101.0, "low": Decimal("99"), "close": Decimal("100"), "volume": Decimal("1")},
        {"open": Decimal("100"), "high": Decimal("101"), "low": Decimal("99"), "close": Decimal("100"), "volume": 1.0},
    ],
)
def test_bar_rejects_float_fields(aapl: Symbol, kwargs: dict) -> None:
    with pytest.raises(TypeError, match="must be a Decimal"):
        Bar(symbol=aapl, ts_ns_utc=0, **kwargs)


def test_signal_rejects_float_r_hat(aapl: Symbol) -> None:
    with pytest.raises(TypeError, match="r_hat_daily.*Decimal"):
        Signal(
            symbol=aapl,
            r_hat_daily=0.0042,  # type: ignore[arg-type]
            confidence=0.5,
            model_version="m1",
            as_of_ns=0,
        )


def test_order_intent_rejects_float_qty(aapl: Symbol) -> None:
    with pytest.raises(TypeError, match="delta_qty.*Decimal"):
        OrderIntent(
            intent_id="i1",
            symbol=aapl,
            side=Side.BUY,
            delta_qty=10.0,  # type: ignore[arg-type]
            mark=Decimal("150"),
            batch_id="b1",
            decision_id="b1",
        )


def test_nav_snapshot_rejects_float(aapl: Symbol, sha256_hash: str) -> None:
    with pytest.raises(TypeError, match="net_liq.*Decimal"):
        NavSnapshot(
            snapshot_ns=0,
            net_liq=100000.0,  # type: ignore[arg-type]
            buying_power=Decimal("100000"),
            positions={},
            snapshot_version=sha256_hash,
            account_id="U1",
        )


def test_money_helper_rejects_float() -> None:
    with pytest.raises(TypeError):
        money(0.1)  # type: ignore[arg-type]
    assert money("0.1") == Decimal("0.1")
    assert money(10) == Decimal(10)


def test_quantity_helper_rejects_float() -> None:
    with pytest.raises(TypeError):
        quantity(1.5)  # type: ignore[arg-type]
    assert quantity("100") == Decimal("100")


def test_validate_decimal_rejects_non_decimal() -> None:
    with pytest.raises(TypeError):
        validate_decimal(1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_decimal("1")  # type: ignore[arg-type]
    assert validate_decimal(Decimal("1")) == Decimal("1")


# ---------------------------------------------------------------------------
# 3. Valid construction works
# ---------------------------------------------------------------------------


def test_symbol_valid(aapl: Symbol) -> None:
    assert aapl.ticker == "AAPL"
    assert aapl.venue is Venue.NASDAQ
    assert aapl.currency == "USD"
    assert str(aapl) == "AAPL@NASDAQ.USD"
    assert hash(aapl) == hash(Symbol(ticker="AAPL", venue=Venue.NASDAQ, currency="USD"))


def test_symbol_rejects_smart() -> None:
    with pytest.raises(ValueError, match="SMART"):
        Symbol(ticker="AAPL", venue=Venue.SMART, currency="USD")


def test_symbol_rejects_bad_currency() -> None:
    with pytest.raises(ValueError, match="currency"):
        Symbol(ticker="AAPL", venue=Venue.NASDAQ, currency="usd")
    with pytest.raises(ValueError, match="currency"):
        Symbol(ticker="AAPL", venue=Venue.NASDAQ, currency="USDOLLAR")


def test_symbol_rejects_empty_ticker() -> None:
    with pytest.raises(ValueError, match="ticker"):
        Symbol(ticker="", venue=Venue.NASDAQ, currency="USD")


def test_bar_ohlc_consistency(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="OHLC"):
        Bar(
            symbol=aapl,
            ts_ns_utc=0,
            open=Decimal("200"),  # > high
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1"),
        )


def test_dollar_bar_valid(aapl: Symbol) -> None:
    db = DollarBar(
        symbol=aapl,
        ts_ns_utc=1,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("500"),
        notional=Decimal("50000"),
        tick_count=42,
    )
    assert db.notional == Decimal("50000")
    assert db.tick_count == 42


def test_tick_trades_valid(aapl: Symbol) -> None:
    t = Tick(
        symbol=aapl,
        ts_ns_utc=0,
        tick_type=TickType.TRADES,
        price=Decimal("150.25"),
        size=Decimal("100"),
        exchange=Venue.NASDAQ,
        mid=None,
    )
    assert t.tick_type is TickType.TRADES


def test_tick_midpoint_valid(aapl: Symbol) -> None:
    t = Tick(
        symbol=aapl,
        ts_ns_utc=0,
        tick_type=TickType.MIDPOINT,
        price=None,
        size=None,
        exchange=None,
        mid=Decimal("150.20"),
    )
    assert t.mid == Decimal("150.20")


def test_tick_trades_missing_price_raises(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="TRADES tick missing"):
        Tick(
            symbol=aapl,
            ts_ns_utc=0,
            tick_type=TickType.TRADES,
            price=None,
            size=Decimal("1"),
            exchange=Venue.NASDAQ,
            mid=None,
        )


def test_tick_midpoint_with_price_raises(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="MIDPOINT"):
        Tick(
            symbol=aapl,
            ts_ns_utc=0,
            tick_type=TickType.MIDPOINT,
            price=Decimal("1"),
            size=None,
            exchange=None,
            mid=Decimal("1"),
        )


def test_signal_valid(aapl: Symbol) -> None:
    s = Signal(
        symbol=aapl,
        r_hat_daily=Decimal("0.0042"),
        confidence=0.7,
        model_version="lgbm-v1",
        as_of_ns=1_700_000_000_000_000_000,
    )
    assert s.r_hat_daily == Decimal("0.0042")


def test_signal_confidence_out_of_range(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Signal(
            symbol=aapl,
            r_hat_daily=Decimal("0.01"),
            confidence=1.5,
            model_version="m",
            as_of_ns=0,
        )


def test_signal_unit_error_guard(aapl: Symbol) -> None:
    # 50% next-day return is almost certainly a unit error
    with pytest.raises(ValueError, match="unit error"):
        Signal(
            symbol=aapl,
            r_hat_daily=Decimal("0.5"),
            confidence=0.5,
            model_version="m",
            as_of_ns=0,
        )


def test_target_position_batch_valid(aapl: Symbol, spy: Symbol) -> None:
    targets = (
        TargetPosition(symbol=aapl, target_qty=Decimal("10"), hold_reason=None, reason_trace={}),
        TargetPosition(symbol=spy, target_qty=Decimal("-5"), hold_reason="halt.luld", reason_trace={}),
    )
    batch = TargetPositionBatch(
        batch_id="01HBATCH",
        decision_id="dec-01HBATCH",
        mode=ChainMode.LIVE,
        solver_status="optimal",
        generated_at_ns=1_700_000_000_000_000_000,
        supersedes=None,
        targets_count=2,
        targets=targets,
        nav_snapshot_version="abc" * 22,  # any non-empty string
    )
    assert len(batch.targets) == 2
    assert batch.decision_id == "dec-01HBATCH"
    assert batch.mode is ChainMode.LIVE
    assert batch.solver_status == "optimal"


def test_target_position_batch_checksum_mismatch(aapl: Symbol) -> None:
    targets = (
        TargetPosition(symbol=aapl, target_qty=Decimal("10"), hold_reason=None, reason_trace={}),
    )
    with pytest.raises(ValueError, match="targets_count"):
        TargetPositionBatch(
            batch_id="b1",
            decision_id="d1",
            mode=ChainMode.PAPER,
            solver_status="optimal",
            generated_at_ns=0,
            supersedes=None,
            targets_count=5,  # wrong
            targets=targets,
            nav_snapshot_version="v1",
        )


def test_target_position_batch_duplicate_symbol(aapl: Symbol) -> None:
    targets = (
        TargetPosition(symbol=aapl, target_qty=Decimal("10"), hold_reason=None, reason_trace={}),
        TargetPosition(symbol=aapl, target_qty=Decimal("20"), hold_reason=None, reason_trace={}),
    )
    with pytest.raises(ValueError, match="duplicate symbol"):
        TargetPositionBatch(
            batch_id="b1",
            decision_id="d1",
            mode=ChainMode.PAPER,
            solver_status="optimal",
            generated_at_ns=0,
            supersedes=None,
            targets_count=2,
            targets=targets,
            nav_snapshot_version="v1",
        )


def test_target_position_batch_rejects_list(aapl: Symbol) -> None:
    with pytest.raises(TypeError, match="tuple"):
        TargetPositionBatch(
            batch_id="b1",
            decision_id="d1",
            mode=ChainMode.PAPER,
            solver_status="optimal",
            generated_at_ns=0,
            supersedes=None,
            targets_count=0,
            targets=[],  # type: ignore[arg-type]
            nav_snapshot_version="v1",
        )


def test_target_position_batch_requires_decision_id(aapl: Symbol) -> None:
    targets = (
        TargetPosition(symbol=aapl, target_qty=Decimal("10"), hold_reason=None, reason_trace={}),
    )
    with pytest.raises(ValueError, match="decision_id"):
        TargetPositionBatch(
            batch_id="b1",
            decision_id="",  # empty
            mode=ChainMode.LIVE,
            solver_status="optimal",
            generated_at_ns=0,
            supersedes=None,
            targets_count=1,
            targets=targets,
            nav_snapshot_version="v1",
        )


def test_target_position_batch_requires_chainmode(aapl: Symbol) -> None:
    targets = (
        TargetPosition(symbol=aapl, target_qty=Decimal("10"), hold_reason=None, reason_trace={}),
    )
    with pytest.raises(TypeError, match="mode must be ChainMode"):
        TargetPositionBatch(
            batch_id="b1",
            decision_id="d1",
            mode="live",  # type: ignore[arg-type]
            solver_status="optimal",
            generated_at_ns=0,
            supersedes=None,
            targets_count=1,
            targets=targets,
            nav_snapshot_version="v1",
        )


def test_target_position_batch_requires_solver_status(aapl: Symbol) -> None:
    targets = (
        TargetPosition(symbol=aapl, target_qty=Decimal("10"), hold_reason=None, reason_trace={}),
    )
    with pytest.raises(ValueError, match="solver_status"):
        TargetPositionBatch(
            batch_id="b1",
            decision_id="d1",
            mode=ChainMode.LIVE,
            solver_status="",  # empty
            generated_at_ns=0,
            supersedes=None,
            targets_count=1,
            targets=targets,
            nav_snapshot_version="v1",
        )


def test_order_intent_valid(aapl: Symbol) -> None:
    oi = OrderIntent(
        intent_id="i1",
        symbol=aapl,
        side=Side.SELL,
        delta_qty=Decimal("3"),
        mark=Decimal("150.25"),
        batch_id="b1",
        decision_id="b1",
    )
    assert oi.side is Side.SELL


def test_order_intent_rejects_negative_delta(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="delta_qty must be positive"):
        OrderIntent(
            intent_id="i1",
            symbol=aapl,
            side=Side.BUY,
            delta_qty=Decimal("-5"),
            mark=Decimal("100"),
            batch_id="b1",
            decision_id="b1",
        )


def test_nav_snapshot_valid(aapl: Symbol, sha256_hash: str) -> None:
    ns = NavSnapshot(
        snapshot_ns=1_700_000_000_000_000_000,
        net_liq=Decimal("100000.50"),
        buying_power=Decimal("99000"),
        positions={aapl: Decimal("10")},
        snapshot_version=sha256_hash,
        account_id="U1234567",
    )
    assert ns.positions[aapl] == Decimal("10")


def test_nav_snapshot_rejects_bad_version_hash(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="sha256"):
        NavSnapshot(
            snapshot_ns=0,
            net_liq=Decimal("1"),
            buying_power=Decimal("1"),
            positions={},
            snapshot_version="too-short",
            account_id="U1",
        )


def test_nav_snapshot_rejects_negative_net_liq(sha256_hash: str) -> None:
    with pytest.raises(ValueError, match="net_liq negative"):
        NavSnapshot(
            snapshot_ns=0,
            net_liq=Decimal("-1"),
            buying_power=Decimal("1"),
            positions={},
            snapshot_version=sha256_hash,
            account_id="U1",
        )


def test_breach_event_valid(sha256_hash: str) -> None:
    be = BreachEvent(
        breach_id="01HBE",
        rule_id="MAX_SINGLE_NAME_NOTIONAL",
        observed=Decimal("60000"),
        threshold=Decimal("50000"),
        mode=ChainMode.PAPER,
        ts_ns=0,
        intent_id="i1",
        snapshot_hash=sha256_hash,
        would_halt=True,
    )
    assert be.mode is ChainMode.PAPER
    # Defaults for the cycle-8 audit fields
    assert be.config_id == ""
    assert be.symbol == ""
    assert be.extra == {}


def test_breach_event_with_audit_fields(sha256_hash: str) -> None:
    """Cycle-8 Epic 17: audit fields lifted from the sidecar onto the contract."""
    be = BreachEvent(
        breach_id="01HBE",
        rule_id="MAX_SINGLE_NAME_NOTIONAL",
        observed=Decimal("60000"),
        threshold=Decimal("50000"),
        mode=ChainMode.LIVE,
        ts_ns=1_700_000_000_000_000_000,
        intent_id="i1",
        snapshot_hash=sha256_hash,
        would_halt=True,
        config_id="sha256:abc123",
        symbol="AAPL",
        extra={"pct": "0.12", "net_liq": "100000"},
    )
    assert be.config_id == "sha256:abc123"
    assert be.symbol == "AAPL"
    assert be.extra == {"pct": "0.12", "net_liq": "100000"}


def test_breach_event_rejects_non_dict_extra(sha256_hash: str) -> None:
    with pytest.raises(TypeError, match="extra must be dict"):
        BreachEvent(
            breach_id="b",
            rule_id="r",
            observed="o",
            threshold="t",
            mode=ChainMode.LIVE,
            ts_ns=0,
            intent_id=None,
            snapshot_hash=sha256_hash,
            would_halt=False,
            extra="not-a-dict",  # type: ignore[arg-type]
        )


def test_breach_event_rejects_non_str_symbol(sha256_hash: str, aapl: Symbol) -> None:
    """``symbol`` on BreachEvent is the ticker string, not a Symbol — ops
    dashboards index by name and account-level breaches use ``''``."""
    with pytest.raises(TypeError, match="symbol must be str"):
        BreachEvent(
            breach_id="b",
            rule_id="r",
            observed="o",
            threshold="t",
            mode=ChainMode.LIVE,
            ts_ns=0,
            intent_id=None,
            snapshot_hash=sha256_hash,
            would_halt=False,
            symbol=aapl,  # type: ignore[arg-type]
        )


def test_breach_event_extra_is_per_instance(sha256_hash: str) -> None:
    """``field(default_factory=dict)`` must give each instance its own dict,
    not share a class-level mutable default (a classic Python footgun)."""
    a = BreachEvent(
        breach_id="a", rule_id="r", observed="o", threshold="t",
        mode=ChainMode.LIVE, ts_ns=0, intent_id=None,
        snapshot_hash=sha256_hash, would_halt=False,
    )
    b = BreachEvent(
        breach_id="b", rule_id="r", observed="o", threshold="t",
        mode=ChainMode.LIVE, ts_ns=0, intent_id=None,
        snapshot_hash=sha256_hash, would_halt=False,
    )
    assert a.extra is not b.extra


def test_corporate_action_valid(aapl: Symbol) -> None:
    ca = CorporateAction(
        ca_id="01HCA",
        symbol=aapl,
        action_type=CorporateActionType.CASH_DIVIDEND,
        effective_date_ns=1_700_000_000_000_000_000,
        ratio_numerator=1,
        ratio_denominator=1,
        cash_amount=Decimal("0.24"),
        multiplier=Decimal(1) / Decimal(1),
        new_symbol=None,
        payable_date_ns=1_700_086_400_000_000_000,
        source="ibkr.reqContractDetails",
        as_of_ns=1_700_000_000_000_000_000,
    )
    assert ca.cash_amount == Decimal("0.24")


def test_corporate_action_multiplier_mismatch(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="multiplier"):
        CorporateAction(
            ca_id="ca",
            symbol=aapl,
            action_type=CorporateActionType.SPLIT,
            effective_date_ns=0,
            ratio_numerator=2,
            ratio_denominator=1,
            cash_amount=Decimal("0"),
            multiplier=Decimal("3"),  # should be 2/1
            new_symbol=None,
            payable_date_ns=None,
            source="src",
            as_of_ns=0,
        )


def test_corporate_action_ticker_change_requires_new_symbol(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="new_symbol"):
        CorporateAction(
            ca_id="ca",
            symbol=aapl,
            action_type=CorporateActionType.TICKER_CHANGE,
            effective_date_ns=0,
            ratio_numerator=1,
            ratio_denominator=1,
            cash_amount=Decimal("0"),
            multiplier=Decimal(1),
            new_symbol=None,
            payable_date_ns=None,
            source="src",
            as_of_ns=0,
        )


def test_cash_dividend_requires_amount(aapl: Symbol) -> None:
    with pytest.raises(ValueError, match="CASH_DIVIDEND"):
        CorporateAction(
            ca_id="ca",
            symbol=aapl,
            action_type=CorporateActionType.CASH_DIVIDEND,
            effective_date_ns=0,
            ratio_numerator=1,
            ratio_denominator=1,
            cash_amount=Decimal("0"),
            multiplier=Decimal(1),
            new_symbol=None,
            payable_date_ns=None,
            source="src",
            as_of_ns=0,
        )


def test_fold_spec_valid() -> None:
    fs = FoldSpec(
        fold_id=0,
        train_idx=np.arange(100),
        val_idx=np.arange(100, 110),
        test_idx=np.arange(110, 130),
        fold_meta={"train_start_ns": 0, "embargo_days": 5},
    )
    assert fs.fold_id == 0
    assert len(fs.train_idx) == 100


def test_fold_spec_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="train_idx"):
        FoldSpec(
            fold_id=0,
            train_idx=[0, 1, 2],  # type: ignore[arg-type]
            val_idx=np.array([3]),
            test_idx=np.array([4]),
            fold_meta={},
        )


# ---------------------------------------------------------------------------
# 4. Enum string values match the spec
# ---------------------------------------------------------------------------


def test_venue_values_match_spec() -> None:
    expected = {
        "NASDAQ": "NASDAQ", "NYSE": "NYSE", "ARCA": "ARCA", "BATS": "BATS",
        "IEX": "IEX", "AMEX": "AMEX",
        "CME": "CME", "COMEX": "COMEX", "NYMEX": "NYMEX", "CBOT": "CBOT", "CFE": "CFE",
        "IDEALPRO": "IDEALPRO", "SMART": "SMART",
    }
    actual = {v.name: v.value for v in Venue}
    assert actual == expected


def test_side_values_match_spec() -> None:
    assert Side.BUY.value == "BUY"
    assert Side.SELL.value == "SELL"
    assert [s.value for s in Side] == ["BUY", "SELL"]


def test_chain_mode_values_match_spec() -> None:
    # Lowercase per CONTEXT.md T11
    assert ChainMode.LIVE.value == "live"
    assert ChainMode.PAPER.value == "paper"
    assert ChainMode.WHATIF.value == "whatif"


def test_tick_type_values_match_spec() -> None:
    assert TickType.TRADES.value == "TRADES"
    assert TickType.MIDPOINT.value == "MIDPOINT"


def test_corporate_action_type_values_match_spec() -> None:
    expected = {
        "SPLIT": "split", "REVERSE_SPLIT": "reverse_split",
        "CASH_DIVIDEND": "cash_dividend", "STOCK_DIVIDEND": "stock_dividend",
        "SPINOFF": "spinoff",
        "MERGER_CASH": "merger_cash", "MERGER_STOCK": "merger_stock",
        "TICKER_CHANGE": "ticker_change", "DELISTING": "delisting",
        "RIGHTS_OFFERING": "rights_offering",
    }
    actual = {c.name: c.value for c in CorporateActionType}
    assert actual == expected


def test_strenum_str_serialization() -> None:
    """KD4: StrEnum instances JSON-serialise as their underlying string."""
    assert str(Side.BUY) == "BUY"
    assert Side.BUY == "BUY"
    assert ChainMode.LIVE == "live"


# ---------------------------------------------------------------------------
# 5. Timestamp helpers round-trip
# ---------------------------------------------------------------------------


def test_now_ns_returns_int() -> None:
    ns = now_ns()
    assert isinstance(ns, int)
    # Sanity: roughly "now" — between 2020 and 2100
    assert 1_577_836_800 * NS_PER_SECOND < ns < 4_102_444_800 * NS_PER_SECOND


def test_ns_to_datetime_round_trip_via_iso() -> None:
    original_ns = 1_700_000_000_123_456_000  # microsecond precision
    dt = ns_to_datetime_utc(original_ns)
    assert dt.tzinfo is timezone.utc
    iso = dt.isoformat()
    back_ns = parse_ts(iso)
    assert back_ns == original_ns


def test_parse_ts_z_suffix() -> None:
    ns_z = parse_ts("2024-01-01T00:00:00Z")
    ns_offset = parse_ts("2024-01-01T00:00:00+00:00")
    assert ns_z == ns_offset


def test_parse_ts_rejects_naive() -> None:
    with pytest.raises(ValueError, match="naive"):
        parse_ts("2024-01-01T00:00:00")


def test_ns_to_datetime_rejects_float() -> None:
    with pytest.raises(TypeError):
        ns_to_datetime_utc(1.0)  # type: ignore[arg-type]


def test_now_ns_monotonic_within_call() -> None:
    a = now_ns()
    b = now_ns()
    assert b >= a


def test_ns_datetime_matches_stdlib() -> None:
    """Cross-check: ns_to_datetime_utc of a known epoch agrees with stdlib."""
    ns = 0  # 1970-01-01T00:00:00Z
    dt = ns_to_datetime_utc(ns)
    assert dt == datetime(1970, 1, 1, tzinfo=timezone.utc)
