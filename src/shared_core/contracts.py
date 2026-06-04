"""shared_core.contracts — canonical domain dataclasses.

Single source of truth for cross-tentacle types. All dataclasses are
``@dataclass(frozen=True, slots=True)`` (KD3). All enums are
``StrEnum`` (KD4). All money-like fields are ``Decimal`` constructed
only from ``str|int`` (KD1). All timestamp fields end ``_ns`` and are
``int64`` UTC nanoseconds (KD2).

Canonical import path:
    from shared_core.contracts import Signal, OrderIntent, ...

This module intentionally contains all 14 ratified types in one file
rather than the per-file sub-module layout sketched in CONTEXT.md — a
single file is easier to keep in lockstep when consumer tentacles need
to introspect the schema, and the import path is the same either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


__all__ = [
    "Bar",
    "BreachEvent",
    "ChainMode",
    "CorporateAction",
    "CorporateActionType",
    "DollarBar",
    "FoldSpec",
    "NavSnapshot",
    "OrderIntent",
    "Side",
    "Signal",
    "Symbol",
    "TargetPosition",
    "TargetPositionBatch",
    "Tick",
    "TickType",
    "Venue",
]


# ---------------------------------------------------------------------------
# Decimal construction guard (KD1)
# ---------------------------------------------------------------------------


def _decimal_guard(value: Decimal, *, field: str) -> Decimal:
    """Reject ``Decimal(float)`` construction at the contract boundary.

    The convention is: every money/quantity field is built from ``str``
    or ``int`` only. If a caller hands us ``Decimal(0.1)`` the value
    already carries 17 digits of binary-float noise — we cannot
    retroactively detect that, so we instead reject any non-Decimal
    input and rely on caller discipline (or the project's mypy plugin)
    to catch float construction at the call site.

    What we *can* catch at runtime: a caller who passes a raw ``float``
    instead of wrapping in ``Decimal`` at all. That's the most common
    footgun and the one this guard explicitly fails on.
    """
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise TypeError(
            f"{field} must be a Decimal constructed from str|int "
            f"(got {type(value).__name__}={value!r}); use Decimal(\"...\") "
            f"or shared_core.types.money/quantity helpers"
        )
    return value


# ---------------------------------------------------------------------------
# T2. Venue
# ---------------------------------------------------------------------------


class Venue(StrEnum):
    """IBKR-supported exchange enum (KD4)."""

    # US equities / ETFs
    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    ARCA = "ARCA"
    BATS = "BATS"
    IEX = "IEX"
    AMEX = "AMEX"
    # US futures (CME Group)
    CME = "CME"
    COMEX = "COMEX"
    NYMEX = "NYMEX"
    CBOT = "CBOT"
    CFE = "CFE"
    # FX
    IDEALPRO = "IDEALPRO"
    # Routing aliases
    SMART = "SMART"  # only valid as a routing hint, NOT as a primary listing


# ---------------------------------------------------------------------------
# T3. Side
# ---------------------------------------------------------------------------


class Side(StrEnum):
    """Order direction (KD4)."""

    BUY = "BUY"
    SELL = "SELL"


# ---------------------------------------------------------------------------
# T1. Symbol
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Symbol:
    """Canonical instrument identifier (hashable, used as dict key)."""

    ticker: str
    venue: Venue
    currency: str

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.isascii():
            raise ValueError(f"ticker must be non-empty ASCII: {self.ticker!r}")
        if not isinstance(self.venue, Venue):
            raise TypeError(f"venue must be Venue (got {type(self.venue).__name__})")
        if self.venue is Venue.SMART:
            raise ValueError(
                "SMART is a routing hint, not a primary listing venue; "
                "use the actual listing venue (NASDAQ/NYSE/etc.)"
            )
        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isupper()
            or not self.currency.isascii()
        ):
            raise ValueError(
                f"currency must be 3-letter ISO 4217 uppercase: {self.currency!r}"
            )

    def __str__(self) -> str:
        return f"{self.ticker}@{self.venue.value}.{self.currency}"


# ---------------------------------------------------------------------------
# T4. Bar + DollarBar
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bar:
    """OHLCV bar over an arbitrary aggregation window."""

    symbol: Symbol
    ts_ns_utc: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        _decimal_guard(self.open, field="open")
        _decimal_guard(self.high, field="high")
        _decimal_guard(self.low, field="low")
        _decimal_guard(self.close, field="close")
        _decimal_guard(self.volume, field="volume")
        if not isinstance(self.ts_ns_utc, int) or isinstance(self.ts_ns_utc, bool):
            raise TypeError(f"ts_ns_utc must be int (got {type(self.ts_ns_utc).__name__})")
        if not (
            self.low <= self.open <= self.high and self.low <= self.close <= self.high
        ):
            raise ValueError(
                f"OHLC inconsistency at {self.ts_ns_utc}: "
                f"open={self.open} high={self.high} low={self.low} close={self.close}"
            )
        if self.volume < 0:
            raise ValueError(f"negative volume at {self.ts_ns_utc}: {self.volume}")


@dataclass(frozen=True, slots=True)
class DollarBar(Bar):
    """Bar aggregated to a fixed dollar-volume threshold.

    Implementation note: we call ``Bar.__post_init__(self)`` explicitly
    rather than ``super().__post_init__()`` to dodge CPython gh-90562 —
    ``@dataclass(slots=True)`` rebuilds the class, which in turn confuses
    a zero-arg ``super()`` lookup inside the generated subclass body.
    """

    notional: Decimal
    tick_count: int

    def __post_init__(self) -> None:
        Bar.__post_init__(self)
        _decimal_guard(self.notional, field="notional")
        if not isinstance(self.tick_count, int) or isinstance(self.tick_count, bool):
            raise TypeError(
                f"tick_count must be int (got {type(self.tick_count).__name__})"
            )
        if self.notional <= 0:
            raise ValueError(f"non-positive dollar-bar notional: {self.notional}")
        if self.tick_count <= 0:
            raise ValueError(f"non-positive tick_count: {self.tick_count}")


# ---------------------------------------------------------------------------
# T5. Tick
# ---------------------------------------------------------------------------


class TickType(StrEnum):
    """Tick variant discriminator."""

    TRADES = "TRADES"
    MIDPOINT = "MIDPOINT"


@dataclass(frozen=True, slots=True)
class Tick:
    """Single trade or quote event from the market-data tentacle."""

    symbol: Symbol
    ts_ns_utc: int
    tick_type: TickType
    price: Decimal | None
    size: Decimal | None
    exchange: Venue | None
    mid: Decimal | None

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns_utc, int) or isinstance(self.ts_ns_utc, bool):
            raise TypeError(f"ts_ns_utc must be int (got {type(self.ts_ns_utc).__name__})")
        if not isinstance(self.tick_type, TickType):
            raise TypeError(
                f"tick_type must be TickType (got {type(self.tick_type).__name__})"
            )

        if self.tick_type == TickType.TRADES:
            if self.price is None or self.size is None or self.exchange is None:
                raise ValueError(
                    f"TRADES tick missing price/size/exchange: "
                    f"price={self.price} size={self.size} exchange={self.exchange}"
                )
            _decimal_guard(self.price, field="price")
            _decimal_guard(self.size, field="size")
            if not isinstance(self.exchange, Venue):
                raise TypeError(
                    f"exchange must be Venue (got {type(self.exchange).__name__})"
                )
            if self.mid is not None:
                raise ValueError(f"TRADES tick must have mid=None (got {self.mid})")
        elif self.tick_type == TickType.MIDPOINT:
            if self.mid is None:
                raise ValueError("MIDPOINT tick missing mid")
            _decimal_guard(self.mid, field="mid")
            if (
                self.price is not None
                or self.size is not None
                or self.exchange is not None
            ):
                raise ValueError(
                    f"MIDPOINT tick must have price/size/exchange = None "
                    f"(got price={self.price} size={self.size} exchange={self.exchange})"
                )


# ---------------------------------------------------------------------------
# T6. Signal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signal:
    """ML inference output (produced by ml-research, consumed by strategy-engine).

    ``confidence`` is intentionally ``float``: a unitless [0, 1] ratio,
    not money. Every other amount-like field is ``Decimal``.
    """

    symbol: Symbol
    r_hat_daily: Decimal
    confidence: float
    model_version: str
    as_of_ns: int

    def __post_init__(self) -> None:
        _decimal_guard(self.r_hat_daily, field="r_hat_daily")
        if not isinstance(self.confidence, float) or isinstance(self.confidence, bool):
            raise TypeError(
                f"confidence must be float (got {type(self.confidence).__name__})"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of [0,1]: {self.confidence}")
        if not self.model_version:
            raise ValueError("model_version must be non-empty")
        if not isinstance(self.as_of_ns, int) or isinstance(self.as_of_ns, bool):
            raise TypeError(f"as_of_ns must be int (got {type(self.as_of_ns).__name__})")
        if abs(self.r_hat_daily) > Decimal("0.20"):
            raise ValueError(
                f"|r_hat_daily| > 20% — almost certainly unit error: {self.r_hat_daily}"
            )


# ---------------------------------------------------------------------------
# T11. ChainMode (moved above TargetPositionBatch in cycle 8 — Epic 16 lifts
# ``mode`` onto the batch envelope, so the enum must be defined first.)
# ---------------------------------------------------------------------------


class ChainMode(StrEnum):
    """Risk-chain execution mode. No default — every callsite must specify."""

    LIVE = "live"
    PAPER = "paper"
    WHATIF = "whatif"


# ---------------------------------------------------------------------------
# T7. TargetPosition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """Declarative end-state for one symbol (strategy-engine output)."""

    symbol: Symbol
    target_qty: Decimal
    hold_reason: str | None
    reason_trace: dict[str, Any]

    def __post_init__(self) -> None:
        _decimal_guard(self.target_qty, field="target_qty")
        if self.hold_reason is not None:
            if not isinstance(self.hold_reason, str):
                raise TypeError(
                    f"hold_reason must be str|None (got {type(self.hold_reason).__name__})"
                )
            if not self.hold_reason:
                raise ValueError("hold_reason must be either None or non-empty string")
        if not isinstance(self.reason_trace, dict):
            raise TypeError(
                f"reason_trace must be dict (got {type(self.reason_trace).__name__})"
            )


# ---------------------------------------------------------------------------
# T8. TargetPositionBatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetPositionBatch:
    """Atomic envelope from strategy-engine to risk-management.

    Cycle-8 decision (Epic 16, shared-core owner): the three cross-cutting
    observability fields ``decision_id``, ``mode``, and ``solver_status``
    were lifted from the strategy-engine local envelope into the canonical
    shared_core form. Rationale:

    - ``decision_id`` is the reproducibility primary key per strategy-engine
      CONTEXT.md and is also a required correlator on every ``OrderIntent``
      and ``BreachEvent``. Two tentacles already depend on it being a
      first-class field on the batch; promoting it eliminates the need for
      a separate sidecar lookup.
    - ``mode: ChainMode`` is frozen at 09:35 wake (strategy-engine I7) and
      required by risk-management's ``check_chain.run`` signature (T11/I7a).
      Carrying it on the batch makes the cross-tentacle handoff explicit;
      every downstream rule sees the same mode value.
    - ``solver_status`` is the Stage-1/2/3 result observability hard-depends
      on (``optimal``/``fallback_hrp_after_3_retries``/``halted_no_universe``
      etc.). Replay tooling cannot reconstruct without it.

    Strategy-engine-internal numerics (``r_hat``, ``sigma_contribution``,
    ``target_weight``, ``confidence``, ``w_prev``, ``sigma_portfolio_annualized``,
    ``edge_case_rules_fired``) intentionally stay OUT of shared_core — those
    are reasoning-trace fields no other tentacle reads. They belong in the
    strategy-engine envelope and/or ``TargetPosition.reason_trace``.

    Epic 15 ratified dropping ``nav_snapshot: Decimal`` and
    ``nav_snapshot_source`` from the envelope in favour of
    ``nav_snapshot_version`` (consumers re-read the full NavSnapshot using
    the version hash); that decision stands.

    ``batch_id`` and ``decision_id`` can be equal in single-emission
    sessions; ``supersedes`` carries the prior batch_id for re-emissions.
    """

    batch_id: str
    decision_id: str
    mode: ChainMode
    solver_status: str
    generated_at_ns: int
    supersedes: str | None
    targets_count: int
    targets: tuple[TargetPosition, ...]
    nav_snapshot_version: str

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty (ULID)")
        if not self.decision_id:
            raise ValueError("decision_id must be non-empty (reproducibility hash)")
        if not isinstance(self.mode, ChainMode):
            raise TypeError(
                f"mode must be ChainMode (got {type(self.mode).__name__}) — "
                f"no default permitted (T11); every batch carries its chain mode"
            )
        if not isinstance(self.solver_status, str) or not self.solver_status:
            raise ValueError(
                "solver_status must be a non-empty string "
                "(e.g. 'optimal' | 'fallback_hrp_after_3_retries' | 'halted_no_universe')"
            )
        if not isinstance(self.generated_at_ns, int) or isinstance(
            self.generated_at_ns, bool
        ):
            raise TypeError(
                f"generated_at_ns must be int (got {type(self.generated_at_ns).__name__})"
            )
        if self.supersedes is not None and not isinstance(self.supersedes, str):
            raise TypeError(
                f"supersedes must be str|None (got {type(self.supersedes).__name__})"
            )
        if not isinstance(self.targets, tuple):
            raise TypeError(
                f"targets must be tuple (got {type(self.targets).__name__}) — "
                f"frozen consistency requires immutable container"
            )
        if self.targets_count != len(self.targets):
            raise ValueError(
                f"targets_count={self.targets_count} != len(targets)={len(self.targets)} "
                f"— batch is corrupt or write was non-atomic"
            )
        if not self.nav_snapshot_version:
            raise ValueError("nav_snapshot_version must be non-empty")
        # Symbol uniqueness: at most one TargetPosition per symbol per batch
        seen: set[Symbol] = set()
        for t in self.targets:
            if not isinstance(t, TargetPosition):
                raise TypeError(
                    f"targets must contain TargetPosition (got {type(t).__name__})"
                )
            if t.symbol in seen:
                raise ValueError(f"duplicate symbol in batch: {t.symbol}")
            seen.add(t.symbol)


# ---------------------------------------------------------------------------
# T9. OrderIntent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """Imperative delta emitted by risk-management to execution."""

    intent_id: str
    symbol: Symbol
    side: Side
    delta_qty: Decimal
    mark: Decimal
    batch_id: str
    decision_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.side, Side):
            raise TypeError(f"side must be Side (got {type(self.side).__name__})")
        _decimal_guard(self.delta_qty, field="delta_qty")
        _decimal_guard(self.mark, field="mark")
        if self.delta_qty <= 0:
            raise ValueError(
                f"delta_qty must be positive (direction is on `side`): {self.delta_qty}"
            )
        if self.mark <= 0:
            raise ValueError(f"mark must be positive: {self.mark}")
        if not self.intent_id or not self.batch_id or not self.decision_id:
            raise ValueError("intent_id, batch_id, decision_id must all be non-empty")


# ---------------------------------------------------------------------------
# T10. NavSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NavSnapshot:
    """Authoritative account state captured at 09:30 ET by risk-management.

    ``positions`` is a ``dict`` — the dataclass is frozen but the dict
    itself is mutable. Callers MUST NOT mutate; the instance is treated
    as deeply immutable. A future epic may switch to ``MappingProxyType``.

    Cycle-8 decision (Epic 17 part 2, shared-core owner): ``positions`` STAYS
    ``dict[Symbol, Decimal]`` (canonical Symbol keying). Risk-management
    cycle 7 raised the option of migrating the shared_core type to
    ``dict[str, Decimal]`` to match the local chain cache; that proposal
    was rejected. Rationale: ``Symbol`` is the cross-tentacle primary key
    (T1), execution requires real venue/currency metadata, and a
    ticker-only key would force every consumer to re-resolve venue via a
    symbology table at every read. Risk-management keeps the str-keyed
    local cache and uses the ``IntentContext.ticker`` bridge documented
    in risk-management/CONTEXT.md; the bridge to shared_core constructs
    real Symbols at the chain → execution boundary.
    """

    snapshot_ns: int
    net_liq: Decimal
    buying_power: Decimal
    positions: dict[Symbol, Decimal]
    snapshot_version: str
    account_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_ns, int) or isinstance(self.snapshot_ns, bool):
            raise TypeError(
                f"snapshot_ns must be int (got {type(self.snapshot_ns).__name__})"
            )
        _decimal_guard(self.net_liq, field="net_liq")
        _decimal_guard(self.buying_power, field="buying_power")
        if self.net_liq < 0:
            raise ValueError(f"net_liq negative — account blown up: {self.net_liq}")
        if self.buying_power < 0:
            raise ValueError(f"buying_power negative: {self.buying_power}")
        if not isinstance(self.positions, dict):
            raise TypeError(
                f"positions must be dict (got {type(self.positions).__name__})"
            )
        for sym, qty in self.positions.items():
            if not isinstance(sym, Symbol):
                raise TypeError(f"positions key must be Symbol (got {type(sym).__name__})")
            _decimal_guard(qty, field=f"positions[{sym}]")
        if not self.snapshot_version or len(self.snapshot_version) != 64:
            raise ValueError(
                f"snapshot_version must be sha256 hex (64 chars): "
                f"{self.snapshot_version!r}"
            )
        if not self.account_id:
            raise ValueError("account_id must be non-empty")


# ---------------------------------------------------------------------------
# T12. BreachEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BreachEvent:
    """Typed event written to risk/breaches.jsonl.

    Cycle-8 decision (Epic 17 part 1, shared-core owner): the three audit
    fields ``config_id``, ``symbol``, and ``extra`` were lifted from the
    risk-management in-memory ``_AUDIT_CTX`` sidecar onto the canonical
    contract. Rationale:

    - The sidecar was process-local mutable state keyed by ``breach_id``,
      brittle across test boundaries and impossible to replay from the
      JSONL alone. Replay tooling MUST reconstruct without out-of-band
      state — the JSONL is the source of truth for compliance.
    - ``config_id`` ties the breach to the exact ``risk/limits.toml`` hash
      that was active when the rule fired (audit traceability).
    - ``symbol`` (ticker string, not ``Symbol``) is read by ops dashboards
      that index breaches by name; using ``str`` rather than ``Symbol``
      lets non-trading-instrument breaches (e.g. account-level kill switch)
      use ``""`` without manufacturing a placeholder ``Symbol``.
    - ``extra: dict[str, Any]`` carries per-rule context
      (``pct``/``net_liq`` on MAX_SINGLE_NAME, ``kelly_optimal`` /
      ``kelly_fraction`` on KELLY_CAP, ``broker``/``internal`` on
      POSITION_RECONCILES). Schema is rule-specific so a typed union would
      explode; ``dict[str, Any]`` is the pragmatic choice for an audit log.

    All three default empty so existing callers (and other tentacles that
    raise breaches without the audit context) keep compiling.

    ``ts_unix`` is intentionally NOT added — KD2 says ``ts_ns`` is the
    canonical timestamp; replay tooling can derive seconds-float at I/O
    time via ``ts_ns / 1e9``.
    """

    breach_id: str
    rule_id: str
    observed: Decimal | str | Any
    threshold: Decimal | str | Any
    mode: ChainMode
    ts_ns: int
    intent_id: str | None
    snapshot_hash: str
    would_halt: bool
    config_id: str = ""
    symbol: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.breach_id or not self.rule_id:
            raise ValueError("breach_id and rule_id must be non-empty")
        if not isinstance(self.mode, ChainMode):
            raise TypeError(f"mode must be ChainMode (got {type(self.mode).__name__})")
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"ts_ns must be int (got {type(self.ts_ns).__name__})")
        if self.intent_id is not None and not isinstance(self.intent_id, str):
            raise TypeError(
                f"intent_id must be str|None (got {type(self.intent_id).__name__})"
            )
        if not isinstance(self.snapshot_hash, str) or not self.snapshot_hash:
            raise ValueError("snapshot_hash must be a non-empty string")
        if not isinstance(self.would_halt, bool):
            raise TypeError(
                f"would_halt must be bool (got {type(self.would_halt).__name__})"
            )
        if not isinstance(self.config_id, str):
            raise TypeError(
                f"config_id must be str (got {type(self.config_id).__name__})"
            )
        if not isinstance(self.symbol, str):
            raise TypeError(
                f"symbol must be str ticker (got {type(self.symbol).__name__}); "
                f"use Symbol.ticker for the string form"
            )
        if not isinstance(self.extra, dict):
            raise TypeError(
                f"extra must be dict (got {type(self.extra).__name__})"
            )


# ---------------------------------------------------------------------------
# T13. CorporateAction
# ---------------------------------------------------------------------------


class CorporateActionType(StrEnum):
    """Corporate-action discriminator (KD4)."""

    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SPINOFF = "spinoff"
    MERGER_CASH = "merger_cash"
    MERGER_STOCK = "merger_stock"
    TICKER_CHANGE = "ticker_change"
    DELISTING = "delisting"
    RIGHTS_OFFERING = "rights_offering"


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """Event that changes ``w_prev`` overnight (split, div, spinoff, ...)."""

    ca_id: str
    symbol: Symbol
    action_type: CorporateActionType
    effective_date_ns: int
    ratio_numerator: int
    ratio_denominator: int
    cash_amount: Decimal
    multiplier: Decimal
    new_symbol: Symbol | None
    payable_date_ns: int | None
    source: str
    as_of_ns: int

    def __post_init__(self) -> None:
        if not self.ca_id:
            raise ValueError("ca_id must be non-empty")
        if not isinstance(self.action_type, CorporateActionType):
            raise TypeError(
                f"action_type must be CorporateActionType (got "
                f"{type(self.action_type).__name__})"
            )
        for ns_field in ("effective_date_ns", "as_of_ns"):
            val = getattr(self, ns_field)
            if not isinstance(val, int) or isinstance(val, bool):
                raise TypeError(f"{ns_field} must be int (got {type(val).__name__})")
        if self.payable_date_ns is not None and (
            not isinstance(self.payable_date_ns, int)
            or isinstance(self.payable_date_ns, bool)
        ):
            raise TypeError(
                f"payable_date_ns must be int|None "
                f"(got {type(self.payable_date_ns).__name__})"
            )
        if (
            not isinstance(self.ratio_numerator, int)
            or isinstance(self.ratio_numerator, bool)
            or not isinstance(self.ratio_denominator, int)
            or isinstance(self.ratio_denominator, bool)
        ):
            raise TypeError("ratio_numerator and ratio_denominator must be int")
        if self.ratio_numerator <= 0 or self.ratio_denominator <= 0:
            raise ValueError(
                f"ratio components must be positive: "
                f"{self.ratio_numerator}/{self.ratio_denominator}"
            )
        _decimal_guard(self.cash_amount, field="cash_amount")
        _decimal_guard(self.multiplier, field="multiplier")
        if self.cash_amount < 0:
            raise ValueError(f"cash_amount must be >= 0: {self.cash_amount}")
        expected_mult = Decimal(self.ratio_numerator) / Decimal(self.ratio_denominator)
        if self.multiplier != expected_mult:
            raise ValueError(
                f"multiplier {self.multiplier} != ratio "
                f"{self.ratio_numerator}/{self.ratio_denominator} (= {expected_mult})"
            )
        if (
            self.action_type
            in (
                CorporateActionType.TICKER_CHANGE,
                CorporateActionType.SPINOFF,
                CorporateActionType.MERGER_STOCK,
            )
            and self.new_symbol is None
        ):
            raise ValueError(f"{self.action_type} requires new_symbol")
        if (
            self.action_type == CorporateActionType.CASH_DIVIDEND
            and self.cash_amount == 0
        ):
            raise ValueError("CASH_DIVIDEND requires cash_amount > 0")
        if not self.source:
            raise ValueError("source must be non-empty")


# ---------------------------------------------------------------------------
# T14. FoldSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """One walk-forward CV fold.

    Emitted by ``ml_research.cv.WalkForwardHarness.split()``. The numpy
    arrays are not hashable; ``FoldSpec`` is iterated, never used as a
    dict key — the ``frozen=True`` flag still prevents reassignment.
    """

    fold_id: int
    train_idx: "np.ndarray"
    val_idx: "np.ndarray"
    test_idx: "np.ndarray"
    fold_meta: dict[str, Any]

    def __post_init__(self) -> None:
        # Late import keeps shared_core import cheap if numpy is unused
        # at module-load time, but FoldSpec construction always needs it.
        import numpy as np

        if not isinstance(self.fold_id, int) or isinstance(self.fold_id, bool):
            raise TypeError(f"fold_id must be int (got {type(self.fold_id).__name__})")
        if self.fold_id < 0:
            raise ValueError(f"fold_id must be >= 0: {self.fold_id}")
        for name in ("train_idx", "val_idx", "test_idx"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"{name} must be np.ndarray (got {type(arr).__name__})"
                )
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D (got ndim={arr.ndim})")
        if not isinstance(self.fold_meta, dict):
            raise TypeError(
                f"fold_meta must be dict (got {type(self.fold_meta).__name__})"
            )
