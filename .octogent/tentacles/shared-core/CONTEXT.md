# shared-core

Domain types (Order, Tick, Symbol, Position), config loader, logging primitives, error types. **This is the leaf dependency** — `shared-core` imports from nothing inside the project; every other tentacle imports from here.

## Status
Greenfield — no source code exists yet, but **types ratified 2026-05-17 (cycle 1)** after the other 4 tentacles (ml-research, strategy-engine, risk-management, backtesting) all flagged in their cycle-2/3 CONTEXTs that they were blocked on this consolidation. The dataclass specs below are the **single source of truth**: when any sibling tentacle's CONTEXT prints a schema, it is mirroring this file. If a field must change, the change lands here first and consumer tentacles update in the same PR.

## Scope
- Canonical domain types: `Symbol`, `Venue`, `Side`, `Bar`, `DollarBar`, `Tick`, `Signal`, `TargetPosition`, `TargetPositionBatch`, `OrderIntent`, `NavSnapshot`, `ChainMode`, `BreachEvent`, `CorporateAction`, `FoldSpec`
- Time + decimal helpers (nanosecond timestamps, `Decimal` arithmetic — no `float` for prices)
- Config loader (load + validate config files into typed structures)
- Logging primitive (structured logging with correlation ids: `decision_id`, `client_order_id`, `batch_id`, `intent_id`, `breach_id`)
- Error types (typed errors with `kind` so callers branch without string-matching)

## Boundaries
- **Owns:** the canonical types and the small set of utilities listed above
- **Consumes from:** nothing inside the project (leaf dependency; CI-enforced via import-linter)
- **Provides to:** every other tentacle

## Key Decisions (ratified 2026-05-17)

### KD1. Decimal type policy — `decimal.Decimal`, NOT fixed-point `int64`
**Decision: use `decimal.Decimal` from the Python standard library for every price, quantity, notional, NAV, return, and cash amount.** Fixed-point `int64` with an implied scale is rejected.

**Citation / rationale:**
- **`ml-research/CONTEXT.md` line 38** locks `r_hat_daily: Decimal` in the canonical `Signal` dataclass and says explicitly "DECIMAL (e.g. `Decimal("0.0042")`), NOT a z-score". The contract has already been published to the rest of the system as `Decimal`.
- **`strategy-engine/CONTEXT.md` Improvement #5** mirrors the same — `CorporateAction.cash_amount: Decimal`, `ratio_*` as `int`.
- **`risk-management/CONTEXT.md`** routinely speaks in dollar notionals (`$50`, `0.15 * NetLiq`); the chain functions are pure functions of `(intent, snapshot, mark)` where price/qty arithmetic must be exact. A fixed-point `int64` policy would require every check formula to carry an implied-scale comment, which is exactly the silent-bug surface area Decimal was designed to remove.
- **`backtesting/CONTEXT.md`** trade-journal parquet uses `float64` as the **wire format** (parquet doesn't natively store Decimal efficiently and Arrow's `decimal128` adds friction for downstream pandas readers). That is a **serialization-boundary** concession — in-memory all arithmetic is `Decimal`, and the conversion to `float64` happens only at `journal.write_fill(...)`. The contract test asserts round-trip parity within `1e-9` for prices and exact equality for integer share counts.
- Knuth, *The Art of Computer Programming Vol. 2* §4.2.2, and IEEE 754 — binary floats cannot exactly represent decimal fractions like `0.1`; for monetary arithmetic this is a category error, not a precision concern. Python's `decimal` module implements IEEE 754-2008 decimal arithmetic with configurable precision.
- Choosing `Decimal` over fixed-point `int64` because: (a) every sibling CONTEXT already wrote `Decimal` into its locked schema; flipping to fixed-point now would require a coordinated 4-tentacle PR with zero consumer benefit, (b) `Decimal` carries its own scale (no implicit `× 1e8` convention to forget), (c) Python's stdlib means zero new dependencies, (d) the M4 Mac mini is nowhere near Decimal performance being a bottleneck for once-per-day rebalance arithmetic.

**Context:** `getcontext().prec = 28` (Python default), `getcontext().rounding = ROUND_HALF_EVEN` (banker's rounding, IEEE 754 default). Set once in `shared_core.decimal_ctx.install()` and called from every tentacle's entrypoint. CI test asserts `Decimal("0.1") + Decimal("0.2") == Decimal("0.3")` to prove the context is installed (a failing test means someone reset to `ROUND_HALF_UP` or a lower precision).

**Construction rule:** ALWAYS construct from `str` or `int`, NEVER from `float`. `Decimal("0.1")` is exact; `Decimal(0.1)` is `Decimal('0.1000000000000000055511151231257827021181583404541015625')`. A custom mypy plugin (or a runtime guard in `shared_core.contracts._decimal_guard`) raises on `Decimal(float_value)` construction inside the project.

### KD2. Timestamp policy — `int64` nanoseconds since Unix epoch, UTC
- **Every** timestamp field is `ts_*_ns: int` (Python `int`, conceptually `int64`). Naming convention: any int-typed timestamp ends in `_ns` (e.g. `as_of_ns`, `ts_decision_ns`, `snapshot_ns`).
- Naive `datetime.datetime`, float-seconds-since-epoch, ISO strings, and pandas `Timestamp` are **rejected at the boundary**. Helpers in `shared_core.time` convert at the I/O edge only:
  - `from_iso(s: str) -> int`
  - `to_iso(ns: int) -> str`
  - `from_datetime(dt: datetime) -> int` (raises if naive)
  - `to_datetime(ns: int) -> datetime` (always tz-aware UTC)
- Rationale: int comparison is exact and tz-unambiguous; the entire substrate (ArcticDB bars, dollar-bar boundaries, fill timestamps, snapshot timestamps) is already int64-ns by convention in `market-data`, and the `ml-research` Signal contract pins `as_of_ns: int`.

### KD3. Dataclass convention — `@dataclass(frozen=True, slots=True)` everywhere
- `frozen=True`: domain types are immutable value objects; mutation is a bug (reassign the field by constructing a new instance, or use `dataclasses.replace`).
- `slots=True`: reduces memory footprint (no `__dict__`) and makes typos raise `AttributeError` at write time. Material when `TargetPositionBatch.targets` is a list of thousands of `TargetPosition` instances.
- All dataclasses live under `shared_core.contracts.<name>` and are re-exported from `shared_core.contracts.__init__`. **Canonical import path:** `from shared_core.contracts import Signal, TargetPosition, OrderIntent, NavSnapshot, ...` — siblings never deep-import a sub-module.

### KD4. Enum convention — `StrEnum` (PEP 663, Python 3.11+)
- All enums (`Venue`, `Side`, `ChainMode`, `OrderType`, `TimeInForce`, `AssetClass`, `CorporateActionType`) subclass `enum.StrEnum`.
- Reason: JSON-serializable without a custom encoder (`StrEnum` instances `__str__` to their value), comparable to plain strings at the I/O boundary (parquet `dictionary<string>` columns deserialize cleanly), and the underlying string is the canonical wire form. Avoids the `IntEnum`-vs-`Enum` confusion in older code.

### KD5. Versioning policy
- Backwards-incompatible schema changes require updating **all consumer tentacles in the same PR**. CI runs a cross-tentacle import test (`tests/contracts/test_consumer_imports.py`) that imports every public dataclass from every consumer's expected import path; a rename breaks the build immediately.
- Every dataclass carries an implicit version via the module's git sha; if a field must be removed or retyped, bump the major version of `shared-core` (no semver discipline beyond "0.x.y where any x bump is breaking" — solo-operator project).
- The contract tests living in sibling tentacles (e.g. ml-research's `test_signal_contract`, backtesting's `test_journal_schema`, risk-management's chain mode test) all import from `shared_core.contracts` and assert field names + types + `frozen`/`slots` flags by introspection.

---

## Ratified Types

All types below live under `shared_core.contracts.*` and are re-exported from `shared_core.contracts.__init__`. Import path: `from shared_core.contracts import <Name>`. All are `@dataclass(frozen=True, slots=True)` unless explicitly marked otherwise. All `Decimal` fields are constructed from `str` or `int` only (KD1). All `*_ns` fields are `int64` nanoseconds UTC (KD2).

### T1. `Symbol` — canonical instrument identifier
```python
# shared_core/contracts/symbol.py
from dataclasses import dataclass
from shared_core.contracts.venue import Venue

@dataclass(frozen=True, slots=True)
class Symbol:
    ticker: str          # e.g. "AAPL", "ESZ6", "EUR.USD"
    venue: Venue         # primary listing venue
    currency: str        # ISO 4217, e.g. "USD", "EUR", "JPY"

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.isascii():
            raise ValueError(f"ticker must be non-empty ASCII: {self.ticker!r}")
        if len(self.currency) != 3 or not self.currency.isupper():
            raise ValueError(f"currency must be 3-letter ISO 4217 uppercase: {self.currency!r}")
```
- Used as a dict key throughout (e.g. `NavSnapshot.positions: dict[Symbol, Decimal]`) — `frozen=True` makes it hashable.
- `__str__` returns `f"{ticker}@{venue.value}.{currency}"` (e.g. `"AAPL@NASDAQ.USD"`) for log lines.

### T2. `Venue` — IBKR-supported exchange enum
```python
# shared_core/contracts/venue.py
from enum import StrEnum

class Venue(StrEnum):
    # US equities / ETFs
    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    ARCA = "ARCA"
    BATS = "BATS"
    IEX = "IEX"
    AMEX = "AMEX"
    # US futures (CME Group)
    CME = "CME"           # E-mini index, FX futures
    COMEX = "COMEX"       # metals
    NYMEX = "NYMEX"       # energy
    CBOT = "CBOT"         # rates, ags
    CFE = "CFE"           # VIX futures
    # FX
    IDEALPRO = "IDEALPRO" # IBKR spot FX venue
    # Routing aliases
    SMART = "SMART"       # IBKR SmartRouter; only valid as a routing hint, not as a primary listing
```
- `SMART` is special-cased: validation in `Symbol.__post_init__` rejects it as a primary listing venue but accepts it in `OrderIntent.route` (separate field, future expansion).

### T3. `Side` — order direction
```python
# shared_core/contracts/side.py
from enum import StrEnum

class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
```

### T4. `Bar` + `DollarBar` — OHLCV substrate
```python
# shared_core/contracts/bar.py
from dataclasses import dataclass
from decimal import Decimal
from shared_core.contracts.symbol import Symbol

@dataclass(frozen=True, slots=True)
class Bar:
    symbol: Symbol
    ts_ns_utc: int          # bar close time, int64 ns UTC (KD2)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal         # shares for equity, contracts for futures, notional units for FX

    def __post_init__(self) -> None:
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(f"OHLC inconsistency at {self.ts_ns_utc}: {self}")
        if self.volume < 0:
            raise ValueError(f"negative volume at {self.ts_ns_utc}: {self.volume}")

@dataclass(frozen=True, slots=True)
class DollarBar(Bar):
    notional: Decimal       # cumulative dollar volume traded in the bar
    tick_count: int         # number of trades aggregated into this bar

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.notional <= 0:
            raise ValueError(f"non-positive dollar-bar notional: {self.notional}")
        if self.tick_count <= 0:
            raise ValueError(f"non-positive tick_count: {self.tick_count}")
```
- `DollarBar` inherits from `Bar` and adds `notional` + `tick_count`. The `ts_ns_utc` is the bar-close time (the moment the dollar-volume threshold was crossed), which is the canonical "as of" time for any feature computed from this bar.

### T5. `Tick` — single trade or quote event
```python
# shared_core/contracts/tick.py
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from shared_core.contracts.symbol import Symbol
from shared_core.contracts.venue import Venue

class TickType(StrEnum):
    TRADES = "TRADES"
    MIDPOINT = "MIDPOINT"

@dataclass(frozen=True, slots=True)
class Tick:
    symbol: Symbol
    ts_ns_utc: int
    tick_type: TickType
    # TRADES fields (None when tick_type == MIDPOINT)
    price: Decimal | None       # trade price (TRADES) — None for MIDPOINT
    size: Decimal | None        # trade size  (TRADES) — None for MIDPOINT
    exchange: Venue | None      # reporting exchange (TRADES) — None for MIDPOINT
    # MIDPOINT fields (None when tick_type == TRADES)
    mid: Decimal | None         # (bid+ask)/2 (MIDPOINT) — None for TRADES

    def __post_init__(self) -> None:
        if self.tick_type == TickType.TRADES:
            if self.price is None or self.size is None or self.exchange is None:
                raise ValueError(f"TRADES tick missing price/size/exchange: {self}")
            if self.mid is not None:
                raise ValueError(f"TRADES tick must have mid=None: {self}")
        elif self.tick_type == TickType.MIDPOINT:
            if self.mid is None:
                raise ValueError(f"MIDPOINT tick missing mid: {self}")
            if self.price is not None or self.size is not None or self.exchange is not None:
                raise ValueError(f"MIDPOINT tick must have price/size/exchange = None: {self}")
```
- Two variants in one frozen dataclass (rather than `Trade` + `Quote` siblings) so the ArcticDB tick stream can be a single typed column without a tagged-union codec. Validation enforces the disjoint-field discriminator.

### T6. `Signal` — ML inference output (consumed by strategy-engine)
**Source spec: `ml-research/CONTEXT.md` Signal contract (lines 29-47).** Mirrored verbatim here.
```python
# shared_core/contracts/signal.py
from dataclasses import dataclass
from decimal import Decimal
from shared_core.contracts.symbol import Symbol

@dataclass(frozen=True, slots=True)
class Signal:
    symbol: Symbol           # canonical, matches market-data universe registry
    r_hat_daily: Decimal     # predicted next-day open-to-close log return, Decimal (e.g. Decimal("0.0042"))
    confidence: float        # in [0.0, 1.0], paired LightGBM (0.1, 0.9) quantile spread
    model_version: str       # opaque ml-research registry handle
    as_of_ns: int            # int64 ns UTC, when features were computed

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of [0,1]: {self.confidence}")
        if not self.model_version:
            raise ValueError("model_version must be non-empty")
        if abs(self.r_hat_daily) > Decimal("0.20"):
            raise ValueError(f"|r_hat_daily| > 20% — almost certainly unit error: {self.r_hat_daily}")
```
- `confidence` is `float` (not `Decimal`) intentionally — it's a unitless ratio, not money. ml-research locked this.

### T7. `TargetPosition` — declarative end-state for one symbol
**Source spec: `strategy-engine/CONTEXT.md` Improvement #6 (lines 213-221).** `hold_reason` is the rule-id summary that risk-management's diff reads.
```python
# shared_core/contracts/target_position.py
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from shared_core.contracts.symbol import Symbol

@dataclass(frozen=True, slots=True)
class TargetPosition:
    symbol: Symbol
    target_qty: Decimal                  # signed: positive = long, negative = short, 0 = flatten
    hold_reason: str | None              # rule-id summary if the position is held by an edge-case rule; NULL otherwise
    reason_trace: dict[str, Any]         # full reasoning trace: r_hat, confidence, w_optimal_*, sigma_p,
                                         # solver_status, edge_case_rules_fired[], etc.

    def __post_init__(self) -> None:
        # reason_trace is a dict for forward-compatibility; freeze it via tuple-of-tuples for hashability
        # (we don't enforce hashability here because TargetPosition itself is rarely hashed; the batch is)
        if self.hold_reason is not None and not self.hold_reason:
            raise ValueError("hold_reason must be either None or non-empty string")
```
- `hold_reason` semantics (per strategy-engine I6):
  - `None` → normally-optimized position. risk-management computes `delta = target_qty - current_qty` and emits an `OrderIntent` if non-zero.
  - non-`None` (e.g. `"halt.luld"`, `"stale_bar.>48h"`, `"new_listing.<60d"`, `"corporate_action.spinoff_pending"`) → "freeze this position". strategy-engine sets `target_qty == current_qty`; risk-management's `INTENT_NONZERO` check naturally skips it.
- `reason_trace` is a free-form `dict` for now. A future epic may tighten it to a typed sub-dataclass once the strategy-engine reasoning fields stabilize.

### T8. `TargetPositionBatch` — atomic envelope from strategy-engine to risk-management
**Source spec: `strategy-engine/CONTEXT.md` lines 49-60 + Improvement #4 (`supersedes`) + `risk-management/CONTEXT.md` I7b (`targets_count` checksum).**
```python
# shared_core/contracts/target_position_batch.py
from dataclasses import dataclass
from shared_core.contracts.target_position import TargetPosition

@dataclass(frozen=True, slots=True)
class TargetPositionBatch:
    batch_id: str                        # ULID
    generated_at_ns: int                 # int64 ns UTC, the 09:35 ET wake timestamp (or re-emit ts)
    supersedes: str | None               # prior batch_id if this is a re-emission (Improvement #4); NULL on morning batch
    targets_count: int                   # checksum: must equal len(targets) (I7b guard against fsync-no-rename race)
    targets: tuple[TargetPosition, ...]  # immutable tuple, not list, for hashability + frozen consistency
    nav_snapshot_version: str            # hash of the NavSnapshot used to compute targets; included in decision_id

    def __post_init__(self) -> None:
        if self.targets_count != len(self.targets):
            raise ValueError(
                f"targets_count={self.targets_count} != len(targets)={len(self.targets)} "
                f"— batch is corrupt or write was non-atomic"
            )
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty (ULID)")
        # Symbol uniqueness: a batch contains at most one TargetPosition per symbol
        seen: set[Symbol] = set()
        for t in self.targets:
            if t.symbol in seen:
                raise ValueError(f"duplicate symbol in batch: {t.symbol}")
            seen.add(t.symbol)
```
- `targets` is a `tuple`, not `list` — frozen dataclass + tuple = fully immutable.
- `nav_snapshot_version` is the hash of the `NavSnapshot` (see T10) used to compute sizing; reproducibility requires it.

### T9. `OrderIntent` — imperative delta emitted by risk-management to execution
**Source spec: `risk-management/CONTEXT.md` I4 (lines 137-155) + cross-tentacle naming ratified in `strategy-engine/CONTEXT.md` (lines 39-43).**
```python
# shared_core/contracts/order_intent.py
from dataclasses import dataclass
from decimal import Decimal
from shared_core.contracts.symbol import Symbol
from shared_core.contracts.side import Side

@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str                       # ULID; PK
    symbol: Symbol
    side: Side                           # BUY or SELL (StrEnum, KD4)
    delta_qty: Decimal                   # always positive; direction is on `side`
    mark: Decimal                        # reference price used for sizing/fat-finger check
    batch_id: str                        # the TargetPositionBatch this intent was lowered from
    decision_id: str                     # upstream strategy decision (same as TargetPositionBatch.batch_id in v1)

    def __post_init__(self) -> None:
        if self.delta_qty <= 0:
            raise ValueError(f"delta_qty must be positive (direction on `side`): {self.delta_qty}")
        if self.mark <= 0:
            raise ValueError(f"mark must be positive: {self.mark}")
        if not self.intent_id or not self.batch_id or not self.decision_id:
            raise ValueError("intent_id, batch_id, decision_id must all be non-empty")
```
- `delta_qty` is **unsigned**; the sign comes from `side`. Matches the `backtesting` trade-journal convention (column comment line 194: "unsigned; sign comes from `side`").
- A `replaces: str | None` field is reserved for Improvement #4's cancel-and-replace (lowering contract step d). Added in a follow-on epic once execution's order-state machine lands; intentionally not in v1 to keep the contract minimal.

### T10. `NavSnapshot` — authoritative account state, owned by risk-management
**Source spec: `risk-management/CONTEXT.md` I3 (lines 130-135) + `strategy-engine/CONTEXT.md` NAV source key decision (lines 63-67).**
```python
# shared_core/contracts/nav_snapshot.py
from dataclasses import dataclass
from decimal import Decimal
from shared_core.contracts.symbol import Symbol

@dataclass(frozen=True, slots=True)
class NavSnapshot:
    snapshot_ns: int                            # int64 ns UTC, 09:30 ET capture time
    net_liq: Decimal                            # NetLiquidation in account currency
    buying_power: Decimal                       # IBKR AvailableFunds
    positions: dict[Symbol, Decimal]            # signed qty per held symbol; empty dict = flat book
    snapshot_version: str                       # sha256 hex of (snapshot_ns || net_liq || sorted positions || ...)
    account_id: str                             # IBKR account id (e.g. "U1234567")

    def __post_init__(self) -> None:
        if self.net_liq < 0:
            raise ValueError(f"net_liq negative — account blown up: {self.net_liq}")
        if self.buying_power < 0:
            raise ValueError(f"buying_power negative: {self.buying_power}")
        if not self.snapshot_version or len(self.snapshot_version) != 64:
            raise ValueError(f"snapshot_version must be sha256 hex (64 chars): {self.snapshot_version!r}")
        if not self.account_id:
            raise ValueError("account_id must be non-empty")
```
- `positions` is a `dict[Symbol, Decimal]`; the dataclass is `frozen=True` but the dict is mutable. **Convention:** callers MUST NOT mutate; the dataclass is treated as deeply immutable. A future hardening epic may switch to `types.MappingProxyType` or a frozen mapping helper.
- `snapshot_version` is `sha256(canonical_json(snapshot_ns, net_liq, sorted_positions, buying_power, account_id))` — a deterministic content hash. Carried in `TargetPositionBatch.nav_snapshot_version` so any snapshot replacement is fully auditable (matches strategy-engine line 67).

### T11. `ChainMode` — risk-chain execution mode (LIVE / PAPER / WHATIF)
**Source spec: `risk-management/CONTEXT.md` I7a (lines 194-213).**
```python
# shared_core/contracts/chain_mode.py
from enum import StrEnum

class ChainMode(StrEnum):
    LIVE = "live"        # writes risk/breaches.jsonl, can touch state/HALT, updates equity_curve.parquet
    PAPER = "paper"      # writes risk/paper/* sidecars; touches state/PAPER_HALT (separate file)
    WHATIF = "whatif"    # pure function; ZERO filesystem side-effects; returns BreachEvent[] to caller
```
- **There is no default value.** Every callsite (`check_chain.run(..., mode=...)`, `diff_targets_to_intents(..., mode=...)`, drawdown evaluator, NAV-delta detector) must specify explicitly. A missing `mode=` raises `TypeError` — eliminates the "I forgot to set the flag" footgun that would let a backtest sweep HALT the live bot.
- `mode` is recorded inside every `BreachEvent` (T12) so post-mortem analysis cannot confuse a replay event with a real one.

### T12. `BreachEvent` — typed event written to risk/breaches.jsonl
**Source spec: `risk-management/CONTEXT.md` Conventions (line 36) + I7a (`mode` field).**
```python
# shared_core/contracts/breach_event.py
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from shared_core.contracts.chain_mode import ChainMode

@dataclass(frozen=True, slots=True)
class BreachEvent:
    breach_id: str                        # ULID; PK
    rule_id: str                          # e.g. "KILL_SWITCH_ACTIVE", "MAX_SINGLE_NAME_NOTIONAL",
                                          #      "PRICE_BAND_FATFINGER", "STRATEGY_INTENTS_TIMEOUT"
    observed: Decimal | str | Any         # the value the check saw (Decimal for numeric checks, str for state checks)
    threshold: Decimal | str | Any        # the limit/threshold that was breached
    mode: ChainMode                       # LIVE | PAPER | WHATIF (I7a — never confuse replay with real)
    ts_ns: int                            # int64 ns UTC
    intent_id: str | None                 # the OrderIntent that triggered the breach, if applicable
    snapshot_hash: str                    # NavSnapshot.snapshot_version for replay correlation
    would_halt: bool                      # True if this breach would have flipped HALT in LIVE mode

    def __post_init__(self) -> None:
        if not self.breach_id or not self.rule_id:
            raise ValueError("breach_id and rule_id must be non-empty")
```
- `observed` / `threshold` are typed as `Decimal | str | Any` because the chain has heterogeneous checks (numeric thresholds, file-existence checks, hash comparisons). The JSONL writer renders each via `repr()` for the audit log; the in-memory type is preserved for assertions.
- `would_halt`: in `LIVE` mode the breach actually flipped HALT; in `PAPER`/`WHATIF` it tags what the live behavior would have been so replay can build a counterfactual HALT timeline.

### T13. `CorporateAction` — events that change `w_prev` overnight
**Source spec: `strategy-engine/CONTEXT.md` Improvement #5 (lines 178-211).** Producer is `compliance` (recommended); consumers are `strategy-engine` (sizing) and `risk-management` (reconciliation).
```python
# shared_core/contracts/corporate_action.py
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from shared_core.contracts.symbol import Symbol

class CorporateActionType(StrEnum):
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
    ca_id: str                            # ULID; PK
    symbol: Symbol
    action_type: CorporateActionType
    effective_date_ns: int                # int64 ns UTC, 09:30 ET of the ex-date session
    ratio_numerator: int                  # split 2-for-1 → num=2, den=1; reverse 1-for-10 → num=1, den=10
    ratio_denominator: int                # both = 1 for non-ratio actions (cash div, ticker change)
    cash_amount: Decimal                  # per pre-action share; Decimal("0") for non-cash actions
    multiplier: Decimal                   # convenience: Decimal(num) / Decimal(den); cached for sizing math
    new_symbol: Symbol | None             # populated for TICKER_CHANGE, SPINOFF (parent's child), MERGER_STOCK
    payable_date_ns: int | None           # int64 ns UTC, when cash actually hits the account (None for non-cash)
    source: str                           # "ibkr.reqContractDetails" | "vendor.<name>" | "manual.<operator_id>"
    as_of_ns: int                         # int64 ns UTC, when the producer published this row

    def __post_init__(self) -> None:
        if self.ratio_numerator <= 0 or self.ratio_denominator <= 0:
            raise ValueError(f"ratio components must be positive: {self.ratio_numerator}/{self.ratio_denominator}")
        if self.cash_amount < 0:
            raise ValueError(f"cash_amount must be >= 0: {self.cash_amount}")
        expected_mult = Decimal(self.ratio_numerator) / Decimal(self.ratio_denominator)
        if self.multiplier != expected_mult:
            raise ValueError(f"multiplier {self.multiplier} != ratio {self.ratio_numerator}/{self.ratio_denominator}")
        if self.action_type in (CorporateActionType.TICKER_CHANGE, CorporateActionType.SPINOFF,
                                CorporateActionType.MERGER_STOCK) and self.new_symbol is None:
            raise ValueError(f"{self.action_type} requires new_symbol")
        if self.action_type == CorporateActionType.CASH_DIVIDEND and self.cash_amount == 0:
            raise ValueError("CASH_DIVIDEND requires cash_amount > 0")
```
- **Consumer-side mismatch flagged:** `strategy-engine/CONTEXT.md` Improvement #5 uses field name `ex_date` (int64 ns). The task brief asks for `effective_date`. **Ratified name here: `effective_date_ns`** (KD2 — int timestamps end in `_ns`). `strategy-engine` consumer code MUST read `ca.effective_date_ns`, not `ca.ex_date`. Strategy-engine's CONTEXT.md needs a minor update on next cycle.
- **Consumer-side mismatch flagged:** strategy-engine spec uses `payable_date`, `ex_date`, `as_of` (no `_ns` suffix). All renamed here to `*_ns` per KD2 — single naming convention beats per-tentacle drift.

### T14. `FoldSpec` — walk-forward CV fold descriptor
**Source spec: `ml-research/CONTEXT.md` Walk-forward harness contract (lines 125-135) + `backtesting/CONTEXT.md` walk-forward table.** The splitter logic itself OWNED by `ml-research` (`ml_research.cv.WalkForwardHarness`); `FoldSpec` is just the data type both tentacles pass around.
```python
# shared_core/contracts/fold_spec.py
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True, slots=True)
class FoldSpec:
    """One walk-forward CV fold. Emitted by ml_research.cv.WalkForwardHarness.split()."""
    fold_id: int                          # 0, 1, 2, ... in chronological order
    train_idx: np.ndarray                 # row indices into the sorted-by-time data frame
    val_idx: np.ndarray                   # in-train validation rows (hyperparam selection)
    test_idx: np.ndarray                  # OOS test rows
    fold_meta: dict[str, Any]             # {"train_start_ns": int, "train_end_ns": int,
                                          #  "val_start_ns": int, "val_end_ns": int,
                                          #  "test_start_ns": int, "test_end_ns": int,
                                          #  "purge_days": int, "embargo_days": int, ...}
```
- **`frozen=True` but with `np.ndarray` fields:** numpy arrays are not hashable; that's accepted — `FoldSpec` instances are iterated, not used as dict keys. The frozen flag prevents reassignment, which is the intent.
- The harness construction params are **not** stored on `FoldSpec` itself — they belong on the harness object. They are listed below for documentation, mirroring the canonical signature ratified in ml-research's CONTEXT (line 129) and backtesting's contract test (lines 73-86):
  - `train_min_years: int = 5`
  - `train_max_years: int = 15`
  - `val_days: int = 252`
  - `embargo_days: int = 5`
  - `purge_days: int = 1`
  - `test_days: int = 21`
  - `refit_cadence: str = "monthly"` (literal `"monthly"` or `"annual"`)
- backtesting's annual-cadence config: `WalkForwardHarness(test_days=252, refit_cadence="annual")` — not a separate splitter (per ml-research line 133).
- `FoldSpec.fold_meta` is a free `dict` to allow ml-research to add fields (regime tags, sample-weight stats per fold) without breaking the dataclass contract.

---

## Conventions
- **Adding a type here requires that at least two other tentacles use it.** One-use code stays in its home tentacle.
- **Never `float` for prices, quantities, notionals, NAV, or cash amounts.** `Decimal` only (KD1). `float` is acceptable for `confidence`, IC, p-values, ratios — anything unitless.
- **Never naive `datetime`.** `int64` ns UTC only (KD2).
- **Construct `Decimal` from `str` or `int`, never `float`.** A runtime guard in `shared_core.contracts._decimal_guard` raises on `Decimal(float_value)` inside project code.
- **All dataclasses are `frozen=True, slots=True`** (KD3). Reassign via `dataclasses.replace`.
- **All enums are `StrEnum`** (KD4). JSON-serializable, parquet-friendly, comparable to strings.
- **Canonical import path:** `from shared_core.contracts import <Name>`. Sub-module deep imports are a convention violation.
- **Backwards-incompatible changes** require updating all consumer tentacles in the same PR (KD5).
- **Every log record carries a correlation id**: `decision_id`, `client_order_id`, `batch_id`, `intent_id`, or `breach_id` — whichever applies to the call path.
- **Anti-circular-dep:** `shared_core/` MUST NOT import from any other tentacle. CI-enforced via import-linter.

## Cross-tentacle mismatches found during ratification (consumer-side updates needed)
1. **`strategy-engine/CONTEXT.md` Improvement #5 `CorporateAction`** uses field names `ex_date`, `payable_date`, `as_of` (no `_ns` suffix). Ratified here as `effective_date_ns`, `payable_date_ns`, `as_of_ns` per KD2 (all int timestamps end in `_ns`). strategy-engine consumer code must read the `_ns`-suffixed names. Renamed `action_type: str (enum string)` → `action_type: CorporateActionType` (StrEnum).
2. **`strategy-engine/CONTEXT.md` line 56** `nav_snapshot: Money` in the TargetPositionBatch envelope — this field is dropped from the ratified `TargetPositionBatch` (T8). The batch carries `nav_snapshot_version: str` only; consumers re-read the full `NavSnapshot` from `risk-management` using that version hash. Avoids embedding a fat object in every batch and avoids the temptation to deserialize it as a different `Money` type. strategy-engine envelope needs to drop the `nav_snapshot` and `nav_snapshot_source` fields and rely on the version hash + risk-management's snapshot read API.
3. **`risk-management/CONTEXT.md` I4** mentions `target_qty: int` in the `TargetPosition` it diffs — ratified here as `Decimal` (KD1). risk-management's diff function signature must accept `Decimal` qty (and round to lot at the broker boundary, not inside the diff).
4. **`ml-research/CONTEXT.md`** is consistent — no changes needed. Its locked Signal spec is mirrored byte-for-byte here.
5. **`backtesting/CONTEXT.md`** parquet trade-journal uses `float64` for `fill_px`, `commission`, etc. — that's the wire format and is fine. In-memory `ExecutionSimulator` arithmetic must use `Decimal` and convert at `execution.journal.write_fill(...)`. A round-trip CI test asserts `Decimal → float64 → Decimal` within `1e-9` for prices and exact equality for integer share counts.
