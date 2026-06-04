"""treasury.balance_ledger — append-only balance snapshot + reconciliation.

Paper-trading MVP, Alpaca-only. Persists each snapshot of the venue's
account summary as one row of ``data/treasury/balances.parquet``.
Schema (all Decimal-typed values stored as strings to preserve KD1
precision — parquet has no native Decimal-256, and float would defeat
the point of treasury):

    timestamp_ns   : int64   — KD2, snapshot wall-clock UTC ns
    account_id     : str
    equity         : str (Decimal)  — total account equity (NLV)
    cash           : str (Decimal)
    buying_power   : str (Decimal)
    long_market_value  : str (Decimal)
    short_market_value : str (Decimal)
    portfolio_value    : str (Decimal)
    daytrade_count : int
    pattern_day_trader : bool
    status         : str

Reconciliation is a pure function:

    ``reconcile(internal_positions, alpaca_positions, *, tolerance) -> list[PositionBreak]``

It compares two ``{ticker: Decimal qty}`` dicts and emits one
``PositionBreak`` per symbol whose absolute difference exceeds the
tolerance (default zero — paper trading should reconcile exactly).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "BalanceLedger",
    "PositionBreak",
    "reconcile",
]


# Fields we pull from Alpaca's TradingAccount object (alpaca-py) into the
# parquet row. Each value gets coerced to ``str(Decimal(...))`` so the
# precision survives a parquet round-trip.
_DECIMAL_FIELDS: tuple[str, ...] = (
    "equity",
    "cash",
    "buying_power",
    "long_market_value",
    "short_market_value",
    "portfolio_value",
)

_OPTIONAL_FIELDS: tuple[str, ...] = (
    "daytrade_count",
    "pattern_day_trader",
    "status",
    "account_id",
)


# ---------------------------------------------------------------------------
# Reconciliation result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionBreak:
    """One symbol whose internal qty disagrees with the venue snapshot.

    All quantities are ``Decimal``. ``diff = internal - venue``: positive
    means treasury thinks we own more than Alpaca says we do (most likely
    a missed fill from the venue side), negative means Alpaca holds more
    than treasury knows about (probably a manual entry or an unrecorded
    fill).
    """

    ticker: str
    internal_qty: Decimal
    venue_qty: Decimal
    diff: Decimal
    tolerance: Decimal

    def to_jsonable(self) -> dict[str, Any]:
        """Render to a JSON-safe dict (Decimal → str) for ``breaks.jsonl``."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Decimal):
                d[k] = str(v)
        return d


# ---------------------------------------------------------------------------
# Reconciliation engine
# ---------------------------------------------------------------------------


def reconcile(
    internal_positions: Mapping[str, Decimal] | Mapping[str, int | str],
    alpaca_positions: Mapping[str, Decimal] | Mapping[str, int | str],
    *,
    tolerance: Decimal = Decimal("0"),
) -> list[PositionBreak]:
    """Compare internal vs. venue positions, return list of breaks.

    A pure function: same inputs → same output, no I/O, no clock reads.
    Replayable from the snapshot pair alone.
    """

    if not isinstance(tolerance, Decimal) or tolerance < 0:
        raise ValueError(
            f"tolerance must be a non-negative Decimal (got {tolerance!r})"
        )

    def _coerce(value: Decimal | int | str) -> Decimal:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, bool):  # bool is an int subclass — exclude.
            raise TypeError(f"position qty cannot be bool: {value!r}")
        if isinstance(value, (int, str)):
            return Decimal(value)
        raise TypeError(
            f"position qty must be Decimal|int|str (got {type(value).__name__})"
        )

    tickers = set(internal_positions) | set(alpaca_positions)
    breaks: list[PositionBreak] = []
    for ticker in sorted(tickers):
        internal_qty = _coerce(internal_positions.get(ticker, Decimal("0")))
        venue_qty = _coerce(alpaca_positions.get(ticker, Decimal("0")))
        diff = internal_qty - venue_qty
        if abs(diff) > tolerance:
            breaks.append(
                PositionBreak(
                    ticker=ticker,
                    internal_qty=internal_qty,
                    venue_qty=venue_qty,
                    diff=diff,
                    tolerance=tolerance,
                )
            )
    return breaks


# ---------------------------------------------------------------------------
# Persistent ledger
# ---------------------------------------------------------------------------


@dataclass
class BalanceLedger:
    """Append-only parquet ledger of Alpaca account snapshots.

    Default path is ``<repo>/data/treasury/balances.parquet`` (resolved
    via the constructor). Writes are *not* atomic — parquet append is
    implemented as ``read_existing → concat → overwrite via tmp+rename``.
    For 15-minute snapshots the table stays small (~96 rows/day) so the
    cost is negligible; if we move to per-fill cadence we'll swap in
    ArcticDB symbol writes.
    """

    parquet_path: Path

    def __post_init__(self) -> None:
        self.parquet_path = Path(self.parquet_path)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # -- snapshot extraction ------------------------------------------------
    @staticmethod
    def _coerce_account_summary(
        timestamp_ns: int,
        account_summary: Any,
    ) -> dict[str, Any]:
        """Pull a flat row out of an Alpaca ``TradingAccount`` object or dict.

        Accepts either an alpaca-py ``TradingAccount`` (attribute access)
        or a plain ``dict`` (key access) so tests can pass either form.
        """

        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            raise TypeError(
                f"timestamp_ns must be int (got {type(timestamp_ns).__name__})"
            )

        def _get(name: str, default: Any = None) -> Any:
            if isinstance(account_summary, Mapping):
                return account_summary.get(name, default)
            return getattr(account_summary, name, default)

        row: dict[str, Any] = {"timestamp_ns": timestamp_ns}

        for f in _DECIMAL_FIELDS:
            raw = _get(f)
            if raw is None:
                row[f] = None
            else:
                # str(Decimal(str(raw))) — round-trips through string so a
                # float input (Alpaca's SDK sometimes returns those) doesn't
                # poison the precision.
                row[f] = str(Decimal(str(raw)))

        account_id = _get("account_id") or _get("id")
        row["account_id"] = "" if account_id is None else str(account_id)

        dt_count = _get("daytrade_count")
        row["daytrade_count"] = (
            int(dt_count) if dt_count is not None else 0
        )

        pdt = _get("pattern_day_trader")
        row["pattern_day_trader"] = bool(pdt) if pdt is not None else False

        status = _get("status")
        row["status"] = "" if status is None else str(status)

        return row

    # -- write --------------------------------------------------------------
    def record_snapshot(
        self,
        timestamp_ns: int,
        account_summary: Any,
    ) -> dict[str, Any]:
        """Append one snapshot row to the parquet ledger; return the row.

        Returns the row that was written, primarily for caller logging
        and unit-test introspection.
        """

        # Import pandas lazily — keeps the module-import cost down when
        # only the ``reconcile`` pure-function is needed (e.g. in unit
        # tests that don't touch parquet I/O at all).
        import pandas as pd

        row = self._coerce_account_summary(timestamp_ns, account_summary)
        new_df = pd.DataFrame([row])

        if self.parquet_path.exists():
            existing = pd.read_parquet(self.parquet_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        tmp = self.parquet_path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(self.parquet_path)
        return row

    # -- read ---------------------------------------------------------------
    def read_all(self) -> "Any":  # pragma: no cover - thin wrapper
        """Return the full ledger as a pandas DataFrame (empty if no file)."""
        import pandas as pd

        if not self.parquet_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.parquet_path)

    # -- reconciliation helper ---------------------------------------------
    @staticmethod
    def reconcile(
        internal_positions: Mapping[str, Decimal] | Mapping[str, int | str],
        alpaca_positions: Mapping[str, Decimal] | Mapping[str, int | str],
        *,
        tolerance: Decimal = Decimal("0"),
    ) -> list[PositionBreak]:
        """Class-method alias for :func:`reconcile` — instance carries no state."""
        return reconcile(
            internal_positions, alpaca_positions, tolerance=tolerance
        )

    # -- breaks → jsonl -----------------------------------------------------
    def write_breaks(
        self,
        timestamp_ns: int,
        breaks: list[PositionBreak],
        *,
        breaks_path: Path | None = None,
    ) -> Path:
        """Append breaks to ``data/treasury/breaks.jsonl`` (one JSON per line).

        Each line: ``{"ts_ns": ..., "break": {...}}``. If ``breaks`` is
        empty we still write nothing — the file only grows when there's
        something to record.
        """
        target = (
            breaks_path
            if breaks_path is not None
            else self.parquet_path.parent / "breaks.jsonl"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not breaks:
            return target
        with target.open("a", encoding="utf-8") as fh:
            for br in breaks:
                fh.write(
                    json.dumps(
                        {"ts_ns": timestamp_ns, "break": br.to_jsonable()},
                        sort_keys=True,
                    )
                )
                fh.write("\n")
        return target
