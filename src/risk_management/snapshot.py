"""snapshot — NavSnapshot builder from IBKR AccountSummary (Epic 2).

The IBKR ``reqAccountSummary`` call is stubbed; this module takes the
already-fetched dict input (so unit tests are deterministic and replay
harnesses can build snapshots directly from fixtures).

CONTEXT.md ratification points implemented:
  - 09:30 ET snapshot is authoritative; chain reads only from snapshot.
  - ``snapshot_version`` is the sha256 hash of the canonical-JSON form
    of the snapshot fields — feeds into ``BreachEvent.snapshot_hash``.
  - I7f: ``limits_resolved`` is materialized into the snapshot (no
    inheritance logic needed at replay time).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from risk_management.limits import Limits


@dataclass(frozen=True)
class NavSnapshot:
    """09:30 ET account-state snapshot, deterministic input to the chain."""

    as_of_ns: int                          # int64-ns UTC timestamp
    net_liq: Decimal                       # NetLiquidation
    available_funds: Decimal               # AvailableFunds
    gross_position_value: Decimal          # GrossPositionValue
    positions: dict[str, Decimal] = field(default_factory=dict)  # symbol -> qty
    cushion: Decimal = Decimal("0")        # IBKR Cushion
    account_ready: bool = True             # AccountReady
    account_status: str = "Funded"         # Funded | Restricted | ...
    prev_close: dict[str, Decimal] = field(default_factory=dict)  # symbol -> prior close
    feature_ok: dict[str, bool] = field(default_factory=dict)     # symbol -> sanity ok
    broker_positions: dict[str, Decimal] = field(default_factory=dict)  # for reconcile
    snapshot_version: str = ""             # sha256(...) hash for breach correlation
    config_id: str = ""                    # from limits.toml header
    limits_hash: str = ""                  # sha256: of limits.toml at snapshot time
    limits_resolved: dict[str, Any] = field(default_factory=dict)  # I7f materialized

    def position(self, symbol: str) -> Decimal:
        return self.positions.get(symbol, Decimal("0"))

    def broker_position(self, symbol: str) -> Decimal:
        # Default to internal book when broker side isn't supplied — same value
        # means a vacuous "reconciled" check rather than a false reject.
        return self.broker_positions.get(symbol, self.positions.get(symbol, Decimal("0")))


def _decimal_safe(o: Any) -> Any:
    """JSON serializer that converts ``Decimal`` to a string for canonical hashing."""

    if isinstance(o, Decimal):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _materialize_limits(limits: Limits | None) -> dict[str, Any]:
    """Flatten ``Limits`` into a plain dict (no inheritance) for replay."""

    if limits is None:
        return {}
    return {
        "schema_version": limits.schema_version,
        "config_id": limits.config_id,
        "account": {
            k: str(getattr(limits.account, k))
            for k in (
                "max_single_name_pct",
                "max_gross_exposure_pct",
                "max_daily_turnover_pct",
                "kelly_fraction",
                "fatfinger_band_pct",
                "min_intent_notional_usd",
                "dd_daily_pct",
                "dd_weekly_pct",
                "dd_ptt_pct",
            )
        },
        "defaults_symbol": {
            "max_notional_pct": str(limits.defaults_symbol.max_notional_pct),
            "kelly_fraction": str(limits.defaults_symbol.kelly_fraction),
            "fatfinger_band_pct": str(limits.defaults_symbol.fatfinger_band_pct),
        },
        "symbol_overrides": {
            sym: {
                "max_notional_pct": str(s.max_notional_pct),
                "kelly_fraction": str(s.kelly_fraction),
                "fatfinger_band_pct": str(s.fatfinger_band_pct),
            }
            for sym, s in limits.symbol_overrides.items()
        },
    }


def _to_decimal(v: Any) -> Decimal:
    if isinstance(v, Decimal):
        return v
    if isinstance(v, bool):
        raise TypeError(f"bool is not a valid decimal value ({v!r})")
    if isinstance(v, (int, str)):
        return Decimal(v)
    if isinstance(v, float):
        raise TypeError(f"float not accepted for snapshot values (got {v!r}); pass str|int|Decimal")
    raise TypeError(f"unsupported type {type(v).__name__} for snapshot decimal")


def _dec_map(d: dict[str, Any]) -> dict[str, Decimal]:
    return {k: _to_decimal(v) for k, v in (d or {}).items()}


def build_snapshot(
    account_summary: dict[str, Any],
    *,
    as_of_ns: int,
    positions: dict[str, Any] | None = None,
    broker_positions: dict[str, Any] | None = None,
    prev_close: dict[str, Any] | None = None,
    feature_ok: dict[str, bool] | None = None,
    limits: Limits | None = None,
) -> NavSnapshot:
    """Construct a NavSnapshot from a (stubbed) IBKR ``AccountSummary`` dict.

    The IBKR call itself is out of scope — Epic 2 will wire this to
    ``reqAccountSummary``. For now ``account_summary`` is the parsed
    dict, with at least ``NetLiquidation``, ``AvailableFunds``,
    ``GrossPositionValue`` keys.
    """

    if "NetLiquidation" not in account_summary:
        raise KeyError("account_summary missing 'NetLiquidation'")

    net_liq = _to_decimal(account_summary["NetLiquidation"])
    available_funds = _to_decimal(account_summary.get("AvailableFunds", net_liq))
    gross_position_value = _to_decimal(account_summary.get("GrossPositionValue", "0"))
    cushion = _to_decimal(account_summary.get("Cushion", "0"))
    account_ready = bool(account_summary.get("AccountReady", True))
    account_status = str(account_summary.get("AccountStatus", "Funded"))

    pos = _dec_map(positions or {})
    broker_pos = _dec_map(broker_positions or {})
    pc = _dec_map(prev_close or {})
    fo = dict(feature_ok or {})

    limits_resolved = _materialize_limits(limits)

    # Canonical-JSON hash for snapshot_version.
    body = {
        "as_of_ns": as_of_ns,
        "net_liq": net_liq,
        "available_funds": available_funds,
        "gross_position_value": gross_position_value,
        "cushion": cushion,
        "account_ready": account_ready,
        "account_status": account_status,
        "positions": pos,
        "broker_positions": broker_pos,
        "prev_close": pc,
        "feature_ok": fo,
        "limits_resolved": limits_resolved,
    }
    canonical = json.dumps(body, sort_keys=True, default=_decimal_safe)
    snapshot_version = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return NavSnapshot(
        as_of_ns=as_of_ns,
        net_liq=net_liq,
        available_funds=available_funds,
        gross_position_value=gross_position_value,
        positions=pos,
        cushion=cushion,
        account_ready=account_ready,
        account_status=account_status,
        prev_close=pc,
        feature_ok=fo,
        broker_positions=broker_pos,
        snapshot_version=snapshot_version,
        config_id=limits.config_id if limits else "",
        limits_hash=limits.limits_hash if limits else "",
        limits_resolved=limits_resolved,
    )


__all__ = ["NavSnapshot", "build_snapshot"]
