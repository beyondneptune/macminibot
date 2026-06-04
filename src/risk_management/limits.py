"""limits — TOML config loader for risk/limits.toml (Epic 4 + I7d).

Schema (CONTEXT.md I7d):
    schema_version = 2
    config_id      = "2026-05-17.a"

    [account]
    max_single_name_pct = 0.15
    max_gross_exposure_pct = 1.00
    max_daily_turnover_pct = 0.50
    kelly_fraction = 0.25
    fatfinger_band_pct = 0.15
    min_intent_notional_usd = 50.00
    dd_daily_pct = -0.02
    dd_weekly_pct = -0.05
    dd_ptt_pct = -0.10

    [defaults.symbol]
    max_notional_pct = 0.15
    kelly_fraction = 0.25
    fatfinger_band_pct = 0.15

    [symbols.TSLA]
    max_notional_pct = 0.05         # tighten OK
    kelly_fraction = 0.15
    # fatfinger_band_pct = 0.25     # loosen => REJECT unless ack_loosen

Invariants (refuse-to-load on violation):
  1. ``schema_version`` is a known value.
  2. All override keys exist in ``defaults.symbol`` (typo guard).
  3. For each ceiling key in symbol overrides: ``override <= default``.
  4. ``symbols.X.max_notional_pct <= account.max_single_name_pct``.
  5. Hot-reload that loosens an existing value requires ``ack_loosen=True``.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (2,)

# Keys for which "higher value = looser limit"; per-symbol overrides must
# be <= the matching default. Same set is the tighten-only set for hot-reload.
SYMBOL_CEILING_KEYS: tuple[str, ...] = (
    "max_notional_pct",
    "kelly_fraction",
    "fatfinger_band_pct",
)

ACCOUNT_CEILING_KEYS: tuple[str, ...] = (
    "max_single_name_pct",
    "max_gross_exposure_pct",
    "max_daily_turnover_pct",
    "kelly_fraction",
    "fatfinger_band_pct",
)

# Drawdown thresholds are negative; "looser" = closer to 0 (less negative).
ACCOUNT_DD_KEYS: tuple[str, ...] = (
    "dd_daily_pct",
    "dd_weekly_pct",
    "dd_ptt_pct",
)


class LimitsLoadError(ValueError):
    """Raised when limits.toml fails validation or loosens without ack."""


@dataclass(frozen=True)
class SymbolLimits:
    """Materialized per-symbol limit envelope (defaults already resolved)."""

    max_notional_pct: Decimal
    kelly_fraction: Decimal
    fatfinger_band_pct: Decimal


@dataclass(frozen=True)
class AccountLimits:
    max_single_name_pct: Decimal
    max_gross_exposure_pct: Decimal
    max_daily_turnover_pct: Decimal
    kelly_fraction: Decimal
    fatfinger_band_pct: Decimal
    min_intent_notional_usd: Decimal
    dd_daily_pct: Decimal
    dd_weekly_pct: Decimal
    dd_ptt_pct: Decimal


@dataclass(frozen=True)
class Limits:
    """Full materialized limit set, with hash for snapshot pinning."""

    schema_version: int
    config_id: str
    account: AccountLimits
    defaults_symbol: SymbolLimits
    symbol_overrides: dict[str, SymbolLimits] = field(default_factory=dict)
    limits_hash: str = ""
    raw_text: str = ""

    def for_symbol(self, symbol: str) -> SymbolLimits:
        """Return the resolved per-symbol envelope (override or default)."""

        return self.symbol_overrides.get(symbol, self.defaults_symbol)


def _to_decimal(value: Any, *, field_name: str) -> Decimal:
    """Coerce numeric TOML values to ``Decimal`` via ``str`` to avoid binary float drift."""

    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LimitsLoadError(
            f"{field_name}: expected number, got {type(value).__name__}={value!r}"
        )
    return Decimal(str(value))


def _validate_schema_version(version: Any) -> int:
    if not isinstance(version, int) or isinstance(version, bool):
        raise LimitsLoadError(f"schema_version must be int (got {version!r})")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise LimitsLoadError(
            f"LIMITS_SCHEMA_UNKNOWN: schema_version={version} (supported {SUPPORTED_SCHEMA_VERSIONS})"
        )
    return version


def _parse_account(table: dict[str, Any]) -> AccountLimits:
    required = (
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
    missing = [k for k in required if k not in table]
    if missing:
        raise LimitsLoadError(f"[account] missing keys: {missing}")
    return AccountLimits(
        **{k: _to_decimal(table[k], field_name=f"account.{k}") for k in required}
    )


def _parse_defaults_symbol(table: dict[str, Any]) -> SymbolLimits:
    required = SYMBOL_CEILING_KEYS
    missing = [k for k in required if k not in table]
    if missing:
        raise LimitsLoadError(f"[defaults.symbol] missing keys: {missing}")
    return SymbolLimits(
        **{
            k: _to_decimal(table[k], field_name=f"defaults.symbol.{k}")
            for k in required
        }
    )


def _parse_symbol_overrides(
    symbols_table: dict[str, Any],
    defaults: SymbolLimits,
    account: AccountLimits,
) -> dict[str, SymbolLimits]:
    out: dict[str, SymbolLimits] = {}
    default_values = {k: getattr(defaults, k) for k in SYMBOL_CEILING_KEYS}
    for sym, raw in symbols_table.items():
        if not isinstance(raw, dict):
            raise LimitsLoadError(f"[symbols.{sym}] must be a table (got {type(raw).__name__})")
        unknown = [k for k in raw if k not in SYMBOL_CEILING_KEYS]
        if unknown:
            raise LimitsLoadError(
                f"[symbols.{sym}] unknown keys (typo guard): {unknown} "
                f"— allowed: {list(SYMBOL_CEILING_KEYS)}"
            )
        merged = dict(default_values)
        for k, v in raw.items():
            d = _to_decimal(v, field_name=f"symbols.{sym}.{k}")
            if d > default_values[k]:
                raise LimitsLoadError(
                    f"LIMITS_OVERRIDE_LOOSENS: symbols.{sym}.{k}={d} "
                    f"> defaults.symbol.{k}={default_values[k]} (overrides may only tighten)"
                )
            merged[k] = d
        # Hard outer bound: per-symbol max_notional_pct cannot exceed account ceiling.
        if merged["max_notional_pct"] > account.max_single_name_pct:
            raise LimitsLoadError(
                f"symbols.{sym}.max_notional_pct={merged['max_notional_pct']} "
                f"> account.max_single_name_pct={account.max_single_name_pct}"
            )
        out[sym] = SymbolLimits(**merged)
    return out


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_loosen(prev: Decimal, new: Decimal, *, is_drawdown: bool) -> bool:
    """Return True if ``new`` loosens ``prev``.

    For ceilings (positive limits), looser = higher value.
    For drawdowns (negative thresholds), looser = closer to zero
    (i.e. less negative), so ``new > prev``.
    """

    if is_drawdown:
        return new > prev
    return new > prev


@dataclass(frozen=True)
class LimitChange:
    """One field-level difference between two ``Limits`` configs."""

    path: str               # e.g. "account.max_single_name_pct"
    prev: Decimal
    new: Decimal
    direction: str          # "tighten" | "loosen" | "same"

    def format(self) -> str:
        return f"{self.path}: {self.prev} -> {self.new} ({self.direction.upper()})"


@dataclass(frozen=True)
class LimitsDiff:
    """Structured diff between a prior and a new ``Limits`` config.

    Designed for pre-deploy review of ``limits.toml`` edits — operators
    eyeball ``format_summary()`` (or pipe it to stderr behind ``--ack-loosen``)
    before accepting a hot-reload. The hot-reload path itself only needs
    ``loosened``; ``tightened`` is reported for situational awareness so a
    reviewer notices "I tightened TSLA but also accidentally loosened MSFT
    by inheriting defaults."
    """

    tightened: tuple[LimitChange, ...]
    loosened: tuple[LimitChange, ...]
    config_id_prev: str
    config_id_new: str

    @property
    def has_changes(self) -> bool:
        return bool(self.tightened) or bool(self.loosened)

    def format_summary(self) -> str:
        """Human-readable multi-line summary (suitable for stderr / digest)."""

        header = (
            f"limits diff {self.config_id_prev or '<none>'} -> "
            f"{self.config_id_new or '<none>'}"
        )
        lines: list[str] = [header]
        if self.tightened:
            lines.append(f"  TIGHTENED ({len(self.tightened)}):")
            for c in self.tightened:
                lines.append(f"    + {c.format()}")
        if self.loosened:
            lines.append(f"  LOOSENED ({len(self.loosened)}):")
            for c in self.loosened:
                lines.append(f"    ! {c.format()}")
        if not self.has_changes:
            lines.append("  (no ceiling changes)")
        return "\n".join(lines)


def _direction(prev: Decimal, new: Decimal, *, is_drawdown: bool, lower_is_looser: bool = False) -> str:
    """Classify a numeric change as tighten / loosen / same.

    ``lower_is_looser`` flips the polarity (used for ``min_intent_notional_usd``
    where a lower threshold admits more orders).
    """

    if prev == new:
        return "same"
    if lower_is_looser:
        return "loosen" if new < prev else "tighten"
    if is_drawdown:
        # Drawdown thresholds are negative; closer to 0 == less negative == looser.
        return "loosen" if new > prev else "tighten"
    # Standard ceiling: higher = looser.
    return "loosen" if new > prev else "tighten"


def diff_limits(prev: "Limits", new: "Limits") -> LimitsDiff:
    """Compute a full tighten/loosen diff between two ``Limits`` configs.

    Polarity is tighten-aware: drawdown thresholds (negative numbers) and
    ``min_intent_notional_usd`` (lower = looser) are classified correctly.
    Per-symbol comparison resolves through ``defaults_symbol`` on both
    sides so symbols added/removed from the override table are still
    diffed against the inherited default (mirroring the chain's runtime
    inheritance, which is what an operator actually cares about).
    """

    tightened: list[LimitChange] = []
    loosened: list[LimitChange] = []

    def _record(path: str, pv: Decimal, nv: Decimal, *, is_drawdown: bool = False,
                lower_is_looser: bool = False) -> None:
        direction = _direction(pv, nv, is_drawdown=is_drawdown, lower_is_looser=lower_is_looser)
        if direction == "same":
            return
        change = LimitChange(path=path, prev=pv, new=nv, direction=direction)
        (tightened if direction == "tighten" else loosened).append(change)

    for k in ACCOUNT_CEILING_KEYS:
        _record(f"account.{k}", getattr(prev.account, k), getattr(new.account, k))
    for k in ACCOUNT_DD_KEYS:
        _record(f"account.{k}", getattr(prev.account, k), getattr(new.account, k),
                is_drawdown=True)
    _record(
        "account.min_intent_notional_usd",
        prev.account.min_intent_notional_usd,
        new.account.min_intent_notional_usd,
        lower_is_looser=True,
    )

    for k in SYMBOL_CEILING_KEYS:
        _record(
            f"defaults.symbol.{k}",
            getattr(prev.defaults_symbol, k),
            getattr(new.defaults_symbol, k),
        )

    for sym in sorted(set(prev.symbol_overrides) | set(new.symbol_overrides)):
        prev_lim = prev.symbol_overrides.get(sym, prev.defaults_symbol)
        new_lim = new.symbol_overrides.get(sym, new.defaults_symbol)
        for k in SYMBOL_CEILING_KEYS:
            _record(f"symbols.{sym}.{k}", getattr(prev_lim, k), getattr(new_lim, k))

    return LimitsDiff(
        tightened=tuple(tightened),
        loosened=tuple(loosened),
        config_id_prev=prev.config_id,
        config_id_new=new.config_id,
    )


def _check_loosen_against_prior(prev: "Limits", new: "Limits") -> list[str]:
    """Return a list of human-readable diffs where ``new`` loosens ``prev``.

    Backwards-compat shim — delegates to ``diff_limits`` and renders the
    loosen subset in the legacy ``"path: prev -> new (LOOSEN)"`` shape so
    the hot-reload error message stays byte-identical.
    """

    diff = diff_limits(prev, new)
    return [f"{c.path}: {c.prev} -> {c.new} (LOOSEN)" for c in diff.loosened]


def load_limits(
    path: str | Path,
    *,
    previous: Limits | None = None,
    ack_loosen: bool = False,
) -> Limits:
    """Load and validate ``limits.toml``.

    Parameters
    ----------
    path
        Path to the TOML file.
    previous
        If supplied (hot-reload), the new file is compared against it and
        any loosening requires ``ack_loosen=True``.
    ack_loosen
        Explicit operator acknowledgement that loosening is intentional
        (corresponds to ``--ack-loosen`` CLI flag).
    """

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise LimitsLoadError(f"invalid TOML in {p}: {e}") from e

    schema_version = _validate_schema_version(data.get("schema_version"))
    config_id = data.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise LimitsLoadError("config_id must be a non-empty string")

    if "account" not in data or not isinstance(data["account"], dict):
        raise LimitsLoadError("missing or invalid [account] table")
    account = _parse_account(data["account"])

    defaults_table = data.get("defaults", {}).get("symbol")
    if not isinstance(defaults_table, dict):
        raise LimitsLoadError("missing or invalid [defaults.symbol] table")
    defaults_symbol = _parse_defaults_symbol(defaults_table)

    # Defaults.symbol.max_notional_pct must respect the account ceiling too
    # (otherwise an unlisted symbol could exceed the account-wide cap via
    # default inheritance).
    if defaults_symbol.max_notional_pct > account.max_single_name_pct:
        raise LimitsLoadError(
            f"defaults.symbol.max_notional_pct={defaults_symbol.max_notional_pct} "
            f"> account.max_single_name_pct={account.max_single_name_pct}"
        )

    symbols_table = data.get("symbols", {}) or {}
    if not isinstance(symbols_table, dict):
        raise LimitsLoadError("[symbols.*] must be a table-of-tables")
    overrides = _parse_symbol_overrides(symbols_table, defaults_symbol, account)

    limits = Limits(
        schema_version=schema_version,
        config_id=config_id,
        account=account,
        defaults_symbol=defaults_symbol,
        symbol_overrides=overrides,
        limits_hash=_hash_text(text),
        raw_text=text,
    )

    if previous is not None:
        loosen_diffs = _check_loosen_against_prior(previous, limits)
        if loosen_diffs and not ack_loosen:
            joined = "\n  ".join(loosen_diffs)
            raise LimitsLoadError(
                "LIMITS_RELOAD_REJECTED: loosening detected; rerun with "
                "ack_loosen=True (--ack-loosen) to accept:\n  " + joined
            )

    return limits


__all__ = [
    "ACCOUNT_CEILING_KEYS",
    "ACCOUNT_DD_KEYS",
    "AccountLimits",
    "LimitChange",
    "Limits",
    "LimitsDiff",
    "LimitsLoadError",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SYMBOL_CEILING_KEYS",
    "SymbolLimits",
    "diff_limits",
    "load_limits",
]
