"""Tests for the TOML limits loader."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from risk_management.limits import (
    LimitChange,
    LimitsDiff,
    LimitsLoadError,
    diff_limits,
    load_limits,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


VALID_TOML = """\
schema_version = 2
config_id = "2026-05-17.a"

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
max_notional_pct = 0.05
kelly_fraction = 0.15

[symbols.AAPL]
max_notional_pct = 0.10
"""


def test_loads_valid_file(tmp_path: Path) -> None:
    p = _write(tmp_path / "limits.toml", VALID_TOML)
    lims = load_limits(p)
    assert lims.schema_version == 2
    assert lims.config_id == "2026-05-17.a"
    assert lims.account.max_single_name_pct == Decimal("0.15")
    assert lims.for_symbol("TSLA").max_notional_pct == Decimal("0.05")
    # Default inheritance: AAPL.kelly_fraction defaults to 0.25
    assert lims.for_symbol("AAPL").kelly_fraction == Decimal("0.25")
    # Unknown symbol falls through to defaults.
    assert lims.for_symbol("UNLISTED").max_notional_pct == Decimal("0.15")
    assert lims.limits_hash.startswith("sha256:")


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    body = VALID_TOML.replace("schema_version = 2", "schema_version = 99")
    p = _write(tmp_path / "limits.toml", body)
    with pytest.raises(LimitsLoadError, match="LIMITS_SCHEMA_UNKNOWN"):
        load_limits(p)


def test_rejects_unknown_symbol_key(tmp_path: Path) -> None:
    body = VALID_TOML + '\n[symbols.NVDA]\nkelly_fration = 0.10\n'  # typo
    p = _write(tmp_path / "limits.toml", body)
    with pytest.raises(LimitsLoadError, match="unknown keys"):
        load_limits(p)


def test_rejects_per_symbol_loosen(tmp_path: Path) -> None:
    """fatfinger_band_pct = 0.25 with default 0.15 must REFUSE — loosens."""

    body = VALID_TOML + '\n[symbols.NVDA]\nfatfinger_band_pct = 0.25\n'
    p = _write(tmp_path / "limits.toml", body)
    with pytest.raises(LimitsLoadError, match="LIMITS_OVERRIDE_LOOSENS"):
        load_limits(p)


def test_per_symbol_cannot_exceed_account_ceiling(tmp_path: Path) -> None:
    """defaults.symbol.max_notional_pct=0.10, account=0.05 -> reject."""

    body = """\
schema_version = 2
config_id = "tight.001"

[account]
max_single_name_pct = 0.05
max_gross_exposure_pct = 1.00
max_daily_turnover_pct = 0.50
kelly_fraction = 0.25
fatfinger_band_pct = 0.15
min_intent_notional_usd = 50.00
dd_daily_pct = -0.02
dd_weekly_pct = -0.05
dd_ptt_pct = -0.10

[defaults.symbol]
max_notional_pct = 0.10
kelly_fraction = 0.25
fatfinger_band_pct = 0.15
"""
    p = _write(tmp_path / "limits.toml", body)
    with pytest.raises(LimitsLoadError, match=r"max_notional_pct"):
        load_limits(p)


def test_hot_reload_rejects_loosening_without_ack(tmp_path: Path) -> None:
    """Per-deliverable test: loosening without ack_loosen=True must fail."""

    p = _write(tmp_path / "limits.toml", VALID_TOML)
    prev = load_limits(p)

    # Loosen the account ceiling (raise dd_daily_pct closer to zero AND
    # raise max_gross_exposure_pct). Both should trigger the loosen check.
    loosened = VALID_TOML.replace(
        "max_gross_exposure_pct = 1.00",
        "max_gross_exposure_pct = 1.50",  # leverage — looser
    )
    _write(p, loosened)

    with pytest.raises(LimitsLoadError, match="LIMITS_RELOAD_REJECTED"):
        load_limits(p, previous=prev)


def test_hot_reload_accepts_loosening_with_ack(tmp_path: Path) -> None:
    p = _write(tmp_path / "limits.toml", VALID_TOML)
    prev = load_limits(p)

    loosened = VALID_TOML.replace(
        "max_gross_exposure_pct = 1.00",
        "max_gross_exposure_pct = 1.50",
    )
    _write(p, loosened)

    new = load_limits(p, previous=prev, ack_loosen=True)
    assert new.account.max_gross_exposure_pct == Decimal("1.50")


def test_hot_reload_accepts_tightening(tmp_path: Path) -> None:
    p = _write(tmp_path / "limits.toml", VALID_TOML)
    prev = load_limits(p)

    tightened = VALID_TOML.replace(
        "max_gross_exposure_pct = 1.00",
        "max_gross_exposure_pct = 0.75",
    )
    _write(p, tightened)

    new = load_limits(p, previous=prev)  # no ack needed for tightening
    assert new.account.max_gross_exposure_pct == Decimal("0.75")


def test_loosening_drawdown_threshold_requires_ack(tmp_path: Path) -> None:
    """dd_daily_pct -0.02 -> -0.01 is LESS negative => looser."""

    p = _write(tmp_path / "limits.toml", VALID_TOML)
    prev = load_limits(p)

    loosened = VALID_TOML.replace("dd_daily_pct = -0.02", "dd_daily_pct = -0.01")
    _write(p, loosened)
    with pytest.raises(LimitsLoadError, match="LIMITS_RELOAD_REJECTED"):
        load_limits(p, previous=prev)


# ---------------------------------------------------------------------------
# diff_limits — pre-deploy review helper
# ---------------------------------------------------------------------------


def test_diff_limits_no_changes(tmp_path: Path) -> None:
    p = _write(tmp_path / "limits.toml", VALID_TOML)
    cfg = load_limits(p)
    d = diff_limits(cfg, cfg)
    assert isinstance(d, LimitsDiff)
    assert d.tightened == ()
    assert d.loosened == ()
    assert d.has_changes is False
    assert "no ceiling changes" in d.format_summary()


def test_diff_limits_detects_account_tighten(tmp_path: Path) -> None:
    p = _write(tmp_path / "limits.toml", VALID_TOML)
    prev = load_limits(p)
    tighter = VALID_TOML.replace(
        "max_gross_exposure_pct = 1.00",
        "max_gross_exposure_pct = 0.75",
    )
    _write(p, tighter)
    new = load_limits(p, previous=prev)

    d = diff_limits(prev, new)
    paths = {c.path: c for c in d.tightened}
    assert "account.max_gross_exposure_pct" in paths
    change = paths["account.max_gross_exposure_pct"]
    assert change.direction == "tighten"
    assert change.prev == Decimal("1.00") and change.new == Decimal("0.75")
    assert d.loosened == ()


def test_diff_limits_detects_drawdown_polarity(tmp_path: Path) -> None:
    """dd_daily -0.02 -> -0.01 (less negative) is LOOSER; -0.03 is TIGHTER."""

    p = _write(tmp_path / "limits.toml", VALID_TOML)
    prev = load_limits(p)
    looser = VALID_TOML.replace("dd_daily_pct = -0.02", "dd_daily_pct = -0.01")
    _write(p, looser)
    new_loose = load_limits(p, previous=prev, ack_loosen=True)
    d = diff_limits(prev, new_loose)
    assert any(c.path == "account.dd_daily_pct" and c.direction == "loosen"
               for c in d.loosened)

    tighter = VALID_TOML.replace("dd_daily_pct = -0.02", "dd_daily_pct = -0.03")
    _write(p, tighter)
    new_tight = load_limits(p)
    d2 = diff_limits(prev, new_tight)
    assert any(c.path == "account.dd_daily_pct" and c.direction == "tighten"
               for c in d2.tightened)


def test_diff_limits_min_intent_notional_polarity(tmp_path: Path) -> None:
    """Lower min_intent_notional_usd admits MORE orders => looser."""

    p = _write(tmp_path / "limits.toml", VALID_TOML)
    prev = load_limits(p)
    looser = VALID_TOML.replace(
        "min_intent_notional_usd = 50.00",
        "min_intent_notional_usd = 25.00",
    )
    _write(p, looser)
    new = load_limits(p, previous=prev, ack_loosen=True)
    d = diff_limits(prev, new)
    assert any(c.path == "account.min_intent_notional_usd" and c.direction == "loosen"
               for c in d.loosened)


def test_diff_limits_per_symbol_override(tmp_path: Path) -> None:
    p = _write(tmp_path / "limits.toml", VALID_TOML)
    prev = load_limits(p)
    # Tighten TSLA from 0.05 -> 0.03.
    tighter = VALID_TOML.replace("max_notional_pct = 0.05", "max_notional_pct = 0.03")
    _write(p, tighter)
    new = load_limits(p, previous=prev)
    d = diff_limits(prev, new)
    assert any(c.path == "symbols.TSLA.max_notional_pct" and c.direction == "tighten"
               for c in d.tightened)


def test_diff_limits_format_summary_has_both_sections() -> None:
    """Manual fabrication: format_summary lists TIGHTENED and LOOSENED both."""

    from risk_management.limits import (
        AccountLimits,
        Limits,
        SymbolLimits,
    )

    def _mk(*, gross: str, dd: str, min_intent: str = "50.00") -> Limits:
        return Limits(
            schema_version=2,
            config_id=f"cfg-{gross}-{dd}",
            account=AccountLimits(
                max_single_name_pct=Decimal("0.15"),
                max_gross_exposure_pct=Decimal(gross),
                max_daily_turnover_pct=Decimal("0.50"),
                kelly_fraction=Decimal("0.25"),
                fatfinger_band_pct=Decimal("0.15"),
                min_intent_notional_usd=Decimal(min_intent),
                dd_daily_pct=Decimal(dd),
                dd_weekly_pct=Decimal("-0.05"),
                dd_ptt_pct=Decimal("-0.10"),
            ),
            defaults_symbol=SymbolLimits(
                max_notional_pct=Decimal("0.15"),
                kelly_fraction=Decimal("0.25"),
                fatfinger_band_pct=Decimal("0.15"),
            ),
        )

    prev = _mk(gross="1.00", dd="-0.02", min_intent="50.00")
    new = _mk(gross="0.75", dd="-0.01", min_intent="25.00")  # gross tighter, dd+intent looser
    d = diff_limits(prev, new)
    out = d.format_summary()
    assert "TIGHTENED" in out
    assert "LOOSENED" in out
    assert "max_gross_exposure_pct" in out
    assert "dd_daily_pct" in out
    assert "min_intent_notional_usd" in out
    # Config ids appear in header.
    assert prev.config_id in out and new.config_id in out


def test_limit_change_format_renders_path_and_direction() -> None:
    c = LimitChange(
        path="account.kelly_fraction",
        prev=Decimal("0.25"),
        new=Decimal("0.20"),
        direction="tighten",
    )
    s = c.format()
    assert "account.kelly_fraction" in s
    assert "0.25" in s and "0.20" in s
    assert "TIGHTEN" in s
