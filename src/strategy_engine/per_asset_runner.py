"""Per-asset trailing-stop strategy runner — production daily orchestrator.

This is the deployment-side counterpart to the trailing-stop bake-off
(``scripts/51_trailing_stop_bakeoff.py``). It runs ONE tick per trading
day (the 09:35 ET wake) and:

  1. For each of the 7 deployable assets (AAPL, MSFT, GOOGL, META, TSLA,
     NVDA, GLD — AMZN excluded on bake-off CI gate), reads the latest
     daily bars from ArcticDB and computes that asset's specialist
     signal via ``ml_research.asset_specialists.compute_signal_for``.
  2. Loads the persistent per-asset position state (FLAT, IN_LONG,
     IN_SHORT) from ``data/per_asset_state.json``.
  3. Applies the trailing-stop state machine:
       - FLAT and signal != 0  →  ENTER in signal direction.
       - IN_LONG and signal < 0 →  EXIT (flat & flip; the new long entry
         is taken on the next wake).
       - IN_SHORT and signal > 0 →  EXIT.
       - IN_LONG and price retraced ``trail_sigma * sigma_21d`` from the
         running peak → EXIT (trailing stop).
       - IN_SHORT and price rallied ``trail_sigma * sigma_21d`` from the
         running trough → EXIT.
       - else: HOLD.
  4. Persists the updated state back to JSON (atomic write).
  5. Returns a list of :class:`shared_core.contracts.OrderIntent` for any
     ENTRY or EXIT triggered this run.

Bar-frequency convention
------------------------
We run on daily bars. The running peak/trough is updated using the
**prior bar's close** (matching the spec — no intraday data). Sigma
is the 21-day rolling std of log-returns measured as-of T-1. The
band width is ``entry_px * sigma_log * trail_sigma`` in absolute price
units, frozen at entry — same convention as the bake-off simulator.

Sizing
------
Equal-weight across the 7 deployable assets: target notional per
position = ``net_liq / 7``, converted to integer shares at the latest
close.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional

import numpy as np
import pandas as pd

from ml_research.asset_specialists import (
    DEPLOYABLE_ASSETS,
    compute_signal_for,
    winner_for,
)
from ml_research.asset_specialists._signals import add_per_asset_features
from shared_core.contracts import OrderIntent, Side, Symbol, Venue


__all__ = [
    "AssetState",
    "PerAssetRunner",
    "PerAssetRunnerConfig",
    "PositionState",
    "RunResult",
    "load_state_file",
    "save_state_file",
]


_LOG = logging.getLogger(__name__)


# Asset → (Arctic library, Arctic symbol, Venue, currency) lookup.
# Mirrors ``TRADABLE`` in scripts/50_per_asset_bakeoff.py.
_ASSET_META: dict[str, tuple[str, str, Venue, str]] = {
    # Legacy v2 universe (now managed only by the intraday-exit-monitor
    # as positions trail-stop out; runner no longer opens new entries here).
    "AAPL":  ("equity.daily_bars", "NASDAQ.AAPL",  Venue.NASDAQ, "USD"),
    "MSFT":  ("equity.daily_bars", "NASDAQ.MSFT",  Venue.NASDAQ, "USD"),
    "GOOGL": ("equity.daily_bars", "NASDAQ.GOOGL", Venue.NASDAQ, "USD"),
    "AMZN":  ("equity.daily_bars", "NASDAQ.AMZN",  Venue.NASDAQ, "USD"),
    "META":  ("equity.daily_bars", "NASDAQ.META",  Venue.NASDAQ, "USD"),
    "TSLA":  ("equity.daily_bars", "NASDAQ.TSLA",  Venue.NASDAQ, "USD"),
    "NVDA":  ("equity.daily_bars", "NASDAQ.NVDA",  Venue.NASDAQ, "USD"),
    "GLD":   ("etf.daily_bars",    "ARCA.GLD",     Venue.ARCA,   "USD"),
    # 1%/day target deployment 2026-06-04: 8 leveraged ETFs. Bars come from
    # Alpaca (not ArcticDB) — see default_bar_loader fallback below.
    "SOXL":  ("alpaca.daily_bars", "SOXL",         Venue.ARCA,   "USD"),
    "NVDL":  ("alpaca.daily_bars", "NVDL",         Venue.NASDAQ, "USD"),
    "TQQQ":  ("alpaca.daily_bars", "TQQQ",         Venue.NASDAQ, "USD"),
    "UPRO":  ("alpaca.daily_bars", "UPRO",         Venue.ARCA,   "USD"),
    "TNA":   ("alpaca.daily_bars", "TNA",          Venue.ARCA,   "USD"),
    "FAS":   ("alpaca.daily_bars", "FAS",          Venue.ARCA,   "USD"),
    "LABU":  ("alpaca.daily_bars", "LABU",         Venue.ARCA,   "USD"),
    "TSLL":  ("alpaca.daily_bars", "TSLL",         Venue.NASDAQ, "USD"),
    # Extended candidate pool for the weekly universe rotation
    # (scripts/110_universe_rotation.py).
    "RETL":  ("alpaca.daily_bars", "RETL",         Venue.ARCA,   "USD"),
    "DRN":   ("alpaca.daily_bars", "DRN",          Venue.ARCA,   "USD"),
    "URTY":  ("alpaca.daily_bars", "URTY",         Venue.ARCA,   "USD"),
    "TECL":  ("alpaca.daily_bars", "TECL",         Venue.ARCA,   "USD"),
    "CURE":  ("alpaca.daily_bars", "CURE",         Venue.ARCA,   "USD"),
    "DPST":  ("alpaca.daily_bars", "DPST",         Venue.ARCA,   "USD"),
    "DUSL":  ("alpaca.daily_bars", "DUSL",         Venue.ARCA,   "USD"),
    "ERX":   ("alpaca.daily_bars", "ERX",          Venue.ARCA,   "USD"),
    "PILL":  ("alpaca.daily_bars", "PILL",         Venue.ARCA,   "USD"),
    "WANT":  ("alpaca.daily_bars", "WANT",         Venue.ARCA,   "USD"),
    "MSFU":  ("alpaca.daily_bars", "MSFU",         Venue.NASDAQ, "USD"),
    "AAPU":  ("alpaca.daily_bars", "AAPU",         Venue.NASDAQ, "USD"),
    "AMZU":  ("alpaca.daily_bars", "AMZU",         Venue.NASDAQ, "USD"),
    "GOOL":  ("alpaca.daily_bars", "GOOL",         Venue.NASDAQ, "USD"),
    "AMDL":  ("alpaca.daily_bars", "AMDL",         Venue.NASDAQ, "USD"),
}

_ALPACA_BARS_PREFIX = "alpaca.daily_bars"


PositionState = Literal["FLAT", "IN_LONG", "IN_SHORT"]


# ---------------------------------------------------------------------------
# Persistent state model
# ---------------------------------------------------------------------------


@dataclass
class AssetState:
    """Per-asset position state. Serialised to JSON on disk.

    The runner is restart-safe: on cold start we re-read this file and
    resume the trailing band exactly where the prior wake left it.
    """

    asset: str
    state: PositionState = "FLAT"
    entry_price: float | None = None
    entry_sigma_log: float | None = None  # 21d log-return std AT entry
    trail_sigma: float | None = None      # asset's deployed trail width
    profit_target_sigmas: float | None = None  # σ-multiple PT (None disables)
    peak: float | None = None              # running max close (IN_LONG)
    trough: float | None = None            # running min close (IN_SHORT)
    qty: int = 0
    entry_ts_ns: int | None = None
    last_close: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AssetState":
        return cls(
            asset=str(raw["asset"]),
            state=raw.get("state", "FLAT"),  # type: ignore[arg-type]
            entry_price=raw.get("entry_price"),
            entry_sigma_log=raw.get("entry_sigma_log"),
            trail_sigma=raw.get("trail_sigma"),
            profit_target_sigmas=raw.get("profit_target_sigmas"),
            peak=raw.get("peak"),
            trough=raw.get("trough"),
            qty=int(raw.get("qty", 0)),
            entry_ts_ns=raw.get("entry_ts_ns"),
            last_close=raw.get("last_close"),
        )


# ---------------------------------------------------------------------------
# Run result (returned to the daily wake)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    intents: tuple[OrderIntent, ...]
    states: dict[str, AssetState]
    decision_id: str
    batch_id: str
    skipped: tuple[tuple[str, str], ...]  # (asset, reason) pairs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerAssetRunnerConfig:
    """Runner configuration with sensible production defaults."""

    state_path: Path
    assets: tuple[str, ...] = DEPLOYABLE_ASSETS
    portfolio_size: int = 7  # equal-weight denominator
    min_bars_required: int = 30  # need this many bars to compute σ_21d
    # Strategy identifier (goes into decision_id + intent metadata)
    strategy_sha: str = "trailing_stop_v1"


# ---------------------------------------------------------------------------
# State I/O — atomic via tempfile + os.replace
# ---------------------------------------------------------------------------


def load_state_file(path: Path) -> dict[str, AssetState]:
    """Read the per-asset state JSON. Missing file → empty dict."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        _LOG.warning("per_asset_state.json unreadable (%s); starting fresh.", e)
        return {}
    out: dict[str, AssetState] = {}
    for asset, payload in raw.items():
        try:
            out[asset] = AssetState.from_dict(payload)
        except Exception as e:  # pragma: no cover — corruption guard
            _LOG.warning("state for %s corrupt (%s); resetting to FLAT.", asset, e)
            out[asset] = AssetState(asset=asset)
    return out


def save_state_file(path: Path, states: Mapping[str, AssetState]) -> None:
    """Atomic write of the per-asset state JSON.

    Writes to a tempfile in the same directory then ``os.replace`` to
    ensure no half-written state survives a crash mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {a: s.to_dict() for a, s in states.items()}
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix=".per_asset_state-", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path_str, path)
    except Exception:
        # Cleanup on failure
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Bar loader protocol — injectable for tests
# ---------------------------------------------------------------------------


BarLoader = Callable[[str], pd.DataFrame]


def default_bar_loader(asset: str) -> pd.DataFrame:
    """Load latest daily bars from ArcticDB or Alpaca IEX.

    Routing: assets with lib_name ``alpaca.daily_bars`` (the leveraged-ETF
    universe added 2026-06-04) pull directly from Alpaca's IEX feed
    because ArcticDB doesn't carry their substrate yet. Everything else
    reads from ArcticDB as before.
    """
    if asset not in _ASSET_META:
        raise ValueError(f"unknown asset {asset!r}")
    lib_name, sym, _venue, _ccy = _ASSET_META[asset]
    if lib_name == _ALPACA_BARS_PREFIX:
        return _alpaca_daily_bars(sym)
    # ArcticDB path — lazy import.
    from scripts._storage import open_store  # type: ignore

    store = open_store()
    df = store[lib_name].read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _alpaca_daily_bars(ticker: str, years: float = 3.0) -> pd.DataFrame:
    """Pull recent daily bars from Alpaca IEX (free tier).

    Cached per-process via :func:`functools.lru_cache` to avoid repeated
    REST hits within one runner cycle.
    """
    return _alpaca_daily_bars_cached(ticker, int(years * 365.25))


import functools as _functools  # noqa: E402


@_functools.lru_cache(maxsize=32)
def _alpaca_daily_bars_cached(ticker: str, days_back: int) -> pd.DataFrame:
    import os as _os
    from datetime import datetime as _dt, timedelta as _td
    from dotenv import load_dotenv as _load_dotenv

    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv(project_root / ".env")
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        _os.environ["ALPACA_API_KEY"],
        _os.environ.get("ALPACA_SECRET") or _os.environ["ALPACA_API_SECRET"],
    )
    end = _dt.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - _td(days=days_back)
    req = StockBarsRequest(
        symbol_or_symbols=[ticker], timeframe=TimeFrame.Day,
        start=start, end=end, feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if hasattr(bars.index, "levels"):
        bars = bars.xs(ticker, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx
    return out.sort_index()


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def _make_symbol(asset: str) -> Symbol:
    _, _, venue, ccy = _ASSET_META[asset]
    return Symbol(ticker=asset, venue=venue, currency=ccy)


def _money(value: float | int | str | Decimal) -> Decimal:
    """Decimal from str|int only (KD1). Floats stringify safely for monetary
    inputs because we always quantize before contract construction."""
    return Decimal(str(value))


def _quantize_price(value: float) -> Decimal:
    """Quantize a price to 2dp Decimal via the str-then-Decimal path."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"price must be finite and positive, got {value!r}")
    return Decimal(f"{value:.4f}").quantize(Decimal("0.01"))


def _decision_id(
    as_of_ns: int, strategy_sha: str, signals: Mapping[str, float]
) -> str:
    """Stable hash of (timestamp, strategy id, signal vector) — reproducibility key."""
    payload = json.dumps(
        {
            "as_of_ns": as_of_ns,
            "strategy": strategy_sha,
            "signals": {k: float(v) for k, v in sorted(signals.items())},
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


@dataclass
class PerAssetRunner:
    """Daily-tick per-asset trailing-stop orchestrator.

    Inject ``bar_loader`` for tests; the production wake uses
    :func:`default_bar_loader` (ArcticDB).
    """

    config: PerAssetRunnerConfig
    bar_loader: BarLoader = field(default_factory=lambda: default_bar_loader)

    # ----- public entry point ---------------------------------------------

    def run(self, *, net_liq: Decimal | None = None) -> RunResult:
        """One daily tick. Returns the order intents triggered this run."""
        now_ns = time.time_ns()
        states = load_state_file(self.config.state_path)
        # Make sure every deployable asset has a state entry.
        for a in self.config.assets:
            states.setdefault(a, AssetState(asset=a))

        # Compute the per-asset signal up-front so the decision_id captures
        # the full signal vector (reproducibility — a replay of this exact
        # vector + state must reproduce the intent list).
        signals: dict[str, float] = {}
        sigmas: dict[str, float] = {}
        last_closes: dict[str, float] = {}
        skipped: list[tuple[str, str]] = []
        per_asset_winners: dict[str, dict[str, Any]] = {}

        for asset in self.config.assets:
            winner = winner_for(asset)
            if winner is None:
                skipped.append((asset, "no_winner"))
                continue
            per_asset_winners[asset] = winner
            try:
                bars = self.bar_loader(asset)
            except Exception as e:  # pragma: no cover — exercised only via fake-loader tests
                _LOG.warning("bar_loader failed for %s: %s", asset, e)
                skipped.append((asset, f"bar_loader_error:{type(e).__name__}"))
                continue
            if bars is None or len(bars) < self.config.min_bars_required:
                skipped.append((asset, "insufficient_bars"))
                continue
            feat = add_per_asset_features(bars)
            sigma = self._latest_sigma(feat)
            if sigma is None:
                skipped.append((asset, "sigma_unavailable"))
                continue
            try:
                sig = compute_signal_for(asset, bars)
            except Exception as e:
                _LOG.warning("compute_signal_for(%s) raised %s", asset, e)
                skipped.append((asset, f"signal_error:{type(e).__name__}"))
                continue
            signals[asset] = float(sig)
            sigmas[asset] = sigma
            last_closes[asset] = float(bars["close"].iloc[-1])

        decision_id = _decision_id(now_ns, self.config.strategy_sha, signals)
        batch_id = f"par-{decision_id[:16]}-{uuid.uuid4().hex[:6]}"

        # Now resolve state transitions per asset.
        intents: list[OrderIntent] = []
        for asset in self.config.assets:
            if asset not in signals:
                continue
            winner = per_asset_winners[asset]
            sig = int(round(signals[asset]))
            sigma_log = sigmas[asset]
            last_close = last_closes[asset]
            state = states[asset]
            state.last_close = last_close
            trail_sigma_w = float(winner["trail_sigma"])
            pt_sigmas_w = winner.get("profit_target_sigmas")
            pt_sigmas_w = float(pt_sigmas_w) if pt_sigmas_w is not None else None

            transition_intent = self._step_state_machine(
                asset=asset,
                state=state,
                signal_dir=sig,
                last_close=last_close,
                sigma_log=sigma_log,
                trail_sigma=trail_sigma_w,
                profit_target_sigmas=pt_sigmas_w,
                net_liq=net_liq,
                batch_id=batch_id,
                decision_id=decision_id,
                now_ns=now_ns,
            )
            if transition_intent is not None:
                intents.append(transition_intent)

        save_state_file(self.config.state_path, states)
        return RunResult(
            intents=tuple(intents),
            states=states,
            decision_id=decision_id,
            batch_id=batch_id,
            skipped=tuple(skipped),
        )

    # ----- internals -------------------------------------------------------

    @staticmethod
    def _latest_sigma(feat: pd.DataFrame) -> float | None:
        """21d log-return std measured as-of T-1 (i.e. value at the latest bar
        in the feature-enriched frame).

        Returns ``None`` if the value is non-finite or non-positive — those
        cases mean we cannot size a trailing band and must skip this asset.
        """
        if "vol_21" not in feat.columns or feat.empty:
            return None
        # vol_21 is rolling std of log-returns; the value at index[-1] reflects
        # the last 21 returns through T-1 (returns are computed with shift(1)).
        val = feat["vol_21"].iloc[-1]
        if not math.isfinite(val) or val <= 0:
            return None
        return float(val)

    def _share_qty(
        self, *, net_liq: Decimal | None, last_close: float
    ) -> int:
        """Equal-weight integer share size; 0 when no NAV is provided."""
        if net_liq is None or net_liq <= 0 or last_close <= 0:
            return 0
        notional_per_position = Decimal(net_liq) / Decimal(self.config.portfolio_size)
        shares = (notional_per_position / Decimal(str(last_close))).to_integral_value(
            rounding=ROUND_DOWN
        )
        return max(0, int(shares))

    def _step_state_machine(
        self,
        *,
        asset: str,
        state: AssetState,
        signal_dir: int,
        last_close: float,
        sigma_log: float,
        trail_sigma: float,
        profit_target_sigmas: float | None,
        net_liq: Decimal | None,
        batch_id: str,
        decision_id: str,
        now_ns: int,
    ) -> OrderIntent | None:
        """Pure transition function. Returns at most one OrderIntent per call.

        Two-event-per-bar limit: if a position would simultaneously trip the
        trailing stop AND the signal flip, we exit on the trailing stop and
        defer the new-direction entry to the next wake (matches the bake-off
        simulator's one-bar grace period).
        """
        # 1) Maintain peak/trough from prior-bar close (the spec).
        if state.state == "IN_LONG":
            if state.peak is None or last_close > state.peak:
                state.peak = last_close
        elif state.state == "IN_SHORT":
            if state.trough is None or last_close < state.trough:
                state.trough = last_close

        # 2) Check trailing stop FIRST (matches the bake-off "sl_first" prio).
        if state.state in ("IN_LONG", "IN_SHORT") and state.entry_price is not None:
            band = state.entry_price * (state.entry_sigma_log or sigma_log) * (
                state.trail_sigma or trail_sigma
            )
            if state.state == "IN_LONG" and state.peak is not None:
                stop_price = state.peak - band
                if last_close <= stop_price:
                    return self._exit_intent(
                        asset=asset, state=state, last_close=last_close,
                        batch_id=batch_id, decision_id=decision_id,
                        now_ns=now_ns, reason="trailing_stop",
                    )
            if state.state == "IN_SHORT" and state.trough is not None:
                stop_price = state.trough + band
                if last_close >= stop_price:
                    return self._exit_intent(
                        asset=asset, state=state, last_close=last_close,
                        batch_id=batch_id, decision_id=decision_id,
                        now_ns=now_ns, reason="trailing_stop",
                    )

        # 2b) Profit target check (σ-multiple of entry sigma off entry price).
        if (state.state in ("IN_LONG", "IN_SHORT")
                and state.entry_price is not None
                and (state.profit_target_sigmas or profit_target_sigmas) is not None):
            pt_mult = state.profit_target_sigmas or profit_target_sigmas
            entry_sig = state.entry_sigma_log or sigma_log
            if state.state == "IN_LONG":
                pt_price = state.entry_price * (1.0 + entry_sig * pt_mult)
                if last_close >= pt_price:
                    return self._exit_intent(
                        asset=asset, state=state, last_close=last_close,
                        batch_id=batch_id, decision_id=decision_id,
                        now_ns=now_ns, reason="profit_target",
                    )
            else:  # IN_SHORT
                pt_price = state.entry_price * (1.0 - entry_sig * pt_mult)
                if last_close <= pt_price:
                    return self._exit_intent(
                        asset=asset, state=state, last_close=last_close,
                        batch_id=batch_id, decision_id=decision_id,
                        now_ns=now_ns, reason="profit_target",
                    )

        # 3) Signal flip → exit (next-wake entry handles flip).
        if state.state == "IN_LONG" and signal_dir < 0:
            return self._exit_intent(
                asset=asset,
                state=state,
                last_close=last_close,
                batch_id=batch_id,
                decision_id=decision_id,
                now_ns=now_ns,
                reason="signal_flip",
            )
        if state.state == "IN_SHORT" and signal_dir > 0:
            return self._exit_intent(
                asset=asset,
                state=state,
                last_close=last_close,
                batch_id=batch_id,
                decision_id=decision_id,
                now_ns=now_ns,
                reason="signal_flip",
            )

        # 4) FLAT + signal → enter.
        if state.state == "FLAT" and signal_dir != 0:
            qty = self._share_qty(net_liq=net_liq, last_close=last_close)
            if qty <= 0:
                return None
            side = Side.BUY if signal_dir > 0 else Side.SELL
            state.state = "IN_LONG" if signal_dir > 0 else "IN_SHORT"
            state.entry_price = last_close
            state.entry_sigma_log = sigma_log
            state.trail_sigma = trail_sigma
            state.profit_target_sigmas = profit_target_sigmas
            state.peak = last_close if signal_dir > 0 else None
            state.trough = last_close if signal_dir < 0 else None
            state.qty = qty
            state.entry_ts_ns = now_ns
            return OrderIntent(
                intent_id=f"par-{decision_id[:12]}-{asset}-entry",
                symbol=_make_symbol(asset),
                side=side,
                delta_qty=Decimal(qty),
                mark=_quantize_price(last_close),
                batch_id=batch_id,
                decision_id=decision_id,
            )

        # 5) Otherwise HOLD.
        return None

    @staticmethod
    def _exit_intent(
        *,
        asset: str,
        state: AssetState,
        last_close: float,
        batch_id: str,
        decision_id: str,
        now_ns: int,
        reason: str,
    ) -> OrderIntent | None:
        if state.qty <= 0:
            # Defensive — should not happen if entries always size positively.
            state.state = "FLAT"
            return None
        was_long = state.state == "IN_LONG"
        exit_side = Side.SELL if was_long else Side.BUY
        qty = state.qty
        # Reset state to FLAT after constructing the intent.
        state.state = "FLAT"
        state.entry_price = None
        state.entry_sigma_log = None
        state.profit_target_sigmas = None
        state.peak = None
        state.trough = None
        state.qty = 0
        state.entry_ts_ns = None
        return OrderIntent(
            intent_id=f"par-{decision_id[:12]}-{asset}-exit-{reason}",
            symbol=_make_symbol(asset),
            side=exit_side,
            delta_qty=Decimal(qty),
            mark=_quantize_price(last_close),
            batch_id=batch_id,
            decision_id=decision_id,
        )
