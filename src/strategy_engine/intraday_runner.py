"""intraday_runner — Bollinger-3σ/3σ mean-reversion state machine on 5-min bars.

Deployed config per the 79_best_per_asset head-to-head:
   AAPL, MSFT  →  bollinger trail=3σ profit_target=3σ on 5-min bars

State machine is identical to ``per_asset_runner.PerAssetRunner`` (trailing-
stop on σ-band of entry, signal-flip exit, profit-target on σ-band of entry)
but consumes 5-minute bars and persists to its own state file so the daily-
bar runner doesn't see/disturb intraday state.

Public API:
    cfg = IntradayRunnerConfig(state_path=Path("data/intraday_state.json"),
                                bar_loader=intraday_bar_loader,
                                net_liq=Decimal("100000"),
                                per_position_size=Decimal("12500"))
    runner = IntradayRunner(cfg)
    result = runner.run()  # one fire

Each fire processes the LATEST 5-min bar per asset and emits at most one
OrderIntent per asset (entry, trail-exit, PT-exit, or signal-flip exit).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from ml_research.asset_specialists._signals import add_per_asset_features
from shared_core.contracts import OrderIntent, Side, Symbol, Venue

_LOG = logging.getLogger(__name__)

INTRADAY_ASSETS = ("AAPL", "MSFT")

_ASSET_META: dict[str, tuple[str, str, Venue, str]] = {
    "AAPL": ("equity.minute_bars", "NASDAQ.AAPL", Venue.NASDAQ, "USD"),
    "MSFT": ("equity.minute_bars", "NASDAQ.MSFT", Venue.NASDAQ, "USD"),
}

WINNER_CONFIG = {
    "signal": "bollinger",
    "trail_sigma": 3.0,
    "profit_target_sigmas": 3.0,
    "bollinger_lookback": 20,
    "bollinger_z_threshold": 2.0,
}

MIN_BARS_REQUIRED = 100  # enough for vol_21 + Bollinger 20-window after warmup


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


@dataclass
class IntradayAssetState:
    asset: str
    state: str = "FLAT"            # FLAT | IN_LONG | IN_SHORT
    entry_price: float | None = None
    entry_sigma_log: float | None = None
    entry_ts_ns: int | None = None
    qty: int = 0
    peak: float | None = None
    trough: float | None = None
    last_close: float | None = None
    trail_sigma: float | None = None
    profit_target_sigmas: float | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "IntradayAssetState":
        return cls(
            asset=raw["asset"],
            state=raw.get("state", "FLAT"),
            entry_price=raw.get("entry_price"),
            entry_sigma_log=raw.get("entry_sigma_log"),
            entry_ts_ns=raw.get("entry_ts_ns"),
            qty=int(raw.get("qty", 0)),
            peak=raw.get("peak"),
            trough=raw.get("trough"),
            last_close=raw.get("last_close"),
            trail_sigma=raw.get("trail_sigma"),
            profit_target_sigmas=raw.get("profit_target_sigmas"),
        )


def load_state_file(path: Path) -> dict[str, IntradayAssetState]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: IntradayAssetState.from_dict(v) for k, v in raw.items()}


def save_state_file(path: Path, states: Mapping[str, IntradayAssetState]) -> None:
    serializable = {k: asdict(v) for k, v in states.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(serializable, indent=2, sort_keys=True))
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Bar loader
# ---------------------------------------------------------------------------


BarLoader = Callable[[str], pd.DataFrame]


def default_intraday_bar_loader(asset: str) -> pd.DataFrame:
    """Load 5-min bars from ArcticDB. The intraday backfill script keeps
    this up to date; the production runner can also pull fresh bars before
    each fire via the Alpaca data API and update ArcticDB."""
    if asset not in _ASSET_META:
        raise ValueError(f"unknown intraday asset {asset!r}")
    lib_name, sym, _venue, _ccy = _ASSET_META[asset]
    from scripts._storage import open_store  # type: ignore

    store = open_store()
    df = store[lib_name].read(sym).data.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Bollinger signal — re-export so tests don't have to re-derive
# ---------------------------------------------------------------------------


def compute_bollinger_signal(bars: pd.DataFrame, lookback: int = 20,
                              z_threshold: float = 2.0) -> int:
    """+1 if previous-bar close z-score < -threshold (oversold → long),
    -1 if > +threshold (overbought → short), else 0."""
    if len(bars) < lookback + 2:
        return 0
    close = bars["close"]
    ma = close.rolling(lookback).mean().shift(1)
    sd = close.rolling(lookback).std().shift(1)
    prev_close = close.shift(1)
    z_last = (prev_close.iloc[-1] - ma.iloc[-1]) / sd.iloc[-1]
    if not math.isfinite(z_last):
        return 0
    if z_last < -z_threshold:
        return 1
    if z_last > z_threshold:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize_price(value: float) -> Decimal:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"price must be finite and positive, got {value!r}")
    return Decimal(f"{value:.4f}").quantize(Decimal("0.01"))


def _make_symbol(asset: str) -> Symbol:
    _, _, venue, ccy = _ASSET_META[asset]
    return Symbol(ticker=asset, venue=venue, currency=ccy)


def _decision_id(as_of_ns: int, strategy_sha: str, signals: Mapping[str, int]) -> str:
    payload = json.dumps(
        {"as_of_ns": as_of_ns, "strategy": strategy_sha,
         "signals": {k: int(v) for k, v in sorted(signals.items())}},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntradayRunnerConfig:
    state_path: Path
    assets: tuple[str, ...] = INTRADAY_ASSETS
    per_position_size: Decimal = Decimal("12500")
    strategy_sha: str = "intraday_bollinger_v1"


@dataclass(frozen=True)
class IntradayRunResult:
    intents: tuple[OrderIntent, ...]
    states: dict[str, IntradayAssetState]
    decision_id: str
    batch_id: str


@dataclass
class IntradayRunner:
    config: IntradayRunnerConfig
    bar_loader: BarLoader = field(default_factory=lambda: default_intraday_bar_loader)

    def run(self) -> IntradayRunResult:
        now_ns = time.time_ns()
        states = load_state_file(self.config.state_path)
        for a in self.config.assets:
            states.setdefault(a, IntradayAssetState(asset=a))

        signals: dict[str, int] = {}
        sigmas: dict[str, float] = {}
        last_closes: dict[str, float] = {}

        trail = WINNER_CONFIG["trail_sigma"]
        pt_sigmas = WINNER_CONFIG["profit_target_sigmas"]
        bb_lookback = WINNER_CONFIG["bollinger_lookback"]
        bb_z = WINNER_CONFIG["bollinger_z_threshold"]

        for asset in self.config.assets:
            try:
                bars = self.bar_loader(asset)
            except Exception as e:
                _LOG.warning("intraday bar_loader failed for %s: %s", asset, e)
                continue
            if bars is None or len(bars) < MIN_BARS_REQUIRED:
                _LOG.info("intraday %s: insufficient bars (%d)", asset, len(bars) if bars is not None else 0)
                continue
            feat = add_per_asset_features(bars)
            sigma = float(feat["vol_21"].iloc[-1])
            if not math.isfinite(sigma) or sigma <= 0:
                continue
            sig = compute_bollinger_signal(bars, lookback=bb_lookback, z_threshold=bb_z)
            signals[asset] = sig
            sigmas[asset] = sigma
            last_closes[asset] = float(bars["close"].iloc[-1])

        decision_id = _decision_id(now_ns, self.config.strategy_sha, signals)
        batch_id = f"intra-{decision_id[:16]}-{uuid.uuid4().hex[:6]}"

        intents: list[OrderIntent] = []
        for asset in self.config.assets:
            if asset not in signals:
                continue
            sig = signals[asset]
            sigma = sigmas[asset]
            close = last_closes[asset]
            state = states[asset]
            state.last_close = close

            transition = self._step_state_machine(
                asset=asset, state=state, signal_dir=sig,
                last_close=close, sigma_log=sigma,
                trail_sigma=float(trail), profit_target_sigmas=float(pt_sigmas),
                batch_id=batch_id, decision_id=decision_id, now_ns=now_ns,
            )
            if transition is not None:
                intents.append(transition)

        save_state_file(self.config.state_path, states)
        return IntradayRunResult(
            intents=tuple(intents), states=states,
            decision_id=decision_id, batch_id=batch_id,
        )

    def _step_state_machine(
        self, *, asset: str, state: IntradayAssetState, signal_dir: int,
        last_close: float, sigma_log: float, trail_sigma: float,
        profit_target_sigmas: float, batch_id: str, decision_id: str, now_ns: int,
    ) -> OrderIntent | None:
        # Maintain peak/trough.
        if state.state == "IN_LONG":
            if state.peak is None or last_close > state.peak:
                state.peak = last_close
        elif state.state == "IN_SHORT":
            if state.trough is None or last_close < state.trough:
                state.trough = last_close

        # Exit checks: trailing stop > profit target > signal flip.
        if state.state in ("IN_LONG", "IN_SHORT") and state.entry_price is not None:
            band = state.entry_price * (state.entry_sigma_log or sigma_log) * (
                state.trail_sigma or trail_sigma
            )
            if state.state == "IN_LONG" and state.peak is not None:
                stop = state.peak - band
                if last_close <= stop:
                    return self._exit_intent(asset, state, last_close,
                                              batch_id, decision_id, now_ns, "trailing_stop")
                if profit_target_sigmas is not None and state.entry_sigma_log is not None:
                    pt_price = state.entry_price * (
                        1.0 + state.entry_sigma_log * (state.profit_target_sigmas or profit_target_sigmas)
                    )
                    if last_close >= pt_price:
                        return self._exit_intent(asset, state, last_close,
                                                  batch_id, decision_id, now_ns, "profit_target")
            elif state.state == "IN_SHORT" and state.trough is not None:
                stop = state.trough + band
                if last_close >= stop:
                    return self._exit_intent(asset, state, last_close,
                                              batch_id, decision_id, now_ns, "trailing_stop")
                if profit_target_sigmas is not None and state.entry_sigma_log is not None:
                    pt_price = state.entry_price * (
                        1.0 - state.entry_sigma_log * (state.profit_target_sigmas or profit_target_sigmas)
                    )
                    if last_close <= pt_price:
                        return self._exit_intent(asset, state, last_close,
                                                  batch_id, decision_id, now_ns, "profit_target")

        # Signal flip exit.
        if state.state == "IN_LONG" and signal_dir < 0:
            return self._exit_intent(asset, state, last_close,
                                      batch_id, decision_id, now_ns, "signal_flip")
        if state.state == "IN_SHORT" and signal_dir > 0:
            return self._exit_intent(asset, state, last_close,
                                      batch_id, decision_id, now_ns, "signal_flip")

        # FLAT + signal → enter.
        if state.state == "FLAT" and signal_dir != 0:
            notional = self.config.per_position_size
            shares = (notional / Decimal(str(last_close))).to_integral_value(rounding=ROUND_DOWN)
            qty = max(0, int(shares))
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
                intent_id=f"intra-{decision_id[:12]}-{asset}-entry",
                symbol=_make_symbol(asset),
                side=side,
                delta_qty=Decimal(qty),
                mark=_quantize_price(last_close),
                batch_id=batch_id,
                decision_id=decision_id,
            )
        return None

    @staticmethod
    def _exit_intent(asset: str, state: IntradayAssetState, last_close: float,
                     batch_id: str, decision_id: str, now_ns: int,
                     reason: str) -> OrderIntent | None:
        if state.qty <= 0:
            state.state = "FLAT"
            return None
        was_long = state.state == "IN_LONG"
        exit_side = Side.SELL if was_long else Side.BUY
        qty = state.qty
        state.state = "FLAT"
        state.entry_price = None
        state.entry_sigma_log = None
        state.peak = None
        state.trough = None
        state.qty = 0
        state.entry_ts_ns = None
        return OrderIntent(
            intent_id=f"intra-{decision_id[:12]}-{asset}-exit-{reason}",
            symbol=_make_symbol(asset),
            side=exit_side,
            delta_qty=Decimal(qty),
            mark=_quantize_price(last_close),
            batch_id=batch_id,
            decision_id=decision_id,
        )
