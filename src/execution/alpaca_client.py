"""execution.alpaca_client — thin wrapper around ``alpaca-py`` TradingClient.

Scope of this module (today): paper-trading adapter only. Credentials are
loaded from ``.env`` via ``python-dotenv``. We expose six primitives the
per-asset trailing-stop runner needs:

  - :func:`AlpacaClient.submit_order(intent)` → returns ``client_order_id``
  - :func:`AlpacaClient.submit_bracket_order(intent, take_profit_pct,
    stop_loss_pct)` — market entry with attached TP & SL legs (OrderClass
    BRACKET); the per-asset variant bake-off used this as the exit policy.
  - :func:`AlpacaClient.submit_trailing_stop(intent, trail_pct)` — a
    pure trailing-stop order (TrailingStopOrderRequest) for *exit-side*
    submissions where we don't need an explicit TP target.
  - :func:`AlpacaClient.get_positions()` → ``{ticker: Decimal qty}``
  - :func:`AlpacaClient.get_account_equity()` → ``Decimal``
  - :func:`AlpacaClient.cancel_all_orders()` — kill-switch primitive

Design notes
------------
* The ``alpaca-py`` ``TradingClient`` is constructed lazily so unit tests
  can monkeypatch the underlying client without `.env` being present.
* ``client_order_id`` is derived deterministically from ``intent_id`` so a
  retry submits the same id (broker rejects duplicates → idempotent).
  See Contract 5 in execution/CONTEXT.md; this is the minimal viable form
  pending the full ULID-from-seed scheme.
* For brackets we attach ``take_profit`` / ``stop_loss`` legs computed in
  *percent* off the intent's ``mark`` so callers don't have to compute
  absolute price levels.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from shared_core.contracts import OrderIntent, Side

if TYPE_CHECKING:  # pragma: no cover — typing only
    from alpaca.trading.client import TradingClient as _AlpacaTradingClient


__all__ = [
    "AlpacaClient",
    "AlpacaConfig",
    "AlpacaCredentialsMissing",
    "TradingClientProtocol",
    "derive_client_order_id",
    "load_alpaca_config",
]


# ---------------------------------------------------------------------------
# Config + credentials
# ---------------------------------------------------------------------------


class AlpacaCredentialsMissing(RuntimeError):
    """Raised when the .env file is missing required ALPACA_* keys."""


@dataclass(frozen=True, slots=True)
class AlpacaConfig:
    api_key: str
    api_secret: str
    base_url: str
    paper: bool = True

    def redact(self) -> str:
        return (
            f"AlpacaConfig(api_key=***{self.api_key[-4:]}, "
            f"base_url={self.base_url}, paper={self.paper})"
        )


def _load_dotenv_if_present(dotenv_path: Path | None) -> None:
    """Load ``.env`` via python-dotenv if available and path exists.

    We import lazily so unit tests that pass explicit credentials don't
    require ``python-dotenv`` to be installed.
    """

    if dotenv_path is None:
        # Default search: repo root .env next to pyproject.toml.
        candidate = Path(__file__).resolve().parents[2] / ".env"
        dotenv_path = candidate if candidate.exists() else None
    if dotenv_path is None or not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover — python-dotenv is in deps
        return
    load_dotenv(dotenv_path, override=False)


def load_alpaca_config(dotenv_path: Path | None = None) -> AlpacaConfig:
    """Read Alpaca credentials from the environment (or .env file).

    Recognised keys: ``ALPACA_API_KEY``, ``ALPACA_SECRET`` (or
    ``ALPACA_API_SECRET``), ``ALPACA_BASE_URL``.
    """

    _load_dotenv_if_present(dotenv_path)
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not api_secret:
        raise AlpacaCredentialsMissing(
            "Missing ALPACA_API_KEY / ALPACA_SECRET in environment "
            "(check .env in repo root)."
        )
    paper = "paper" in base_url.lower()
    return AlpacaConfig(
        api_key=api_key, api_secret=api_secret, base_url=base_url, paper=paper
    )


# ---------------------------------------------------------------------------
# Client-order-id derivation (Contract 5, minimal form)
# ---------------------------------------------------------------------------


def derive_client_order_id(
    intent_id: str,
    *,
    account: str = "PAPER",
    attempt: int = 0,
    strategy_sha: str = "v0",
) -> str:
    """Deterministic client-order-id derived from ``intent_id``.

    The full Contract 5 form is a ULID-from-seed; for the first paper
    trade we use a 24-char hex prefix of sha256 which is:

    - deterministic (same inputs → same id, retries are idempotent),
    - unique across restarts (intent_id encodes session),
    - short enough to satisfy Alpaca's 48-char client_order_id cap.
    """

    seed = f"{intent_id}|{strategy_sha}|{account}|{attempt}".encode()
    digest = hashlib.sha256(seed).hexdigest()
    return f"mmb-{digest[:24]}"


# ---------------------------------------------------------------------------
# TradingClient protocol — what the wrapper needs from alpaca-py
# ---------------------------------------------------------------------------


class TradingClientProtocol(Protocol):
    """Subset of ``alpaca.trading.client.TradingClient`` we depend on.

    Defined as a Protocol so unit tests can supply a hand-rolled fake
    instead of monkey-patching the real class.
    """

    def submit_order(self, order_data):  # noqa: ANN001 — opaque request type
        ...

    def get_all_positions(self): ...

    def get_account(self): ...

    def cancel_orders(self): ...


# ---------------------------------------------------------------------------
# The wrapper itself
# ---------------------------------------------------------------------------


@dataclass
class AlpacaClient:
    """Wrapper around an ``alpaca-py`` TradingClient (paper trading).

    Construction policy:
      - ``AlpacaClient.from_env()`` — production path, reads .env.
      - ``AlpacaClient(client=<fake>)`` — test path, inject a stub
        satisfying :class:`TradingClientProtocol`.
    """

    client: "TradingClientProtocol"
    config: AlpacaConfig | None = None

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "AlpacaClient":
        config = load_alpaca_config(dotenv_path)
        from alpaca.trading.client import TradingClient

        tc = TradingClient(
            api_key=config.api_key,
            secret_key=config.api_secret,
            paper=config.paper,
        )
        return cls(client=tc, config=config)

    # -- order submission ---------------------------------------------------
    def submit_order(
        self,
        intent: OrderIntent,
        *,
        account: str = "PAPER",
        attempt: int = 0,
        strategy_sha: str = "v0",
    ) -> str:
        """Submit a market DAY order; return the deterministic client_order_id.

        We always paper-trade today, so ``order_type=MARKET`` and
        ``time_in_force=DAY``. We submit ``qty`` as the unsigned
        ``delta_qty`` integer — Alpaca requires positive ints for share
        quantities.
        """

        coid = derive_client_order_id(
            intent.intent_id,
            account=account,
            attempt=attempt,
            strategy_sha=strategy_sha,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = (
            OrderSide.BUY if intent.side is Side.BUY else OrderSide.SELL
        )
        # delta_qty is a Decimal; Alpaca accepts int or str for whole shares.
        qty = int(intent.delta_qty)
        if qty <= 0:
            raise ValueError(
                f"submit_order: non-positive qty after int() cast: "
                f"{intent.delta_qty!r}"
            )
        req = MarketOrderRequest(
            symbol=intent.symbol.ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
        )
        self.client.submit_order(order_data=req)
        return coid

    # -- bracket order ------------------------------------------------------
    def submit_bracket_order(
        self,
        intent: OrderIntent,
        *,
        take_profit_pct: float,
        stop_loss_pct: float,
        account: str = "PAPER",
        attempt: int = 0,
        strategy_sha: str = "v0",
    ) -> str:
        """Submit a market entry with attached TP + SL legs (OrderClass BRACKET).

        ``take_profit_pct`` and ``stop_loss_pct`` are fractions (e.g.
        ``0.05`` = 5%) measured off ``intent.mark`` and applied in the
        direction implied by the entry side:

          * BUY:  TP = mark * (1 + tp_pct), SL = mark * (1 - sl_pct)
          * SELL: TP = mark * (1 - tp_pct), SL = mark * (1 + sl_pct)

        Returns the deterministic ``client_order_id``. The Alpaca API
        rejects bracket orders for non-positive prices or for ``tp_pct``
        / ``sl_pct`` that produce non-positive limit prices — we validate
        the percentages here so the caller gets a clear local error
        rather than an opaque broker rejection.
        """
        if take_profit_pct <= 0 or take_profit_pct >= 1:
            raise ValueError(
                f"take_profit_pct must be in (0, 1); got {take_profit_pct!r}"
            )
        if stop_loss_pct <= 0 or stop_loss_pct >= 1:
            raise ValueError(
                f"stop_loss_pct must be in (0, 1); got {stop_loss_pct!r}"
            )
        coid = derive_client_order_id(
            intent.intent_id,
            account=account,
            attempt=attempt,
            strategy_sha=strategy_sha,
        )
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        side = OrderSide.BUY if intent.side is Side.BUY else OrderSide.SELL
        qty = int(intent.delta_qty)
        if qty <= 0:
            raise ValueError(
                f"submit_bracket_order: non-positive qty after int() cast: "
                f"{intent.delta_qty!r}"
            )
        mark = Decimal(intent.mark)
        tp = Decimal(str(take_profit_pct))
        sl = Decimal(str(stop_loss_pct))
        if intent.side is Side.BUY:
            tp_price = (mark * (Decimal("1") + tp)).quantize(Decimal("0.01"))
            sl_price = (mark * (Decimal("1") - sl)).quantize(Decimal("0.01"))
        else:
            tp_price = (mark * (Decimal("1") - tp)).quantize(Decimal("0.01"))
            sl_price = (mark * (Decimal("1") + sl)).quantize(Decimal("0.01"))
        if tp_price <= 0 or sl_price <= 0:
            raise ValueError(
                f"bracket TP/SL produced non-positive price: "
                f"tp={tp_price} sl={sl_price} (mark={mark})"
            )

        req = MarketOrderRequest(
            symbol=intent.symbol.ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=float(tp_price)),
            stop_loss=StopLossRequest(stop_price=float(sl_price)),
        )
        self.client.submit_order(order_data=req)
        return coid

    # -- trailing-stop order ------------------------------------------------
    def submit_trailing_stop(
        self,
        intent: OrderIntent,
        *,
        trail_pct: float,
        account: str = "PAPER",
        attempt: int = 0,
        strategy_sha: str = "v0",
    ) -> str:
        """Submit an Alpaca TrailingStopOrderRequest.

        Per Alpaca's API, trailing-stop orders are placed in the
        *opposite* direction of the resting position: to protect a long
        you submit a SELL trailing stop, and vice versa. The ``intent``'s
        ``side`` therefore encodes the exit side — the caller is expected
        to pass an exit-side intent (the runner constructs these via
        the trailing-stop state machine).

        ``trail_pct`` is the percent (e.g. ``0.05`` = 5%); we map to
        ``trail_percent`` on the request body.
        """
        if trail_pct <= 0 or trail_pct >= 1:
            raise ValueError(f"trail_pct must be in (0, 1); got {trail_pct!r}")
        coid = derive_client_order_id(
            intent.intent_id,
            account=account,
            attempt=attempt,
            strategy_sha=strategy_sha,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import TrailingStopOrderRequest

        side = OrderSide.BUY if intent.side is Side.BUY else OrderSide.SELL
        qty = int(intent.delta_qty)
        if qty <= 0:
            raise ValueError(
                f"submit_trailing_stop: non-positive qty after int() cast: "
                f"{intent.delta_qty!r}"
            )
        req = TrailingStopOrderRequest(
            symbol=intent.symbol.ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
            trail_percent=float(round(trail_pct * 100, 4)),
        )
        self.client.submit_order(order_data=req)
        return coid

    # -- positions ----------------------------------------------------------
    def get_positions(self) -> dict[str, Decimal]:
        """Return current open positions as ``{ticker: Decimal qty}``.

        Signed: long = positive, short = negative.
        """

        result: dict[str, Decimal] = {}
        for pos in self.client.get_all_positions():
            ticker = getattr(pos, "symbol", None)
            qty_raw = getattr(pos, "qty", None)
            if ticker is None or qty_raw is None:
                continue
            qty = Decimal(str(qty_raw))
            side = getattr(pos, "side", "long")
            side_str = str(getattr(side, "value", side)).lower()
            if side_str == "short" and qty > 0:
                qty = -qty
            result[str(ticker)] = qty
        return result

    # -- equity -------------------------------------------------------------
    def get_account_equity(self) -> Decimal:
        """Return account equity as ``Decimal`` (KD1)."""

        acct = self.client.get_account()
        raw = getattr(acct, "equity", None)
        if raw is None:
            raise RuntimeError("Alpaca account response missing 'equity' field")
        return Decimal(str(raw))

    # -- kill switch --------------------------------------------------------
    def cancel_all_orders(self) -> None:
        """Cancel every working order. Synchronous; honoured mid-batch."""

        self.client.cancel_orders()
