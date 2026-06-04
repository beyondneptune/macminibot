"""Alpaca paper-trading connection test.

Verifies API credentials work, queries account state + tradable contract details
for our 8-contract universe, and confirms market clock. No orders placed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "GLD"]


def main() -> int:
    api_key = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET"]

    trading = TradingClient(api_key, secret, paper=True)
    data = StockHistoricalDataClient(api_key, secret)

    # 1. Account
    acct = trading.get_account()
    print(f"=== ALPACA PAPER ACCOUNT ===")
    print(f"  account_id      : {acct.id}")
    print(f"  status          : {acct.status}")
    print(f"  cash            : ${float(acct.cash):>12,.2f}")
    print(f"  buying_power    : ${float(acct.buying_power):>12,.2f}")
    print(f"  portfolio_value : ${float(acct.portfolio_value):>12,.2f}")
    print(f"  pattern_day_trader: {acct.pattern_day_trader}")
    print(f"  trading_blocked : {acct.trading_blocked}")

    # 2. Market clock
    clock = trading.get_clock()
    print(f"\n=== MARKET CLOCK ===")
    print(f"  timestamp : {clock.timestamp}")
    print(f"  is_open   : {clock.is_open}")
    print(f"  next_open : {clock.next_open}")
    print(f"  next_close: {clock.next_close}")

    # 3. Per-symbol tradability + latest quote
    print(f"\n=== UNIVERSE (8 symbols) ===")
    for sym in UNIVERSE:
        try:
            asset = trading.get_asset(sym)
            tradable = "✓" if asset.tradable else "✗"
            shortable = "✓" if asset.shortable else "✗"
            fractionable = "✓" if asset.fractionable else "✗"
            print(
                f"  {sym:6s}  tradable={tradable}  shortable={shortable}  "
                f"fractionable={fractionable}  class={asset.asset_class}"
            )
        except Exception as exc:
            print(f"  {sym:6s}  ERROR: {type(exc).__name__}: {exc}")

    # 4. Quote sanity (use IEX free feed for paper)
    print(f"\n=== LATEST QUOTES (IEX feed) ===")
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=UNIVERSE)
        quotes = data.get_stock_latest_quote(req)
        for sym, q in quotes.items():
            print(f"  {sym:6s}  bid=${float(q.bid_price):>8.2f}  ask=${float(q.ask_price):>8.2f}  ts={q.timestamp}")
    except Exception as exc:
        print(f"  quote fetch FAILED: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
