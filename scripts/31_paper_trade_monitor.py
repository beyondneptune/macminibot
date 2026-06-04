"""31_paper_trade_monitor — live observability for the paper bot.

Polls Alpaca paper account every POLL_SECONDS and prints:
- Account equity + buying power + day P&L
- Open positions (qty, mark, unrealized P&L)
- Open orders (status, filled qty)
- Bot HALT state from ~/macminibot/state/HALT + heartbeat age
- Last 5 entries from data/journal/paper/

Run alongside the runner:

    .venv/bin/python scripts/31_paper_trade_monitor.py

Ctrl-C to stop. Safe to run anytime; read-only.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

POLL_SECONDS = 30
STATE_DIR = Path(os.environ.get("MACMINIBOT_STATE_DIR", str(Path.home() / "macminibot" / "state")))


def colour(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m"


def banner(s: str) -> str:
    return colour(s, "1;36")


def good(s: str) -> str:
    return colour(s, "32")


def warn(s: str) -> str:
    return colour(s, "33")


def bad(s: str) -> str:
    return colour(s, "31")


def fmt_pct(x: float) -> str:
    s = f"{x:+.2%}"
    return good(s) if x >= 0 else bad(s)


def fmt_money(x: float) -> str:
    return f"${x:>12,.2f}"


def fmt_pnl(x: float) -> str:
    s = fmt_money(x)
    return good(s) if x >= 0 else bad(s)


def read_halt_state() -> tuple[bool, float | None]:
    """Returns (is_halted, heartbeat_age_seconds_or_None)."""
    is_halted = (STATE_DIR / "HALT").exists()
    hb = STATE_DIR / "heartbeat.json"
    if hb.exists():
        age = time.time() - hb.stat().st_mtime
    else:
        age = None
    return is_halted, age


def read_recent_journal() -> list[dict]:
    """Read the most recent entries from data/journal/paper/ partition."""
    jp = PROJECT_ROOT / "data" / "journal" / "paper"
    if not jp.exists():
        return []
    parquets = sorted(jp.rglob("*.parquet"))[-5:]
    if not parquets:
        return []
    import pandas as pd
    rows = []
    for p in parquets[-3:]:
        try:
            df = pd.read_parquet(p)
            rows.extend(df.tail(3).to_dict("records"))
        except Exception:
            pass
    return rows[-5:]


def poll_once(trading: TradingClient) -> None:
    now = datetime.now(timezone.utc).astimezone()
    print()
    print(banner("=" * 72))
    print(banner(f"  PAPER BOT MONITOR  •  {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"))
    print(banner("=" * 72))

    # 1. Account
    try:
        acct = trading.get_account()
        equity = float(acct.equity)
        last_eq = float(acct.last_equity)
        day_pnl = equity - last_eq
        day_pnl_pct = day_pnl / last_eq if last_eq else 0.0
        print()
        print(f"  Equity         {fmt_money(equity)}")
        print(f"  Day P&L        {fmt_pnl(day_pnl)}  ({fmt_pct(day_pnl_pct)})")
        print(f"  Cash           {fmt_money(float(acct.cash))}")
        print(f"  Buying Power   {fmt_money(float(acct.buying_power))}")
        if acct.trading_blocked:
            print(f"  {bad('TRADING BLOCKED')}")
    except Exception as exc:
        print(f"  {bad(f'account fetch failed: {exc}')}")

    # 2. Market clock
    try:
        clock = trading.get_clock()
        status = good("OPEN") if clock.is_open else warn("CLOSED")
        print(f"  Market         {status}")
        if not clock.is_open:
            print(f"  Next open      {clock.next_open}")
    except Exception:
        pass

    # 3. Positions
    print()
    print(banner("  POSITIONS"))
    try:
        positions = trading.get_all_positions()
        if not positions:
            print(f"    {warn('(no open positions)')}")
        for p in positions:
            qty = float(p.qty)
            mark = float(p.current_price)
            avg = float(p.avg_entry_price)
            unr = float(p.unrealized_pl)
            unr_pct = float(p.unrealized_plpc)
            side = "LONG " if qty > 0 else "SHORT"
            print(
                f"    {side} {p.symbol:<6s} qty={qty:>+8.0f}  "
                f"mark=${mark:>8.2f}  avg=${avg:>8.2f}  "
                f"unreal={fmt_pnl(unr)} ({fmt_pct(unr_pct)})"
            )
    except Exception as exc:
        print(f"    {bad(f'positions fetch failed: {exc}')}")

    # 4. Open orders
    print()
    print(banner("  OPEN ORDERS"))
    try:
        orders = trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        if not orders:
            print(f"    {warn('(no open orders)')}")
        for o in orders:
            qty = float(o.qty)
            filled = float(o.filled_qty or 0)
            print(
                f"    {o.side.value:<4s} {o.symbol:<6s} qty={qty:>+8.0f}  "
                f"filled={filled:>+5.0f}  status={o.status.value}  "
                f"type={o.order_type.value}"
            )
    except Exception as exc:
        print(f"    {bad(f'orders fetch failed: {exc}')}")

    # 5. Bot state
    print()
    print(banner("  BOT STATE"))
    is_halted, hb_age = read_halt_state()
    halt_str = bad("HALTED") if is_halted else good("RUNNING")
    print(f"    Kill switch    {halt_str}")
    if hb_age is None:
        print(f"    Heartbeat      {bad('MISSING')}")
    elif hb_age < 300:
        print(f"    Heartbeat      {good(f'{hb_age:.0f}s ago')}")
    else:
        print(f"    Heartbeat      {bad(f'STALE ({hb_age:.0f}s ago)')}")

    # 6. Recent journal fills
    print()
    print(banner("  RECENT JOURNAL (data/journal/paper/)"))
    rows = read_recent_journal()
    if not rows:
        print(f"    {warn('(no journal entries yet)')}")
    for r in rows[-5:]:
        sym = r.get("symbol", "?")
        side = r.get("side", "?")
        fill_qty = r.get("fill_qty", 0) or 0
        fill_price = r.get("fill_price", 0) or 0
        status = r.get("status", "?")
        print(f"    {sym:<6s} {side:<4s} qty={fill_qty:>+5.0f}  fill=${float(fill_price or 0):>8.2f}  {status}")


def main() -> int:
    api_key = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET"]
    trading = TradingClient(api_key, secret, paper=True)
    print(f"Monitoring Alpaca paper account every {POLL_SECONDS}s. Ctrl-C to stop.")
    try:
        while True:
            poll_once(trading)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n[monitor] stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
