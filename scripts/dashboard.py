"""macminibot — Streamlit paper-trading dashboard.

Run from project root:

    .venv/bin/streamlit run scripts/dashboard.py --server.address 0.0.0.0 --server.port 8501

Then open http://<this-mac-ip>:8501 from any device on the local network.
Auto-refreshes every 15s.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

STATE_DIR = Path(os.environ.get("MACMINIBOT_STATE_DIR", str(Path.home() / "macminibot" / "state")))
JOURNAL_PAPER = PROJECT_ROOT / "data" / "journal" / "paper"
REFRESH_SECONDS = 15

st.set_page_config(
    page_title="macminibot — Paper Trading",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def _trading_client() -> TradingClient:
    return TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET"], paper=True)


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def fmt_pct(x: float) -> str:
    return f"{x:+.2%}"


def fmt_delta(x: float) -> str:
    return f"{x:+,.2f}"


def read_halt_state():
    is_halted = (STATE_DIR / "HALT").exists()
    hb = STATE_DIR / "heartbeat.json"
    age = time.time() - hb.stat().st_mtime if hb.exists() else None
    return is_halted, age


def read_recent_fills(limit: int = 50) -> pd.DataFrame:
    if not JOURNAL_PAPER.exists():
        return pd.DataFrame()
    parquets = sorted(JOURNAL_PAPER.rglob("*.parquet"))
    if not parquets:
        return pd.DataFrame()
    frames = []
    for p in parquets[-10:]:
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "submitted_at_ns" in df.columns:
        df = df.sort_values("submitted_at_ns", ascending=False)
    return df.head(limit)


def main() -> None:
    # Auto-refresh
    st.markdown(
        f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">',
        unsafe_allow_html=True,
    )

    # Header
    col_title, col_time = st.columns([3, 1])
    with col_title:
        st.title("🤖 macminibot")
        st.caption("Paper trading on Alpaca — Ridge baseline, vix-gated, dollar-neutral long-short")
    with col_time:
        now = datetime.now(timezone.utc).astimezone()
        st.caption(f"**Local**: {now.strftime('%H:%M:%S %Z')}")
        st.caption(f"**Auto-refresh**: every {REFRESH_SECONDS}s")

    trading = _trading_client()

    # ===== Account row =====
    try:
        acct = trading.get_account()
        equity = float(acct.equity)
        last_eq = float(acct.last_equity)
        day_pnl = equity - last_eq
        day_pnl_pct = day_pnl / last_eq if last_eq else 0.0
        cash = float(acct.cash)
        bp = float(acct.buying_power)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity", fmt_money(equity), fmt_delta(day_pnl))
        c2.metric("Day P&L", fmt_money(day_pnl), fmt_pct(day_pnl_pct))
        c3.metric("Cash", fmt_money(cash))
        c4.metric("Buying Power", fmt_money(bp))

        if acct.trading_blocked:
            st.error("⚠️ TRADING BLOCKED on Alpaca account")
    except Exception as exc:
        st.error(f"Account fetch failed: {exc}")
        return

    # ===== Market clock + bot state =====
    col_market, col_bot = st.columns(2)
    with col_market:
        st.subheader("📊 Market")
        try:
            clock = trading.get_clock()
            if clock.is_open:
                st.success(f"🟢 OPEN — closes {clock.next_close.strftime('%H:%M %Z')}")
            else:
                st.warning(f"🟡 CLOSED — next open {clock.next_open.strftime('%Y-%m-%d %H:%M %Z')}")
        except Exception as exc:
            st.error(f"clock fetch failed: {exc}")

    with col_bot:
        st.subheader("🛡️ Bot State")
        is_halted, hb_age = read_halt_state()
        cols = st.columns(2)
        with cols[0]:
            if is_halted:
                st.error("🔴 HALTED")
            else:
                st.success("🟢 RUNNING")
        with cols[1]:
            if hb_age is None:
                st.error("⚪ heartbeat MISSING")
            elif hb_age < 300:
                st.success(f"💓 heartbeat {hb_age:.0f}s ago")
            else:
                st.error(f"💔 heartbeat STALE ({hb_age:.0f}s)")

    # ===== Positions =====
    st.subheader("📈 Open Positions")
    try:
        positions = trading.get_all_positions()
        if not positions:
            st.info("No open positions")
        else:
            pos_rows = []
            total_unrealized = 0.0
            for p in positions:
                qty = float(p.qty)
                mark = float(p.current_price)
                avg = float(p.avg_entry_price)
                unr = float(p.unrealized_pl)
                unr_pct = float(p.unrealized_plpc)
                total_unrealized += unr
                pos_rows.append({
                    "Side": "LONG" if qty > 0 else "SHORT",
                    "Symbol": p.symbol,
                    "Qty": qty,
                    "Avg Entry": f"${avg:,.2f}",
                    "Mark": f"${mark:,.2f}",
                    "Notional": f"${abs(qty) * mark:,.2f}",
                    "Unrealized P&L": f"${unr:+,.2f}",
                    "Unrealized %": f"{unr_pct:+.2%}",
                })
            df = pd.DataFrame(pos_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            colA, colB = st.columns(2)
            colA.metric("Total Unrealized P&L", fmt_money(total_unrealized))
            colB.metric("Position Count", len(positions))
    except Exception as exc:
        st.error(f"positions fetch failed: {exc}")

    # ===== Open orders =====
    st.subheader("📝 Open Orders")
    try:
        orders = trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50))
        if not orders:
            st.info("No open orders")
        else:
            ord_rows = []
            for o in orders:
                ord_rows.append({
                    "Symbol": o.symbol,
                    "Side": o.side.value.upper(),
                    "Qty": float(o.qty),
                    "Filled": float(o.filled_qty or 0),
                    "Type": o.order_type.value,
                    "Status": o.status.value,
                    "Client Order ID": (o.client_order_id or "")[:48],
                })
            df = pd.DataFrame(ord_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"orders fetch failed: {exc}")

    # ===== Closed orders (today) =====
    st.subheader("✅ Filled Orders (last 50)")
    try:
        orders = trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=50))
        if not orders:
            st.info("No filled orders yet today")
        else:
            fill_rows = []
            for o in orders:
                fill_rows.append({
                    "Time": o.filled_at.strftime("%H:%M:%S") if o.filled_at else "-",
                    "Symbol": o.symbol,
                    "Side": o.side.value.upper(),
                    "Qty": float(o.filled_qty or 0),
                    "Avg Fill": f"${float(o.filled_avg_price or 0):,.4f}" if o.filled_avg_price else "-",
                    "Status": o.status.value,
                })
            df = pd.DataFrame(fill_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"closed orders fetch failed: {exc}")

    # ===== Recent journal fills (from our local parquet) =====
    with st.expander("📋 Recent Journal Entries (data/journal/paper/)", expanded=False):
        df = read_recent_fills(limit=20)
        if df.empty:
            st.info("No journal entries yet")
        else:
            cols = [c for c in ["symbol", "side", "qty", "fill_qty", "fill_price", "status", "intent_id"] if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

    st.caption(f"📍 macminibot — dashboard auto-refreshes every {REFRESH_SECONDS}s. Last poll: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
