"""106_all_trades_export — assemble every paper trade since the v2
deployment (2026-05-19) into a single CSV opens-able in Excel / Numbers.

Sources merged in time order:
  * data/paper/dt=YYYY-MM-DD/per_asset_runner_*.parquet   (daily entries + exits)
  * data/paper/dt=YYYY-MM-DD/intraday_exits_*.parquet      (intraday trail/PT)
  * Alpaca order history (filled orders → executed price + commission)

Output: data/all_trades_since_deployment.csv with one row per fill,
plus a paired-trades sheet (entries with exit info) and a daily summary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402


DEPLOY_DATE = "2026-05-19"
OUT_DIR = PROJECT_ROOT / "data"


def load_journal_fills() -> pd.DataFrame:
    """Walk data/paper/dt=*/ and concat every parquet."""
    paper = PROJECT_ROOT / "data" / "paper"
    if not paper.exists():
        return pd.DataFrame()
    rows = []
    for parquet in sorted(paper.rglob("*.parquet")):
        try:
            df = pd.read_parquet(parquet)
        except Exception as exc:
            print(f"WARN: failed {parquet}: {exc}", file=sys.stderr)
            continue
        # Tag source file + date dir
        date_dir = parquet.parent.name.replace("dt=", "")
        df["source_file"] = parquet.name
        df["dt"] = date_dir
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Convert ts_unix to readable timestamp if present
    if "ts_unix" in out.columns:
        out["ts_iso"] = pd.to_datetime(out["ts_unix"], unit="s", utc=True)
    return out


def fetch_alpaca_fills(start_date: str) -> pd.DataFrame:
    """Pull every filled order from Alpaca paper since deploy_date.

    Walk backward 500 at a time using ``until`` only (no ``after`` filter —
    that combo returned empty in 2026-05). Stop once we cross ``start_date``.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    from datetime import datetime, timezone, timedelta

    tc = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
        paper=True,
    )
    cutoff = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    page_until = datetime.now(timezone.utc) + timedelta(hours=1)

    all_orders = []
    seen_ids = set()
    while True:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,
            until=page_until,
            direction="desc",
        )
        try:
            page = tc.get_orders(filter=req)
        except Exception as exc:
            print(f"WARN: get_orders failed: {exc}", file=sys.stderr)
            break
        if not page:
            break
        added_any = False
        oldest = page_until
        for o in page:
            oid = str(getattr(o, "id", ""))
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            all_orders.append(o)
            added_any = True
            if o.submitted_at and o.submitted_at < oldest:
                oldest = o.submitted_at
        if not added_any or oldest <= cutoff or len(page) < 500:
            break
        page_until = oldest

    rows = []
    for o in all_orders:
        status = str(getattr(getattr(o, "status", ""), "value", o.status)).lower()
        if status not in {"filled", "partially_filled"}:
            continue
        filled_at = getattr(o, "filled_at", None)
        if filled_at and filled_at < cutoff:
            continue
        rows.append({
            "client_order_id": str(getattr(o, "client_order_id", "")),
            "symbol": str(getattr(o, "symbol", "")),
            "side": str(getattr(getattr(o, "side", ""), "value", o.side)),
            "qty": float(o.qty) if o.qty else 0.0,
            "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else 0.0,
            "submitted_at": pd.Timestamp(o.submitted_at).tz_convert(None) if o.submitted_at else pd.NaT,
            "filled_at": pd.Timestamp(o.filled_at).tz_convert(None) if o.filled_at else pd.NaT,
            "order_class": str(getattr(getattr(o, "order_class", ""), "value", "")),
            "order_type": str(getattr(getattr(o, "order_type", ""), "value", "")),
            "status": status,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("filled_at").reset_index(drop=True)
        df["notional"] = df["filled_qty"] * df["filled_avg_price"]
    return df


def pair_round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    """FIFO-pair BUYs with SELLs per symbol; compute realized P&L per trade."""
    rows = []
    if fills.empty:
        return pd.DataFrame()
    for sym, sub in fills.groupby("symbol"):
        sub = sub.sort_values("filled_at").reset_index(drop=True)
        long_q = []   # list of (qty, price, fill_ts, coid)
        short_q = []
        for _, fill in sub.iterrows():
            side = fill["side"].upper()
            qty = float(fill["filled_qty"])
            price = float(fill["filled_avg_price"])
            ts = fill["filled_at"]
            coid = fill["client_order_id"]
            if side == "BUY":
                # Cover shorts first
                rem = qty
                while rem > 0 and short_q:
                    sq, sp, sts, sc = short_q[0]
                    take = min(rem, sq)
                    pnl = (sp - price) * take
                    rows.append({
                        "symbol": sym, "trade_side": "COVER_SHORT",
                        "qty": take,
                        "entry_price": sp, "exit_price": price,
                        "entry_ts": sts, "exit_ts": ts,
                        "hold_hours": ((ts - sts).total_seconds() / 3600.0) if pd.notna(ts) and pd.notna(sts) else None,
                        "gross_pnl": pnl,
                        "entry_coid": sc, "exit_coid": coid,
                    })
                    if take == sq:
                        short_q.pop(0)
                    else:
                        short_q[0] = (sq - take, sp, sts, sc)
                    rem -= take
                if rem > 0:
                    long_q.append((rem, price, ts, coid))
            elif side == "SELL":
                rem = qty
                while rem > 0 and long_q:
                    lq, lp, lts, lc = long_q[0]
                    take = min(rem, lq)
                    pnl = (price - lp) * take
                    rows.append({
                        "symbol": sym, "trade_side": "CLOSE_LONG",
                        "qty": take,
                        "entry_price": lp, "exit_price": price,
                        "entry_ts": lts, "exit_ts": ts,
                        "hold_hours": ((ts - lts).total_seconds() / 3600.0) if pd.notna(ts) and pd.notna(lts) else None,
                        "gross_pnl": pnl,
                        "entry_coid": lc, "exit_coid": coid,
                    })
                    if take == lq:
                        long_q.pop(0)
                    else:
                        long_q[0] = (lq - take, lp, lts, lc)
                    rem -= take
                if rem > 0:
                    short_q.append((rem, price, ts, coid))
    return pd.DataFrame(rows)


def daily_summary(fills: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame()
    f = fills.copy()
    f["date"] = pd.to_datetime(f["filled_at"]).dt.date
    fills_by_day = f.groupby("date").agg(
        n_fills=("symbol", "count"),
        n_buys=("side", lambda x: (x.str.upper() == "BUY").sum()),
        n_sells=("side", lambda x: (x.str.upper() == "SELL").sum()),
        notional_usd=("notional", "sum"),
        symbols=("symbol", lambda x: ",".join(sorted(set(x)))),
    )
    if not paired.empty:
        p = paired.copy()
        p["date"] = pd.to_datetime(p["exit_ts"]).dt.date
        pnl_by_day = p.groupby("date").agg(
            n_closed_trades=("gross_pnl", "count"),
            realized_pnl_usd=("gross_pnl", "sum"),
            wins=("gross_pnl", lambda x: (x > 0).sum()),
            losses=("gross_pnl", lambda x: (x <= 0).sum()),
        )
        summary = fills_by_day.join(pnl_by_day, how="outer").fillna(0)
    else:
        summary = fills_by_day
    return summary.reset_index().sort_values("date")


def main() -> int:
    print(f"Loading journal fills since {DEPLOY_DATE}...")
    journal = load_journal_fills()
    print(f"  journal rows: {len(journal)}")

    print(f"Pulling Alpaca order history since {DEPLOY_DATE}...")
    alpaca = fetch_alpaca_fills(DEPLOY_DATE)
    print(f"  alpaca filled orders: {len(alpaca)}")

    if alpaca.empty:
        print("Alpaca returned no fills — falling back to the strategy journal.")
        if journal.empty:
            print("ERROR: no data in either source")
            return 1
        # Build a minimal "alpaca-shaped" frame from the journal
        # so the rest of the pipeline (pairing, summary, xlsx) keeps working.
        j = journal.copy()
        rename_map = {"mark": "filled_avg_price", "fill_qty": "filled_qty",
                       "qty": "qty"}
        for src, dst in rename_map.items():
            if src in j.columns and dst not in j.columns:
                j[dst] = j[src]
        # filled_at from ts_unix
        if "ts_unix" in j.columns:
            j["filled_at"] = pd.to_datetime(j["ts_unix"], unit="s", utc=True).dt.tz_convert(None)
            j["submitted_at"] = j["filled_at"]
        # Default columns
        for col in ("filled_qty", "filled_avg_price", "qty"):
            if col not in j.columns:
                j[col] = 0.0
        j["notional"] = j.get("filled_qty", 0.0) * j.get("filled_avg_price", 0.0)
        j["order_class"] = j.get("order_class", "")
        j["order_type"] = "market"
        j["status"] = j.get("status", "FILLED").astype(str).str.lower()
        keep = ["client_order_id", "symbol", "side", "qty", "filled_qty",
                "filled_avg_price", "submitted_at", "filled_at",
                "order_class", "order_type", "status", "notional"]
        alpaca = j[[c for c in keep if c in j.columns]].copy()
        # Filter to FILLED only
        alpaca = alpaca[alpaca["status"].isin(["filled", "partially_filled"])]
        alpaca = alpaca.sort_values("filled_at").reset_index(drop=True)
        print(f"  journal-fallback fills: {len(alpaca)}")

    print("Pairing round-trips (FIFO per symbol)...")
    paired = pair_round_trips(alpaca)
    print(f"  closed round-trips: {len(paired)}")
    if not paired.empty:
        total_pnl = paired["gross_pnl"].sum()
        win_rate = (paired["gross_pnl"] > 0).mean() * 100
        print(f"  total realized P&L: ${total_pnl:+,.2f}  win rate: {win_rate:.1f}%")

    print("Building daily summary...")
    daily = daily_summary(alpaca, paired)

    # Excel doesn't accept tz-aware datetimes — strip tz from all datetime cols.
    def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                if hasattr(out[col].dt, "tz") and out[col].dt.tz is not None:
                    out[col] = out[col].dt.tz_convert(None) if out[col].dt.tz else out[col].dt.tz_localize(None)
        return out

    # Output: multi-sheet xlsx if openpyxl is available; else 3 CSVs
    try:
        xlsx_path = OUT_DIR / "all_trades_since_deployment.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            _strip_tz(alpaca).to_excel(xw, sheet_name="all_fills", index=False)
            if not paired.empty:
                _strip_tz(paired).to_excel(xw, sheet_name="paired_round_trips", index=False)
            if not daily.empty:
                _strip_tz(daily).to_excel(xw, sheet_name="daily_summary", index=False)
            if not journal.empty:
                keep = [c for c in ["dt", "ts_iso", "symbol", "side", "qty",
                                      "fill_qty", "mark", "status", "client_order_id",
                                      "strategy_sha", "exit_reason"]
                        if c in journal.columns]
                _strip_tz(journal[keep]).to_excel(xw, sheet_name="strategy_journal", index=False)
        print(f"\n  XLSX → {xlsx_path}")
    except ImportError:
        print("  (openpyxl not installed — writing CSVs instead)")
        csv1 = OUT_DIR / "trades_all_fills.csv"
        alpaca.to_csv(csv1, index=False)
        print(f"  CSV → {csv1}")
        if not paired.empty:
            csv2 = OUT_DIR / "trades_paired.csv"
            paired.to_csv(csv2, index=False)
            print(f"  CSV → {csv2}")
        if not daily.empty:
            csv3 = OUT_DIR / "trades_daily_summary.csv"
            daily.to_csv(csv3, index=False)
            print(f"  CSV → {csv3}")
        if not journal.empty:
            csv4 = OUT_DIR / "trades_strategy_journal.csv"
            journal.to_csv(csv4, index=False)
            print(f"  CSV → {csv4}")

    # Print headline numbers
    print("\n" + "=" * 80)
    print("HEADLINE NUMBERS")
    print("=" * 80)
    print(f"Total fills (Alpaca):        {len(alpaca)}")
    print(f"Total closed round-trips:    {len(paired)}")
    if not paired.empty:
        wins = paired[paired["gross_pnl"] > 0]
        losses = paired[paired["gross_pnl"] <= 0]
        print(f"  wins:  {len(wins):>3}  avg=${wins['gross_pnl'].mean():+.2f}  total=${wins['gross_pnl'].sum():+.2f}")
        print(f"  losses:{len(losses):>3}  avg=${losses['gross_pnl'].mean():+.2f}  total=${losses['gross_pnl'].sum():+.2f}")
        print(f"  net:   ${paired['gross_pnl'].sum():+.2f}")
        print(f"  win rate:    {(paired['gross_pnl'] > 0).mean() * 100:.1f}%")
    print(f"\nDate range: {alpaca['filled_at'].min()} → {alpaca['filled_at'].max()}")
    print(f"Symbols traded: {sorted(alpaca['symbol'].unique())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
