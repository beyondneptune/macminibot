"""46_operator_eod_summary — single-page dashboard for when the operator returns.

Aggregates all post-close evidence in one terminal-friendly view:
  • Account equity + day P&L vs deploy + cumulative
  • Open positions with MTM, stop distance, PT distance
  • Today's runner + intraday-exit-monitor activity (entries, exits, blocks)
  • Day's benchmark journal entry (bot vs B&H vs UnifStop-{1,2,3}σ)
  • Time-of-day filter firing events (was it active when expected?)
  • Halt audit log (any HALT events today?)
  • Tomorrow's expected behavior

Run:
    .venv/bin/python scripts/46_operator_eod_summary.py
    .venv/bin/python scripts/46_operator_eod_summary.py --date 2026-05-21
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402


def fetch_alpaca():
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.trading.client import TradingClient
    tc = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
        paper=True,
    )
    a = tc.get_account()
    positions = tc.get_all_positions()
    return tc, a, positions


def read_state():
    p = PROJECT_ROOT / "data" / "per_asset_state.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def read_log(asset_pattern: str, date_iso: str, limit: int = 30) -> list[str]:
    p = PROJECT_ROOT / "data" / f"{asset_pattern}_{date_iso}.log"
    if not p.exists():
        return []
    return p.read_text().splitlines()[-limit:]


def read_benchmark_today(date_iso: str) -> dict | None:
    journal = PROJECT_ROOT / "data" / "benchmark_journal.parquet"
    if not journal.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(journal)
    df["date_str"] = df["date"].astype(str).str[:10]
    matches = df[df["date_str"] == date_iso]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    target = args.date or date.today().isoformat()

    print()
    print("═" * 78)
    print(f"  macminibot OPERATOR EOD SUMMARY — {target}")
    print("═" * 78)

    # 1. Account
    try:
        tc, acct, positions = fetch_alpaca()
    except Exception as exc:
        print(f"  ALPACA FETCH FAILED: {exc}")
        return 1
    equity = float(acct.portfolio_value)
    last_eq = float(acct.last_equity)
    deploy_eq = 100088.35
    day_pl = equity - last_eq
    cum_pl = equity - deploy_eq
    print()
    print("📊 ACCOUNT")
    print(f"   Equity now:       ${equity:>12,.2f}")
    print(f"   Yesterday close:  ${last_eq:>12,.2f}")
    print(f"   Today's P&L:      ${day_pl:>+12,.2f}  ({day_pl/last_eq*100:+.3f}%)")
    print(f"   Cumulative P&L:   ${cum_pl:>+12,.2f}  ({cum_pl/deploy_eq*100:+.3f}% since 2026-05-19)")
    print(f"   Buying power:     ${float(acct.buying_power):>12,.2f}")
    print(f"   Status:           {acct.status.value}")

    # 2. Open positions
    state = read_state()
    print()
    print(f"📈 OPEN POSITIONS ({len(positions)})")
    if not positions:
        print("   (none)")
    for p in positions:
        sym = p.symbol
        side = p.side.value
        entry = float(p.avg_entry_price)
        cur = float(p.current_price)
        mtm = float(p.unrealized_pl)
        st = state.get(sym, {})
        peak = st.get("peak")
        trough = st.get("trough")
        sigma = st.get("entry_sigma_log")
        trail = st.get("trail_sigma")
        pt = st.get("profit_target_sigmas")
        stop_str = "—"
        if sigma and trail and entry:
            band = entry * sigma * trail
            if side == "long" and peak:
                stop_str = f"${peak - band:.2f} (dist {(cur - (peak - band))/cur*100:+.2f}%)"
            elif side == "short" and trough:
                stop_str = f"${trough + band:.2f} (dist {((trough + band) - cur)/cur*100:+.2f}%)"
        pt_str = "—"
        if pt and sigma and entry:
            if side == "long":
                pt_price = entry * (1 + sigma * pt)
                pt_str = f"${pt_price:.2f} (need {(pt_price - cur)/cur*100:+.2f}%)"
            else:
                pt_price = entry * (1 - sigma * pt)
                pt_str = f"${pt_price:.2f} (need {(pt_price - cur)/cur*100:+.2f}%)"
        mtm_emoji = "🟢" if mtm > 0 else "🔴" if mtm < 0 else "⚪"
        print(f"   {mtm_emoji} {sym:<5} {side:<5} qty={p.qty:<4}  "
              f"entry=${entry:.2f}  cur=${cur:.2f}  MTM=${mtm:+,.2f}")
        print(f"          trail-stop @ {stop_str}    PT @ {pt_str}")

    # 3. Today's runner activity
    print()
    print("⚙️  TODAY'S BOT ACTIVITY")
    runner_log = read_log("30_per_asset_run", target, 50)
    if runner_log:
        # Extract key lines
        for line in runner_log:
            if any(k in line for k in ["SUBMITTED", "REJECTED", "APPROVED", "emitted", "WARNING", "ERROR"]):
                print(f"   {line}")
    else:
        print("   (no daily-runner log for today yet)")

    # 4. Intraday exit monitor activity
    print()
    print("⏰ INTRADAY EXIT MONITOR")
    exit_log = read_log("36_intraday_exit", target, 200)
    morning_blocks = sum(1 for ln in exit_log if "MORNING BLOCK" in ln)
    exits = [ln for ln in exit_log if "EXIT " in ln]
    print(f"   Morning-block (09:30-10:00 ET) fires: {morning_blocks}  ← time-of-day filter activity")
    print(f"   Exits triggered today: {len(exits)}")
    for ln in exits[-10:]:
        print(f"     {ln}")

    # 5. Benchmark journal today
    print()
    print("🏁 BENCHMARK COMPARISON (cumulative P&L since deploy)")
    bm = read_benchmark_today(target)
    if bm is not None:
        bot_cum = bm.get("bot_cum_pnl", 0)
        bh_cum = bm.get("bh_cum_pnl", 0)
        us2_cum = bm.get("unifstop_2sigma_cum_pnl", 0)
        print(f"   Bot:              ${bot_cum:>+12,.2f}")
        print(f"   Buy & Hold:       ${bh_cum:>+12,.2f}  α=${bot_cum - bh_cum:+,.2f}")
        print(f"   UnifStop-2σ:      ${us2_cum:>+12,.2f}  α=${bot_cum - us2_cum:+,.2f}")
    else:
        print(f"   (no benchmark row for {target} yet — logger fires at 21:30 BST)")

    # 6. Halt audit
    print()
    print("🚨 HALT EVENTS TODAY")
    halt_log = PROJECT_ROOT / "state" / "halt_audit.log"
    if halt_log.exists():
        lines = halt_log.read_text().strip().splitlines()
        today_events = []
        for ln in lines:
            try:
                rec = json.loads(ln)
                ts = datetime.fromtimestamp(rec["ts_unix"], timezone.utc)
                if ts.date().isoformat() == target:
                    today_events.append((ts.strftime("%H:%M"), rec["action"], rec["reason"]))
            except Exception:
                pass
        if not today_events:
            print(f"   ✓ no halt events today")
        for t, act, rsn in today_events:
            print(f"   {t} UTC  {act}  — {rsn}")

    # 7. Tomorrow forecast
    print()
    print("🔮 TOMORROW")
    print(f"   per-asset-runner fires:    Mon-Fri 14:35 BST")
    print(f"   intraday-exit-monitor:     every 5 min during US session")
    print(f"     - exits BLOCKED 09:30-10:00 ET (= 14:30-15:00 BST)")
    print(f"   benchmark-logger fires:    21:30 BST")

    print()
    print("═" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
