"""70_replay_simulation — animated bar-by-bar replay of the per-asset
trailing-stop state machine. Generates an interactive HTML per asset (and a
combined index page) with a Play button + slider that scrubs through the
last N daily bars, revealing the price path and BUY / SELL markers as the
state machine would have triggered them in real time.

Each asset also gets a realized P&L breakdown: per-round-trip $ P&L,
% return, holding-period, exit reason, and running cumulative P&L (sized
at the production runner's allocation: net_liq / portfolio_size per name).

Run:
    .venv/bin/python scripts/70_replay_simulation.py [--lookback-days N]
                                                     [--net-liq N]

Output:
    data/replay/<ASSET>.html   — per-asset animated chart + trade table
    data/replay/index.html     — landing page with portfolio P&L summary
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from ml_research.asset_specialists import (  # noqa: E402
    DEPLOYABLE_ASSETS,
    compute_signal_for,
    winner_for,
)
from ml_research.asset_specialists._signals import add_per_asset_features  # noqa: E402
from strategy_engine.per_asset_runner import default_bar_loader  # noqa: E402


def _latest_sigma(feat: pd.DataFrame) -> float | None:
    if "vol_21" not in feat.columns or feat.empty:
        return None
    v = feat["vol_21"].iloc[-1]
    if not math.isfinite(v) or v <= 0:
        return None
    return float(v)


def replay_one(asset: str, lookback_days: int,
               notional_per_position: float) -> dict:
    """Bar-by-bar replay. Returns the plot_window slice + per-bar series
    (close, stop_band) + the trade list with ts/side/price/reason, plus a
    paired round-trip list with realized P&L."""
    winner = winner_for(asset)
    if winner is None:
        return {"asset": asset, "skipped": "no_winner"}

    trail_sigma_w = float(winner["trail_sigma"])
    bars = default_bar_loader(asset)
    if len(bars) < 30:
        return {"asset": asset, "skipped": "insufficient_bars"}

    warmup = 252
    start_idx = max(0, len(bars) - (lookback_days + warmup))
    window = bars.iloc[start_idx:].copy()
    plot_start_idx = max(warmup, len(window) - lookback_days)

    state = "FLAT"
    entry_price = None
    entry_sigma = None
    entry_ts = None
    entry_qty = 0
    peak = None
    trades: list[dict] = []
    round_trips: list[dict] = []
    stop_band: dict[pd.Timestamp, float] = {}

    for i in range(warmup, len(window)):
        bars_so_far = window.iloc[: i + 1]
        feat = add_per_asset_features(bars_so_far)
        sigma = _latest_sigma(feat)
        if sigma is None:
            continue
        try:
            sig = compute_signal_for(asset, bars_so_far)
        except Exception:
            continue
        sig_dir = int(round(float(sig)))
        close = float(window["close"].iloc[i])
        ts = window.index[i]

        if state == "IN_LONG":
            if peak is None or close > peak:
                peak = close
            band = entry_price * entry_sigma * trail_sigma_w
            stop_price = peak - band
            stop_band[ts] = stop_price
            exit_reason = None
            if close <= stop_price:
                exit_reason = "trailing_stop"
            elif sig_dir < 0:
                exit_reason = "signal_flip"
            if exit_reason is not None:
                pnl_dollar = (close - entry_price) * entry_qty
                pnl_pct = (close / entry_price - 1.0) * 100.0
                hold_days = (ts - entry_ts).days if entry_ts else 0
                trades.append({"ts": ts, "side": "SELL", "price": close,
                               "reason": exit_reason, "qty": entry_qty,
                               "pnl": pnl_dollar, "pnl_pct": pnl_pct,
                               "hold_days": hold_days,
                               "entry_ts": entry_ts, "entry_price": entry_price})
                round_trips.append({
                    "entry_ts": entry_ts, "entry_price": entry_price,
                    "exit_ts": ts, "exit_price": close, "qty": entry_qty,
                    "hold_days": hold_days, "reason": exit_reason,
                    "pnl": pnl_dollar, "pnl_pct": pnl_pct,
                })
                state = "FLAT"
                entry_price = entry_sigma = peak = entry_ts = None
                entry_qty = 0
                continue

        if state == "FLAT" and sig_dir > 0:
            qty = int(notional_per_position // close) if close > 0 else 0
            if qty <= 0:
                continue
            trades.append({"ts": ts, "side": "BUY", "price": close,
                           "reason": "entry", "qty": qty})
            state = "IN_LONG"
            entry_price = close
            entry_sigma = sigma
            entry_ts = ts
            entry_qty = qty
            peak = close
            stop_band[ts] = close - close * sigma * trail_sigma_w

    plot_window = window.iloc[plot_start_idx:].copy()
    plot_window = plot_window.assign(
        stop_band=plot_window.index.map(lambda t: stop_band.get(t, None))
    )
    trades_in_window = [t for t in trades if t["ts"] in plot_window.index]
    round_trips_in_window = [
        rt for rt in round_trips if rt["exit_ts"] in plot_window.index
    ]
    open_unrealized = 0.0
    open_qty = 0
    if state == "IN_LONG" and entry_price is not None:
        last_close = float(plot_window["close"].iloc[-1])
        open_unrealized = (last_close - entry_price) * entry_qty
        open_qty = entry_qty

    realized = sum(rt["pnl"] for rt in round_trips_in_window)
    wins = [rt for rt in round_trips_in_window if rt["pnl"] > 0]
    losses = [rt for rt in round_trips_in_window if rt["pnl"] <= 0]

    return {
        "asset": asset,
        "winner": winner,
        "trail_sigma": trail_sigma_w,
        "plot_window": plot_window,
        "trades": trades_in_window,
        "round_trips": round_trips_in_window,
        "realized_pnl": realized,
        "open_unrealized": open_unrealized,
        "open_qty": open_qty,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": (len(wins) / len(round_trips_in_window)) if round_trips_in_window else 0.0,
        "gross_profit": sum(rt["pnl"] for rt in wins),
        "gross_loss": sum(rt["pnl"] for rt in losses),
        "all_trades_count": len(trades),
        "notional_per_position": notional_per_position,
    }


def build_animated_figure(result: dict) -> go.Figure:
    """One animated Plotly figure: price reveals bar-by-bar, BUY/SELL
    markers pop in when their bar arrives, trailing-stop band tracks
    the active position."""
    asset = result["asset"]
    win = result["plot_window"]
    trades = result["trades"]
    winner = result["winner"]
    trail = result["trail_sigma"]

    xs = list(win.index)
    closes = win["close"].astype(float).tolist()
    stops = win["stop_band"].astype(float).tolist()  # NaN where no position

    trade_by_ts: dict[pd.Timestamp, dict] = {t["ts"]: t for t in trades}

    # One frame per bar. Each frame draws price[0..i], stop[0..i], and the
    # cumulative set of buys/sells whose timestamps are <= xs[i].
    frames = []
    for i in range(len(xs)):
        ts_cut = xs[i]
        cum_buys = [t for t in trades if t["ts"] <= ts_cut and t["side"] == "BUY"]
        cum_sells = [t for t in trades if t["ts"] <= ts_cut and t["side"] == "SELL"]
        frame_traces = [
            go.Scatter(x=xs[: i + 1], y=closes[: i + 1], mode="lines",
                       line=dict(color="#222", width=1.4), name="close"),
            go.Scatter(x=xs[: i + 1], y=stops[: i + 1], mode="lines",
                       line=dict(color="#cc0000", width=1.0, dash="dash"),
                       name=f"trailing stop ({trail}σ)", connectgaps=False),
            go.Scatter(
                x=[t["ts"] for t in cum_buys], y=[t["price"] for t in cum_buys],
                mode="markers", marker=dict(symbol="triangle-up", size=14,
                                            color="#1a9850",
                                            line=dict(color="white", width=1)),
                name="BUY", hovertemplate="BUY %{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
            ),
            go.Scatter(
                x=[t["ts"] for t in cum_sells], y=[t["price"] for t in cum_sells],
                mode="markers", marker=dict(symbol="triangle-down", size=14,
                                            color="#d73027",
                                            line=dict(color="white", width=1)),
                name="SELL",
                hovertemplate=("SELL %{x|%Y-%m-%d}<br>$%{y:.2f}"
                               "<br>%{customdata}<extra></extra>"),
                customdata=[
                    f"{t['reason']}<br>qty={t.get('qty',0)} "
                    f"P&L=${t.get('pnl',0):,.2f} ({t.get('pnl_pct',0):+.2f}%)"
                    f"<br>held {t.get('hold_days',0)}d"
                    for t in cum_sells
                ],
            ),
        ]
        # Banner: today's event (if any) + running P&L.
        ev = trade_by_ts.get(ts_cut)
        cum_pnl = sum(t.get("pnl", 0.0) for t in cum_sells)
        ev_txt = ""
        if ev is not None:
            if ev["side"] == "SELL":
                ev_txt = (f"  ←  SELL {ev['reason']} @ ${ev['price']:.2f}  "
                          f"P&L=${ev.get('pnl',0):,.2f} ({ev.get('pnl_pct',0):+.2f}%)")
            else:
                ev_txt = f"  ←  BUY @ ${ev['price']:.2f} (qty={ev.get('qty',0)})"
        pnl_color = "#1a9850" if cum_pnl >= 0 else "#d73027"
        ann_text = (
            f"<b>{asset}</b>  {winner['model']} / trail={trail}σ  •  "
            f"bar {i+1}/{len(xs)}  •  {ts_cut.strftime('%Y-%m-%d')}  •  "
            f"BUY={len(cum_buys)} SELL={len(cum_sells)}  •  "
            f"<span style='color:{pnl_color}'>realized P&L=${cum_pnl:,.2f}</span>"
            f"{ev_txt}"
        )
        frames.append(go.Frame(
            data=frame_traces,
            name=str(i),
            layout=go.Layout(annotations=[dict(
                text=ann_text, x=0.5, xref="paper", y=1.04, yref="paper",
                showarrow=False, font=dict(size=12),
            )]),
        ))

    # Initial / final-state traces (so the chart isn't blank before Play).
    base_traces = frames[-1].data

    # Slider — one step per frame, every Kth labelled to avoid clutter.
    label_every = max(1, len(xs) // 20)
    steps = [
        dict(method="animate",
             args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                  transition=dict(duration=0))],
             label=xs[i].strftime("%Y-%m-%d") if i % label_every == 0 else "")
        for i in range(len(xs))
    ]

    fig = go.Figure(
        data=list(base_traces),
        frames=frames,
        layout=go.Layout(
            template="plotly_white",
            title=dict(
                text=(f"{asset} — {winner['model']} / trail={trail}σ "
                      f"<br><span style='font-size:11px;color:#666'>"
                      f"Per-asset trailing-stop replay • last {len(xs)} trading days "
                      f"• Press ▶ to animate</span>"),
                x=0.5, xanchor="center",
            ),
            xaxis=dict(title="date", range=[xs[0], xs[-1]], type="date"),
            yaxis=dict(title="price ($)"),
            hovermode="x unified",
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
            updatemenus=[dict(
                type="buttons", direction="left", x=0.05, y=-0.07, xanchor="left",
                showactive=False, pad=dict(t=4, r=4),
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=80, redraw=True),
                                          fromcurrent=True,
                                          transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate",
                                            transition=dict(duration=0))]),
                    dict(label="⏮ Reset", method="animate",
                         args=[["0"], dict(frame=dict(duration=0, redraw=True),
                                           mode="immediate",
                                           transition=dict(duration=0))]),
                ],
            )],
            sliders=[dict(
                active=len(xs) - 1, x=0.18, len=0.78, y=-0.05,
                pad=dict(t=20),
                currentvalue=dict(prefix="bar: ", visible=True, xanchor="left"),
                steps=steps,
            )],
            margin=dict(l=60, r=30, t=70, b=110),
            height=560,
        ),
    )
    return fig


def _pnl_color(v: float) -> str:
    return "#1a9850" if v >= 0 else "#d73027"


def render_asset_page(result: dict, fig: go.Figure, out_dir: Path) -> Path:
    """Embed the animated figure + a per-round-trip P&L table on one page."""
    asset = result["asset"]
    rts = result["round_trips"]
    win = result["plot_window"]

    rows = []
    running = 0.0
    for rt in rts:
        running += rt["pnl"]
        rows.append(
            f'<tr>'
            f'<td>{rt["entry_ts"].strftime("%Y-%m-%d")}</td>'
            f'<td>${rt["entry_price"]:.2f}</td>'
            f'<td>{rt["exit_ts"].strftime("%Y-%m-%d")}</td>'
            f'<td>${rt["exit_price"]:.2f}</td>'
            f'<td>{rt["qty"]}</td>'
            f'<td>{rt["hold_days"]}d</td>'
            f'<td>{rt["reason"]}</td>'
            f'<td style="color:{_pnl_color(rt["pnl"])};text-align:right">'
            f'${rt["pnl"]:,.2f}</td>'
            f'<td style="color:{_pnl_color(rt["pnl_pct"])};text-align:right">'
            f'{rt["pnl_pct"]:+.2f}%</td>'
            f'<td style="color:{_pnl_color(running)};text-align:right">'
            f'${running:,.2f}</td>'
            f'</tr>'
        )
    if result["open_qty"] > 0:
        last_close = float(win["close"].iloc[-1])
        last_bar = win.index[-1].strftime("%Y-%m-%d")
        rows.append(
            f'<tr style="background:#fff8e1">'
            f'<td colspan="6"><i>OPEN POSITION (mark-to-market on {last_bar} '
            f'@ ${last_close:.2f})</i></td>'
            f'<td>—</td>'
            f'<td style="color:{_pnl_color(result["open_unrealized"])};text-align:right">'
            f'${result["open_unrealized"]:,.2f}</td>'
            f'<td>—</td>'
            f'<td style="color:{_pnl_color(running + result["open_unrealized"])};text-align:right">'
            f'${running + result["open_unrealized"]:,.2f}</td>'
            f'</tr>'
        )

    net = result["realized_pnl"] + result["open_unrealized"]
    notional = result["notional_per_position"]
    return_pct = (net / notional * 100.0) if notional else 0.0

    chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False,
                             auto_play=False, config={"displayModeBar": False})

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>{asset} — trailing-stop replay</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro", sans-serif;
          max-width: 1100px; margin: 30px auto; padding: 0 20px; color: #222; }}
  h2 {{ font-size: 18px; margin-top: 32px; }}
  .pnl-card {{ display: flex; gap: 24px; margin: 12px 0 4px 0; }}
  .pnl-card .box {{ background: #fafafa; padding: 10px 16px; border-radius: 6px;
                    min-width: 120px; }}
  .pnl-card .box .label {{ font-size: 11px; color: #666; text-transform: uppercase;
                           letter-spacing: 0.04em; }}
  .pnl-card .box .val {{ font-size: 18px; font-weight: 600; margin-top: 2px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  a {{ color: #0366d6; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
<p><a href="index.html">← back to all assets</a></p>
{chart_html}
<h2>Realized P&L breakdown</h2>
<div class="pnl-card">
  <div class="box"><div class="label">Round trips</div>
       <div class="val">{len(rts)}</div></div>
  <div class="box"><div class="label">Win rate</div>
       <div class="val">{result["win_rate"]*100:.1f}%
         <small style="font-size:11px;color:#888">
           ({result["win_count"]}W / {result["loss_count"]}L)
         </small>
       </div></div>
  <div class="box"><div class="label">Realized P&L</div>
       <div class="val" style="color:{_pnl_color(result["realized_pnl"])}">
         ${result["realized_pnl"]:,.2f}</div></div>
  <div class="box"><div class="label">Unrealized (open)</div>
       <div class="val" style="color:{_pnl_color(result["open_unrealized"])}">
         ${result["open_unrealized"]:,.2f}</div></div>
  <div class="box" style="background:#eef7e8">
       <div class="label">Net P&L</div>
       <div class="val" style="color:{_pnl_color(net)}">${net:,.2f}
         <small style="font-size:12px;color:#666">({return_pct:+.2f}% on ${notional:,.0f})</small>
       </div></div>
</div>
<table>
  <tr><th>entry date</th><th>entry $</th><th>exit date</th><th>exit $</th>
      <th>qty</th><th>held</th><th>reason</th>
      <th style="text-align:right">$ P&L</th><th style="text-align:right">% return</th>
      <th style="text-align:right">cum P&L</th></tr>
  {"".join(rows) if rows else '<tr><td colspan="10" style="color:#999"><i>No round trips yet — position may be open or no entries fired.</i></td></tr>'}
</table>
"""
    p = out_dir / f"{asset}.html"
    p.write_text(html)
    return p


def render_index(results: list[dict], out_dir: Path,
                 notional_per_position: float) -> Path:
    rows = []
    total_realized = 0.0
    total_open = 0.0
    total_wins = 0
    total_losses = 0
    deployed_count = 0
    for r in results:
        if "skipped" in r:
            rows.append(f'<tr><td><b>{r["asset"]}</b></td><td colspan="9">'
                        f'<i>skipped: {r["skipped"]}</i></td></tr>')
            continue
        deployed_count += 1
        w = r["winner"]
        win = r["plot_window"]
        last = float(win["close"].iloc[-1])
        last_bar = win.index[-1].strftime("%Y-%m-%d")
        net = r["realized_pnl"] + r["open_unrealized"]
        total_realized += r["realized_pnl"]
        total_open += r["open_unrealized"]
        total_wins += r["win_count"]
        total_losses += r["loss_count"]
        rows.append(
            f'<tr>'
            f'<td><b><a href="{r["asset"]}.html">{r["asset"]}</a></b></td>'
            f'<td>{w["model"]}</td><td>{r["trail_sigma"]}σ</td>'
            f'<td style="text-align:right">{r["win_count"] + r["loss_count"]}</td>'
            f'<td style="text-align:right">'
            f'{r["win_rate"]*100:.1f}% <small style="color:#888">'
            f'({r["win_count"]}W/{r["loss_count"]}L)</small></td>'
            f'<td style="text-align:right;color:{_pnl_color(r["realized_pnl"])}">'
            f'${r["realized_pnl"]:,.2f}</td>'
            f'<td style="text-align:right;color:{_pnl_color(r["open_unrealized"])}">'
            f'${r["open_unrealized"]:,.2f}</td>'
            f'<td style="text-align:right;font-weight:600;color:{_pnl_color(net)}">'
            f'${net:,.2f}</td>'
            f'<td style="text-align:right;color:{_pnl_color(net)}">'
            f'{net/notional_per_position*100:+.2f}%</td>'
            f'<td>${last:.2f} <small>({last_bar})</small></td>'
            f'</tr>'
        )
    portfolio_net = total_realized + total_open
    portfolio_notional = deployed_count * notional_per_position
    portfolio_return = (portfolio_net / portfolio_notional * 100.0) if portfolio_notional else 0.0
    total_trades = total_wins + total_losses
    portfolio_winrate = (total_wins / total_trades * 100.0) if total_trades else 0.0

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>macminibot — trailing-stop replay</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro", sans-serif;
          max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ font-size: 22px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #fafafa; font-weight: 600; }}
  tr.total {{ background: #eef7e8; font-weight: 600; }}
  a {{ color: #0366d6; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .note {{ background: #fff8e1; padding: 12px 14px; border-left: 4px solid #ffb300;
           font-size: 13px; line-height: 1.5; }}
  .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
  .summary .box {{ background: #fafafa; padding: 14px 20px; border-radius: 6px;
                   flex: 1; }}
  .summary .label {{ font-size: 11px; color: #666; text-transform: uppercase;
                     letter-spacing: 0.04em; }}
  .summary .val {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
</style>
<h1>macminibot — per-asset trailing-stop replay</h1>
<p class="note">Each row links to an animated chart (press <b>▶ Play</b> to scrub
through the last 252 trading days). P&L below is <b>realized</b> from closed
round trips plus mark-to-market on any open position, sized at the production
runner's per-name allocation of <b>${notional_per_position:,.0f}</b>
(net_liq / portfolio_size). This is the deployed state machine running on
historical bars — <i>not</i> a backtest with statistical promotion gates
(those live in <code>data/bakeoff_variants/FINAL_ANALYSIS.md</code>).</p>
<div class="summary">
  <div class="box"><div class="label">Portfolio net P&L</div>
    <div class="val" style="color:{_pnl_color(portfolio_net)}">${portfolio_net:,.2f}</div>
    <small style="color:#666">{portfolio_return:+.2f}% on ${portfolio_notional:,.0f} deployed</small>
  </div>
  <div class="box"><div class="label">Realized</div>
    <div class="val" style="color:{_pnl_color(total_realized)}">${total_realized:,.2f}</div>
  </div>
  <div class="box"><div class="label">Unrealized (open)</div>
    <div class="val" style="color:{_pnl_color(total_open)}">${total_open:,.2f}</div>
  </div>
  <div class="box"><div class="label">Round trips</div>
    <div class="val">{total_trades}</div>
    <small style="color:#666">{portfolio_winrate:.1f}% win rate ({total_wins}W / {total_losses}L)</small>
  </div>
</div>
<table>
  <tr><th>asset</th><th>model</th><th>trail</th>
      <th style="text-align:right">round trips</th>
      <th style="text-align:right">win rate</th>
      <th style="text-align:right">realized $</th>
      <th style="text-align:right">unrealized $</th>
      <th style="text-align:right">net $</th>
      <th style="text-align:right">return %</th>
      <th>last close</th></tr>
  {"".join(rows)}
</table>
"""
    p = out_dir / "index.html"
    p.write_text(html)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=252)
    ap.add_argument("--net-liq", type=float, default=100000.0,
                    help="Account net liquidation (default 100k); per-position "
                         "notional = net_liq / 7.")
    args = ap.parse_args()

    notional = args.net_liq / 7.0  # matches PerAssetRunnerConfig.portfolio_size
    out_dir = PROJECT_ROOT / "data" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Replay window = last {args.lookback_days} trading days. Output → {out_dir}")
    print(f"Net liq = ${args.net_liq:,.0f}  →  per-position notional = ${notional:,.2f}")
    print("=" * 92)
    print(f"{'asset':<6} {'model':<20} {'trail':<6} {'trades':<7} "
          f"{'winrate':<8} {'realized':<14} {'open':<13} {'net':<14}")
    print("-" * 92)
    results = []
    for asset in DEPLOYABLE_ASSETS:
        try:
            r = replay_one(asset, args.lookback_days, notional)
        except Exception as exc:
            print(f"{asset:<6} ERROR: {type(exc).__name__}: {exc}")
            continue
        results.append(r)
        if "skipped" in r:
            print(f"{asset:<6} SKIPPED: {r['skipped']}")
            continue
        fig = build_animated_figure(r)
        out = render_asset_page(r, fig, out_dir)
        n_rt = r["win_count"] + r["loss_count"]
        net = r["realized_pnl"] + r["open_unrealized"]
        print(f"  {r['asset']:<5} {r['winner']['model']:<20} "
              f"{r['trail_sigma']}σ{'':<4}{n_rt:<7} "
              f"{r['win_rate']*100:>5.1f}%  "
              f"${r['realized_pnl']:>11,.2f}  "
              f"${r['open_unrealized']:>10,.2f}  "
              f"${net:>11,.2f}  → {out.name}")

    print("-" * 92)
    total_realized = sum(r["realized_pnl"] for r in results if "skipped" not in r)
    total_open = sum(r["open_unrealized"] for r in results if "skipped" not in r)
    n_deployed = sum(1 for r in results if "skipped" not in r)
    net_total = total_realized + total_open
    portfolio_notional = n_deployed * notional
    print(f"{'PORTFOLIO':<6} {'(7 assets)':<20} {'':<6} {'':<7} {'':<8} "
          f"${total_realized:>11,.2f}  ${total_open:>10,.2f}  ${net_total:>11,.2f}")
    print(f"  return on ${portfolio_notional:,.0f} deployed: "
          f"{net_total/portfolio_notional*100:+.2f}%")
    print("=" * 92)
    idx = render_index(results, out_dir, notional)
    print(f"\nindex → {idx}")
    print(f"open in browser:\n  open {idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
