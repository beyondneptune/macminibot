# macminibot Operator Runbook

**Last updated:** 2026-05-21 (autonomous mode iteration 4)
**Active deployment:** v2 daily + time-of-day exit filter
**Account:** Alpaca paper, $100k seed

---

## Quick status check (run this first when you return)

```bash
.venv/bin/python /Users/weeoto/macminibot/scripts/46_operator_eod_summary.py
```

Single-page view of: account equity, day P&L, open positions with stop/PT distances, today's runner+exit-monitor activity, benchmark journal, halt events.

## What's deployed and live

| job | schedule | purpose |
|---|---|---|
| `com.macminibot.per-asset-runner` | Mon-Fri 14:35 BST | Daily fire — evaluates 8 assets for new entries |
| `com.macminibot.intraday-exit-monitor` | every 5 min during US session | Closes positions on trail-stop/PT hits **EXCEPT during 09:30-10:00 ET block** |
| `com.macminibot.benchmark-logger` | Mon-Fri 21:30 BST | Logs bot vs B&H vs UnifStop-{1,2,3}σ daily |
| `com.macminibot.auto-halt` | every 5 min | Reconciles broker vs journal; halts on mismatch |
| `com.macminibot.heartbeat` | continuous | Liveness signal for kill-switch |
| `com.macminibot.equity-recorder` | every 5 min | NAV snapshots to parquet |
| `com.macminibot.treasury-recorder` | every 5 min | Treasury yield snapshots |
| `com.macminibot.nightly-tick-pull` | daily | Refresh ArcticDB bars |
| `com.macminibot.eod-summary` | Mon-Fri 21:30 BST | Daily summary file |
| `com.macminibot.security-check` | daily | Cert/env validation |
| `com.macminibot.dashboard` | continuous | Streamlit on port 8501 |

## Per-asset configuration

| asset | signal | trail σ | profit target | source |
|---|---|---|---|---|
| AAPL  | buy_and_hold     | 3.0σ | 3.0σ | daily v2 (comprehensive search) |
| MSFT  | rsi2_mr_10_90    | 1.0σ | 5.0σ | daily v2 |
| GOOGL | buy_and_hold     | 2.0σ | —    | bake-off original |
| AMZN  | buy_and_hold     | 0.5σ | 5.0σ | daily v2 |
| META  | lgbm             | 2.0σ | —    | bake-off original |
| TSLA  | macd_signal      | 0.5σ | 3.0σ | daily v2 |
| NVDA  | buy_and_hold     | 2.0σ | —    | bake-off original |
| GLD   | buy_and_hold     | 2.0σ | 3.0σ | daily v2 |

Per-name notional: $12,500 ($100k / 8).
Cost model: ~3 bp per side realistic.

## What was tested and rejected

| variant | OOS test | verdict |
|---|---|---|
| Widen AMZN/TSLA trails (0.5σ → 2σ) | −$7,016 vs baseline | rejected |
| Profit-lock at +1σ before trailing | −$1,723 vs baseline | rejected |
| Multi-position pyramiding (5×) | regime-dependent, marginal | rejected |
| Kelly weighted sizing | overfit, COVID worse | rejected |
| Naive long basket (no signals) | −$23k vs v2 | confirmed v2 |
| Fixed TP/SL (1%/0.5%, 2:1 ratio) | −$30k vs baseline | catastrophic |
| 3-strike SL | marginal, worse than N=1 | rejected |
| Intraday Bollinger (AAPL/MSFT 5-min) | costed-to-zero by spread | killed |
| Leveraged ETF intraday | severe train→test slippage | killed |
| **Time-of-day filter (09:30-10:00 ET)** | **+$7,995 / +8% on 2-year OOS** | **DEPLOYED** |

## v3 Leveraged Sleeve — ready but NOT live

Infrastructure built, awaiting your activation decision. See `memory/project_v3_leveraged_research.md`.

**To activate:**

1. Decide sizing approach:
   - **Path A (conservative):** REPLACE 3 v2 drags (AMZN/GOOGL/TSLA) with SOXL/NVDL/TSLL → stays at $100k total
   - **Path B (aggressive):** ADD 3 leveraged names alongside v2 8 → $160k deployed = 1.6× margin

2. Wire risk chain in `scripts/37_leveraged_sleeve_runner.py` (real submission path NOT yet connected — intentional)

3. Live paper validate at ¼ size ($5k/name) for 10+ trading days

4. Create + bootstrap `com.macminibot.leveraged-sleeve-runner.plist`

5. Update `memory/project_daily_v2_deployment.md` to reference v3 sleeve

**Expected returns:**
- Path A: similar to v2 (~+0.2%/day) with higher per-name volatility
- Path B: ~+0.5%/day combined but with ~−10% account DD in stress

**Reality check:** Even Path B does NOT hit the 1-3%/day target. That requires options or different asset class — not currently researched.

## When something goes wrong

### Bot is HALTED
Check halt audit:
```bash
tail -20 /Users/weeoto/macminibot/state/halt_audit.log
```

Common causes:
- `BROKER_ACCOUNT_INACTIVE` — market closed overnight, will auto-recover
- `POSITION_RECONCILE` — broker positions ≠ journal; investigate before unhalting

Unhalt:
```bash
PYTHONPATH=/Users/weeoto/macminibot/src .venv/bin/python -c "
from risk_management.kill_switch import unhalt
unhalt('manual unhalt — <REASON>')
"
```

### Bot positions diverged from state file
Edit `data/per_asset_state.json` to match broker positions (use Alpaca dashboard as source of truth).

### Need to STOP the bot entirely
```bash
launchctl bootout gui/$UID/com.macminibot.per-asset-runner
launchctl bootout gui/$UID/com.macminibot.intraday-exit-monitor
```
Positions stay open until you close them manually on Alpaca.

### Need to RESTART the bot
```bash
launchctl bootstrap gui/$UID /Users/weeoto/macminibot/scripts/launchd/com.macminibot.per-asset-runner.plist
launchctl bootstrap gui/$UID /Users/weeoto/macminibot/scripts/launchd/com.macminibot.intraday-exit-monitor.plist
```

### Need to dry-run without submitting orders
**SAFE** — `--force-window` now auto-enables WHATIF unless `--i-mean-it` is passed:
```bash
.venv/bin/python /Users/weeoto/macminibot/scripts/30_per_asset_runner.py --force-window
.venv/bin/python /Users/weeoto/macminibot/scripts/36_intraday_exit_monitor.py --force-window
```

## Realistic expectations

- **Daily mean:** +0.2%/day net after costs
- **Daily std:** 0.8%/day
- **Annualized:** ~+50-60% gross, ~+20-30% honest forward expectation
- **67% of days positive**
- **Max DD (no crisis):** −3-5%
- **Crisis DD (2022-style):** −7 to −12%
- **Statistical signal:** need 30-60 trading days before live data is meaningful

**The 2-3%/day target is NOT reachable on this universe** at current Sharpe regardless of strategy variant. Confirmed by 11 agent investigations and >100 backtest cells.

## Pre-committed decision thresholds (added 2026-05-22 after 5-bot debate)

We will NOT change v2 config based on pattern recognition or short-window losses. Changes trigger ONLY when these tripwires breach:

| metric | threshold | source |
|---|---|---|
| Day 10 cumulative P&L | < −$700 | 5th percentile of OOS Monte Carlo |
| Day 15 cumulative P&L | < −$1,000 | 5th percentile |
| Day 21 cumulative P&L | < −$1,500 | 5th percentile |
| Drawdown from peak | ≥ −$3,500 | 5th percentile DD |

Currently at Day 5: cumulative −$244 = 20th percentile = normal noise band.

The risk-manager "halve AMZN/TSLA notional to $6,250" change was REJECTED because halving on a drawdown ratchets — once cut, never confidently restored. Holding v2 untouched is the empirically-correct call until a pre-commit threshold breaches.

## Honest reading of current state

The bot is at the empirical ceiling for daily-bar US large-cap trading. Every plausible strategy variant has been tested. The architecture earns its keep specifically in DRAWDOWN windows (saved $24k in 2022 RATESHOCK vs B&H's −$32k). In bull markets it slightly underperforms naive long.

Paths to genuinely higher returns require structural changes:
1. Options overlay (uncapped downside but capped upside; needs new research cycle)
2. Crypto / FX (different asset class, 24/7 trading)
3. Higher leverage at the broker level (Alpaca caps at 4× intraday)

The v3 leveraged ETF sleeve is the closest near-term path and is ready to deploy.
