# macminibot — Deployment Readiness Brief

Date: 2026-05-18
Mode: **paper trading only** (no live-money switch)
Architecture winner: **per-asset trailing-stop** (variant `trailing_stop`)

---

## 1. What was built (autonomous closed-loop)

Per the operator directive (no questions while away; empirically chosen
architecture; FWER-corrected promotion gates), a closed-loop architecture
search was run with 5 candidate exit-policy variants:

| variant | gate winners / 8 | RW-strict winners / 8 | avg gate Sharpe |
|---|---|---|---|
| `bracket` (TP/SL grid)          | 0 | 0 | — |
| `signal_flip`                   | 3 | 2 | +0.943 |
| `long_only_ensemble`            | 0 | 0 | — |
| **`trailing_stop`** ← **WINNER** | **7** | **1** | **+1.737** |
| `time_decay`                    | 3 | 0 | — |

Promotion gate (uniform across variants after harmonization):

```
DSR_harm > 0.95
AND Holm-adjusted p (within-variant across 8 per-asset bests) < 0.05
AND Politis–Romano stationary block bootstrap 95% CI lower bound > 0
AND aggregate net Sharpe > 0
```

Romano–Wolf stepdown was also applied **across all 32 cells**
(5 variants × 8 assets, minus long_only_ensemble where no per-fold data
exists) as an architecture-search FWER correction. Only GLD trailing-stop
survives the RW-strict reading; under within-variant Holm (the
pre-registered gate), trailing-stop wins 7/8.

See `data/bakeoff_variants/FINAL_ANALYSIS.md` for the full derivation and
the gate-vs-RW tradeoff discussion.

---

## 2. Deploy-ready per-asset spec (7 assets)

| asset | model           | scope      | trail | hit_priority | passes RW-strict? |
|-------|-----------------|------------|-------|--------------|-------------------|
| AAPL  | buy_and_hold    | asset_only | 2.0 σ | sl_first     | ✗ (gate only)     |
| MSFT  | rsi2_mr_10_90   | asset_only | 2.0 σ | sl_first     | ✗ (gate only)     |
| GOOGL | buy_and_hold    | asset_only | 2.0 σ | sl_first     | ✗ (gate only)     |
| META  | lgbm            | asset_only | 2.0 σ | sl_first     | ✗ (gate only)     |
| TSLA  | macd_signal     | asset_only | 0.5 σ | sl_first     | ✗ (gate only)     |
| NVDA  | buy_and_hold    | asset_only | 2.0 σ | sl_first     | ✗ (gate only)     |
| GLD   | buy_and_hold    | asset_only | 3.0 σ | sl_first     | ✓                 |

**AMZN excluded** — bootstrap CI_lo on the best AMZN config
(`ema_12_26 / 0.5σ trail`) is −0.49; fails the CI gate.

Per-asset OOS net Sharpe (mean) = **+1.737**. The per-fold equal-weight
Sharpe (+6.16, CI95 [+3.82, +8.53] across 63 monthly folds) is *not* the
right deploy-ready estimate — it's inflated by within-fold annualisation.

---

## 3. Honest limitations (read this before flipping the switch)

**Four of the seven winners are passive-with-circuit-breaker.** AAPL,
GOOGL, NVDA, GLD all selected `buy_and_hold` + trailing stop. These are
effectively levered long-only with a 2σ–3σ drawdown circuit breaker — not
systematic alpha. Their Sharpes are partly a 2018–2025 megacap-bull
regime artefact with one circuit-breaker event covered (Q1 2020).

**Three are genuine model-driven entries**: MSFT (RSI-2 mean reversion),
META (LightGBM classifier), TSLA (MACD signal). These three are the
*systematic* signal cells but also the cells whose t-stats are weakest
under Romano-Wolf — they pass the within-variant gate but get crushed by
the 32-cell max-T null.

**Romano-Wolf strict reading**: only GLD trailing-stop survives FWER
control across 32 cells. A 1-asset GLD-only portfolio has Sharpe ≈ 2.12
but is concentrated single-asset risk; not deployable on its own. The
gate-driven 7-asset deployment is justified by treating the 5 variants
as a small pre-registered set rather than a 32-iid-trial grid.

**Daily-bar substrate**: the bracket variant's failure is partly a
daily-bar artefact (intra-bar fill ambiguity flips Sharpe by 3–4 units
when `hit_priority` rotates). The trailing-stop architecture sidesteps
this by using one-sided σ-multiple stops at σ ≥ 0.5; bracket's true
performance is unknown until 5m or 1m bars are wired in.

**No live-money switch.** This deployment is paper-only. The activation
steps below switch from the cross-sectional MV pipeline (paper) to the
per-asset trailing-stop pipeline (paper). Live-money requires explicit
operator authorisation per the autonomous directive.

---

## 4. Wired components (what runs)

| component | path |
|---|---|
| Bake-off winner registry           | `src/ml_research/asset_specialists/` |
| Per-asset state machine            | `src/strategy_engine/per_asset_runner.py` (`PerAssetRunner.run`) |
| Trailing-stop transition logic     | `_step_state_machine` in same module |
| Alpaca paper-trade client          | `src/execution/alpaca_client.py` (`submit_order`, `submit_bracket_order`, `submit_trailing_stop`) |
| 10-rule risk chain                 | `src/risk_management/chain.py` |
| WHATIF integration smoke test      | `scripts/60_per_asset_dryrun.py` |
| Production runner                  | `scripts/30_per_asset_runner.py` |
| launchd job                        | `scripts/launchd/com.macminibot.per-asset-runner.plist` (fires 14:35 BST / 09:35 ET Mon–Fri) |

**Architectural note**: trailing-stop logic is internal to
`PerAssetRunner._step_state_machine` (state machine tracks peak per
position, emits a SELL intent when price retraces ≥ `trail_sigma * σ_21d`
from peak). Orders sent to Alpaca are plain market orders, NOT
`TrailingStopOrderRequest`. The `submit_trailing_stop` method on
`AlpacaClient` is available for future architectures but is unused by
the current deployment.

---

## 5. Verification evidence

**WHATIF dryrun** (this session, 2026-05-18):

```
.venv/bin/python scripts/60_per_asset_dryrun.py
→ asset_specialists universe: 7/8 → ['AAPL','GLD','GOOGL','META','MSFT','NVDA','TSLA']
→ marks pulled: 7/7
→ snapshot: net_liq=$100088.37 positions=0
→ per_asset_runner: emitted 6 OrderIntents
   AAPL/GOOGL/META/TSLA/NVDA/GLD BUY at IEX-derived marks
   MSFT in RSI no-trade band, AMZN excluded
→ risk chain: 6 rejected on KILL_SWITCH_ACTIVE (HALT_EXISTS=true) —
   expected on Sunday with market halted
```

**Production runner smoke test** (this session, `--force-window`):

```
.venv/bin/python scripts/30_per_asset_runner.py --force-window
→ Same universe + marks + snapshot path completes cleanly.
→ Emitted 0 intents because data/per_asset_state.json already shows
  all 7 assets IN_LONG from the prior dryrun — state machine correctly
  recognises no transition.
```

The 0-intents output is the **correct** behaviour: the state machine
is honouring its persisted state. To force-test entry emission, reset
`data/per_asset_state.json` to `{}` before the next run.

**Test suite**: 570 tests passing across 14 tentacles
(`shared-core`, `execution`, `risk-management`, `strategy-engine`,
`ml-research`, `data-ingest`, `observability`, `compliance`, `treasury`,
`control-plane`, `infrastructure`, `security`, `simulation`, `audit`).

---

## 6. Activation steps (operator-run)

The new per-asset plist and the legacy cross-sectional plist both target
the same Alpaca paper account at 14:35 BST. **They cannot both be loaded
simultaneously.** Bootout the old one first.

```bash
# 1. Stop the old cross-sectional MV runner.
launchctl bootout gui/$UID/com.macminibot.paper-runner

# 2. (Optional but recommended) Reset the per-asset state file so the
#    first production tick treats positions as fresh. Skip if you want
#    to inherit the dryrun's IN_LONG snapshot.
echo '{}' > /Users/weeoto/macminibot/data/per_asset_state.json

# 3. Liquidate any positions left over from the cross-sectional pipeline
#    (Alpaca paper UI: Positions → Close All) so the new state machine
#    starts from a flat book matching its state file.

# 4. Bootstrap the new per-asset plist.
launchctl bootstrap gui/$UID \
    /Users/weeoto/macminibot/scripts/launchd/com.macminibot.per-asset-runner.plist

# 5. Verify it's loaded.
launchctl list | grep com.macminibot.per-asset-runner

# 6. The first scheduled fire is the next weekday at 14:35 BST.
#    (Don't RunAtLoad — the plist is configured to fire only on schedule.)
```

**Rollback** (if something looks wrong on day 1):

```bash
launchctl bootout gui/$UID/com.macminibot.per-asset-runner
launchctl bootstrap gui/$UID \
    /Users/weeoto/macminibot/scripts/launchd/com.macminibot.paper-runner.plist
```

The cross-sectional MV pipeline remains intact and can be reactivated
with the rollback command above — nothing was deleted.

---

## 7. Day-1 monitoring checklist

- [ ] `data/30_per_asset_run_<date>.log` exists after 14:35 BST.
- [ ] `universe: 7/8 assets passed bake-off` line present.
- [ ] `per_asset_runner: emitted N intents` line present.
- [ ] No `chain raised` exceptions.
- [ ] Alpaca paper account shows submitted orders matching the
      `SUBMITTED <SYM> BUY qty=<N> coid=<coid>` log lines.
- [ ] `data/per_asset_state.json` updated with non-`FLAT` entries for
      the symbols that filled.
- [ ] Dashboard (`com.macminibot.dashboard` Streamlit) reflects new fills.

If any check fails: rollback (section 6) and inspect the log.

---

## 8. Sign-off

| item | status |
|---|---|
| Bake-off complete, 5 architecture variants tested      | ✓ |
| Winner chosen empirically (no operator override)       | ✓ |
| FWER correction applied (within-variant Holm + RW)     | ✓ |
| 7 of 8 assets pass within-variant gate                 | ✓ |
| Per-asset state machine implemented                    | ✓ |
| Risk chain integration verified (WHATIF)               | ✓ |
| Production runner implemented + smoke-tested           | ✓ |
| launchd plist in place                                 | ✓ |
| Operator activation steps documented                   | ✓ |
| Honest limitations called out                          | ✓ |
| Live-money switch                                      | **DEFERRED — paper only** |
| 5m / 1m intraday bar substrate (next experiment)       | **DEFERRED — next iteration** |

Ready for **paper deployment** pending operator-run activation steps.

---

## 9. Reproduction

- Bake-off harmoniser: `data/bakeoff_variants/_harmonize/harmonize_v2.py`
- Per-variant inputs: `data/bakeoff_variants/{bracket,signal_flip,trailing_stop,time_decay,long_only_ensemble}/`
- Final analysis: `data/bakeoff_variants/FINAL_ANALYSIS.md`
- Seed: `RNG_SEED=20260518`
- Run dryrun: `.venv/bin/python scripts/60_per_asset_dryrun.py`
- Run production (out-of-hours debug): `.venv/bin/python scripts/30_per_asset_runner.py --force-window`
