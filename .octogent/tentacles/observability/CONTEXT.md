# observability

P&L attribution, trade journals, latency histograms, Prometheus/Grafana dashboards, alerting.

## Status
Greenfield — no source code exists yet. The notes below are a **starting proposal**, not ratified design. This dept's job is to make problems visible *before* they become incidents.

## Scope
- Metric emission library used by every other dept (counters, gauges, histograms with stable names)
- Time-series metric backend (Prometheus or equivalent) and dashboards (Grafana or equivalent)
- Trade journal: structured record of every decision → intent → order → fill chain, queryable for post-trade analysis
- P&L attribution: per-strategy, per-symbol, per-venue, realized vs. unrealized, gross vs. net of fees
- Latency tracking across the whole pipeline (tick-in → decision → intent → ack → fill)
- Alerting on SLO breaches (feed gaps, decision-latency overruns, abnormal P&L, kill-switch trips, breach events)

## Boundaries
- **Owns:** metrics library, dashboards, alerting rules, trade journal storage + query
- **Consumes from:** all other tentacles emit metrics, events, and journal entries here
- **Provides to:** `control-plane` (current state for UI); humans (dashboards + alerts)

## Key Decisions (to be made)
- Metrics backend (Prometheus, VictoriaMetrics, InfluxDB)
- Dashboard tool (Grafana most common)
- Trade journal storage (Postgres, ClickHouse, append-only Parquet)
- Alert routing (Pagerduty, Opsgenie, Slack, email, Telegram) — appropriate for a Mac mini home setup
- Metric naming taxonomy — once names are out there, renaming is painful

## Conventions (proposed)
- All metric names are stable and namespaced: `bot_<dept>_<noun>_<unit>` (e.g. `bot_execution_orders_total`, `bot_strategy_decision_latency_seconds`)
- Every metric has a comment explaining what it measures and when it changes
- Trade journal entries link via shared `decision_id` and `client_order_id` so a full trace can be reconstructed
- Alerts have a runbook link in their description — no orphan alerts that no one knows how to act on
- Dashboards are checked into the repo as JSON, not maintained only in Grafana's UI
