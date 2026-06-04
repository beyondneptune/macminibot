# Todo

Bootstrap backlog for `observability`. Each item is a self-contained agent-sized epic.

- [ ] **Metric library + naming taxonomy** — Wrapper around the chosen client (prometheus_client / opentelemetry) that enforces naming convention `bot_<dept>_<noun>_<unit>`. Document the taxonomy in `CONTEXT.md`.
- [ ] **Stand up metrics backend** — Run Prometheus (or chosen alternative) locally on the Mac mini. Scrape every component. Persist with a sensible retention.
- [ ] **Stand up Grafana** — Provision dashboards as code from the repo. Add a starter dashboard covering bot health (uptime, feed status, decision-latency p50/p99, kill-switch state).
- [ ] **Trade journal store** — Persist every decision/intent/order/fill chain to the chosen store. Indexed by `decision_id`, `client_order_id`, `symbol`, `strategy_id`.
- [ ] **P&L attribution view** — Real-time P&L per strategy/symbol/venue, realized vs. unrealized, gross vs. net of fees. Backed by the journal.
- [ ] **End-to-end latency histogram** — Stitch latencies tick-in → decision → intent → ack → fill into a single trace per `decision_id`. Display as a stacked histogram.
- [ ] **Alert pack v1** — Feed-gap, decision-latency overrun, kill-switch trip, drawdown breach, no-trades-in-N-minutes. Each alert has a runbook link.
- [ ] **Pick alert routing** — Wire alerts to chosen channel (Telegram / Slack / Pushover) suitable for a home Mac mini deployment.
