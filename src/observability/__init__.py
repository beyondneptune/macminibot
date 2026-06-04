"""observability — alert sinks, equity curve, P&L attribution, dashboard state.

Surface:

- :mod:`observability.alert_sink` — multi-backend alert dispatcher
  (file / desktop / webhook). Battle-tested by the auto-halt daemon
  and EOD summary.
- :mod:`observability.equity_curve` — append-only NetLiq parquet log
  with drawdown helpers; written by ``scripts/42_equity_recorder.py``.
- :mod:`observability.pnl_attribution` — per-symbol realized/unrealized
  P&L from a journal-fills DataFrame.
- :mod:`observability.metrics_emitter` — re-exports the schema-bound
  emitter from ``risk_management.metrics_emitter`` and adds the
  dashboard-facing :func:`combined_dashboard_state` aggregator.
"""

from observability.alert_sink import (
    AlertEvent,
    AlertSeverity,
    AlertSink,
    DesktopAlertSink,
    FileAlertSink,
    MultiSink,
    WebhookAlertSink,
    build_sink_from_uri,
    load_sinks_from_config,
)
from observability.equity_curve import (
    DEFAULT_PARQUET_PATH,
    EQUITY_COLUMNS,
    load_curve,
    max_drawdown_pct,
    recent,
    record,
)
from observability.metrics_emitter import (
    METRIC_NAMES,
    SCHEMA,
    MetricSample,
    MetricsSampler,
    MetricsValidationError,
    combined_dashboard_state,
    new_sampler,
)
from observability.pnl_attribution import attribute_pnl, total_row

__all__ = [
    # alert_sink
    "AlertEvent",
    "AlertSeverity",
    "AlertSink",
    "DesktopAlertSink",
    "FileAlertSink",
    "MultiSink",
    "WebhookAlertSink",
    "build_sink_from_uri",
    "load_sinks_from_config",
    # equity_curve
    "DEFAULT_PARQUET_PATH",
    "EQUITY_COLUMNS",
    "load_curve",
    "max_drawdown_pct",
    "recent",
    "record",
    # metrics_emitter
    "METRIC_NAMES",
    "SCHEMA",
    "MetricSample",
    "MetricsSampler",
    "MetricsValidationError",
    "combined_dashboard_state",
    "new_sampler",
    # pnl_attribution
    "attribute_pnl",
    "total_row",
]
