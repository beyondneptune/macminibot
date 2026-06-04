"""treasury — account-level state: balances, fees, reconciliation, capital deployed.

This is the **paper-trading MVP** adaptation of the spec in
``.octogent/tentacles/treasury/CONTEXT.md``. The original spec is
IBKR-oriented and multi-venue; today the live bot trades exclusively
through Alpaca paper, so the MVP scope is:

  - one venue (Alpaca paper), one currency (USD),
  - balance ledger persisted to ``data/treasury/balances.parquet``,
  - reconciliation: ``execution``-side internal positions vs. Alpaca's
    ``get_all_positions`` snapshot — breaks land in ``breaks.jsonl``,
  - fee ledger: Alpaca paper is commission-free, but SEC §31 fees on
    sells and FINRA TAF on sells are still tracked per the regulatory
    rate table (so when we move to live they're already accounted for),
  - capital-deployed helpers: gross/net notional, leverage, cash
    utilisation, consumed by ``observability`` and ``risk-management``.

Treasury is **read-only** with respect to the rest of the system — it
reports state, it never cancels orders or trips the kill-switch
(per CONTEXT.md conventions).
"""

from __future__ import annotations

from treasury.balance_ledger import (
    BalanceLedger,
    PositionBreak,
    reconcile,
)
from treasury.capital_deployed import (
    cash_utilization,
    leverage,
    total_gross_notional,
    total_net_notional,
)
from treasury.fees import (
    FeeBreakdown,
    SEC_FEE_RATE,
    TAF_PER_SHARE,
    TAF_MAX_PER_TRADE,
    compute_trade_cost,
)

__all__ = [
    "BalanceLedger",
    "FeeBreakdown",
    "PositionBreak",
    "SEC_FEE_RATE",
    "TAF_MAX_PER_TRADE",
    "TAF_PER_SHARE",
    "cash_utilization",
    "compute_trade_cost",
    "leverage",
    "reconcile",
    "total_gross_notional",
    "total_net_notional",
]
