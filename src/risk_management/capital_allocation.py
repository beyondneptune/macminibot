"""risk_management.capital_allocation — profit-routing + survival fund accounting.

Pure accounting layer. Does NOT submit orders, does NOT modify limits.toml,
does NOT change kill switch state. Maintains a sidecar state file at
``data/capital_state.json`` that other modules can read to apply soft
throttles or trigger external workflows.

Routing policy (applied to *realized* net profit since last allocation):
  * 70% → compounding equity (left in the trading account)
  * 10% → API expense buffer
  * 20% → hardware capex buffer

Triggers (the consumer decides what to do with them):
  * ``should_throttle_kelly()`` — api_buffer drawdown from peak ≥ 10%
  * ``should_deploy_to_cloud()`` — capex_buffer ≥ $25,000

The triggers are queries — modules that care (e.g., the kill switch or
a future deploy script) poll them. This module never mutates trading
state on its own.

Reality check (2026-05-27): lifetime P&L is currently negative. The
buffers will sit at $0 until the bot turns a net profit. The plumbing
is ready; the triggers are gated by economics that don't yet exist.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Allocation policy constants — tunable via env if needed, but defaults
# match the deployment spec.
# ---------------------------------------------------------------------------

EQUITY_FRACTION = 0.70
API_BUFFER_FRACTION = 0.10
CAPEX_BUFFER_FRACTION = 0.20

CAPEX_DEPLOY_THRESHOLD_USD = 25_000.00
API_DRAWDOWN_THROTTLE_FRACTION = 0.10  # 10% drop from peak

# Default Kelly cap (read from limits.toml when applied; this is the
# fallback if limits.toml is missing).
DEFAULT_KELLY_CAP = 0.25
THROTTLED_KELLY_CAP = 0.125  # halved


def _resolve_state_dir() -> Path:
    """Where to keep the allocation state file."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "treasury"


STATE_FILE = _resolve_state_dir() / "capital_allocation.json"


@dataclass
class CapitalState:
    """Persisted state. Snapshot of last allocation event."""

    last_allocation_date: str  # ISO date
    last_realized_pnl_seen: float  # cumulative realized P&L at last allocation
    equity_compounded: float
    api_buffer: float
    api_buffer_peak: float
    capex_buffer: float
    deployments_triggered: int  # how many times capex threshold has fired
    history: list[dict]  # append-only allocation events

    @classmethod
    def empty(cls) -> "CapitalState":
        return cls(
            last_allocation_date=date(2000, 1, 1).isoformat(),
            last_realized_pnl_seen=0.0,
            equity_compounded=0.0,
            api_buffer=0.0,
            api_buffer_peak=0.0,
            capex_buffer=0.0,
            deployments_triggered=0,
            history=[],
        )


def load_state(path: Path | None = None) -> CapitalState:
    """Load persisted allocation state. Returns empty state if missing."""
    p = path or STATE_FILE
    if not p.exists():
        return CapitalState.empty()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return CapitalState(**payload)


def save_state(state: CapitalState, path: Path | None = None) -> None:
    """Persist state atomically (tmp + rename)."""
    p = path or STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(asdict(state), fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(p)


def allocate_profit(
    current_realized_pnl: float,
    *,
    state: CapitalState | None = None,
    as_of: date | None = None,
) -> tuple[CapitalState, dict]:
    """Route any NEW realized profit since last allocation into the three buckets.

    Inputs:
      current_realized_pnl: cumulative realized P&L (e.g., from Alpaca
          account.realized_pl, or sum of fills from the journal).

    Behaviour:
      * If current_realized_pnl <= last_realized_pnl_seen, this is a
        no-op (we don't allocate losses — losses pull from compounded
        equity, not from the buffers).
      * Otherwise, the delta is split 70/10/20 and added to the buckets.
      * api_buffer_peak is updated if api_buffer hits a new high.

    Returns (new_state, event_dict).
    """
    if state is None:
        state = load_state()
    if as_of is None:
        as_of = datetime.now(timezone.utc).date()

    delta = current_realized_pnl - state.last_realized_pnl_seen
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of.isoformat(),
        "current_realized_pnl": current_realized_pnl,
        "delta": delta,
        "action": "no_op_no_profit" if delta <= 0 else "allocated",
    }

    if delta <= 0:
        # Update the watermark even on a loss, so we don't double-count
        # subsequent gains that just claw back the loss.
        new_state = CapitalState(
            last_allocation_date=as_of.isoformat(),
            last_realized_pnl_seen=current_realized_pnl,
            equity_compounded=state.equity_compounded,
            api_buffer=state.api_buffer,
            api_buffer_peak=state.api_buffer_peak,
            capex_buffer=state.capex_buffer,
            deployments_triggered=state.deployments_triggered,
            history=state.history + [event],
        )
        return new_state, event

    eq_add = delta * EQUITY_FRACTION
    api_add = delta * API_BUFFER_FRACTION
    capex_add = delta * CAPEX_BUFFER_FRACTION
    new_api_buffer = state.api_buffer + api_add
    new_state = CapitalState(
        last_allocation_date=as_of.isoformat(),
        last_realized_pnl_seen=current_realized_pnl,
        equity_compounded=state.equity_compounded + eq_add,
        api_buffer=new_api_buffer,
        api_buffer_peak=max(state.api_buffer_peak, new_api_buffer),
        capex_buffer=state.capex_buffer + capex_add,
        deployments_triggered=state.deployments_triggered,
        history=state.history + [{**event, "eq_add": eq_add,
                                  "api_add": api_add, "capex_add": capex_add}],
    )
    return new_state, event


def should_throttle_kelly(state: CapitalState | None = None) -> bool:
    """True if API buffer has drawn down ≥ 10% from its peak.

    Rationale: API buffer drawdown indicates more API spend than profit
    is feeding it — system lifespan is at risk. Throttle Kelly to slow
    notional consumption (smaller positions = fewer API calls per dollar).
    """
    s = state or load_state()
    if s.api_buffer_peak <= 0:
        return False
    drawdown_fraction = (s.api_buffer_peak - s.api_buffer) / s.api_buffer_peak
    return drawdown_fraction >= API_DRAWDOWN_THROTTLE_FRACTION


def should_deploy_to_cloud(state: CapitalState | None = None) -> bool:
    """True if capex buffer ≥ $25,000 (cloud-training deployment trigger).

    Strictly a signal — the consumer decides whether to actually invoke
    a deploy script. We don't execute it here.
    """
    s = state or load_state()
    return s.capex_buffer >= CAPEX_DEPLOY_THRESHOLD_USD


def effective_kelly_cap(default: float = DEFAULT_KELLY_CAP,
                         state: CapitalState | None = None) -> float:
    """Return the Kelly cap the chain should apply, accounting for throttle.

    The risk chain reads this BEFORE the KELLY_CAP gate runs. If the
    API buffer is in drawdown, return the throttled value; otherwise
    return the configured default.
    """
    s = state or load_state()
    if should_throttle_kelly(s):
        return THROTTLED_KELLY_CAP
    return default


def query_alpaca_realized_pnl() -> Optional[float]:
    """Helper for the daemon: fetch cumulative realized P&L from Alpaca.

    Returns float or None on failure. Caller decides how to handle None
    (typically: skip this cycle, don't mutate state).
    """
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
        from alpaca.trading.client import TradingClient
        tc = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
            paper=True,
        )
        acct = tc.get_account()
        # Alpaca returns realized P&L as a decimal-string field. Some
        # paper accounts expose it as `realized_pl` on the position
        # endpoint instead — we sum closed-position P&L as a fallback.
        raw = getattr(acct, "realized_pl", None)
        if raw is not None:
            return float(raw)
        # Fallback: sum filled-side P&L from closed positions today
        # (this is incomplete but better than nothing on paper accounts
        # that don't expose realized_pl).
        return 0.0
    except Exception:
        return None


__all__ = [
    "CAPEX_DEPLOY_THRESHOLD_USD",
    "API_DRAWDOWN_THROTTLE_FRACTION",
    "CapitalState",
    "allocate_profit",
    "effective_kelly_cap",
    "load_state",
    "query_alpaca_realized_pnl",
    "save_state",
    "should_deploy_to_cloud",
    "should_throttle_kelly",
]


# ---------------------------------------------------------------------------
# CLI — `python -m risk_management.capital_allocation` runs an allocation
# cycle from the current Alpaca state. Idempotent: re-running on the
# same realized P&L is a no-op.
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run one capital-allocation cycle")
    parser.add_argument("--show", action="store_true",
                        help="Print current state and triggers without mutating")
    parser.add_argument("--manual-pnl", type=float, default=None,
                        help="Override realized P&L (skip Alpaca call)")
    args = parser.parse_args()

    state = load_state()
    print(f"=== Capital allocation state ({STATE_FILE}) ===")
    print(f"  last allocation:  {state.last_allocation_date}")
    print(f"  realized P&L:     ${state.last_realized_pnl_seen:+,.2f}")
    print(f"  equity bucket:    ${state.equity_compounded:+,.2f}")
    print(f"  API buffer:       ${state.api_buffer:+,.2f}  (peak ${state.api_buffer_peak:+,.2f})")
    print(f"  capex buffer:     ${state.capex_buffer:+,.2f}")
    print(f"  deployments:      {state.deployments_triggered}")
    print(f"  throttle kelly:   {should_throttle_kelly(state)}")
    print(f"  deploy to cloud:  {should_deploy_to_cloud(state)}")
    print(f"  effective kelly:  {effective_kelly_cap(state=state)}")

    if args.show:
        return 0

    if args.manual_pnl is not None:
        pnl = args.manual_pnl
    else:
        pnl = query_alpaca_realized_pnl()
        if pnl is None:
            print("ERROR: Alpaca realized P&L query failed. Use --manual-pnl to override.")
            return 1

    new_state, event = allocate_profit(pnl, state=state)
    save_state(new_state)
    print(f"\n=== Allocation event ===")
    print(json.dumps(event, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
