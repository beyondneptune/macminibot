"""pnl_attribution — per-symbol realized + unrealized P&L from fills.

Input: a journal-fills DataFrame matching ``data/journal/paper/dt=YYYY-MM-DD/``
(see ``scripts/30_paper_trade_runner.py`` for the writer side). The relevant
columns are:

    symbol            str
    side              category ("BUY"|"SELL")
    fill_qty          float64
    fill_price        float64
    status            str    ("SUBMITTED"|"FILLED"|...)
    filled_at_ns      int64 / float64

Rows where ``status != "FILLED"`` (or ``fill_qty`` is null/zero) are
treated as un-executed and ignored.

Cost-basis convention: **quantity-weighted average cost on the long side**.
When a position flips sign within the day we book the residual against the
average and start a new basis on the flipped side. This is the same
convention used by IBKR's TWS for unrealized P&L on day-trade flows.

Output: a DataFrame keyed by symbol with columns

    qty              float64   net position after fills
    avg_cost         float64   quantity-weighted avg entry on the open side
    realized_pnl     float64   booked from sell-into-long / buy-into-short
    unrealized_pnl   float64   (mark - avg_cost) * qty  (long-positive)
    total_pnl        float64   realized + unrealized
    last_mark        float64   the mark used (NaN if absent)

A convenience ``total_row()`` helper sums the per-symbol numbers.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


__all__ = [
    "attribute_pnl",
    "total_row",
]


def _filtered_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that represent an executed fill."""

    if fills.empty:
        return fills
    df = fills.copy()
    if "status" in df.columns:
        df = df.loc[df["status"].astype(str).str.upper() == "FILLED"]
    if "fill_qty" in df.columns:
        df = df.loc[df["fill_qty"].notna() & (df["fill_qty"] > 0)]
    if "fill_price" in df.columns:
        df = df.loc[df["fill_price"].notna() & (df["fill_price"] > 0)]
    return df


def _sort_fills(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by fill time so the running cost basis processes in order."""

    if "filled_at_ns" in df.columns and df["filled_at_ns"].notna().any():
        return df.sort_values("filled_at_ns", kind="mergesort").reset_index(drop=True)
    if "submitted_at_ns" in df.columns:
        return df.sort_values("submitted_at_ns", kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def _per_symbol(
    rows: pd.DataFrame, mark: float | None
) -> dict[str, float]:
    """Single-symbol reduction; returns the output dict for one symbol."""

    qty = 0.0
    avg_cost = 0.0
    realized = 0.0

    for _, r in rows.iterrows():
        side = str(r["side"]).upper()
        f_qty = float(r["fill_qty"])
        f_px = float(r["fill_price"])
        signed = f_qty if side == "BUY" else -f_qty

        if qty == 0:
            qty = signed
            avg_cost = f_px
            continue

        same_direction = (qty > 0 and signed > 0) or (qty < 0 and signed < 0)
        if same_direction:
            new_qty = qty + signed
            # Weighted-average cost only counts the absolute exposures.
            avg_cost = (
                (abs(qty) * avg_cost) + (abs(signed) * f_px)
            ) / abs(new_qty)
            qty = new_qty
        else:
            # Reducing or flipping the position.
            closing_qty = min(abs(qty), abs(signed))
            # Long->sell: realized = (sell_px - avg_cost) * closing
            # Short->buy: realized = (avg_cost - buy_px) * closing
            if qty > 0:
                realized += (f_px - avg_cost) * closing_qty
            else:
                realized += (avg_cost - f_px) * closing_qty
            new_qty = qty + signed
            if new_qty == 0:
                qty = 0.0
                avg_cost = 0.0
            elif (new_qty > 0) == (qty > 0):
                # Position reduced but not flipped — keep avg_cost.
                qty = new_qty
            else:
                # Flipped — residual sits at the fill price.
                qty = new_qty
                avg_cost = f_px

    unrealized = 0.0
    last_mark = float("nan") if mark is None else float(mark)
    if qty != 0 and mark is not None:
        unrealized = (float(mark) - avg_cost) * qty

    return {
        "qty": qty,
        "avg_cost": avg_cost if qty != 0 else 0.0,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "last_mark": last_mark,
    }


def attribute_pnl(
    fills: pd.DataFrame,
    marks: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Per-symbol realized + unrealized P&L with quantity-weighted avg cost.

    Parameters
    ----------
    fills:
        Journal-style fills DataFrame (see module docstring).
    marks:
        Optional mapping symbol -> current mark. Missing symbols have
        unrealized P&L of 0 and ``last_mark = NaN``.
    """

    marks = dict(marks or {})
    df = _filtered_fills(fills)
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame(
            columns=[
                "qty",
                "avg_cost",
                "realized_pnl",
                "unrealized_pnl",
                "total_pnl",
                "last_mark",
            ]
        )

    df = _sort_fills(df)
    rows: dict[str, dict[str, float]] = {}
    for symbol, group in df.groupby("symbol", sort=True):
        rows[str(symbol)] = _per_symbol(group, marks.get(str(symbol)))

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "symbol"
    return out[
        [
            "qty",
            "avg_cost",
            "realized_pnl",
            "unrealized_pnl",
            "total_pnl",
            "last_mark",
        ]
    ]


def total_row(attribution: pd.DataFrame) -> dict[str, float]:
    """Aggregate the per-symbol attribution into one summary dict."""

    if attribution.empty:
        return {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "n_symbols": 0,
        }
    return {
        "realized_pnl": float(attribution["realized_pnl"].sum()),
        "unrealized_pnl": float(attribution["unrealized_pnl"].sum()),
        "total_pnl": float(attribution["total_pnl"].sum()),
        "n_symbols": int((attribution["qty"] != 0).sum()),
    }
