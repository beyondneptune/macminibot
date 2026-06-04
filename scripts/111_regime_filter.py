"""111_regime_filter — daily SPY-vs-200dMA regime detector.

Writes ``data/regime_state.json`` that the per-asset-runner consults
before submitting any new entry orders. When the regime flips to
RISK_OFF, the runner auto-flattens all longs (closes positions at
market) and stops opening new ones until the regime flips back.

Decision:
  RISK_ON  if  SPY_close > SPY_200d_SMA  AND  SPY_close > SPY_50d_SMA
  RISK_OFF otherwise

Fires Mon-Fri 13:55 BST (5 min before the runner at 14:00 morning-unhalt,
35 min before runner fire at 14:35).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "regime_state.json"


def _logger() -> logging.Logger:
    log = logging.getLogger("macminibot.regime_filter")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [regime_filter] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def pull_spy_daily(years: float = 1.5):
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    import pandas as pd

    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
    )
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(years * 365.25))
    req = StockBarsRequest(symbol_or_symbols=["SPY"], timeframe=TimeFrame.Day,
                            start=start, end=end, feed=DataFeed.IEX)
    bars = client.get_stock_bars(req).df
    if hasattr(bars.index, "levels"):
        bars = bars.xs("SPY", level=0)
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is not None:
        bars.index = bars.index.tz_convert(None)
    return bars.sort_index()


def main() -> int:
    log = _logger()
    try:
        spy = pull_spy_daily()
    except Exception as exc:
        log.error("SPY bar fetch failed: %s", exc)
        return 1

    if len(spy) < 200:
        log.error("not enough SPY bars (%d) for 200d MA", len(spy))
        return 1

    close = float(spy["close"].iloc[-1])
    sma_50 = float(spy["close"].rolling(50).mean().iloc[-1])
    sma_200 = float(spy["close"].rolling(200).mean().iloc[-1])
    spy_date = spy.index[-1].date().isoformat()

    above_200 = close > sma_200
    above_50 = close > sma_50
    risk_on = above_200 and above_50

    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "regime": "RISK_ON" if risk_on else "RISK_OFF",
        "spy_close": close,
        "spy_date": spy_date,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "ratio_to_200d": close / sma_200,
        "above_50d": above_50,
        "above_200d": above_200,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(OUT_PATH)

    log.info("SPY $%.2f vs SMA200 $%.2f (%.2fx) | SMA50 $%.2f | regime=%s",
              close, sma_200, payload["ratio_to_200d"], sma_50, payload["regime"])
    log.info("written → %s", OUT_PATH)
    return 0 if risk_on else 0   # success either way


if __name__ == "__main__":
    sys.exit(main())
