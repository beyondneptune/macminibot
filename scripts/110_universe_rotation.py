"""110_universe_rotation — weekly leveraged-ETF universe selector.

Scans a candidate list of ~25 liquid leveraged ETFs, scores each by a
composite of trailing-21d Sharpe + 63d momentum − ATR penalty, picks
the top-8 and writes them to ``data/active_universe.json``.

The per-asset-runner reads this file each fire to decide which 8 names
to trade. Fallback: if file missing or older than 14 days, the runner
uses the hardcoded ``SUPPORTED_ASSETS`` constant.

Fires Sunday 22:00 BST.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "active_universe.json"
N_PICK = 8

# Liquid 2×/3× leveraged ETF candidate pool
CANDIDATES = [
    # 3x equity sector
    "SOXL",  # 3x semis
    "TQQQ",  # 3x QQQ
    "UPRO",  # 3x SPX
    "TNA",   # 3x Russell 2000
    "FAS",   # 3x financials
    "LABU",  # 3x biotech
    "RETL",  # 3x retail
    "DRN",   # 3x real estate
    "URTY",  # 3x small caps (alt)
    "TECL",  # 3x technology
    "CURE",  # 3x healthcare
    "DPST",  # 3x regional banks
    "DUSL",  # 3x industrials
    "ERX",   # 2x energy (only 2x available)
    "PILL",  # 3x pharma
    "WANT",  # 3x consumer discretionary
    # 2x single-stock
    "NVDL",  # 2x NVDA
    "TSLL",  # 2x TSLA
    "MSFU",  # 2x MSFT
    "AAPU",  # 2x AAPL
    "AMZU",  # 2x AMZN
    "GOOL",  # 2x GOOG (legacy ticker; may be sparse)
    "AMDL",  # 2x AMD
    # Volatility (intentionally NOT included: UVXY, VXX, UVIX — pure-vol carry)
]


def _logger() -> logging.Logger:
    log = logging.getLogger("macminibot.universe_rotation")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [universe_rotation] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def load_alpaca_daily(ticker: str, years: float = 1.0) -> pd.DataFrame:
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_API_SECRET"],
    )
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(years * 365.25))
    req = StockBarsRequest(
        symbol_or_symbols=[ticker], timeframe=TimeFrame.Day,
        start=start, end=end, feed=DataFeed.IEX,
    )
    try:
        bars = client.get_stock_bars(req).df
    except Exception:
        return pd.DataFrame()
    if bars is None or bars.empty:
        return pd.DataFrame()
    if hasattr(bars.index, "levels"):
        bars = bars.xs(ticker, level=0)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx
    return out.sort_index()


def score_asset(bars: pd.DataFrame) -> dict | None:
    if len(bars) < 80:
        return None
    close = bars["close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < 70:
        return None
    last_21 = log_ret.iloc[-21:]
    last_63 = log_ret.iloc[-63:]
    mean_21 = float(last_21.mean())
    std_21 = float(last_21.std()) if last_21.std() > 0 else 1e-9
    sharpe_21 = mean_21 / std_21 * math.sqrt(252)
    momentum_63 = float(last_63.sum())
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                     (high - prev_close).abs(),
                     (low - prev_close).abs()], axis=1).max(axis=1)
    atr_21 = float(tr.iloc[-21:].mean())
    atr_pct = atr_21 / float(close.iloc[-1]) if close.iloc[-1] > 0 else 1.0
    avg_dollar_vol = float((close.iloc[-21:] * bars["volume"].astype(float).iloc[-21:]).mean())

    return {
        "sharpe_21d": sharpe_21,
        "momentum_63d_log": momentum_63,
        "atr_pct_21d": atr_pct,
        "avg_dollar_vol_21d": avg_dollar_vol,
        "last_close": float(close.iloc[-1]),
        "n_bars": int(len(bars)),
    }


def main() -> int:
    log = _logger()

    scored: list[tuple[str, dict]] = []
    skipped: list[tuple[str, str]] = []
    for ticker in CANDIDATES:
        try:
            bars = load_alpaca_daily(ticker)
            if bars.empty:
                skipped.append((ticker, "no_bars"))
                continue
            metrics = score_asset(bars)
            if metrics is None:
                skipped.append((ticker, "insufficient_history"))
                continue
            if metrics["avg_dollar_vol_21d"] < 500_000:
                skipped.append((ticker, f"thin_volume({metrics['avg_dollar_vol_21d']:.0f})"))
                continue
            scored.append((ticker, metrics))
        except Exception as exc:
            skipped.append((ticker, f"error:{exc}"))

    if len(scored) < N_PICK:
        log.warning("only %d candidates scored — padding with deployed defaults",
                     len(scored))
        # Pad with the 8 deployed defaults that aren't already in scored.
        scored_tickers = {t for t, _ in scored}
        defaults = ["SOXL", "NVDL", "TQQQ", "UPRO", "TNA", "FAS", "LABU", "TSLL"]
        for d in defaults:
            if d in scored_tickers:
                continue
            try:
                bars = load_alpaca_daily(d)
                m = score_asset(bars)
                if m is not None:
                    scored.append((d, m))
                    if len(scored) >= N_PICK:
                        break
            except Exception:
                continue
        if len(scored) < N_PICK:
            log.error("still only %d after padding — aborting", len(scored))
            return 1

    # Composite score: Sharpe (z) + momentum (z) − ATR% penalty (z)
    df = pd.DataFrame([m for _, m in scored], index=[t for t, _ in scored])
    for col in ["sharpe_21d", "momentum_63d_log", "atr_pct_21d"]:
        s = df[col]
        df[f"z_{col}"] = (s - s.mean()) / s.std()
    df["composite"] = (df["z_sharpe_21d"]
                        + df["z_momentum_63d_log"]
                        - df["z_atr_pct_21d"])
    df = df.sort_values("composite", ascending=False)

    picks = df.head(N_PICK)
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "method": "trailing_21d_sharpe + 63d_momentum - atr_pct_penalty",
        "n_candidates_scanned": len(CANDIDATES),
        "n_scored": len(scored),
        "n_picked": N_PICK,
        "skipped": [{"ticker": t, "reason": r} for t, r in skipped],
        "universe": picks.index.tolist(),
        "scores": picks[["sharpe_21d", "momentum_63d_log", "atr_pct_21d",
                          "composite"]].to_dict(orient="index"),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(OUT_PATH)

    log.info("scanned %d candidates, picked top %d: %s",
              len(CANDIDATES), N_PICK, payload["universe"])
    log.info("written → %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
