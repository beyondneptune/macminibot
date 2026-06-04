"""factory.generator — autonomous strategy synthesizer.

Two modes:
  * DETERMINISTIC (default): emits a templated strategy parameterised by the
    detected anomaly. Reproducible, testable, no external API dependency.
    Used for orchestrator demos and CI.
  * LLM (opt-in via ``MACMINIBOT_FACTORY_USE_ANTHROPIC=1`` env): calls the
    Anthropic API with the anomaly context and asks for a strategy. Disabled
    by default because (a) no API key has been authorised for this factory,
    (b) live LLM output is non-deterministic and would break the validator's
    audit trail.

Each generated strategy is written to ``src/factory/live_strategies/``
with a stamped filename. The Crucible (``factory.validator``) is then
called to backtest it; if it fails, the file is deleted.

Output contract — every generated file MUST define:

    class LiveAnomalyStrategy:
        STRATEGY_ID: str
        PARAMS: dict[str, Any]

        def compute_signal(self, bars: pd.DataFrame) -> int:
            '''Return -1 / 0 / +1.'''

This is intentionally narrow — generated strategies cannot import the
execution layer, the broker client, or the risk chain. They are pure
signal functions; the router decides whether to allocate.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "src" / "factory" / "live_strategies"


@dataclass(frozen=True, slots=True)
class Anomaly:
    """Market anomaly extracted from the live data stream.

    Fed to the generator. The exact set of features here is the contract
    between the upstream anomaly detector and the generator's prompt.
    """

    symbol: str
    detected_at_ns: int
    type: str   # e.g. "vwap_deviation", "imbalance_spike", "rsi_extreme"
    magnitude: float   # standardised z-score
    context: dict[str, Any]


def detect_anomalies(microstructure_dir: Path | None = None,
                      *, top_k: int = 3) -> list[Anomaly]:
    """Read the latest microstructure snapshots produced by async_streamer
    and surface the top-k anomalies by absolute z-score.

    Sources:
      - data/microstructure/<asset>_<date>.parquet (rolling VWAP deviation
        and L1 imbalance from execution.async_streamer)

    A real production version would also pull live RSI/MACD from the
    runtime feature pipeline. For the orchestrator demo we use whichever
    microstructure parquets exist.
    """
    if microstructure_dir is None:
        microstructure_dir = PROJECT_ROOT / "data" / "microstructure"
    if not microstructure_dir.exists():
        return []
    rows = []
    for p in sorted(microstructure_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue
            asset = p.stem.split("_")[0]
            latest = df.iloc[-1]
            vwap_dev_bps = float(latest.get("vwap_dev_bps", 0.0))
            imbalance = float(latest.get("imbalance", 0.0))
            ts_ns = int(latest.get("ts_ns", 0))
            # Standardise vs trailing window
            vwap_z = (vwap_dev_bps - df["vwap_dev_bps"].mean()) / (df["vwap_dev_bps"].std() or 1.0)
            imb_z = (imbalance - df["imbalance"].mean()) / (df["imbalance"].std() or 1.0)
            if abs(float(vwap_z)) >= 1.0:
                rows.append(Anomaly(
                    symbol=asset, detected_at_ns=ts_ns,
                    type="vwap_deviation", magnitude=float(vwap_z),
                    context={"vwap_dev_bps": vwap_dev_bps,
                              "n_samples": int(len(df))},
                ))
            if abs(float(imb_z)) >= 1.0:
                rows.append(Anomaly(
                    symbol=asset, detected_at_ns=ts_ns,
                    type="imbalance_spike", magnitude=float(imb_z),
                    context={"imbalance": imbalance,
                              "n_samples": int(len(df))},
                ))
        except Exception:
            continue
    rows.sort(key=lambda a: abs(a.magnitude), reverse=True)
    return rows[:top_k]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _deterministic_strategy_source(anomaly: Anomaly, strategy_id: str) -> str:
    """Template-based code generation. Parametrises a Bollinger-mean-reversion
    strategy around the anomaly's symbol + magnitude direction.

    Reproducible: same anomaly → same source.
    """
    direction = "long" if anomaly.magnitude < 0 else "short"
    # Magnitude → entry threshold (more extreme anomaly → tighter threshold)
    entry_z = max(1.0, min(3.0, abs(anomaly.magnitude)))
    window = 20 if anomaly.type == "vwap_deviation" else 14
    return f'''"""Auto-generated strategy {strategy_id}.

Generated 2026-05-28 from anomaly:
  symbol = {anomaly.symbol!r}
  type = {anomaly.type!r}
  magnitude_z = {anomaly.magnitude:.3f} ({direction})
  context = {anomaly.context!r}
"""

from __future__ import annotations

import pandas as pd


class LiveAnomalyStrategy:
    STRATEGY_ID = "{strategy_id}"
    PARAMS = {{
        "symbol": "{anomaly.symbol}",
        "anomaly_type": "{anomaly.type}",
        "direction": "{direction}",
        "entry_z_threshold": {entry_z:.3f},
        "window": {window},
    }}

    def compute_signal(self, bars: pd.DataFrame) -> int:
        """Mean-reversion signal: enter against the anomaly direction.

        Returns -1, 0, or +1 (no fractional sizing — router handles size).
        """
        if len(bars) < self.PARAMS["window"]:
            return 0
        close = bars["close"].astype(float)
        ma = close.rolling(self.PARAMS["window"]).mean()
        sd = close.rolling(self.PARAMS["window"]).std()
        z = (close.iloc[-1] - ma.iloc[-1]) / (sd.iloc[-1] if sd.iloc[-1] > 0 else 1.0)
        thr = self.PARAMS["entry_z_threshold"]
        if self.PARAMS["direction"] == "long":
            return 1 if z < -thr else 0
        else:
            return -1 if z > thr else 0
'''


def _anthropic_strategy_source(anomaly: Anomaly, strategy_id: str) -> str:
    """LLM-driven generator. Disabled by default — only fires when
    ``MACMINIBOT_FACTORY_USE_ANTHROPIC=1`` AND ``ANTHROPIC_API_KEY`` is set.

    The prompt asks for a single ``LiveAnomalyStrategy`` class with the
    contract this module enforces. The response is the file body.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — LLM generator unauthorised")
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(f"anthropic SDK not installed: {exc}") from exc

    client = Anthropic()
    prompt = f"""You generate Python trading strategy code.

Detected market anomaly:
  symbol={anomaly.symbol}
  type={anomaly.type}
  magnitude_z={anomaly.magnitude:.3f}
  context={anomaly.context}

Emit ONE Python file. The file MUST define:

  class LiveAnomalyStrategy:
      STRATEGY_ID = "{strategy_id}"
      PARAMS: dict[str, Any]
      def compute_signal(self, bars: pd.DataFrame) -> int:
          # returns -1, 0, or +1

Constraints:
- No imports of execution / broker / risk modules.
- No I/O, no network, no subprocess.
- Output must be importable as a module.
- Return only the code, no markdown fences.
"""
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def generate_strategy(anomaly: Anomaly, *,
                       force_deterministic: bool = False) -> Path:
    """Generate a strategy file from an anomaly. Returns the file path.

    Uses deterministic templating by default. If env var
    ``MACMINIBOT_FACTORY_USE_ANTHROPIC=1`` is set AND ``force_deterministic``
    is False, calls the Anthropic API instead.
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    use_llm = (not force_deterministic
                and os.environ.get("MACMINIBOT_FACTORY_USE_ANTHROPIC") == "1"
                and os.environ.get("ANTHROPIC_API_KEY"))
    strategy_id_seed = f"{anomaly.symbol}_{anomaly.type}_{anomaly.detected_at_ns}"
    strategy_id = "strat_" + hashlib.sha256(strategy_id_seed.encode()).hexdigest()[:12]
    if use_llm:
        source = _anthropic_strategy_source(anomaly, strategy_id)
    else:
        source = _deterministic_strategy_source(anomaly, strategy_id)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = LIVE_DIR / f"{stamp}_{strategy_id}.py"
    out_path.write_text(source)
    return out_path


__all__ = ["Anomaly", "detect_anomalies", "generate_strategy"]
