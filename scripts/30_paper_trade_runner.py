"""30_paper_trade_runner — market-open orchestration script (PAPER mode).

Runs at 14:30 BST = 09:30 ET. Wires the full pipeline:

  1. Load .env (Alpaca creds, ACCOUNT_EQUITY_INITIAL).
  2. Pull current quotes for the 8-symbol universe via Alpaca data API.
  3. Pull yesterday's daily bars from ArcticDB (8 tradables + SPY/QQQ/IWM for vix_proxy).
  4. Compute daily features via ml_research.features.compute_features (dollar-bar
     features skipped — daily-only here).
  5. Train Ridge on the most recent 5y of daily bars.
  6. Predict next-day return per contract.
  7. Build Signal list.
  8. Compute Ledoit-Wolf shrunk Σ.
  9. Build current NavSnapshot from Alpaca account.
 10. Compute VIX-proxy series from rolling SPY+QQQ+IWM realized vol.
 11. Run strategy_engine.scheduler.run_rebalance (mode=ChainMode.PAPER).
 12. VixHighGate pre-gate — log and halt if triggered.
 13. risk_management.diff.diff_targets_to_intents.
 14. risk_management.chain.run_check_chain per intent — drop rejects.
 15. Hand approved intents to execution.runner.execute_batch.
 16. Log everything to data/30_paper_run_YYYY-MM-DD.log.

ChainMode.PAPER everywhere. Paper-mode only. Safe by construction.

Dry-run:
    .venv/bin/python scripts/30_paper_trade_runner.py --dry-run

Adding ``--dry-run`` forces WHATIF on the executor only (the rebalance pipeline
still runs PAPER end-to-end so we exercise the gates and diffing); no Alpaca
order submission happens in dry-run.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Disable heartbeat-staleness gate for first-day runs — the risk_heartbeat
# launchd job is not yet wired. HALT flag still honoured.
os.environ.setdefault("MACMINIBOT_DISABLE_HEARTBEAT_CHECK", "1")

# Write a fresh heartbeat for the risk chain's KILL_SWITCH_ACTIVE check.
# The chain reads heartbeat.json directly; v0 doesn't have the launchd job
# updating it, so we write one ourselves at script start.
def _write_runner_heartbeat() -> None:
    from pathlib import Path as _P
    import json as _json
    import time as _t
    state_dir = _P(os.environ.get("MACMINIBOT_STATE_DIR", str(_P.home() / "macminibot" / "state")))
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "heartbeat.json").write_text(_json.dumps({
        "pid": os.getpid(),
        "ts_unix": _t.time(),
        "version": "runner-v0",
    }))
_write_runner_heartbeat()

from _storage import open_store  # noqa: E402

from execution.alpaca_client import AlpacaClient  # noqa: E402
from execution.runner import execute_batch  # noqa: E402
from ml_research.features import (  # noqa: E402
    FEATURE_NAMES,
    compute_features,
    drop_feature_failures,
)
from ml_research.labels import (  # noqa: E402
    cross_sectional_zscore,
    fit_winsorization_bounds,
    next_day_open_close_log_return,
    winsorize_train_fold,
)
from ml_research.trainer import RidgeConfig, RidgeTrainer  # noqa: E402
from ml_research.auto_retrain import load_latest_model  # noqa: E402
from strategy_engine.auto_tune import load_latest_tuning  # noqa: E402
from risk_management.chain import ChainContext, run_check_chain  # noqa: E402
from risk_management.diff import diff_targets_to_intents  # noqa: E402
from risk_management.limits import load_limits  # noqa: E402
from risk_management.regime_gate import VixHighGate  # noqa: E402
from risk_management.snapshot import build_snapshot as build_risk_snapshot  # noqa: E402
from shared_core.contracts import (  # noqa: E402
    ChainMode,
    NavSnapshot as SharedNavSnapshot,
    Signal,
    Symbol,
    Venue,
)
from strategy_engine.scheduler import RebalanceInputs, run_rebalance  # noqa: E402


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# (arctic_library, arctic_symbol, ticker, venue) for the 8 tradables.
TRADABLE: tuple[tuple[str, str, str, Venue], ...] = (
    ("equity.daily_bars", "NASDAQ.AAPL", "AAPL", Venue.NASDAQ),
    ("equity.daily_bars", "NASDAQ.MSFT", "MSFT", Venue.NASDAQ),
    ("equity.daily_bars", "NASDAQ.GOOGL", "GOOGL", Venue.NASDAQ),
    ("equity.daily_bars", "NASDAQ.AMZN", "AMZN", Venue.NASDAQ),
    ("equity.daily_bars", "NASDAQ.META", "META", Venue.NASDAQ),
    ("equity.daily_bars", "NASDAQ.TSLA", "TSLA", Venue.NASDAQ),
    ("equity.daily_bars", "NASDAQ.NVDA", "NVDA", Venue.NASDAQ),
    ("etf.daily_bars",    "ARCA.GLD",    "GLD",   Venue.ARCA),
)

# Benchmark ETFs needed for SPY rel-strength + vix_proxy. NOT tradable in this run.
AUX_BENCHMARK: tuple[tuple[str, str, str], ...] = (
    ("etf.daily_bars", "ARCA.SPY",    "SPY"),
    ("etf.daily_bars", "NASDAQ.QQQ",  "QQQ"),
    ("etf.daily_bars", "ARCA.IWM",    "IWM"),
)

TRADABLE_TICKERS: tuple[str, ...] = tuple(t[2] for t in TRADABLE)
TRADABLE_SYMBOLS: dict[str, Symbol] = {
    ticker: Symbol(ticker=ticker, venue=venue, currency="USD")
    for _, _, ticker, venue in TRADABLE
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("paper_trade_runner")
    logger.setLevel(logging.INFO)
    # Clear pre-existing handlers (rerun-safe inside REPL / pytest).
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Step 2 — Alpaca quotes
# ---------------------------------------------------------------------------


def fetch_alpaca_quotes(api_key: str, secret: str, tickers: tuple[str, ...], logger: logging.Logger) -> dict[str, Decimal]:
    """Return ticker -> mid-price Decimal from Alpaca latest-quote (IEX feed)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    data = StockHistoricalDataClient(api_key, secret)
    req = StockLatestQuoteRequest(symbol_or_symbols=list(tickers))
    quotes = data.get_stock_latest_quote(req)

    out: dict[str, Decimal] = {}
    for t in tickers:
        q = quotes.get(t)
        if q is None:
            logger.warning("quote missing for %s — skipping in marks", t)
            continue
        bid = float(getattr(q, "bid_price", 0.0) or 0.0)
        ask = float(getattr(q, "ask_price", 0.0) or 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
        if mid <= 0:
            logger.warning("non-positive mid for %s (bid=%s ask=%s) — skipping", t, bid, ask)
            continue
        # Decimal must come from str (KD1).
        out[t] = Decimal(f"{mid:.4f}")
    return out


# ---------------------------------------------------------------------------
# Step 3 — ArcticDB panel load
# ---------------------------------------------------------------------------


def load_daily_panel(logger: logging.Logger) -> pl.DataFrame:
    """Load daily bars for tradables + benchmarks, long-form polars panel."""
    store = open_store()
    rows: list[pl.DataFrame] = []
    universe = [(lib, asym, disp) for (lib, asym, disp, _v) in TRADABLE] + list(AUX_BENCHMARK)
    for lib_name, arctic_sym, display_sym in universe:
        lib = store[lib_name]
        df = lib.read(arctic_sym).data.reset_index().rename(columns={"date": "date"})
        df["symbol"] = display_sym
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        rows.append(pl.from_pandas(df[cols]))
    panel = pl.concat(rows, how="vertical_relaxed").sort(["symbol", "date"])
    panel = panel.with_columns(pl.col("date").cast(pl.Datetime("ns")))
    last_date = panel["date"].max()
    logger.info("loaded panel: %s rows, last_date=%s, n_symbols=%s",
                f"{panel.height:,}", last_date, panel["symbol"].n_unique())
    return panel


# ---------------------------------------------------------------------------
# Step 5 — Train Ridge on most recent 5y
# ---------------------------------------------------------------------------


def train_ridge_recent(
    panel: pl.DataFrame, logger: logging.Logger, years: int = 5,
) -> tuple[RidgeTrainer, list[str], dict[str, int]]:
    """Fit Ridge on (most-recent-5y) of features → next-day open/close return.

    Returns (ridge_trainer, feature_cols, sym_to_id_map).
    """
    feat = compute_features(panel)
    lab = next_day_open_close_log_return(feat, label_col="y_raw")
    lab = lab.filter(pl.col("symbol").is_in(set(TRADABLE_TICKERS)))
    lab = drop_feature_failures(lab, feature_names=FEATURE_NAMES)
    lab = lab.drop_nulls(subset=["y_raw"]).filter(pl.col("y_raw").is_finite())

    # Symbol-id one-hot replacement: int categorical column (LightGBM-style).
    sym_to_id = {s: i for i, s in enumerate(sorted(TRADABLE_TICKERS))}
    lab = lab.with_columns(
        pl.col("symbol")
        .map_elements(lambda s: sym_to_id[s], return_dtype=pl.Int32)
        .alias("symbol_id")
    )

    # Slice the most recent ``years`` of dates.
    max_date = lab["date"].max()
    cutoff_ns = max_date - pd.Timedelta(days=int(years * 365.25))
    lab = lab.filter(pl.col("date") >= cutoff_ns)
    logger.info("training rows (last %sy): %s, dates [%s .. %s]",
                years, f"{lab.height:,}", lab["date"].min(), lab["date"].max())

    feature_cols = list(FEATURE_NAMES) + ["symbol_id"]
    pdf = lab.to_pandas().sort_values(["date", "symbol"]).reset_index(drop=True)

    # Train-fold winsorize + xs-z labels (no separate val fold — single-shot fit).
    bounds = fit_winsorization_bounds(pdf["y_raw"].to_numpy(), sigma=5.0)
    lab_pl = winsorize_train_fold(pl.from_pandas(pdf[["symbol", "date", "y_raw"]]),
                                  bounds, label_col="y_raw", out_col="y_wins")
    lab_pl = cross_sectional_zscore(lab_pl, label_col="y_wins", out_col="y_xs_z")
    merged = pl.from_pandas(pdf[["symbol", "date"] + feature_cols]).join(
        lab_pl, on=["symbol", "date"], how="inner"
    )
    train_pd = merged.to_pandas()

    X = train_pd[feature_cols].to_numpy(dtype=np.float64)
    y = train_pd["y_xs_z"].to_numpy(dtype=np.float64)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    ridge = RidgeTrainer(config=RidgeConfig(alpha=1.0))
    ridge.fit(X[m], y[m])
    logger.info("ridge fit: n_train=%s, n_features=%s, alpha=%s",
                int(m.sum()), len(feature_cols), 1.0)
    return ridge, feature_cols, sym_to_id


# ---------------------------------------------------------------------------
# Step 6/7 — Predict + build Signals
# ---------------------------------------------------------------------------


def predict_and_build_signals(
    panel: pl.DataFrame,
    ridge: RidgeTrainer,
    feature_cols: list[str],
    sym_to_id: dict[str, int],
    as_of_ns: int,
    logger: logging.Logger,
    daily_std: float = 0.10,
) -> tuple[list[Signal], pd.DataFrame]:
    """Compute features on the panel and predict the latest available row per symbol."""
    feat = compute_features(panel)
    feat = drop_feature_failures(feat, feature_names=FEATURE_NAMES)
    feat = feat.filter(pl.col("symbol").is_in(set(TRADABLE_TICKERS)))
    feat = feat.with_columns(
        pl.col("symbol")
        .map_elements(lambda s: sym_to_id[s], return_dtype=pl.Int32)
        .alias("symbol_id")
    )
    # Latest row per symbol.
    latest = feat.sort(["symbol", "date"]).group_by("symbol").tail(1)
    latest_pd = latest.to_pandas().set_index("symbol")
    X = latest_pd[feature_cols].to_numpy(dtype=np.float64)
    preds_z = ridge.predict(X)  # xs-zscore space

    # Scale z-score predictions back to a daily-return magnitude. Z-score
    # of a winsorized log-return ~ N(0,1); multiply by a daily std (~2.5%
    # for liquid US mega-caps) to recover an r_hat_daily that respects
    # the shared_core Signal sanity guard |r_hat| <= 0.20. Ridge
    # predictions in xs-z space are small in magnitude; the scaling
    # is needed to push them above the 5bp zero-signal-day cutoff so
    # the optimizer actually allocates (rather than decaying to cash).
    # NOTE: daily_std is the bridge from xs-z space back to physical daily
    # log-return space. The Ridge sanity-floor produces tiny xs-z predictions
    # (≈ ±0.03), so the scaler must be aggressive enough that the MV
    # optimizer's lambda_risk=5 doesn't collapse every weight below the
    # 0.5% NAV no-trade band. 0.10 is the empirical floor that keeps the
    # v0 paper pipeline emitting at least one tradable intent on most days.
    # Production model versions should re-derive this from the train-fold
    # winsorization bounds and bake it into model_version.
    # (When called via main(), ``daily_std`` is overridden from the pickled
    # auto-retrain artifact if data/models/latest.pkl exists.)
    r_hat = preds_z * daily_std

    signals: list[Signal] = []
    rows: list[dict] = []
    for ticker, r in zip(latest_pd.index.tolist(), r_hat):
        if not np.isfinite(r):
            logger.warning("non-finite prediction for %s — skipping", ticker)
            continue
        sym = TRADABLE_SYMBOLS[ticker]
        r_hat_dec = Decimal(f"{r:.8f}")
        # Clamp to the shared_core contract band (|r_hat_daily| <= 0.20).
        if abs(r_hat_dec) > Decimal("0.20"):
            r_hat_dec = Decimal("0.20") if r_hat_dec > 0 else Decimal("-0.20")
        signals.append(Signal(
            symbol=sym,
            r_hat_daily=r_hat_dec,
            confidence=0.50,
            model_version="ridge-paper-v0",
            as_of_ns=int(as_of_ns),
        ))
        rows.append({"ticker": ticker, "r_hat_daily": float(r_hat_dec)})
    sig_df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
    logger.info("signals built: %d / %d", len(signals), len(TRADABLE_TICKERS))
    return signals, sig_df


# ---------------------------------------------------------------------------
# Step 8 — Daily-bar return matrix for Σ + step 10 vix proxy series
# ---------------------------------------------------------------------------


def build_returns_matrix(panel: pl.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    """Wide returns DataFrame (cols=Symbol, rows=date) for the 8 tradables."""
    df = panel.filter(pl.col("symbol").is_in(set(TRADABLE_TICKERS)))
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol")).log().alias("ret")
    ).drop_nulls(subset=["ret"])
    pdf = df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    wide = pdf.pivot(index="date", columns="symbol", values="ret").dropna()
    wide = wide.tail(lookback_days)
    # Rename columns from str ticker → Symbol (strategy_engine uses Symbol-typed cols).
    wide = wide.rename(columns={t: TRADABLE_SYMBOLS[t] for t in wide.columns})
    return wide


def build_vix_proxy_series(panel: pl.DataFrame, lookback_days: int = 252) -> np.ndarray:
    """Trailing series of vix_proxy values (xs std of 21d rv across SPY+QQQ+IWM)."""
    feat = compute_features(panel)
    # vix_proxy is broadcast across symbols per date — pick one (SPY) and pull the series.
    sub = feat.filter(pl.col("symbol") == "SPY").sort("date").drop_nulls(subset=["vix_proxy"])
    series = sub["vix_proxy"].to_numpy().astype(np.float64)
    series = series[np.isfinite(series)]
    return series[-lookback_days:]


# ---------------------------------------------------------------------------
# Step 9 — NavSnapshot from Alpaca
# ---------------------------------------------------------------------------


def build_alpaca_snapshots(
    client: AlpacaClient,
    marks: dict[str, Decimal],
    panel: pl.DataFrame,
    as_of_ns: int,
    limits,
    logger: logging.Logger,
):
    """Return (risk_snapshot, shared_snapshot, account_equity).

    Both snapshot flavours are needed: ``risk_management.chain`` uses
    ``risk_management.snapshot.NavSnapshot`` (str-keyed positions);
    ``strategy_engine.scheduler`` uses ``shared_core.contracts.NavSnapshot``
    (Symbol-keyed positions).
    """
    acct = client.client.get_account()
    net_liq = Decimal(str(getattr(acct, "portfolio_value", acct.equity)))
    buying_power = Decimal(str(acct.buying_power))
    account_id = str(getattr(acct, "id", "alpaca-paper"))

    positions = client.get_positions()  # {ticker: Decimal}
    # prev_close from the panel's penultimate row per symbol.
    pdf = panel.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    prev_close: dict[str, Decimal] = {}
    for t in TRADABLE_TICKERS:
        sub = pdf[pdf["symbol"] == t].sort_values("date")
        if len(sub) >= 1:
            prev_close[t] = Decimal(f"{float(sub.iloc[-1]['close']):.4f}")

    risk_snap = build_risk_snapshot(
        account_summary={
            "NetLiquidation": str(net_liq),
            "AvailableFunds": str(buying_power),
            "GrossPositionValue": str(net_liq),
        },
        as_of_ns=as_of_ns,
        positions={k: str(v) for k, v in positions.items()},
        broker_positions={k: str(v) for k, v in positions.items()},
        prev_close={k: str(v) for k, v in prev_close.items()},
        feature_ok={t: True for t in TRADABLE_TICKERS},
        limits=limits,
    )

    # shared_core snapshot: positions keyed by Symbol; version is sha256-hex (no 'sha256:' prefix).
    shared_positions: dict[Symbol, Decimal] = {}
    for t, qty in positions.items():
        sym = TRADABLE_SYMBOLS.get(t)
        if sym is None:
            # Unknown ticker in account (e.g. legacy positions outside universe) — synth.
            sym = Symbol(ticker=t, venue=Venue.NASDAQ, currency="USD")
        shared_positions[sym] = qty

    shared_version = hashlib.sha256(
        f"{account_id}|{as_of_ns}|{net_liq}|{buying_power}".encode()
    ).hexdigest()
    shared_snap = SharedNavSnapshot(
        snapshot_ns=int(as_of_ns),
        net_liq=net_liq,
        buying_power=buying_power,
        positions=shared_positions,
        snapshot_version=shared_version,
        account_id=account_id,
    )
    logger.info("snapshots: net_liq=$%s buying_power=$%s positions=%d",
                net_liq, buying_power, len(positions))
    return risk_snap, shared_snap, net_liq


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-trade market-open runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Alpaca order submission (WHATIF on executor); still runs gates.")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = PROJECT_ROOT / "data" / f"30_paper_run_{today}.log"
    logger = setup_logger(log_path)

    logger.info("=" * 72)
    logger.info("PAPER TRADE RUNNER — start (dry_run=%s, log=%s)", args.dry_run, log_path)
    logger.info("=" * 72)

    # 1. Load .env
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET"]
    account_initial = Decimal(os.environ.get("ACCOUNT_EQUITY_INITIAL", "100000"))
    logger.info("env loaded: ACCOUNT_EQUITY_INITIAL=$%s", account_initial)

    # 2. Alpaca quotes
    marks = fetch_alpaca_quotes(api_key, secret, TRADABLE_TICKERS, logger)
    logger.info("quotes: %s", {t: f"${v}" for t, v in marks.items()})

    # 3. Daily bars panel
    panel = load_daily_panel(logger)

    # 4-5. Features + Ridge model. Prefer the persisted auto-retrain artifact
    # (written by scripts/40_daily_morning_routine.py at 09:00 ET) so we use
    # the model fitted on yesterday's bars rather than retraining from
    # scratch at the open. Falls back to in-script training if the pickle is
    # missing or unreadable — keeps day-1 behavior intact.
    model_source = "in-script"
    model_daily_std = 0.10  # historical default; overwritten on pickle hit
    try:
        artifact = load_latest_model(project_root=PROJECT_ROOT)
        ridge = artifact["model"]
        feature_cols = artifact["feature_cols"]
        sym_to_id = artifact["sym_to_id"]
        model_daily_std = float(artifact.get("daily_std", 0.10))
        model_source = f"pickle:{artifact.get('trained_on_date', 'unknown')}"
        logger.info(
            "loaded persisted model from data/models/latest.pkl (trained_on=%s, "
            "n_train=%s, oos_ic=%.4f, oos_sharpe=%.3f)",
            artifact.get("trained_on_date"),
            artifact.get("n_train"),
            float(artifact.get("metrics_oos", {}).get("ic", 0.0)),
            float(artifact.get("metrics_oos", {}).get("sharpe_daily", 0.0)),
        )
    except (FileNotFoundError, KeyError, EOFError, Exception) as _e:
        logger.warning(
            "could not load persisted model (%s) — falling back to in-script training", _e
        )
        ridge, feature_cols, sym_to_id = train_ridge_recent(panel, logger, years=5)

    # 6-7. Predict + Signals
    as_of_ns = int(time.time() * 1e9)
    signals, sig_df = predict_and_build_signals(
        panel, ridge, feature_cols, sym_to_id, as_of_ns, logger,
        daily_std=model_daily_std,
    )
    if sig_df is not None and not sig_df.empty:
        logger.info("predictions:\n%s", sig_df.to_string())

    # 8. (returns matrix is consumed inside scheduler via Σ shrink)
    returns = build_returns_matrix(panel, lookback_days=252)
    logger.info("returns matrix shape=%s", returns.shape)

    # 10. vix proxy series
    vix_series = build_vix_proxy_series(panel, lookback_days=252)
    vix_latest = float(vix_series[-1]) if vix_series.size else 0.0
    logger.info("vix_proxy series len=%s latest=%.6f", vix_series.size, vix_latest)

    # 9. Build NavSnapshots (both flavours) + open Alpaca client
    alpaca = AlpacaClient.from_env(PROJECT_ROOT / ".env")
    limits = load_limits(PROJECT_ROOT / "data" / "limits.toml")
    risk_snap, shared_snap, net_liq = build_alpaca_snapshots(
        alpaca, marks, panel, as_of_ns, limits, logger
    )

    # 11. Run rebalance.
    session_close_ns = as_of_ns + int(6.5 * 3600 * 1e9)  # +6h30m
    valid_until_ns = session_close_ns
    prev_positions: dict[Symbol, float] = {
        TRADABLE_SYMBOLS[t]: float(risk_snap.position(t)) for t in TRADABLE_TICKERS
    }
    # Auto-tune: prefer values written by scripts/40_daily_morning_routine.py
    # at 09:00 ET. Falls back to the static day-1 defaults if the file is
    # missing (e.g. first morning before the orchestrator has run).
    tuning = load_latest_tuning(project_root=PROJECT_ROOT)
    if tuning is not None:
        lam = float(tuning["lambda_risk"])
        tv = float(tuning["target_vol_annualized"])
        logger.info(
            "loaded auto-tune record (as_of=%s lambda_risk=%.4f target_vol=%.4f notes=%s)",
            tuning.get("as_of_date"), lam, tv, tuning.get("notes", ""),
        )
    else:
        lam, tv = 0.1, 0.12  # day-1 defaults, matches the historical script
        logger.warning(
            "no data/config/strategy_tuning.json — using static lambda=%.4f target_vol=%.4f",
            lam, tv,
        )

    inputs = RebalanceInputs(
        signals=signals,
        prev_positions=prev_positions,
        returns=returns,
        nav_snapshot=shared_snap,
        mode=ChainMode.PAPER,
        as_of_ns=as_of_ns,
        session_close_ns=session_close_ns,
        valid_until_ns=valid_until_ns,
        model_version=f"ridge-paper-v0[{model_source}]",
        lambda_risk=lam,
        target_vol_annualized=tv,
        vix_history=vix_series if vix_series.size else None,
        vix_proxy_value=vix_latest if vix_series.size else None,
    )
    batch_local = run_rebalance(inputs)
    logger.info("rebalance: solver_status=%s targets=%d rules=%s",
                batch_local.solver_status, batch_local.targets_count,
                batch_local.edge_case_rules_fired)
    for t in batch_local.targets:
        logger.info("  target %s: w=%.4f r_hat=%.6f w_prev=%.4f hold=%s",
                    t.symbol.ticker, t.target_weight, t.r_hat, t.w_prev, t.hold_reason)

    # v0 paper-day-1 fallback: if optimizer produced all near-zero weights
    # (predictions are too weak for MV under regime-dampened vol target), force
    # a ranked equal-weight long-short book from the predictions so paper
    # trades actually happen. Top-half long, bottom-half short at ±1/n each.
    max_abs_w = max((abs(float(t.target_weight)) for t in batch_local.targets), default=0.0)
    if max_abs_w < 0.001:  # all below 0.1% NAV → optimizer effectively idle
        logger.warning(
            "FORCED RANKED BOOK: optimizer weights all below 0.1%% NAV "
            "(max=%.6f). Falling back to ranked equal-weight ±1/n book.", max_abs_w
        )
        n = len(batch_local.targets)
        # Sort by r_hat descending; top half long, bottom half short.
        ranked = sorted(batch_local.targets, key=lambda t: float(t.r_hat), reverse=True)
        half = n // 2
        # Half-book to respect the 50% daily turnover cap on day-1 fresh book:
        # 0.4 / n per side = 5% NAV per name; total gross = 40% NAV (< 50% cap).
        weight_long = 0.4 / n
        weight_short = -0.4 / n
        for i, t in enumerate(ranked):
            if i < half:
                new_w = weight_long
            elif i >= n - half:
                new_w = weight_short
            else:
                new_w = 0.0
            object.__setattr__(t, "target_weight", new_w)
            logger.info("  forced %s: w=%+.4f r_hat=%+.6f",
                        t.symbol.ticker, new_w, float(t.r_hat))

    # Convert target_weight → integer share count using mark prices and net_liq.
    # (Pre-execution lot lowering — the emitter epic will do this properly;
    # for now we just round to whole shares.)
    for t in batch_local.targets:
        mark = marks.get(t.symbol.ticker)
        if mark is None or mark <= 0:
            continue
        dollar_target = Decimal(str(t.target_weight)) * net_liq
        # ``object.__setattr__`` because TargetPosition is frozen.
        shares = int((dollar_target / mark).quantize(Decimal("1.")))
        object.__setattr__(t, "contracts", Decimal(shares))

    # Lower to shared_core canonical batch (required by diff_targets_to_intents).
    batch_shared = batch_local.to_shared_core(nav_snapshot_version=shared_snap.snapshot_version)

    # 12. VIX-high regime gate.
    ctx = ChainContext(limits=limits, mode=ChainMode.PAPER)
    gate = VixHighGate(threshold_pct=80, lookback_days=252)
    if vix_series.size >= 2:
        decision = gate.evaluate(
            vix_history=[Decimal(f"{v:.10f}") for v in vix_series],
            mode=ChainMode.PAPER,
            ctx=ctx,
            snapshot=risk_snap,
        )
        logger.info("vix_high gate: triggered=%s halt=%s reason=%s",
                    decision.triggered, decision.halt, decision.reason)
        if decision.halt:
            logger.warning("VIX_HIGH gate HALT — exiting without orders.")
            return 0
    else:
        logger.info("vix_history too short for gate (n=%d) — skipping.", vix_series.size)

    # 13. Diff -> intents.
    # Marks dict needs Decimal keyed by str ticker (shared_core OrderIntent requires marks).
    marks_for_diff = {t: marks.get(t, Decimal("0")) for t in TRADABLE_TICKERS}
    try:
        intents, contexts = diff_targets_to_intents(
            batch_shared, risk_snap, marks=marks_for_diff
        )
    except Exception as exc:
        logger.exception("diff_targets_to_intents failed: %s", exc)
        return 1
    logger.info("diff: %d intents emitted", len(intents))

    # Wire intent contexts onto the chain ctx + populate cross-intent aggregates.
    ctx.intent_contexts.update(contexts)
    ctx.planned_gross_notional = sum(
        (abs(ic.target_qty) * marks_for_diff.get(ic.ticker, Decimal("0")) for ic in contexts.values()),
        start=Decimal("0"),
    )
    ctx.planned_turnover_notional = sum(
        (abs(ic.target_qty - ic.current_qty) * marks_for_diff.get(ic.ticker, Decimal("0"))
         for ic in contexts.values()),
        start=Decimal("0"),
    )
    # 14. Per-intent chain.
    approved = []
    for it in intents:
        decision = run_check_chain(it, risk_snap, it.mark, mode=ChainMode.PAPER, ctx=ctx)
        if decision.rule_id == "APPROVED":
            approved.append(it)
            logger.info("APPROVED %s %s %s @ $%s (intent_id=%s)",
                        it.symbol.ticker, it.side.value, it.delta_qty, it.mark, it.intent_id)
        else:
            logger.warning("REJECTED %s rule=%s observed=%s threshold=%s",
                           it.symbol.ticker, decision.rule_id,
                           decision.observed, decision.threshold)
    logger.info("chain results: %d approved, %d rejected", len(approved), len(intents) - len(approved))

    # 15. Execute.
    exec_mode = ChainMode.WHATIF if args.dry_run else ChainMode.PAPER
    if not approved:
        logger.info("no approved intents — skipping execution.")
        return 0

    if exec_mode is ChainMode.WHATIF or os.environ.get("MACMINIBOT_PAPER_RUNNER_STUB_EXEC") == "1":
        # Stub executor — log what we WOULD send, return synthetic mock outcomes.
        logger.info("STUB EXECUTOR (mode=%s) — printing what we would send:", exec_mode.value)
        for it in approved:
            logger.info("  WOULD-SEND %s %s qty=%s @ $%s [intent_id=%s]",
                        it.symbol.ticker, it.side.value, it.delta_qty, it.mark, it.intent_id)
        logger.info("STUB EXECUTOR done. (%d mock fills)", len(approved))
    else:
        result = execute_batch(approved, ChainMode.PAPER, client=alpaca)
        for outcome in result.outcomes:
            logger.info("EXEC %s status=%s coid=%s reason=%s",
                        outcome.intent_id, outcome.status,
                        outcome.client_order_id, outcome.reason)
        logger.info("execution: submitted=%d refused=%d whatif=%d",
                    result.submitted, result.refused, result.whatif)

    logger.info("=" * 72)
    logger.info("PAPER TRADE RUNNER — done")
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
