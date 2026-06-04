"""Walk-forward driver — wraps a signal-generator callable into per-fold
trades, fees, slippage, equity curve, and metrics.

Per CONTEXT.md, the *split* logic is owned by ``ml_research.cv.WalkForward
Harness``; this driver does NOT reimplement splitting. When ml-research's
harness is importable we use it; otherwise we fall back to a local
``ExpandingWindowFolder`` that produces the same ``FoldSpec`` 5-tuple
shape so callers can develop against the driver in isolation.

The driver is intentionally small: per-fold loop, fee/slippage hookup,
metric aggregation. Bootstrap CIs (Epic 8) and the FDF gate (Epic 11)
build on top of this driver — they are NOT inlined here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Protocol

import numpy as np
import pandas as pd

from backtesting import fees, metrics, slippage
from backtesting.regime import REGIME_LABELS, RegimeTagger

# ---- shared_core.contracts FoldSpec — stub if absent -----------------------
# Canonical signature: FoldSpec(fold_id, train_idx, val_idx, test_idx, fold_meta)
# with ndarray index fields (see shared_core/contracts.py and CONTEXT.md
# cycle-3 reconciliation). We import the real type when shared_core is on
# PYTHONPATH; otherwise fall back to an identically-shaped local stub.
try:  # pragma: no cover - import-guard
    from shared_core.contracts import FoldSpec  # type: ignore
except ImportError:  # pragma: no cover - fallback
    @dataclass(frozen=True, slots=True)
    class FoldSpec:  # type: ignore[no-redef]
        """Local stub mirroring shared_core.contracts.FoldSpec.

        Field order: (fold_id, train_idx, val_idx, test_idx, fold_meta).
        Index fields are ``np.ndarray`` (1-D, datetime64[ns, UTC] for our
        use case).
        """

        fold_id: int
        train_idx: np.ndarray
        val_idx: np.ndarray
        test_idx: np.ndarray
        fold_meta: dict[str, Any] = field(default_factory=dict)

        def __iter__(self) -> Iterator[Any]:
            yield self.fold_id
            yield self.train_idx
            yield self.val_idx
            yield self.test_idx
            yield self.fold_meta


__all__ = [
    "ExpandingWindowFolder",
    "FoldMetrics",
    "FoldSpec",
    "RegimeMetrics",
    "SignalGenerator",
    "WalkForwardDriver",
    "WalkForwardResult",
]


@dataclass(frozen=True, slots=True)
class RegimeMetrics:
    """Per-regime breakdown of OOS performance (cycle-9 Epic 17).

    Computed only when the driver is given a regime tagger. The cycle-5
    CONTEXT spec calls this out as the gate against "DSR passes on
    aggregate but collapses in (e.g.) high-vol regimes" — the renderer
    flags any bucket where DSR drops by >0.3 vs the aggregate.
    """

    regime: str
    n_obs: int
    sharpe_net: float
    dsr: float
    fee_drag_sharpe: float  # gross - net Sharpe within the regime

    def as_dict(self) -> dict[str, float | int | str]:
        return {f: getattr(self, f) for f in self.__slots__}


class SignalGenerator(Protocol):
    """Callable contract for a strategy signal-generator.

    Called once per *bar* in the test fold with the row of features (a
    ``pandas.Series`` indexed by symbol → close) plus the train-window
    bars (a ``pandas.DataFrame``) and must return a ``dict[symbol, target_
    weight]`` where weights sum (in absolute value) to ``<= 1.0``.

    A constant-long generator looks like::

        def always_long(bar, train_window) -> dict[str, float]:
            return {sym: 1.0 / len(bar) for sym in bar.index}

    Returning ``{}`` means "go flat for the next session".
    """

    def __call__(
        self,
        bar: pd.Series,
        train_window: pd.DataFrame,
    ) -> Mapping[str, float]:
        ...


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    fold_id: int
    n_obs: int
    sharpe_gross: float
    sharpe_net: float
    sortino_net: float
    calmar_net: float
    max_drawdown: float
    psr_net: float
    fee_drag_sharpe: float  # gross - net
    total_fees_usd: float
    total_slippage_usd: float
    turnover: float
    final_equity: float

    def as_dict(self) -> dict[str, float | int]:
        return {f: getattr(self, f) for f in self.__slots__}


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    fold_metrics: list[FoldMetrics]
    aggregate_sharpe_net: float
    aggregate_sharpe_gross: float
    aggregate_max_drawdown: float
    aggregate_dsr: float
    aggregate_pbo: float
    equity_curve: np.ndarray  # full out-of-sample concatenation, 1.0-normalised
    regime_metrics: list[RegimeMetrics] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_metrics": [fm.as_dict() for fm in self.fold_metrics],
            "aggregate_sharpe_net": self.aggregate_sharpe_net,
            "aggregate_sharpe_gross": self.aggregate_sharpe_gross,
            "aggregate_max_drawdown": self.aggregate_max_drawdown,
            "aggregate_dsr": self.aggregate_dsr,
            "aggregate_pbo": self.aggregate_pbo,
            "n_folds": len(self.fold_metrics),
            "regime_metrics": [rm.as_dict() for rm in self.regime_metrics],
        }


class ExpandingWindowFolder:
    """Local fallback splitter — used only when ``ml_research.cv`` is absent.

    Mirrors the *shape* of ``WalkForwardHarness.split(index)``: expanding
    train window starting from ``train_min_days``, embargo + purge around
    the boundary, fixed test_days per fold, sliding by ``test_days``.

    NOT a substitute for the canonical splitter (CONTEXT.md is explicit:
    backtesting NEVER owns splitter code). This exists purely so unit
    tests of the driver can run without depending on ml-research being on
    PYTHONPATH.
    """

    def __init__(
        self,
        train_min_days: int = 30,
        test_days: int = 21,
        embargo_days: int = 5,
        purge_days: int = 1,
    ) -> None:
        if train_min_days < 1 or test_days < 1:
            raise ValueError("train_min_days and test_days must be >= 1")
        self.train_min_days = train_min_days
        self.test_days = test_days
        self.embargo_days = embargo_days
        self.purge_days = purge_days

    def split(self, index: pd.DatetimeIndex) -> Iterator[FoldSpec]:
        index_arr = np.asarray(index)
        n = len(index_arr)
        if n < self.train_min_days + self.test_days:
            return
        fold_id = 0
        train_end = self.train_min_days
        while train_end + self.purge_days + self.test_days <= n:
            test_start = train_end + self.purge_days
            test_end = min(test_start + self.test_days, n)
            train_slice_end = train_end - self.purge_days if self.purge_days > 0 else train_end
            train_idx = index_arr[:train_slice_end]
            # val_idx is empty in this fallback — ml-research's harness uses it
            # for inner-loop hyperparam search; we don't here.
            val_idx = index_arr[0:0]
            test_idx = index_arr[test_start:test_end]
            yield FoldSpec(
                fold_id=fold_id,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                fold_meta={
                    "embargo_days": self.embargo_days,
                    "purge_days": self.purge_days,
                },
            )
            fold_id += 1
            train_end = test_end + self.embargo_days


class WalkForwardDriver:
    """Walk-forward replay over daily bars with fee + slippage charged.

    Inputs
    ------
    bars : pd.DataFrame
        MultiIndex (date, symbol) → at least columns ``open``, ``close``,
        ``volume``, ``adv_21d_shares``, ``daily_vol_bps``. Index level 0
        must be a ``DatetimeIndex``.
    signal_generator : SignalGenerator
        Called once per test-fold bar; returns target weights.
    splitter : object with ``.split(index) -> Iterator[FoldSpec]``
        Defaults to :class:`ExpandingWindowFolder`. Pass an
        ``ml_research.cv.WalkForwardHarness`` instance in production.
    asset_class : str
        Used to look up the spread-half floor and pick the fee function.
    initial_capital_usd : float
        Starting NAV per fold for sizing target shares.

    The driver computes target weights at ``close[T-1]``, fills at
    ``open[T]`` (pitfall #4: gap PnL is broken out from session PnL), and
    charges per-trade fees + Almgren-Chriss slippage with the asset-class
    spread floor.
    """

    REQUIRED_COLUMNS = ("open", "close", "volume", "adv_21d_shares", "daily_vol_bps")
    _SUPPORTED_BROKERS = frozenset({"ibkr_pro", "alpaca"})

    def __init__(
        self,
        bars: pd.DataFrame,
        signal_generator: SignalGenerator,
        splitter: Any | None = None,
        *,
        asset_class: str = "equity",
        initial_capital_usd: float = 100_000.0,
        slippage_k: float = 0.1,
        broker: str = "ibkr_pro",
        regime_tagger: RegimeTagger | None = None,
        vix_proxy: pd.Series | None = None,
    ) -> None:
        self._validate_bars(bars)
        self.bars = bars.sort_index()
        self.signal_generator = signal_generator
        self.splitter = splitter or ExpandingWindowFolder()
        self.asset_class = asset_class
        self.initial_capital_usd = float(initial_capital_usd)
        self.slippage_k = float(slippage_k)
        broker_norm = broker.lower()
        if broker_norm not in self._SUPPORTED_BROKERS:
            raise ValueError(
                f"unknown broker {broker!r}; expected one of "
                f"{sorted(self._SUPPORTED_BROKERS)}"
            )
        if broker_norm == "alpaca" and asset_class != "equity":
            # Alpaca's commission-free retail tier is US-equities-only; the
            # cycle-5 CONTEXT calls this out as the IEX-data-caveat path.
            raise ValueError(
                f"broker='alpaca' is only valid for asset_class='equity'; "
                f"got asset_class={asset_class!r}"
            )
        self.broker = broker_norm
        # Optional regime stratification (cycle-9 Epic 17). Tagger + vix
        # series are both required when either is provided; tagger is the
        # fitted percentile cutoffs, vix_proxy is the per-date value the
        # tagger buckets against.
        if (regime_tagger is None) != (vix_proxy is None):
            raise ValueError(
                "regime_tagger and vix_proxy must both be provided or both omitted"
            )
        self.regime_tagger = regime_tagger
        if vix_proxy is not None:
            if not isinstance(vix_proxy.index, pd.DatetimeIndex):
                raise TypeError(
                    "vix_proxy must have DatetimeIndex; "
                    f"got {type(vix_proxy.index).__name__}"
                )
            self.vix_proxy = vix_proxy.sort_index()
        else:
            self.vix_proxy = None

    @staticmethod
    def _validate_bars(bars: pd.DataFrame) -> None:
        if not isinstance(bars.index, pd.MultiIndex) or bars.index.nlevels != 2:
            raise ValueError(
                "bars must have a 2-level MultiIndex (date, symbol); "
                f"got {bars.index!r}"
            )
        missing = set(WalkForwardDriver.REQUIRED_COLUMNS) - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing required columns: {sorted(missing)}")

    # ---- main loop ---------------------------------------------------------

    def run(self) -> WalkForwardResult:
        dates = self.bars.index.get_level_values(0).unique().sort_values()
        fold_results: list[FoldMetrics] = []
        is_sr_per_fold: list[float] = []
        oos_sr_per_fold: list[float] = []
        full_oos_returns: list[float] = []
        full_oos_gross: list[float] = []
        full_oos_dates: list[pd.Timestamp] = []

        for fold in self.splitter.split(dates):
            fm, fold_returns, fold_gross, fold_dates = self._run_fold(fold)
            fold_results.append(fm)
            is_sr_per_fold.append(fm.sharpe_gross)
            oos_sr_per_fold.append(fm.sharpe_net)
            full_oos_returns.extend(fold_returns.tolist())
            full_oos_gross.extend(fold_gross.tolist())
            full_oos_dates.extend(fold_dates)

        oos_arr = np.asarray(full_oos_returns, dtype=np.float64)
        if oos_arr.size == 0:
            equity = np.array([1.0])
            agg_sharpe_net = float("nan")
            agg_max_dd = 0.0
            agg_dsr = float("nan")
        else:
            equity = np.cumprod(1.0 + oos_arr)
            agg_sharpe_net = metrics.sharpe(oos_arr)
            agg_max_dd = metrics.max_drawdown(equity)
            # DSR uses the cross-fold count as n_trials proxy at the per-run
            # aggregate level (cycle-2: full DSR-vs-FDF is Epic 11; this is
            # the per-run gate signal).
            agg_dsr = metrics.dsr(
                sr_observed=agg_sharpe_net,
                n_trials=max(1, len(fold_results)),
                returns=oos_arr,
            )

        # Gross aggregate: re-run the dot-product but skip fee/slip charge.
        # Cheap because we already have the gross per-fold Sharpe.
        gross_sharpes = np.asarray([fm.sharpe_gross for fm in fold_results])
        agg_sharpe_gross = float(np.nanmean(gross_sharpes)) if gross_sharpes.size else float("nan")

        # PBO across folds — IS = gross, OOS = net (proxy: same fold's net
        # is the "after-cost OOS" perf for the IS-selected signal). The full
        # combinatorial PBO is Epic 8; this is the per-run signal.
        agg_pbo = (
            metrics.pbo(is_sr_per_fold, oos_sr_per_fold)
            if len(fold_results) >= 2
            else float("nan")
        )

        regime_metrics = self._compute_regime_metrics(
            full_oos_dates, full_oos_gross, full_oos_returns
        )

        return WalkForwardResult(
            fold_metrics=fold_results,
            aggregate_sharpe_net=agg_sharpe_net,
            aggregate_sharpe_gross=agg_sharpe_gross,
            aggregate_max_drawdown=agg_max_dd,
            aggregate_dsr=agg_dsr,
            aggregate_pbo=agg_pbo,
            equity_curve=equity,
            regime_metrics=regime_metrics,
        )

    # ---- regime stratification (cycle-9 Epic 17) --------------------------

    def _compute_regime_metrics(
        self,
        dates: list[pd.Timestamp],
        gross_returns: list[float],
        net_returns: list[float],
    ) -> list[RegimeMetrics]:
        """Bucket OOS daily returns by VIX regime and compute Sharpe/DSR/
        fee-drag per bucket. No-op if no tagger was provided.

        Implementation notes:
        - Sharpe + fee-drag are straight per-bucket reductions on the
          subset of daily returns whose date tags into that bucket.
        - DSR per bucket uses ``n_trials=1`` (this is a *replication* of
          the same fitted model in a sub-sample, not an independent
          search), so DSR collapses to PSR vs zero — the truthful per-
          regime "is this >0 with statistical confidence?" answer.
          The cycle-8 fix in CONTEXT.md improvement #18 applies here.
        - Returns an empty list if no tagger or if no dates produce
          warm-cutoff tags.
        """
        if self.regime_tagger is None or self.vix_proxy is None:
            return []
        if not dates:
            return []
        # Bucket each session date.
        vix_map = self.vix_proxy.to_dict()
        labels: list[str] = []
        for d in dates:
            ts = pd.Timestamp(d)
            v = vix_map.get(ts, float("nan"))
            labels.append(self.regime_tagger.tag(ts, v))
        arr_labels = np.asarray(labels)
        arr_net = np.asarray(net_returns, dtype=np.float64)
        arr_gross = np.asarray(gross_returns, dtype=np.float64)

        out: list[RegimeMetrics] = []
        for regime in REGIME_LABELS:
            mask = arr_labels == regime
            n = int(mask.sum())
            if n < 2:
                out.append(
                    RegimeMetrics(
                        regime=regime,
                        n_obs=n,
                        sharpe_net=float("nan"),
                        dsr=float("nan"),
                        fee_drag_sharpe=float("nan"),
                    )
                )
                continue
            net_slice = arr_net[mask]
            gross_slice = arr_gross[mask]
            sr_net = metrics.sharpe(net_slice)
            sr_gross = metrics.sharpe(gross_slice)
            # Per-regime DSR: n_trials=1 because we're *replicating* the
            # fitted strategy on a regime sub-sample, not searching trials
            # (cycle-8 fix, CONTEXT improvement #18).
            dsr_val = metrics.dsr(
                sr_observed=sr_net, n_trials=1, returns=net_slice,
            )
            fee_drag = (
                sr_gross - sr_net
                if (math.isfinite(sr_gross) and math.isfinite(sr_net))
                else float("nan")
            )
            out.append(
                RegimeMetrics(
                    regime=regime,
                    n_obs=n,
                    sharpe_net=sr_net,
                    dsr=dsr_val,
                    fee_drag_sharpe=fee_drag,
                )
            )
        return out

    # ---- per-fold ----------------------------------------------------------

    def _run_fold(
        self, fold: FoldSpec
    ) -> tuple[FoldMetrics, np.ndarray, np.ndarray, list[pd.Timestamp]]:
        train_bars = self.bars.loc[(fold.train_idx, slice(None)), :] if len(fold.train_idx) else self.bars.iloc[0:0]
        test_dates = fold.test_idx

        nav = self.initial_capital_usd
        positions: dict[str, float] = {}  # symbol → shares held entering session
        ref_close_prev: dict[str, float] = {}

        session_returns: list[float] = []
        gross_session_returns: list[float] = []
        session_dates: list[pd.Timestamp] = []
        total_fees = 0.0
        total_slippage_usd = 0.0
        total_turnover_usd = 0.0

        # Seed prev_close from the last train day (or first test day if no train).
        seed_dates = fold.train_idx[-1:] if len(fold.train_idx) else test_dates[:1]
        if len(seed_dates):
            seed_row = self.bars.loc[(seed_dates[-1], slice(None)), :]
            ref_close_prev = {
                sym: float(row["close"])
                for (_d, sym), row in seed_row.iterrows()
            }

        for date in test_dates:
            try:
                day_bars = self.bars.loc[(date, slice(None)), :]
            except KeyError:
                continue
            day_close = day_bars["close"].droplevel(0)
            day_open = day_bars["open"].droplevel(0)
            day_adv = day_bars["adv_21d_shares"].droplevel(0)
            day_vol_bps = day_bars["daily_vol_bps"].droplevel(0)

            # Strategy signal — uses T-1 close (positional 'bar' arg points to
            # the previous-close series for that day).
            prev_close_series = pd.Series(ref_close_prev) if ref_close_prev else day_close
            target_weights = self.signal_generator(prev_close_series, train_bars)

            # Gap PnL: open[T] vs close[T-1] on currently-held positions.
            gap_pnl = 0.0
            for sym, qty in positions.items():
                prev = ref_close_prev.get(sym)
                opn = day_open.get(sym, np.nan)
                if prev is not None and not math.isnan(opn):
                    gap_pnl += qty * (opn - prev)

            # Compute target shares at open[T] given current nav.
            target_shares: dict[str, float] = {}
            for sym, w in target_weights.items():
                if sym not in day_open.index:
                    continue
                px = float(day_open[sym])
                if px <= 0:
                    continue
                target_shares[sym] = (w * nav) / px

            # Trade to target. Charge fees + slippage on the *delta*.
            session_fees = 0.0
            session_slip_usd = 0.0
            session_turnover_usd = 0.0
            new_positions: dict[str, float] = {}
            all_syms = set(positions) | set(target_shares)
            for sym in all_syms:
                cur = positions.get(sym, 0.0)
                tgt = target_shares.get(sym, 0.0)
                delta_shares = tgt - cur
                if delta_shares == 0.0:
                    new_positions[sym] = cur
                    continue
                px_open = float(day_open[sym])
                abs_delta = abs(delta_shares)
                trade_value = abs_delta * px_open
                session_turnover_usd += trade_value

                # Fee charge (asset-class + broker dispatch).
                if self.asset_class == "equity":
                    if self.broker == "alpaca":
                        # Alpaca commission-free: sell-side regs only. Side is
                        # derived from delta sign so longs flipping to flat
                        # correctly attract the sell-side reg fees.
                        side = "SELL" if delta_shares < 0 else "BUY"
                        fb = fees.equity_alpaca(
                            abs_delta, px_open, side=side, year=int(date.year),
                        )
                    else:
                        fb = fees.equity_tiered(abs_delta, px_open)
                    fee = fb.total
                elif self.asset_class == "future":
                    ff = fees.future_cme(abs_delta)
                    fee = ff.total
                elif self.asset_class == "fx":
                    fxb = fees.fx_idealpro(trade_value)
                    fee = fxb.total
                else:
                    raise ValueError(f"unknown asset_class {self.asset_class!r}")
                session_fees += fee

                # Slippage charge via Almgren-Chriss + floor.
                adv = float(day_adv.get(sym, np.nan))
                vol_bps = float(day_vol_bps.get(sym, np.nan))
                if adv > 0 and not math.isnan(vol_bps):
                    slip = slippage.slippage_bps_with_floor(
                        volume_traded=abs_delta,
                        adv=adv,
                        daily_vol_bps=vol_bps,
                        asset_class=self.asset_class,
                        k=self.slippage_k,
                    )
                    slip_bps = slip.total_bps
                else:
                    slip_bps = slippage.SPREAD_HALF_FLOOR_BPS[self.asset_class]
                session_slip_usd += (slip_bps / 1e4) * trade_value

                new_positions[sym] = tgt

            positions = {sym: q for sym, q in new_positions.items() if q != 0.0}

            # Session PnL: open[T] → close[T] on new positions.
            session_pnl = 0.0
            for sym, qty in positions.items():
                opn = float(day_open[sym])
                cls = float(day_close[sym])
                session_pnl += qty * (cls - opn)

            gross_pnl = gap_pnl + session_pnl
            net_pnl = gross_pnl - session_fees - session_slip_usd

            if nav <= 0:
                # Wipeout: stop trading this fold.
                break
            gross_return = gross_pnl / nav
            net_return = net_pnl / nav

            gross_session_returns.append(gross_return)
            session_returns.append(net_return)
            session_dates.append(pd.Timestamp(date))
            total_fees += session_fees
            total_slippage_usd += session_slip_usd
            total_turnover_usd += session_turnover_usd

            nav += net_pnl
            ref_close_prev = {sym: float(day_close[sym]) for sym in day_close.index}

        net_arr = np.asarray(session_returns, dtype=np.float64)
        gross_arr = np.asarray(gross_session_returns, dtype=np.float64)
        n = net_arr.size

        if n == 0:
            sharpe_net = float("nan")
            sharpe_gross = float("nan")
            sortino_net = float("nan")
            calmar_net = float("nan")
            mdd = 0.0
            psr_val = float("nan")
            final_eq = self.initial_capital_usd
            turnover = 0.0
        else:
            sharpe_net = metrics.sharpe(net_arr)
            sharpe_gross = metrics.sharpe(gross_arr)
            sortino_net = metrics.sortino(net_arr)
            calmar_net = metrics.calmar(net_arr)
            equity = np.cumprod(1.0 + net_arr)
            mdd = metrics.max_drawdown(equity)
            psr_val = metrics.psr(net_arr, benchmark_sr=0.0)
            final_eq = self.initial_capital_usd * float(equity[-1])
            turnover = total_turnover_usd / self.initial_capital_usd

        fee_drag = (sharpe_gross - sharpe_net) if (math.isfinite(sharpe_gross) and math.isfinite(sharpe_net)) else float("nan")
        fm = FoldMetrics(
            fold_id=fold.fold_id,
            n_obs=n,
            sharpe_gross=sharpe_gross,
            sharpe_net=sharpe_net,
            sortino_net=sortino_net,
            calmar_net=calmar_net,
            max_drawdown=mdd,
            psr_net=psr_val,
            fee_drag_sharpe=fee_drag,
            total_fees_usd=total_fees,
            total_slippage_usd=total_slippage_usd,
            turnover=turnover,
            final_equity=final_eq,
        )
        return fm, net_arr, gross_arr, session_dates
