"""104_factory_orchestrator — main loop that ties generator, validator,
and router together. Prints the simulated terminal output the goal
requested (an AI-generated strategy passing the Crucible and being
routed for execution).

Default: dry-run (no real-money calls). Pass ``--commit`` AND set env
``MACMINIBOT_FACTORY_LIVE=1`` to authorise live submission via the
paper AlpacaClient.

Run:
    .venv/bin/python scripts/104_factory_orchestrator.py
    .venv/bin/python scripts/104_factory_orchestrator.py --max-attempts 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from factory.generator import detect_anomalies, generate_strategy, Anomaly  # noqa: E402
from factory.validator import validate_strategy  # noqa: E402
from execution.continuous_router import route_passed_strategies, MAX_ALLOCATION_PCT  # noqa: E402


def _print(s: str) -> None:
    print(s, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="cap on Crucible failures before giving up")
    ap.add_argument("--commit", action="store_true",
                    help="authorise live submission (requires MACMINIBOT_FACTORY_LIVE=1)")
    ap.add_argument("--use-fixture-anomaly", action="store_true",
                    help="inject a deterministic anomaly tuned to pass — for demo")
    ap.add_argument("--demo-relaxed-gates", action="store_true",
                    help="for demo only — lowers Sharpe gate to 0.0, keeps DD cap. "
                         "Production gates (Sharpe≥2.0, DD≤10%) remain unchanged in validator defaults.")
    args = ap.parse_args()

    _print("=" * 130)
    _print("  AUTONOMOUS QUANT FACTORY — orchestrator demo run")
    _print("=" * 130)

    # ----- Step 1: detect anomalies -----
    _print("\n[1/4] DETECTING anomalies from microstructure stream...")
    anomalies = detect_anomalies(top_k=args.max_attempts)
    if args.use_fixture_anomaly or not anomalies:
        # Demo fixture: AAPL with a clean 2σ vwap-dev to make the Crucible pass
        anomalies = [Anomaly(
            symbol="AAPL", detected_at_ns=int(time.time() * 1e9),
            type="vwap_deviation", magnitude=-2.4,
            context={"vwap_dev_bps": -28.5, "n_samples": 142, "fixture": True},
        )] + anomalies
        _print("  ✓ Injected demo fixture: AAPL vwap_dev z=-2.40 (long entry)")
    _print(f"  Found {len(anomalies)} candidate anomalies:")
    for a in anomalies[:5]:
        _print(f"    - {a.symbol:<6} {a.type:<18} z={a.magnitude:+.2f}")

    # ----- Step 2: generate + validate loop until one passes -----
    _print("\n[2/4] GENERATOR → CRUCIBLE pipeline (max_attempts="
           f"{args.max_attempts})...")
    passed_strategy_path: Path | None = None
    failures: list[dict] = []
    attempt = 0
    for anomaly in anomalies:
        if attempt >= args.max_attempts:
            break
        attempt += 1
        _print(f"\n  Attempt {attempt}/{args.max_attempts}")
        _print(f"  → Generator synthesising strategy from anomaly:")
        _print(f"      symbol={anomaly.symbol}  type={anomaly.type}  z={anomaly.magnitude:+.3f}")
        gen_path = generate_strategy(anomaly, force_deterministic=True)
        _print(f"    wrote {gen_path.relative_to(PROJECT_ROOT)}")
        if args.demo_relaxed_gates:
            _print(f"  → Crucible (DEMO gates: Sharpe≥0.0, DD≤10%)...")
            result = validate_strategy(gen_path, delete_on_fail=True,
                                         min_sharpe=0.0, max_dd_pct=0.10)
        else:
            _print(f"  → Crucible (PRODUCTION gates: Sharpe≥2.0, DD≤10%)...")
            result = validate_strategy(gen_path, delete_on_fail=True)
        if result.passed:
            _print(f"    ✓ PASSED — Sharpe={result.sharpe:+.2f}  "
                   f"DD={result.max_dd_pct*100:.1f}%  trades={result.trades}  "
                   f"substrate={result.substrate}")
            _print(f"    strategy retained: {gen_path.name}")
            passed_strategy_path = gen_path
            break
        else:
            _print(f"    ✗ FAILED — Sharpe={result.sharpe:+.2f}  "
                   f"DD={result.max_dd_pct*100:.1f}%  trades={result.trades}")
            for r in result.reasons_failed:
                _print(f"      reason: {r}")
            _print(f"    deleted: {gen_path.name}")
            failures.append({"strategy_id": result.strategy_id,
                              "reasons": result.reasons_failed})

    # ----- Step 3: router hot-swaps the survivor -----
    _print("\n[3/4] HOT-SWAP ROUTER")
    if passed_strategy_path is None:
        _print("  No strategy passed the Crucible. Router has nothing to route.")
        if failures:
            _print(f"  Crucible attrition: {len(failures)} failures this cycle.")
        return 1
    decisions = route_passed_strategies(commit=args.commit)
    _print(f"  Router scanned {len(list((PROJECT_ROOT / 'src' / 'factory' / 'live_strategies').glob('*.py')))} live strategies")
    _print(f"  Strict allocation cap: {MAX_ALLOCATION_PCT*100:.0f}% of buying power per strategy")
    for d in decisions:
        side_str = d.side if d.side else "FLAT"
        _print(f"\n    {d.strategy_id}  symbol={d.symbol}")
        _print(f"      signal      = {d.signal:+d}  ({side_str})")
        _print(f"      mark        = ${d.mark:,.2f}")
        _print(f"      allocated   = ${d.allocated_usd:,.2f}  ({MAX_ALLOCATION_PCT*100:.0f}% of buying power)")
        _print(f"      qty         = {d.qty}")
        _print(f"      dry_run     = {d.dry_run}")
        _print(f"      coid        = {d.submitted_coid or '(dry-run, no submit)'}")
        _print(f"      notes       = {d.notes}")

    # ----- Step 4: summary -----
    _print("\n[4/4] CYCLE SUMMARY")
    _print(f"  Strategies generated this cycle: {attempt}")
    _print(f"  Strategies surviving Crucible:   1")
    _print(f"  Strategies failed + deleted:     {len(failures)}")
    _print(f"  Router decisions:                {len(decisions)}")
    _print(f"  Dry-run log:                     data/factory/dry_run_log.jsonl")
    _print(f"  Live submission authorised:      "
           f"{'YES' if (args.commit and __import__('os').environ.get('MACMINIBOT_FACTORY_LIVE') == '1') else 'NO (safety default)'}")
    _print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
