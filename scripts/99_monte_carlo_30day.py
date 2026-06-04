"""99_monte_carlo_30day — Monte Carlo simulation of 30-day forward P&L.

Question under test
===================
Is the live bot underperforming the OOS projection, or is the current 4-day
draw ( -$127 ) inside the noise envelope we should expect from a strategy
with the OOS test's mean / std?

Inputs (from OOS test, n=103 days)
==================================
    mu     = +$220 / day       (was +0.22% on roughly $100k notional)
    sigma  = +$791 / day       (was 0.79%/day std)
    n_oos  = 103 days          (used to bootstrap parametric confidence)

Method
======
1. 10,000 Monte Carlo paths × 30 trading days of i.i.d. Normal(mu, sigma).
2. Cumulative P&L = cumsum across each path's 30 daily draws.
3. Report percentile band (5 / 25 / 50 / 75 / 95) of cumulative day-30 P&L.
4. P(negative at day 30), P(any-time drawdown < -$5,000 from peak).
5. Locate the live 4-day actual ( -$127 ) inside the simulated day-4
   marginal distribution → percentile rank.
6. Bonus: in a random 30-day window, how many 4-day sub-windows hit
   -$127 or worse by pure chance? This tells us if our current draw is
   normal or unusual on its own terms.

All numbers are P&L dollars (matches the user's day-by-day log).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# -----------------------------------------------------------------------------
# 1. Parameters (operator-supplied)
# -----------------------------------------------------------------------------
MU_DAY = 220.0          # mean daily P&L from OOS test ($/day)
SIGMA_DAY = 791.0       # std of daily P&L from OOS test ($/day)
N_OOS_SAMPLE = 103      # OOS sample size (for SE on the mean)
HORIZON_DAYS = 30
N_PATHS = 10_000
RNG_SEED = 20260522

# Live observations to date (4 days)
LIVE_DAILY = np.array([-100.0, +250.0, -290.0, +13.0])
LIVE_CUM = LIVE_DAILY.cumsum()  # [-100, +150, -140, -127]
ACTUAL_DAY4 = float(LIVE_CUM[-1])   # -$127

# Reference drawdown threshold (the brief asked about -5%; on ~$100k that is
# -$5,000, which is the more meaningful "ouch" line for this bot)
DRAWDOWN_THRESHOLD = -5_000.0
BREAK_EVEN_OR_BETTER = 0.0
BAD_DRAWDOWN = -1_000.0


# -----------------------------------------------------------------------------
# 2. Run the simulation
# -----------------------------------------------------------------------------
def simulate(mu: float, sigma: float, days: int, paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    daily = rng.normal(loc=mu, scale=sigma, size=(paths, days))
    cum = daily.cumsum(axis=1)   # shape: (paths, days)
    return cum


def main() -> None:
    print("=" * 72)
    print("MONTE CARLO — 30-day forward P&L, n=10,000 paths")
    print("=" * 72)
    print(f"Inputs   : mu=${MU_DAY:.0f}/day  sigma=${SIGMA_DAY:.0f}/day  "
          f"n_oos={N_OOS_SAMPLE}  horizon={HORIZON_DAYS}d  paths={N_PATHS:,}")
    print()

    cum = simulate(MU_DAY, SIGMA_DAY, HORIZON_DAYS, N_PATHS, RNG_SEED)
    final = cum[:, -1]   # cumulative P&L at day 30

    # ---- Percentile band of day-30 cumulative ----------------------------------
    print("-" * 72)
    print("Day-30 cumulative P&L percentiles")
    print("-" * 72)
    qs = [5, 25, 50, 75, 95]
    pcts = np.percentile(final, qs)
    for q, v in zip(qs, pcts):
        print(f"  p{q:<2d}  ${v:>9,.0f}")
    print(f"  mean ${final.mean():>9,.0f}     (analytic = ${MU_DAY*HORIZON_DAYS:,.0f})")
    print(f"  std  ${final.std(ddof=1):>9,.0f}     "
          f"(analytic = ${SIGMA_DAY*np.sqrt(HORIZON_DAYS):,.0f})")
    print()

    # ---- Rough histogram for the report ----------------------------------------
    print("-" * 72)
    print("Histogram of day-30 cumulative P&L (1,000-wide bins)")
    print("-" * 72)
    bin_edges = np.arange(-8_000, 18_001, 1_000)
    counts, _ = np.histogram(final, bins=bin_edges)
    max_count = counts.max()
    for i, c in enumerate(counts):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bar = "#" * int(round(40 * c / max_count))
        pct = 100 * c / N_PATHS
        print(f"  [${lo:>+6,d} .. ${hi:>+6,d})  {c:>4d}  {pct:4.1f}%  {bar}")
    print()

    # ---- Probabilities ---------------------------------------------------------
    p_neg_day30 = float(np.mean(final < 0))
    p_break_even = float(np.mean(final >= BREAK_EVEN_OR_BETTER))
    p_worse_than_1k = float(np.mean(final <= BAD_DRAWDOWN))

    # Running peak → drawdown analysis along each path
    running_peak = np.maximum.accumulate(cum, axis=1)
    drawdown = cum - running_peak   # always <= 0
    max_dd_per_path = drawdown.min(axis=1)
    p_dd_below_5k = float(np.mean(max_dd_per_path <= DRAWDOWN_THRESHOLD))
    p_dd_below_1k = float(np.mean(max_dd_per_path <= -1_000.0))

    print("-" * 72)
    print("Probabilities at / during the 30-day window")
    print("-" * 72)
    print(f"  P(net negative at day 30)             {p_neg_day30:6.1%}")
    print(f"  P(break-even or better at day 30)     {p_break_even:6.1%}")
    print(f"  P(end day 30 worse than -$1,000)      {p_worse_than_1k:6.1%}")
    print(f"  P(any drawdown worse than -$1,000)    {p_dd_below_1k:6.1%}")
    print(f"  P(any drawdown worse than -$5,000)    {p_dd_below_5k:6.1%}")
    print(f"  Expected end-of-day-30 cumulative     ${final.mean():,.0f}")
    print()

    # ---- Where does live -$127 at day 4 sit? ----------------------------------
    print("-" * 72)
    print("Live performance localisation (4 days, cumulative = -$127)")
    print("-" * 72)
    day4_marginal = cum[:, 3]   # index 3 = day 4
    pct_rank = float(stats.percentileofscore(day4_marginal, ACTUAL_DAY4, kind="mean"))
    mu_d4 = MU_DAY * 4
    sigma_d4 = SIGMA_DAY * np.sqrt(4)
    z_actual = (ACTUAL_DAY4 - mu_d4) / sigma_d4
    p_value_two_sided = float(2 * (1 - stats.norm.cdf(abs(z_actual))))

    print(f"  Day-4 marginal mean      ${day4_marginal.mean():>+8,.0f}")
    print(f"  Day-4 marginal std       ${day4_marginal.std(ddof=1):>+8,.0f}")
    print(f"  Live cumulative day 4    ${ACTUAL_DAY4:>+8,.0f}")
    print(f"  Percentile rank          {pct_rank:5.1f}  (50 = median)")
    print(f"  Z-score                  {z_actual:+5.2f}  (mu_4d=${mu_d4:.0f}, "
          f"sigma_4d=${sigma_d4:.0f})")
    print(f"  Two-sided p              {p_value_two_sided:6.3f}    "
          f"(probability of |result| >= |actual| under H0)")
    print()

    # ---- Bonus: how unusual is -$127-or-worse in a 4-day window? --------------
    print("-" * 72)
    print("Bonus: 4-day windows at or below -$127 inside a 30-day path")
    print("-" * 72)
    # rolling 4-day sum across each simulated 30-day path
    daily = np.diff(cum, axis=1, prepend=0)   # recovers per-day P&L
    # 4-day rolling sums:
    csum = np.cumsum(daily, axis=1)
    rolling4 = csum[:, 3:] - np.concatenate(
        [np.zeros((N_PATHS, 1)), csum[:, :-4]], axis=1
    )
    # rolling4 shape: (paths, 27)  → 27 overlapping 4-day windows per path
    bad_windows_per_path = (rolling4 <= ACTUAL_DAY4).sum(axis=1)

    print(f"  Mean 4-day windows ≤ -$127 in a 30-day path:  "
          f"{bad_windows_per_path.mean():4.1f}")
    print(f"  Median:                                       "
          f"{int(np.median(bad_windows_per_path)):d}")
    print(f"  P(zero such windows in 30 days):              "
          f"{(bad_windows_per_path == 0).mean():5.1%}")
    print(f"  P(≥ 1 such window in 30 days):                "
          f"{(bad_windows_per_path >= 1).mean():5.1%}")
    print(f"  P(≥ 3 such windows in 30 days):               "
          f"{(bad_windows_per_path >= 3).mean():5.1%}")
    print()

    # ---- How many more bad days to flip the verdict? --------------------------
    print("-" * 72)
    print("Sensitivity: how many more days of THIS performance flips the verdict?")
    print("-" * 72)
    # Extrapolate current pace (-$127 over 4 days = -$31.75/day) forward and
    # ask: at what cumulative day N does the live track sit at the 5th
    # percentile of the OOS-projected marginal (true underperformance)?
    current_pace = ACTUAL_DAY4 / 4.0
    print(f"  Current observed pace:   ${current_pace:+.2f}/day")
    print(f"  OOS-projected pace:      ${MU_DAY:+.2f}/day")
    print(f"  Pace gap:                ${current_pace - MU_DAY:+.2f}/day "
          f"(~ {(current_pace - MU_DAY)/SIGMA_DAY:+.2f} sigma_daily)")
    print()
    for n_days in [4, 7, 10, 15, 20, 25, 30]:
        marginal = cum[:, n_days - 1]
        projected_actual = current_pace * n_days
        pct = float(stats.percentileofscore(marginal, projected_actual, kind="mean"))
        sigma_n = SIGMA_DAY * np.sqrt(n_days)
        z_n = (projected_actual - MU_DAY * n_days) / sigma_n
        print(f"  After {n_days:>2d} days at current pace: cum ${projected_actual:>+8,.0f}  "
              f"-> percentile {pct:5.1f}  z={z_n:+5.2f}")
    print()

    # ---- Standard-error sanity on the OOS mean --------------------------------
    se_mean = SIGMA_DAY / np.sqrt(N_OOS_SAMPLE)
    ci_lo = MU_DAY - 1.96 * se_mean
    ci_hi = MU_DAY + 1.96 * se_mean
    print("-" * 72)
    print("OOS-mean uncertainty (parametric)")
    print("-" * 72)
    print(f"  SE(mean)  ${se_mean:.0f}/day")
    print(f"  95% CI    [${ci_lo:+.0f}, ${ci_hi:+.0f}] $/day")
    print(f"  i.e. the OOS test cannot rule out a true mean as low as "
          f"${ci_lo:+.0f}/day.")
    print()

    print("=" * 72)
    print("done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
