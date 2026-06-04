"""Long-only top-N sizing primitive (regime-conditional construction).

Cycle 12 (strategy-engine owner, 2026-05-18): the dollar-neutral
mean-variance optimizer in ``optimizer.py`` is the default construction
path, but cycle-11's regime study showed it fails (negative Sharpe) in a
handful of regimes where short-leg crowding or borrow-cost spikes
dominate the long-side alpha. For those regimes we fall back to a
long-only top-N equal-weight sleeve, vol-targeted the same way as the
MV path so the risk-management diff layer sees comparable risk-units.

Pipeline shape
--------------
The function is intentionally *pure* (no Σ matrix, no solver) so it can
be used as a drop-in alternative when the scheduler's regime tagger
flags a "MV-hostile" day:

1. Pick the top-N symbols by predicted return ``r_hat`` (default
   ``ceil(len(universe) / 2)``).
2. Equal-weight them long — each gets ``1 / N`` of the long sleeve.
3. Scale the sleeve to the requested annualized vol target using
   per-symbol historical vol; portfolio vol uses a *zero-correlation
   diagonal approximation* (``σ_p^2 = Σ_i w_i^2 σ_i^2``) because in
   regimes where MV fails the off-diagonal Σ estimate is the part we
   don't trust. The diagonal approximation is conservative — true
   portfolio vol with positive average correlation will be *higher*
   than the diagonal estimate, so the realized sleeve tends to size
   *below* the target rather than above it. This is the correct sign
   for a fallback primitive.
4. Convert to dollar allocations: each symbol gets ``w_i * nav``.

Return value is a ``dict[Symbol, Decimal]`` of dollar allocations
(long-only — all entries ≥ 0). The caller (scheduler) is responsible
for lowering dollars to share quantities via the contract-spec emitter.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Optional

from shared_core.contracts import Symbol

DEFAULT_VOL_TARGET: float = 0.12  # 12% annualized — matches sizing.DEFAULT_TARGET_VOL
MAX_LEVERAGE_MULTIPLIER: float = 5.0  # mirror sizing.vol_target_scale safety cap


def long_only_top_n(
    r_hat: dict[Symbol, Decimal],
    *,
    nav: Decimal,
    sigma_annualized: dict[Symbol, Decimal],
    n: Optional[int] = None,
    vol_target: float = DEFAULT_VOL_TARGET,
) -> dict[Symbol, Decimal]:
    """Long-only top-N equal-weight construction, vol-targeted.

    Parameters
    ----------
    r_hat:
        Predicted returns by symbol. Empty dict returns an empty dict.
    nav:
        Net asset value (denominator currency assumed consistent across
        the universe; cross-currency NAV conversion is the scheduler's
        responsibility, not this primitive's).
    sigma_annualized:
        Per-symbol annualized historical vol (e.g. 0.25 for a 25% name).
        MUST cover every symbol selected by the top-N pick. A missing
        key raises ``KeyError`` — silently zero-filling would mask a
        data-quality bug at the scheduler layer.
    n:
        Number of names to long-only hold. Defaults to
        ``ceil(len(r_hat) / 2)`` (the median split — keeps the sleeve
        diversified while concentrating on the higher-conviction half
        of the universe). Clamped to ``[1, len(r_hat)]``.
    vol_target:
        Annualized vol target (e.g. 0.12 = 12%). Matches the MV path's
        default so the regime-switched construction does not change
        the risk envelope the scheduler emits.

    Returns
    -------
    dict[Symbol, Decimal]
        Dollar allocations per symbol (long-only, all ≥ 0). Symbols
        that were not picked are *omitted* from the result rather than
        emitted with a zero allocation — the scheduler's diff layer
        treats missing keys as "flatten this name" anyway, and the
        smaller dict is easier to eyeball in the rebalance ledger.

    Raises
    ------
    ValueError
        If ``nav <= 0`` or ``vol_target <= 0``.
    KeyError
        If ``sigma_annualized`` is missing a symbol selected by the
        top-N pick.
    """
    if nav <= Decimal(0):
        raise ValueError(f"nav must be positive: {nav}")
    if vol_target <= 0:
        raise ValueError(f"vol_target must be positive: {vol_target}")

    if not r_hat:
        return {}

    universe_size = len(r_hat)
    if n is None:
        n_eff = math.ceil(universe_size / 2)
    else:
        n_eff = max(1, min(int(n), universe_size))

    # Sort by predicted return descending; ties broken by ticker for
    # determinism (downstream replay must be reproducible).
    ranked = sorted(
        r_hat.items(),
        key=lambda kv: (-float(kv[1]), kv[0].ticker),
    )
    picked = ranked[:n_eff]

    # Equal-weight long sleeve: each gets 1/N (sums to 1.0 long exposure).
    base_weight = 1.0 / n_eff

    # Diagonal-approximation portfolio vol: σ_p = sqrt(Σ w_i^2 σ_i^2).
    # See module docstring for why we ignore off-diagonal Σ here.
    var_p = 0.0
    for sym, _ in picked:
        if sym not in sigma_annualized:
            raise KeyError(
                f"sigma_annualized missing entry for selected symbol {sym}"
            )
        sigma_i = float(sigma_annualized[sym])
        var_p += (base_weight * sigma_i) ** 2
    sigma_p = math.sqrt(var_p)

    if sigma_p <= 1e-12:
        # Degenerate: every picked name has zero historical vol. Skip
        # scaling rather than divide by zero — the scheduler will likely
        # reject this batch upstream but we don't crash here.
        scaling = 1.0
    else:
        raw_scaling = vol_target / sigma_p
        # Mirror the MV path's leverage clamp so the long-only sleeve
        # cannot blow gross-exposure caps via a low-vol universe.
        scaling = min(raw_scaling, MAX_LEVERAGE_MULTIPLIER)

    final_weight = base_weight * scaling

    # Convert weights → dollar allocations. Decimal arithmetic to keep
    # rounding consistent with the rest of the contract layer.
    weight_dec = Decimal(str(final_weight))
    return {sym: nav * weight_dec for sym, _ in picked}


__all__ = [
    "DEFAULT_VOL_TARGET",
    "MAX_LEVERAGE_MULTIPLIER",
    "long_only_top_n",
]
