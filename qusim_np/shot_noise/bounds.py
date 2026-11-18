r"""
qusim_np/shot_noise/bounds.py
Shot-count bounds for finite-shot estimation (spec section 3)

Three bounds for the number of shots N needed to estimate an
observable with outcomes in a bounded range:

    CLT       (asymptotic):  N = ceil( z^2 sigma2 / eps^2 )
    Hoeffding (rigorous,    N >= R^2 ln(2/delta) / (2 eps^2)
               variance-blind)
    Bernstein (rigorous,    N >= (2 sigma2 + (2/3) b eps)
               variance-aware)      * ln(2/delta) / eps^2

where z = z_{1-delta/2} is the two-sided Gaussian quantile,
sigma2 the single-shot variance, R = b - a the outcome range,
and b the max deviation of one shot from the mean.

Convention (spec section 0): `eps` is a STANDARD ERROR by default.
For a confidence half-width at level 1 - delta, pass `delta` so the
z^2 (CLT) or ln(2/delta) (Hoeffding/Bernstein) factor is included.
"""

import warnings

import numpy as np
from scipy.stats import norm

from qusim_np.config import ln_2_over_delta


def _validate_eps_delta(eps, delta):
    """Validate target error and failure probability."""
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}")
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    return eps, delta


def gaussian_z(delta):
    r"""
    Two-sided Gaussian quantile z_{1 - delta/2}.

    Examples: delta=0.10 -> ~1.64, 0.05 -> ~1.96, 0.01 -> ~2.58.
    """
    _, delta = _validate_eps_delta(1.0, delta)
    return float(norm.ppf(1.0 - delta / 2.0))


def shots_clt(sigma2, eps, delta=None):
    r"""
    CLT (asymptotic) shot count: N = ceil(z^2 sigma2 / eps^2).

    Args:
        sigma2: float >= 0 - single-shot variance <O^2> - <O>^2
        eps: float > 0 - target error
        delta: float in (0, 1) or None - failure probability.
            If None, eps is treated as a standard error and the
            z^2 factor is omitted (N = ceil(sigma2 / eps^2)).
            Otherwise eps is a confidence half-width at 1 - delta.

    Returns:
        int: number of shots

    Warns:
        RuntimeWarning if the resulting N is small (< 100), where
        the CLT approximation is unreliable - consider Hoeffding or
        Bernstein instead.
    """
    sigma2 = float(sigma2)
    if sigma2 < 0:
        raise ValueError(f"sigma2 must be non-negative, got {sigma2!r}")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}")

    if delta is None:
        n = int(np.ceil(sigma2 / eps ** 2))
    else:
        _, delta = _validate_eps_delta(eps, delta)
        z = gaussian_z(delta)
        n = int(np.ceil(z ** 2 * sigma2 / eps ** 2))

    if 0 < n < 100:
        warnings.warn(
            f"CLT shot count N={n} is small; the normal approximation "
            "is unreliable for small N or outcomes near the boundary. "
            "Consider shots_hoeffding or shots_bernstein for a "
            "distribution-free bound.",
            RuntimeWarning, stacklevel=2,
        )
    return n


def shots_hoeffding(eps, delta, outcome_range=2.0):
    r"""
    Hoeffding shot count: N >= R^2 ln(2/delta) / (2 eps^2).

    Rigorous and variance-blind: valid for any distribution of
    outcomes in [a, b] with range R = b - a.  For Pauli +/-1
    outcomes (R = 2) this reduces to N >= 2 ln(2/delta) / eps^2,
    about 2x looser than CLT at sigma = 1.

    Args:
        eps: float > 0 - confidence half-width
        delta: float in (0, 1) - failure probability
        outcome_range: float > 0 - R = b - a (default 2 for Pauli)

    Returns:
        int: number of shots (ceil)
    """
    eps, delta = _validate_eps_delta(eps, delta)
    R = float(outcome_range)
    if not np.isfinite(R) or R <= 0:
        raise ValueError(f"outcome_range must be positive, got {R!r}")
    n = int(np.ceil(R ** 2 * ln_2_over_delta(delta) / (2.0 * eps ** 2)))
    return n


def shots_bernstein(sigma2, eps, delta, max_deviation=2.0):
    r"""
    Bernstein shot count:
        N >= (2 sigma2 + (2/3) b eps) ln(2/delta) / eps^2

    Rigorous like Hoeffding but variance-aware: beats Hoeffding when
    the single-shot variance sigma2 is small.  `max_deviation` b is
    the maximum distance of one shot's outcome from the mean
    (<= 2 for Pauli +/-1 outcomes).

    Args:
        sigma2: float >= 0 - single-shot variance
        eps: float > 0 - confidence half-width
        delta: float in (0, 1) - failure probability
        max_deviation: float > 0 - b (default 2 for Pauli)

    Returns:
        int: number of shots (ceil)
    """
    sigma2 = float(sigma2)
    if sigma2 < 0:
        raise ValueError(f"sigma2 must be non-negative, got {sigma2!r}")
    eps, delta = _validate_eps_delta(eps, delta)
    b = float(max_deviation)
    if not np.isfinite(b) or b <= 0:
        raise ValueError(f"max_deviation must be positive, got {b!r}")
    n = int(np.ceil(
        (2.0 * sigma2 + (2.0 / 3.0) * b * eps)
        * ln_2_over_delta(delta) / eps ** 2
    ))
    return n
