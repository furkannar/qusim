r"""
qusim_np/shot_noise/pauli_stats.py
Shot statistics for Pauli observables (spec section 2)

For a Pauli operator P with eigenvalues +/-1:

    sigma2_P = 1 - <P>^2          # because P^2 = I
    p_plus   = (1 + <P>) / 2
    p_minus  = (1 - <P>) / 2
    P_hat    = (N_plus - N_minus) / N

The estimator is a binomial proportion: N_plus + N_minus = N.
"""

import warnings

import numpy as np


def pauli_expval_from_counts(n_plus, n_minus):
    r"""
    Estimate <P> from +/-1 outcome counts of a Pauli measurement.

    Args:
        n_plus: int - number of +1 outcomes
        n_minus: int - number of -1 outcomes

    Returns:
        tuple: (P_hat, stderr)
            P_hat  = (n_plus - n_minus) / N
            stderr = sqrt((1 - P_hat^2) / N)

    Note:
        When |P_hat| = 1 the plug-in variance is exactly zero, which
        would report a zero standard error.  We floor the variance at
        the single-shot worst case sigma2 = 1 and warn, giving the
        conservative stderr = 1/sqrt(N).
    """
    n_plus = int(n_plus)
    n_minus = int(n_minus)
    n = n_plus + n_minus
    if n <= 0:
        raise ValueError("total shots must be positive")
    if n_plus < 0 or n_minus < 0:
        raise ValueError("counts cannot be negative")

    p_hat = (n_plus - n_minus) / n
    var = 1.0 - p_hat ** 2
    if var <= 0.0:
        warnings.warn(
            f"|P_hat| = 1 (all {n} shots agreed); plug-in variance is "
            "zero. Returning the conservative worst-case stderr "
            "1/sqrt(N). Use a Wilson or Clopper-Pearson interval for "
            "a rigorous bound.",
            RuntimeWarning, stacklevel=2,
        )
        var = 1.0
    stderr = float(np.sqrt(var / n))
    return float(p_hat), stderr


def pauli_outcome_probs(expval):
    r"""
    Outcome probabilities of a Pauli measurement given <P>.

    Args:
        expval: float in [-1, 1] - the expectation value <P>

    Returns:
        tuple: (p_plus, p_minus)
    """
    expval = float(expval)
    if not -1.0 <= expval <= 1.0:
        raise ValueError(f"<P> must lie in [-1, 1], got {expval!r}")
    p_plus = (1.0 + expval) / 2.0
    return p_plus, 1.0 - p_plus


def shots_needed_pauli_std_error(expval, eps):
    r"""
    Shots needed for standard error eps on a Pauli observable.

    Uses sigma2 = 1 - <P>^2 (exact for a Pauli since P^2 = I):
        N = sigma2 / eps^2     (standard error)
        N = z^2 sigma2 / eps^2 (confidence half-width; see bounds.py)

    Args:
        expval: float in [-1, 1] - expectation value <P>
        eps: float > 0 - target standard error

    Returns:
        int: number of shots (ceil)
    """
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}")
    p_plus, _ = pauli_outcome_probs(expval)
    sigma2 = 4.0 * p_plus * (1.0 - p_plus)  # = 1 - <P>^2
    return int(np.ceil(sigma2 / eps ** 2))
