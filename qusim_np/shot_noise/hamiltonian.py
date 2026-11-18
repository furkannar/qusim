r"""
qusim_np/shot_noise/hamiltonian.py
Shot allocation for Hamiltonians H = sum_i c_i P_i (spec section 4)

When each Pauli term is measured on separate shots (covariances
vanish):

    Var(H_hat) = sum_i c_i^2 sigma_i^2 / N_i,   sum_i N_i = N

Uniform allocation (N_i = N/M):
    Var = (M / N) * sum_i c_i^2 sigma_i^2

Optimal allocation (Lagrange multipliers):
    N_i     = N * |c_i| sigma_i / sum_j |c_j| sigma_j
    Var_min = ( sum_i |c_i| sigma_i )^2 / N

Total shots for standard error eps:
    N = ( sum_i |c_i| sigma_i )^2 / eps^2  <=  ||c||_1^2 / eps^2
"""

import numpy as np

from qusim_np.shot_noise.bounds import gaussian_z


def _validate_coeffs_sigmas(coefficients, sigmas):
    """Validate and broadcast coefficients and single-shot sigmas."""
    c = np.asarray(coefficients, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("coefficients must not be empty")
    if sigmas is None:
        s = np.ones_like(c)
    else:
        s = np.asarray(sigmas, dtype=float).ravel()
        if s.shape != c.shape:
            raise ValueError(
                f"sigmas must have shape {c.shape}, got {s.shape}"
            )
    if np.any(s < 0):
        raise ValueError("sigmas must be non-negative")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
        raise ValueError("coefficients and sigmas must be finite")
    return c, s


def _validate_eps(eps):
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}")
    return eps


def hamiltonian_variance(coefficients, sigmas, shots_per_term):
    r"""
    Variance of the Hamiltonian estimate for a given shot allocation.

    Args:
        coefficients: array-like - Pauli coefficients c_i
        sigmas: array-like or None - single-shot std devs sigma_i
            (default 1 = worst case for +/-1 Pauli outcomes)
        shots_per_term: array-like of ints - N_i shots on term i

    Returns:
        float: Var(H_hat) = sum_i c_i^2 sigma_i^2 / N_i
    """
    c, s = _validate_coeffs_sigmas(coefficients, sigmas)
    n_i = np.asarray(shots_per_term, dtype=float).ravel()
    if n_i.shape != c.shape:
        raise ValueError(
            f"shots_per_term must have shape {c.shape}, got {n_i.shape}"
        )
    if np.any(n_i <= 0):
        raise ValueError("every term needs a positive shot count")
    return float(np.sum(c ** 2 * s ** 2 / n_i))


def uniform_allocation_variance(coefficients, sigmas, total_shots):
    r"""
    Var(H_hat) under uniform allocation N_i = N / M:
        Var = (M / N) * sum_i c_i^2 sigma_i^2
    """
    c, s = _validate_coeffs_sigmas(coefficients, sigmas)
    m = c.size
    total_shots = int(total_shots)
    if total_shots < m:
        raise ValueError(
            f"total_shots={total_shots} is less than the number of "
            f"terms M={m}"
        )
    return float(m / total_shots * np.sum(c ** 2 * s ** 2))


def optimal_allocation_variance(coefficients, sigmas, total_shots):
    r"""
    Minimum Var(H_hat) over all allocations:
        Var_min = ( sum_i |c_i| sigma_i )^2 / N
    """
    c, s = _validate_coeffs_sigmas(coefficients, sigmas)
    total_shots = int(total_shots)
    if total_shots < 1:
        raise ValueError("total_shots must be positive")
    return float(np.sum(np.abs(c) * s) ** 2 / total_shots)


def allocate_shots(coefficients, sigmas, total_shots):
    r"""
    Optimal shot allocation N_i proportional to |c_i| sigma_i.

    Uses largest-remainder rounding so the integer allocations sum
    exactly to `total_shots`, and guarantees every term with a
    nonzero coefficient gets at least 1 shot.  Terms with c_i = 0
    (or sigma_i = 0) receive 0 shots.

    Args:
        coefficients: array-like - Pauli coefficients c_i
        sigmas: array-like or None - single-shot std devs (default 1)
        total_shots: int - total shots to distribute

    Returns:
        list[int]: shot allocation per term, length M

    Raises:
        ValueError: if total_shots is smaller than the number of
            terms that need at least one shot
    """
    c, s = _validate_coeffs_sigmas(coefficients, sigmas)
    total_shots = int(total_shots)
    weights = np.abs(c) * s
    needs_shot = weights > 0
    n_needed = int(np.count_nonzero(needs_shot))
    if total_shots < n_needed:
        raise ValueError(
            f"total_shots={total_shots} cannot give 1 shot to each of "
            f"the {n_needed} nonzero terms"
        )

    alloc = np.zeros(c.size, dtype=int)
    if n_needed == 0 or total_shots == 0:
        return alloc.tolist()

    # Every contributing term gets 1 shot; distribute the rest
    # proportionally to |c_i| sigma_i via largest remainder.
    alloc[needs_shot] = 1
    remaining = total_shots - n_needed
    w = weights[needs_shot]
    w_sum = w.sum()
    if remaining > 0 and w_sum > 0:
        raw = remaining * w / w_sum
        base = np.floor(raw).astype(int)
        alloc[needs_shot] += base
        leftover = remaining - int(base.sum())
        if leftover > 0:
            order = np.argsort(-(raw - base))
            idx = np.flatnonzero(needs_shot)[order[:leftover]]
            alloc[idx] += 1
    return alloc.tolist()


def shots_for_hamiltonian(coefficients, sigmas=None, eps=None,
                          delta=None):
    r"""
    Total shots for standard error (or confidence half-width) eps on
    H = sum_i c_i P_i under optimal allocation:

        N = ( sum_i |c_i| sigma_i )^2 / eps^2        (standard error)
        N = z^2 ( sum_i |c_i| sigma_i )^2 / eps^2    (confidence 1-delta)

    Bounded by ||c||_1^2 / eps^2 since sigma_i <= 1 for Pauli terms.

    Args:
        coefficients: array-like - Pauli coefficients c_i
        sigmas: array-like or None - single-shot std devs (default 1)
        eps: float > 0 - target error
        delta: float in (0, 1) or None - failure probability.
            If given, eps is a confidence half-width.

    Returns:
        int: total number of shots (ceil)
    """
    c, s = _validate_coeffs_sigmas(coefficients, sigmas)
    eps = _validate_eps(eps)
    base = float(np.sum(np.abs(c) * s))
    if delta is None:
        n = base ** 2 / eps ** 2
    else:
        delta = float(delta)
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta!r}")
        z = gaussian_z(delta)
        n = z ** 2 * base ** 2 / eps ** 2
    return int(np.ceil(n))
