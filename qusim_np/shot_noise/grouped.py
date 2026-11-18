r"""
qusim_np/shot_noise/grouped.py
Grouped (commuting) Pauli measurements (spec section 5)

When commuting terms are measured on the SAME shots, their outcomes
are correlated and covariances matter:

    Var(H_hat) = sum_g Var(H_g) / N_g
    Var(H_g)   = sum_{i,j in g} c_i c_j Cov(P_i, P_j)
    Cov(P_i,P_j) = <P_i P_j> - <P_i><P_j>

For commuting Paulis, P_i P_j is another Pauli (up to sign), so the
covariance is computable from the state or from shared samples.

Covariance can be negative (grouping reduces variance) or positive
(grouping hurts) - it is not automatically a win.

Implementation note (spec): estimate Var(H_g) from the shared shots
by computing the per-shot value h_g = sum_{i in g} c_i o_i and
taking its sample variance.  This accounts for covariances
automatically.
"""

import numpy as np


def pauli_covariance(expval_pi_pj, expval_pi, expval_pj):
    r"""
    Covariance of two Pauli outcomes:
        Cov(P_i, P_j) = <P_i P_j> - <P_i><P_j>

    Args:
        expval_pi_pj: float - <P_i P_j> (product is a Pauli up to sign
            for commuting Paulis)
        expval_pi: float - <P_i>
        expval_pj: float - <P_j>

    Returns:
        float: the covariance
    """
    return float(expval_pi_pj - expval_pi * expval_pj)


def group_shot_values(coefficients, outcomes):
    r"""
    Per-shot Hamiltonian contribution h_g = sum_i c_i o_i.

    Args:
        coefficients: array-like of shape (M,) - term coefficients
        outcomes: numpy.ndarray of shape (N_g, M) - outcome of each
            term on each shot (e.g. +/-1 for Paulis)

    Returns:
        numpy.ndarray: shape (N_g,) - h_g per shot
    """
    c = np.asarray(coefficients, dtype=float).ravel()
    o = np.asarray(outcomes, dtype=float)
    if o.ndim != 2 or o.shape[1] != c.size:
        raise ValueError(
            f"outcomes must have shape (n_shots, {c.size}), "
            f"got {o.shape}"
        )
    return o @ c


def group_variance(coefficients, outcomes):
    r"""
    Variance of a grouped Hamiltonian estimate from shared shots.

    Computes h_g = sum_i c_i o_i per shot and returns its sample
    variance (ddof=1), which automatically includes all covariances
    between the grouped terms.

    Args:
        coefficients: array-like of shape (M,) - term coefficients
        outcomes: numpy.ndarray of shape (N_g, M) - shared-shot
            outcomes for the group's terms

    Returns:
        tuple: (var_h_g, stderr_h_g_mean)
            var_h_g       - single-shot variance of h_g
            stderr_h_g_mean - standard error of the group mean,
                             sqrt(var_h_g / N_g)
    """
    h = group_shot_values(coefficients, outcomes)
    n = h.size
    if n < 2:
        raise ValueError("need at least 2 shots to estimate variance")
    var_h = float(np.var(h, ddof=1))
    return var_h, float(np.sqrt(var_h / n))


def covariance_matrix_from_samples(outcomes):
    r"""
    Empirical covariance matrix of per-shot Pauli outcomes.

    Args:
        outcomes: numpy.ndarray of shape (N, M) - outcomes of M
            observables on N shared shots

    Returns:
        numpy.ndarray: shape (M, M) covariance matrix (ddof=1)
    """
    o = np.asarray(outcomes, dtype=float)
    if o.ndim != 2:
        raise ValueError(f"outcomes must be 2-D, got shape {o.shape}")
    if o.shape[0] < 2:
        raise ValueError("need at least 2 shots to estimate covariance")
    return np.cov(o, rowvar=False, ddof=1)


def allocate_group_shots(group_variances, total_shots):
    r"""
    Allocate shots across measurement groups.

    Optimal allocation across groups: N_g proportional to
    sqrt(Var(H_g)) where Var(H_g) is the single-shot variance of the
    group's per-shot Hamiltonian contribution.

    Uses largest-remainder rounding so integer allocations sum exactly
    to `total_shots`, and gives every group at least 1 shot.

    Args:
        group_variances: array-like - Var(H_g) per group (>= 0)
        total_shots: int - total shots to distribute

    Returns:
        list[int]: shot allocation per group
    """
    v = np.asarray(group_variances, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("group_variances must not be empty")
    if np.any(v < 0):
        raise ValueError("group variances must be non-negative")
    total_shots = int(total_shots)
    n_groups = v.size
    if total_shots < n_groups:
        raise ValueError(
            f"total_shots={total_shots} cannot give 1 shot to each of "
            f"the {n_groups} groups"
        )

    alloc = np.ones(n_groups, dtype=int)
    remaining = total_shots - n_groups
    weights = np.sqrt(v)
    w_sum = weights.sum()
    if remaining > 0 and w_sum > 0:
        raw = remaining * weights / w_sum
        base = np.floor(raw).astype(int)
        alloc += base
        leftover = remaining - int(base.sum())
        if leftover > 0:
            order = np.argsort(-(raw - base))
            alloc[order[:leftover]] += 1
    return alloc.tolist()
