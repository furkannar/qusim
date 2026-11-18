r"""
qusim_np/shot_noise/estimation.py
Finite-shot estimation of expectation values (spec section 1)

Given N i.i.d. measurement outcomes o_k (eigenvalues of the measured
observable read on shot k), the sample mean is unbiased:

    O_hat = (1/N) * sum_k o_k
    Var    = sigma2 / N,   sigma2 = <O^2> - <O>^2
    stderr = sigma / sqrt(N)

Shots reduce variance only; they never remove hardware bias.
"""

import warnings

import numpy as np

from qusim_np.vqe import expectation_value


def estimate_expval(outcomes):
    r"""
    Estimate an expectation value from i.i.d. measurement outcomes.

    Args:
        outcomes: array-like of shape (N,) - the eigenvalue read on
            each shot (e.g. +1/-1 for a Pauli, or bit values 0/1)

    Returns:
        tuple: (mean, stderr) where stderr uses the sample standard
        deviation with ddof=1, i.e. s / sqrt(N)

    Raises:
        ValueError: if outcomes is empty
    """
    outcomes = np.asarray(outcomes, dtype=float).ravel()
    n = outcomes.size
    if n == 0:
        raise ValueError("outcomes must contain at least one shot")
    mean = float(np.mean(outcomes))
    if n < 2:
        warnings.warn(
            "stderr is undefined for a single shot; returning nan",
            RuntimeWarning, stacklevel=2,
        )
        return mean, float("nan")
    stderr = float(np.std(outcomes, ddof=1) / np.sqrt(n))
    return mean, stderr


def observable_expval(state, observable):
    r"""
    Exact expectation value <psi|O|psi> (or Tr(rho O) for a density
    matrix), used as ground truth when validating shot estimates.

    Args:
        state: numpy.ndarray - state vector (2^n,) or density matrix
            (2^n, 2^n)
        observable: numpy.ndarray of shape (2^n, 2^n) - Hermitian
            observable

    Returns:
        float: the expectation value
    """
    state = np.asarray(state)
    N = state.shape[-1] if state.ndim == 1 else state.shape[0]
    op = np.asarray(observable, dtype=complex)
    if state.ndim == 1:
        if op.shape != (N, N):
            raise ValueError(
                f"observable must have shape ({N}, {N}), got {op.shape}"
            )
        return expectation_value(state, op)
    if state.shape != (N, N):
        raise ValueError(f"density matrix must have shape ({N}, {N}), got {state.shape}")
    if op.shape != (N, N):
        raise ValueError(
            f"observable must have shape ({N}, {N}), got {op.shape}"
        )
    return float(np.real(np.trace(state @ op)))


def sample_observable_outcomes(state, observable, shots, rng=None):
    r"""
    Simulate projective measurement of an observable in its eigenbasis.

    Diagonalizes O = V diag(o_i) V^dagger, computes the outcome
    probabilities p_i = |<v_i|psi>|^2 (or <v_i|rho|v_i> for a density
    matrix), and draws `shots` eigenvalues accordingly.

    Args:
        state: numpy.ndarray - state vector (2^n,) or density matrix
        observable: numpy.ndarray of shape (2^n, 2^n) - Hermitian
        shots: int - number of measurement shots
        rng: numpy.random.Generator or None

    Returns:
        numpy.ndarray: shape (shots,) array of sampled eigenvalues
    """
    state = np.asarray(state)
    op = np.asarray(observable, dtype=complex)
    if state.ndim == 1:
        N = state.shape[0]
        if op.shape != (N, N):
            raise ValueError(
                f"observable must have shape ({N}, {N}), got {op.shape}"
            )
        eigvals, eigvecs = np.linalg.eigh(op)
        # p_i = |<v_i|psi>|^2 = |V^dagger psi|^2
        amps = eigvecs.conj().T @ state
        probs = np.abs(amps) ** 2
    else:
        N = state.shape[0]
        if state.shape != (N, N):
            raise ValueError(
                f"density matrix must have shape ({N}, {N}), got {state.shape}"
            )
        if op.shape != (N, N):
            raise ValueError(
                f"observable must have shape ({N}, {N}), got {op.shape}"
            )
        eigvals, eigvecs = np.linalg.eigh(op)
        # p_i = <v_i|rho|v_i> = diag(V^dagger rho V)
        rotated = eigvecs.conj().T @ state @ eigvecs
        probs = np.real(np.diag(rotated))

    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"measurement probabilities sum to {total}, expected 1 "
            "(is the state normalized?)"
        )
    probs = probs / total

    if rng is None:
        rng = np.random.default_rng()
    shots = int(shots)
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    indices = rng.choice(len(eigvals), size=shots, p=probs)
    return eigvals[indices].astype(float)
