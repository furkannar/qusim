r"""
qusim_np/shot_noise/sampling.py
Computational-basis measurement sampling

Simulates projective measurement in the computational basis:
each shot independently draws a basis state |k> with probability
|<k|psi>|^2 (or rho_kk for a density matrix), returning outcome
counts.  This is the shot-level counterpart of the exact
probability distribution |psi_k|^2.
"""

import numpy as np


def measurement_probabilities(state):
    r"""
    Born-rule probabilities of the computational basis.

    Args:
        state: numpy.ndarray - state vector (2^n,) or density matrix
            (2^n, 2^n)

    Returns:
        numpy.ndarray: shape (2^n,) probabilities p_k
    """
    state = np.asarray(state)
    if state.ndim == 1:
        probs = np.abs(state) ** 2
    elif state.ndim == 2 and state.shape[0] == state.shape[1]:
        probs = np.real(np.diag(state))
    else:
        raise ValueError(
            f"state must be a vector (2^n,) or square matrix, "
            f"got shape {state.shape}"
        )
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("state has no measurement probability mass")
    return probs / total


def sample_bitstrings(state, shots, rng=None):
    r"""
    Sample computational-basis measurement outcomes.

    Draws `shots` basis states independently from the Born
distribution and returns per-state counts.

    Args:
        state: numpy.ndarray - state vector (2^n,) or density matrix
        shots: int - number of measurement shots
        rng: numpy.random.Generator or None - defaults to a fresh
            nondeterministic generator

    Returns:
        numpy.ndarray: shape (2^n,) counts; counts[k] is the number
            of times basis state |k> was observed
    """
    probs = measurement_probabilities(state)
    shots = int(shots)
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    if rng is None:
        rng = np.random.default_rng()
    draws = rng.choice(probs.size, size=shots, p=probs)
    return np.bincount(draws, minlength=probs.size).astype(int)


def estimate_distribution(state, shots, rng=None):
    r"""
    Estimated outcome distribution from finite shots.

    Args:
        state: numpy.ndarray - state vector or density matrix
        shots: int - number of measurement shots
        rng: numpy.random.Generator or None

    Returns:
        tuple: (probs_hat, counts)
            probs_hat - shape (2^n,) empirical frequencies
            counts    - shape (2^n,) integer shot counts
    """
    counts = sample_bitstrings(state, shots, rng=rng)
    return counts / counts.sum(), counts
