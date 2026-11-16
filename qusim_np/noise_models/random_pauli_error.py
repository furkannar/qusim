r"""
qusim_np/noise_models/random_pauli_error.py
Random Pauli error channel

A Pauli channel applies Pauli operators to the state with classical
probabilities:

    E(rho) = sum_i q_i P_i rho P_i^dagger,   P_i in {I, X, Y, Z}

with q_i >= 0 and sum_i q_i = 1.  Kraus operators are {sqrt(q_i) P_i}.

Probability vector convention: probs = (q_I, q_X, q_Y, q_Z), i.e. the
order of the labels "IXYZ".  This subsumes several named channels:
- bit flip:        probs = (1-p, p, 0, 0)
- phase flip:      probs = (1-p, 0, 0, p)
- bit-phase flip:  probs = (1-p, 0, p, 0)
- depolarizing:    probs = (1-3p/4, p/4, p/4, p/4)

This module also provides shot-level helpers: sample_pauli() draws one
Pauli according to probs, and apply_sampled_pauli() applies that definite
Pauli to a state (useful for Monte-Carlo trajectory simulations).
"""

import numpy as np

from qusim_np.config import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
from qusim_np.noise_models.channels import apply_channel

_PAULI_MATRICES = (PAULI_I, PAULI_X, PAULI_Y, PAULI_Z)
_PAULI_LABELS = "IXYZ"


def _validate_probs(probs):
    """Validate a probability vector (q_I, q_X, q_Y, q_Z)."""
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (4,):
        raise ValueError(f"probs must have shape (4,), got {probs.shape}")
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(f"probs must lie in [0, 1], got {probs}")
    if not np.isclose(probs.sum(), 1.0, atol=1e-10):
        raise ValueError(f"probs must sum to 1, got {probs.sum()}")
    return probs


def random_pauli_kraus(probs):
    r"""
    Return the Kraus operators of a random Pauli channel.

    Args:
        probs: (q_I, q_X, q_Y, q_Z) - probabilities for each Pauli

    Returns:
        tuple: four 2x2 Kraus operators (sqrt(q_i) P_i); zero-probability
        Paulis still appear (as zero matrices) to keep the order fixed
    """
    probs = _validate_probs(probs)
    return tuple(np.sqrt(q) * P for q, P in zip(probs, _PAULI_MATRICES))


def random_pauli(rho, probs, qubit=0):
    r"""
    Apply the random Pauli channel E(rho) = sum_i q_i P_i rho P_i.

    Accepts a density matrix or state vector; returns a density matrix.

    Args:
        rho: numpy.ndarray - density matrix or state vector
        probs: (q_I, q_X, q_Y, q_Z) - Pauli probabilities summing to 1
        qubit: int - qubit to act on, in [0, n) (default 0)

    Returns:
        numpy.ndarray: The output density matrix of shape (2^n, 2^n)
    """
    ops = random_pauli_kraus(probs)
    return apply_channel(rho, ops, qubit=qubit)


def sample_pauli(probs, rng=None):
    r"""
    Draw one Pauli index according to probs (shot-level simulation).

    Args:
        probs: (q_I, q_X, q_Y, q_Z) - Pauli probabilities summing to 1
        rng: numpy.random.Generator or None - defaults to a fresh
            default_rng() (nondeterministic)

    Returns:
        int: Pauli index in {0, 1, 2, 3} corresponding to "IXYZ"
    """
    probs = _validate_probs(probs)
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(4, p=probs))


def apply_sampled_pauli(rho, pauli, qubit=0):
    r"""
    Apply one definite Pauli operator to a state (a single-error trajectory).

    This is a channel with one Kraus operator {P}, i.e. the unitary
    conjugation rho -> P rho P^dagger.  Accepts a density matrix or state
    vector.

    Args:
        rho: numpy.ndarray - density matrix or state vector
        pauli: int index into "IXYZ" or single-character label
        qubit: int - qubit to act on, in [0, n) (default 0)

    Returns:
        numpy.ndarray: The output density matrix of shape (2^n, 2^n)
    """
    if isinstance(pauli, str):
        if len(pauli) != 1 or pauli.upper() not in _PAULI_LABELS:
            raise ValueError(
                f"pauli must be one of 'I', 'X', 'Y', 'Z', got {pauli!r}"
            )
        pauli = _PAULI_LABELS.index(pauli.upper())
    if not isinstance(pauli, (int, np.integer)) or isinstance(pauli, bool) \
            or not 0 <= pauli < 4:
        raise ValueError(f"pauli index must be in [0, 4), got {pauli!r}")
    return apply_channel(rho, [_PAULI_MATRICES[int(pauli)]], qubit=qubit)
