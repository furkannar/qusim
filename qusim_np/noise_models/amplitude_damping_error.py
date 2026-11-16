r"""
qusim_np/noise_models/amplitude_damping_error.py
Amplitude damping noise model

Models energy relaxation: the excited state |1> decays to |0> with
probability gamma per channel application, while coherences (off-diagonal
density-matrix elements) decay by the factor sqrt(1 - gamma).

Kraus operators (single qubit):

    K_0 = [[1, 0], [0, sqrt(1 - gamma)]]
    K_1 = [[0, sqrt(gamma)], [0, 0]]

Channel action:

    E(rho) = K_0 rho K_0^dagger + K_1 rho K_1^dagger

For a single-qubit density matrix rho = [[p00, p01], [p10, p11]] this gives

    E(rho) = [[p00 + gamma*p11,          sqrt(1-gamma)*p01],
              [sqrt(1-gamma)*p10,       (1-gamma)*p11   ]]

Relaxation-time form: for an idle time t and energy-relaxation time T1,

    gamma = 1 - exp(-t / T1)

Physical constraint: the coherence (dephasing) time obeys T2 <= 2*T1;
amplitude damping alone saturates this bound.
"""

import numpy as np

from qusim_np.config import sqrt_one_minus_gamma
from qusim_np.noise_models.channels import apply_channel


def _validate_gamma(gamma):
    """Validate that gamma is a real number in [0, 1]."""
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma!r}")
    return gamma


def amplitude_damping_kraus(gamma):
    r"""
    Return the Kraus operators of the amplitude damping channel.

    Args:
        gamma: float in [0, 1] - probability of |1> -> |0> decay

    Returns:
        tuple: (K_0, K_1) of 2x2 complex numpy arrays
    """
    gamma = _validate_gamma(gamma)
    s = sqrt_one_minus_gamma(gamma)
    K0 = np.array([[1.0, 0.0], [0.0, s]], dtype=complex)
    K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return K0, K1


def gamma_from_relaxation(t, T1):
    r"""
    Compute the damping probability for an idle time t and relaxation time T1.

        gamma = 1 - exp(-t / T1)

    The coherence decay time must satisfy the physical bound T2 <= 2*T1;
    amplitude damping alone corresponds to T2 = 2*T1.

    Args:
        t: float > 0 - idle time
        T1: float > 0 - energy relaxation time

    Returns:
        float: gamma in [0, 1)
    """
    t = float(t)
    T1 = float(T1)
    if t <= 0:
        raise ValueError(f"t must be positive, got {t!r}")
    if T1 <= 0:
        raise ValueError(f"T1 must be positive, got {T1!r}")
    return float(1.0 - np.exp(-t / T1))


def amplitude_damping(rho, gamma, qubit=0):
    r"""
    Apply the amplitude damping channel to a density matrix.

    Accepts either an n-qubit density matrix of shape (2^n, 2^n) or a
    state vector of shape (2^n,) (converted via rho = |psi><psi|).  The
    channel acts on `qubit` and returns the output density matrix.

    Args:
        rho: numpy.ndarray - density matrix or state vector
        gamma: float in [0, 1] - damping probability (or use
            gamma_from_relaxation(t, T1) to compute it)
        qubit: int - qubit to damp, in [0, n) (default 0)

    Returns:
        numpy.ndarray: The damped density matrix of shape (2^n, 2^n)
    """
    K0, K1 = amplitude_damping_kraus(gamma)
    return apply_channel(rho, [K0, K1], qubit=qubit)
