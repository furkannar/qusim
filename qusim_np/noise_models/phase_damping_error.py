r"""
qusim_np/noise_models/phase_damping_error.py
Phase damping (pure dephasing) noise model

Models loss of coherence without energy relaxation: the populations
(diagonal elements) of the density matrix are unchanged while the
coherences (off-diagonal elements) are multiplied by sqrt(1 - lambda).

Kraus form (canonical):

    K_0 = [[1, 0], [0, sqrt(1 - lambda)]]
    K_1 = [[0, 0], [0, sqrt(lambda)]]

Channel action on rho = [[r00, r01], [r10, r11]]:

    E(rho) = [[r00,               sqrt(1-lambda) * r01],
              [sqrt(1-lambda) * r10, r11              ]]

Equivalent Pauli (random-Z) form:

    E(rho) = (1 - p) rho + p Z rho Z
          = [[r00, (1-2p) r01], [(1-2p) r10, r11]]

with Kraus operators {sqrt(1-p) I, sqrt(p) Z}.  The two forms are
unitarily equivalent and give identical output when

    1 - 2p = sqrt(1 - lambda)
    p     = (1 - sqrt(1 - lambda)) / 2
    lambda = 1 - (1 - 2p)^2          (valid for 0 <= p <= 1/2)

Parameter naming convention (do not mix silently):
- `lam` : phase damping parameter in the Kraus form, 0 <= lam <= 1
- `p`   : phase-flip probability in the Pauli form
- toqito calls the phase damping parameter `gamma`; depolarizing has its
  own p vs 4p/3 convention.

Time dependence (Markovian, pure dephasing time T_phi):

    c = sqrt(1 - lambda) = exp(-t / T_phi)
    lambda              = 1 - exp(-2 t / T_phi)
    p                   = (1 - exp(-t / T_phi)) / 2

Device relation: 1/T2 = 1/(2 T1) + 1/T_phi, i.e.
1/T_phi = 1/T2 - 1/(2 T1), requiring the physical bound T2 <= 2 T1.
When T2 == 2 T1 there is no pure dephasing (lam = 0, "T1-limited").

Combining amplitude damping (gamma from T1) followed by phase damping
produces the thermal relaxation channel: total coherence factor
exp(-t/(2 T1)) * exp(-t/T_phi) = exp(-t/T2), populations follow T1 only.
"""

import numpy as np

from qusim_np.noise_models.channels import apply_channel
from qusim_np.noise_models.amplitude_damping_error import (
    amplitude_damping, gamma_from_relaxation,
)


def _validate_lam(lam):
    """Validate that lambda is a real number in [0, 1]."""
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [0, 1], got {lam!r}")
    return lam


def _validate_p(p):
    """Validate that p is a real number in [0, 1]."""
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    return p


def phase_damping_kraus(lam):
    r"""
    Return the Kraus operators of the phase damping channel.

    Args:
        lam: float in [0, 1] - phase damping parameter

    Returns:
        tuple: (K_0, K_1) of 2x2 complex numpy arrays
    """
    lam = _validate_lam(lam)
    s = np.sqrt(1.0 - lam)
    K0 = np.array([[1.0, 0.0], [0.0, s]], dtype=complex)
    K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(lam)]], dtype=complex)
    return K0, K1


def phase_flip_kraus(p):
    r"""
    Return the Kraus operators of the phase-flip channel (Pauli form).

    E(rho) = (1 - p) rho + p Z rho Z with Kraus operators
    {sqrt(1-p) I, sqrt(p) Z}.  Equivalent to phase damping with
    lambda = 1 - (1 - 2p)^2 for 0 <= p <= 1/2.

    Args:
        p: float in [0, 1] - probability of a Z (phase-flip) error

    Returns:
        tuple: (sqrt(1-p) I, sqrt(p) Z) of 2x2 complex numpy arrays
    """
    p = _validate_p(p)
    I = np.eye(2, dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    return np.sqrt(1.0 - p) * I, np.sqrt(p) * Z


def lambda_from_p(p):
    r"""
    Convert a phase-flip probability p to the phase damping parameter lambda.

        lambda = 1 - (1 - 2p)^2

    Valid for 0 <= p <= 1/2 (outside this range the Pauli form does not
    correspond to pure dephasing).

    Args:
        p: float in [0, 1/2]

    Returns:
        float: lambda in [0, 1]
    """
    p = _validate_p(p)
    if p > 0.5:
        raise ValueError(
            f"p must be <= 1/2 for the phase damping mapping, got {p!r}"
        )
    return float(1.0 - (1.0 - 2.0 * p) ** 2)


def p_from_lambda(lam):
    r"""
    Convert the phase damping parameter lambda to a phase-flip probability.

        p = (1 - sqrt(1 - lambda)) / 2

    Args:
        lam: float in [0, 1]

    Returns:
        float: p in [0, 1/2]
    """
    lam = _validate_lam(lam)
    return float((1.0 - np.sqrt(1.0 - lam)) / 2.0)


def phase_damping_from_times(t, T_phi):
    r"""
    Compute the phase damping parameter for idle time t and dephasing time T_phi.

        lambda = 1 - exp(-2 t / T_phi)

    Args:
        t: float > 0 - idle time
        T_phi: float > 0 - pure dephasing time

    Returns:
        float: lambda in [0, 1)
    """
    t = float(t)
    T_phi = float(T_phi)
    if t <= 0:
        raise ValueError(f"t must be positive, got {t!r}")
    if T_phi <= 0:
        raise ValueError(f"T_phi must be positive, got {T_phi!r}")
    return float(1.0 - np.exp(-2.0 * t / T_phi))


def phase_damping(rho, lam, qubit=0):
    r"""
    Apply the phase damping channel to a density matrix.

    Accepts either an n-qubit density matrix of shape (2^n, 2^n) or a
    state vector of shape (2^n,) (converted via rho = |psi><psi|).

    Args:
        rho: numpy.ndarray - density matrix or state vector
        lam: float in [0, 1] - phase damping parameter
        qubit: int - qubit to dephase, in [0, n) (default 0)

    Returns:
        numpy.ndarray: The output density matrix of shape (2^n, 2^n)
    """
    K0, K1 = phase_damping_kraus(lam)
    return apply_channel(rho, [K0, K1], qubit=qubit)


def phase_flip(rho, p, qubit=0):
    r"""
    Apply the phase-flip channel E(rho) = (1-p) rho + p Z rho Z.

    Equivalent to phase damping with lambda = 1 - (1 - 2p)^2 for
    p <= 1/2.  Accepts a density matrix or state vector.

    Args:
        rho: numpy.ndarray - density matrix or state vector
        p: float in [0, 1] - probability of a Z error
        qubit: int - qubit to act on, in [0, n) (default 0)

    Returns:
        numpy.ndarray: The output density matrix of shape (2^n, 2^n)
    """
    K0, K1 = phase_flip_kraus(p)
    return apply_channel(rho, [K0, K1], qubit=qubit)


def thermal_relaxation(rho, t, T1, T2, qubit=0):
    r"""
    Apply the thermal relaxation channel for an idle time t.

    This is amplitude damping (energy relaxation, governed by T1)
    followed by phase damping (pure dephasing, governed by T_phi derived
    from T1 and T2).  The combined channel reproduces the measured
    coherence decay exp(-t/T2) and population decay exp(-t/T1).

    Requires the physical bound T2 <= 2 T1; T2 > 2 T1 raises ValueError.
    T2 == 2 T1 corresponds to the T1-limited case with no pure dephasing.

    Args:
        rho: numpy.ndarray - density matrix or state vector
        t: float > 0 - idle time
        T1: float > 0 - energy relaxation time
        T2: float > 0 - coherence (dephasing) time, must be <= 2*T1
        qubit: int - qubit to relax, in [0, n) (default 0)

    Returns:
        numpy.ndarray: The output density matrix of shape (2^n, 2^n)
    """
    t = float(t)
    T1 = float(T1)
    T2 = float(T2)
    if t <= 0:
        raise ValueError(f"t must be positive, got {t!r}")
    if T1 <= 0:
        raise ValueError(f"T1 must be positive, got {T1!r}")
    if T2 <= 0:
        raise ValueError(f"T2 must be positive, got {T2!r}")
    if T2 > 2.0 * T1:
        raise ValueError(
            f"T2 must satisfy T2 <= 2*T1 (got T2={T2}, 2*T1={2.0 * T1}); "
            f"T2 > 2*T1 is unphysical"
        )

    # Amplitude damping part: gamma = 1 - exp(-t/T1)
    gamma = gamma_from_relaxation(t, T1)

    # Phase damping part: pure dephasing rate 1/T_phi = 1/T2 - 1/(2 T1).
    # T2 == 2 T1 -> 1/T_phi = 0 -> lambda = 0 (T1-limited, no pure dephasing).
    if np.isclose(T2, 2.0 * T1):
        lam = 0.0
    else:
        inv_T_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
        c_pd = np.exp(-t * inv_T_phi)          # = exp(-t/T_phi)
        lam = float(1.0 - c_pd ** 2)

    rho = amplitude_damping(rho, gamma, qubit=qubit)
    return phase_damping(rho, lam, qubit=qubit)
