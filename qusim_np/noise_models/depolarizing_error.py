r"""
qusim_np/noise_models/depolarizing_error.py
Depolarizing noise model

The depolarizing channel shrinks the Bloch vector toward the origin:
with probability p the state is replaced by the maximally mixed state
I/2, and with probability 1-p it is left unchanged:

    E(rho) = (1 - p) rho + (p/2) I

Convention note (important - conventions differ across sources):
- THIS module uses the "replace with maximally mixed state" convention
  above, with p in [0, 1]; p = 1 gives E(rho) = I/2 for every rho.
- The alternative Pauli-channel convention is
  E(rho) = (1 - q) rho + (q/3) (X rho X + Y rho Y + Z rho Z),
  related by p = 4q/3 (equivalently q = 3p/4).
- Kraus operators here are {sqrt(1 - 3p/4) I, sqrt(p/4) X, sqrt(p/4) Y,
  sqrt(p/4) Z}; note the I coefficient is 1 - 3p/4, NOT 1 - p.
"""

import numpy as np

from qusim_np.config import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
from qusim_np.noise_models.channels import apply_channel

_PAULI_MATRICES = (PAULI_I, PAULI_X, PAULI_Y, PAULI_Z)


def _validate_p(p):
    """Validate that p is a real number in [0, 1]."""
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    return p


def depolarizing_kraus(p):
    r"""
    Return the Kraus operators of the depolarizing channel.

    E(rho) = (1-p) rho + (p/2) I has Kraus operators
    {sqrt(1 - 3p/4) I, sqrt(p/4) X, sqrt(p/4) Y, sqrt(p/4) Z}.

    Args:
        p: float in [0, 1] - probability of replacement by I/2

    Returns:
        tuple: four 2x2 complex numpy arrays
    """
    p = _validate_p(p)
    return tuple(
        np.sqrt(c) * P
        for c, P in zip((1 - 3 * p / 4, p / 4, p / 4, p / 4), _PAULI_MATRICES)
    )


def depolarizing(rho, p, qubit=0):
    r"""
    Apply the depolarizing channel E(rho) = (1-p) rho + (p/2) I.

    Accepts a density matrix or state vector; returns a density matrix.
    The channel is unital and isotropic: it shrinks the Bloch vector by
    the factor (1 - p) in every direction.

    Args:
        rho: numpy.ndarray - density matrix or state vector
        p: float in [0, 1] - depolarizing probability
        qubit: int - qubit to depolarize, in [0, n) (default 0)

    Returns:
        numpy.ndarray: The output density matrix of shape (2^n, 2^n)
    """
    ops = depolarizing_kraus(p)
    return apply_channel(rho, ops, qubit=qubit)
