r"""
Quantum Computing Configuration

This module contains constants and configuration values used across
the quantum computing implementation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class QuantumConfig:
    """Configuration constants for quantum computing implementation."""
    
    # Mathematical constants
    SQRT_2: float = np.sqrt(2)
    INV_SQRT_2: float = 1 / SQRT_2
    
    # Quantum gate constants
    HADAMARD_SCALE: float = INV_SQRT_2
    
    # Grover's algorithm constants
    # These will be computed dynamically based on n and M
    # N = 2^n (total number of basis states)
    # M = number of marked solutions
    
    # Rotation gate constants
    # theta parameter for Rx, Ry, Rz gates
    DEFAULT_ROTATION_ANGLE: float = np.pi / 4
    
    # SU(2) constraints
    # Unitary: U^dagger * U = I
    # Special Unitary: det(U) = 1
    SU2_TOLERANCE: float = 1e-10

    # Shot-noise / measurement sampling
    # Default number of measurement shots for sampling routines
    TOTAL_SHOTS: int = 8192

    # Shor's algorithm memory estimate
    # Peak bytes per state amplitude: 16 for the complex128 state, plus up
    # to 16 of temporaries (half the state, copied twice while a
    # controlled-U^{2^j} permutes it, or the FFT output). Measured with
    # tracemalloc; checked in tests/test_algorithms/test_shor.py.
    BYTES_PER_AMPLITUDE: int = 32

    # System resource guards (qusim_np/resources.py)
    # Refuse to run when free RAM is, or would become, lower than this
    MIN_FREE_RAM_MB: int = 128
    # Reuse a RAM reading this young for small allocations (psutil ~40 us)
    MEMORY_CHECK_CACHE_S: float = 0.1
    # Re-check RAM during long loops only for states this large (2^20 = 16 MiB)
    MONITOR_MIN_AMPLITUDES: int = 2 ** 20
    # How often to re-check RAM during a long loop
    MONITOR_INTERVAL_S: float = 0.5


# Global config instance
config = QuantumConfig()


# Memory units, binary convention (as psutil and the OS report RAM):
# 1 KB = 2^10 bytes, 1 MB = 2^20 bytes, 1 GB = 2^30 bytes (KiB, MiB, GiB).
# Byte counts in the package are converted with these, e.g.
# config.MIN_FREE_RAM_MB * MB, or memory_gb * GB.
KB = 2 ** 10
MB = 2 ** 20
GB = 2 ** 30


# Single-qubit Pauli matrices (used by VQE Hamiltonians and Pauli strings)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def sqrt_one_minus_gamma(gamma):
    r"""
    Compute sqrt(1 - gamma), the coherence factor of amplitude damping.

    This factor multiplies the off-diagonal elements of a density matrix
    under the amplitude damping channel and appears in both Kraus
    operators, so it is factored out here.

    Args:
        gamma: float in [0, 1] - damping probability

    Returns:
        float: sqrt(1 - gamma)
    """
    return float(np.sqrt(1.0 - gamma))


def ln_2_over_delta(delta):
    r"""
    Compute ln(2 / delta), the confidence-log factor in concentration bounds.

    This factor appears in the Hoeffding and Bernstein shot-count bounds
    (qusim_np.shot_noise.bounds) and in the median-of-means count of
    classical shadows, so it is factored out here.

    Args:
        delta: float in (0, 1) - failure probability

    Returns:
        float: ln(2 / delta)
    """
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    return float(np.log(2.0 / delta))


def two_ln_2_over_delta(delta):
    r"""
    Compute 2 * ln(2 / delta).

    This is the numerator of the Hoeffding shot bound for outcomes
    bounded in [-1, 1] (range R = 2): N >= 2 ln(2/delta) / eps^2.

    Args:
        delta: float in (0, 1) - failure probability

    Returns:
        float: 2 * ln(2 / delta)
    """
    return 2.0 * ln_2_over_delta(delta)
