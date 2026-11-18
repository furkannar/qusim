r"""
qusim_np/qft.py
Quantum Fourier Transform

This module implements the Quantum Fourier Transform (QFT) as an
N x N unitary matrix acting on the amplitude vector of a quantum state.

Definition (N = 2^n, omega = e^{2*pi*i/N} the N-th root of unity):
- Matrix entries: F[j,k] = (1/sqrt(N)) * omega^{jk}
- On a basis state: QFT|x> = (1/sqrt(N)) * sum_k omega^{xk} |k>
- On a general state |psi> = sum_j x_j |j>:
  y_k = (1/sqrt(N)) * sum_j x_j * omega^{jk}
- The QFT matrix is symmetric (F = F^T) and unitary (F^dagger F = I)
- Inverse QFT: F^{-1} = F^dagger, entries (1/sqrt(N)) * omega^{-jk}

Relation to the classical DFT:
- The classical DFT uses omega^{-jk}; this QFT convention uses +jk,
  so QFT acting on amplitudes equals sqrt(N) * numpy.fft.ifft.

Examples:
- n = 1: QFT = Hadamard gate H
- n = 2: F_4 = (1/2) [[1,1,1,1],[1,i,-1,-i],[1,-1,1,-1],[1,-i,-1,i]]
"""

import numpy as np
from qusim_np.grover_search import _validate_n


def qft_matrix(n):
    r"""
    Construct the n-qubit Quantum Fourier Transform matrix.

    F[j,k] = (1/sqrt(2^n)) * exp(2*pi*i*j*k / 2^n), for j,k = 0..2^n - 1

    Args:
        n: Number of qubits

    Returns:
        numpy.ndarray: The QFT unitary matrix of shape (2^n, 2^n)
    """
    _validate_n(n)
    N = 2 ** n
    # omega^{jk} depends only on jk mod N; reducing first keeps the
    # exp() argument in [0, 2 pi) and avoids precision loss for large jk.
    jk = np.outer(np.arange(N), np.arange(N)) % N
    return np.exp(2j * np.pi * jk / N) / np.sqrt(N)


def _check_power_of_two_state(state):
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError(f"state must be a 1-D vector, got shape {state.shape}")
    N = state.shape[0]
    n = int(round(np.log2(N))) if N > 0 else 0
    if n < 1 or 2 ** n != N:
        raise ValueError(f"state dimension {N} is not a power of 2 >= 2")
    return state


def inverse_qft_matrix(n):
    r"""
    Construct the inverse Quantum Fourier Transform matrix F^dagger.

    F^{-1}[j,k] = (1/sqrt(2^n)) * exp(-2*pi*i*j*k / 2^n)

    Args:
        n: Number of qubits

    Returns:
        numpy.ndarray: The inverse QFT matrix of shape (2^n, 2^n)
    """
    return qft_matrix(n).conj().T


def qft_state(state):
    r"""
    Apply the Quantum Fourier Transform to a state vector.

    The number of qubits n is inferred from the state dimension
    (len(state) must be a power of 2).

    Args:
        state: numpy.ndarray of shape (2^n,) - quantum state amplitudes

    Computed with the FFT identity  y_k = (1/sqrt(N)) sum_j x_j omega^{jk}
    = numpy.fft.ifft(x, norm="ortho")[k]  in O(N log N) time and O(N)
    memory, instead of an O(N^2) dense matrix-vector product.

    Returns:
        numpy.ndarray: The QFT-transformed state of shape (2^n,)
    """
    state = _check_power_of_two_state(state)
    return np.fft.ifft(state, norm="ortho")


def inverse_qft_state(state):
    r"""
    Apply the inverse Quantum Fourier Transform to a state vector.

    Args:
        state: numpy.ndarray of shape (2^n,) - quantum state amplitudes

    Uses F^dagger x = numpy.fft.fft(x, norm="ortho"), O(N log N).

    Returns:
        numpy.ndarray: The inverse-QFT-transformed state of shape (2^n,)
    """
    state = _check_power_of_two_state(state)
    return np.fft.fft(state, norm="ortho")
