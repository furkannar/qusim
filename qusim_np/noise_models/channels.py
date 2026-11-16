r"""
qusim_np/noise_models/channels.py
Generic quantum channel machinery

A quantum channel E acts on a density matrix rho via its Kraus
representation

    E(rho) = sum_k K_k rho K_k^dagger

where the Kraus operators satisfy sum_k K_k^dagger K_k = I for a
trace-preserving channel.

This module provides:
- to_density_matrix(state): |psi> -> rho = |psi><psi|
- apply_channel(rho, kraus_ops, qubit): apply a single-qubit channel to
  one qubit of an n-qubit density matrix, in O(n * 4^n) time without
  building any 2^n x 2^n operator matrices

The density matrix rho of an n-qubit system is stored as a (2^n, 2^n)
matrix and reshaped internally to a tensor of shape (2,)*2n whose first
n axes are the row (ket) indices and last n axes are the column (bra)
indices — qubit q sits at axis q on the row side and axis n+q on the
column side.
"""

import warnings

import numpy as np

from qusim_np.grover_search import _validate_n

_TRACE_TOL = 1e-8


def to_density_matrix(state):
    r"""
    Convert a state vector to its density matrix rho = |psi><psi|.

    Args:
        state: numpy.ndarray of shape (2^n,) - state amplitudes

    Returns:
        numpy.ndarray: The density matrix of shape (2^n, 2^n)
    """
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError(f"state must be a 1-D vector, got shape {state.shape}")
    return np.outer(state, state.conj())


def _check_density_matrix(rho):
    """Validate a density matrix and return (rho, n)."""
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be a square matrix, got shape {rho.shape}")
    N = rho.shape[0]
    n = int(round(np.log2(N)))
    if n < 1 or 2 ** n != N:
        raise ValueError(f"rho dimension {N} is not a power of 2 >= 2")
    return rho, n


def _apply_1q_to_axis(tensor, U, axis):
    """
    Apply a 2x2 matrix U to one axis of an arbitrary tensor.

    out[...,a,...] = sum_b U[a,b] * in[...,b,...]  along `axis`.
    Cost is O(size(tensor)); no enlarged matrix is built.
    """
    out = np.tensordot(U, tensor, axes=([1], [axis]))
    return np.moveaxis(out, 0, axis)


def _validate_kraus_ops(kraus_ops):
    """Validate and return a list of 2x2 complex Kraus operators."""
    ops = [np.asarray(K, dtype=complex) for K in kraus_ops]
    if not ops:
        raise ValueError("kraus_ops must contain at least one operator")
    for i, K in enumerate(ops):
        if K.shape != (2, 2):
            raise ValueError(f"Kraus operator {i} must have shape (2, 2), got {K.shape}")
    return ops


def apply_channel(rho, kraus_ops, qubit=0):
    r"""
    Apply a single-qubit quantum channel to one qubit of a density matrix.

    Computes E(rho) = sum_k (K_k on qubit) rho (K_k^dagger on qubit) by
    contracting each Kraus operator onto the row axis q and its conjugate
    onto the column axis n+q of the (2,)*2n tensor form of rho.

    A state vector |psi> may be passed instead of rho; it is first
    converted via rho = |psi><psi|.  Because channels generally produce
    mixed states, the return value is always a density matrix.

    The result is checked for trace preservation:
    |Tr(E(rho)) - Tr(rho)| must be below tolerance, otherwise a warning
    is emitted and a ValueError is raised.

    Args:
        rho: numpy.ndarray - density matrix (2^n, 2^n) or state vector (2^n,)
        kraus_ops: sequence of 2x2 matrices - the channel's Kraus operators
        qubit: int - qubit to act on, in [0, n) (default 0)

    Returns:
        numpy.ndarray: The output density matrix of shape (2^n, 2^n)
    """
    if np.asarray(rho).ndim == 1:
        rho = to_density_matrix(rho)
    rho, n = _check_density_matrix(rho)

    if not isinstance(qubit, (int, np.integer)) or isinstance(qubit, bool) or not 0 <= qubit < n:
        raise ValueError(f"qubit must be an integer in [0, {n}), got {qubit!r}")

    ops = _validate_kraus_ops(kraus_ops)

    t = rho.reshape((2,) * (2 * n))
    out = np.zeros_like(t)
    for K in ops:
        t_k = _apply_1q_to_axis(t, K, qubit)
        t_k = _apply_1q_to_axis(t_k, K.conj(), n + qubit)
        out += t_k
    out = out.reshape(rho.shape)

    in_trace = np.trace(rho)
    out_trace = np.trace(out)
    if not np.isclose(out_trace, in_trace, atol=_TRACE_TOL):
        warnings.warn(
            f"channel is not trace-preserving: "
            f"Tr(E(rho)) = {out_trace}, Tr(rho) = {in_trace}",
            stacklevel=2,
        )
        raise ValueError(
            f"channel is not trace-preserving: "
            f"Tr(E(rho)) = {out_trace}, Tr(rho) = {in_trace}"
        )

    return out
