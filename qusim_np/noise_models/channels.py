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

import numpy as np

from qusim_np.grover_search import _validate_n
from qusim_np.resources import check_memory, state_bytes

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
    check_memory(state_bytes(state.size ** 2))
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
    r"""
    Validate and return a list of 2x2 complex Kraus operators.

    Checks the completeness relation sum_k K_k^dagger K_k = I, which is
    equivalent to the channel being trace-preserving on EVERY input
    (Tr E(rho) = Tr(rho sum_k K_k^dagger K_k) = Tr(rho) for all rho).
    Checking the output trace for one particular rho is not enough: a
    non-TP set can preserve the trace of a specific state by accident.
    """
    ops = [np.asarray(K, dtype=complex) for K in kraus_ops]
    if not ops:
        raise ValueError("kraus_ops must contain at least one operator")
    for i, K in enumerate(ops):
        if K.shape != (2, 2):
            raise ValueError(f"Kraus operator {i} must have shape (2, 2), got {K.shape}")
    completeness = sum(K.conj().T @ K for K in ops)
    if not np.allclose(completeness, np.eye(2), atol=_TRACE_TOL):
        raise ValueError(
            "channel is not trace-preserving: "
            f"sum_k K_k^dagger K_k = {completeness.tolist()} != I"
        )
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

    The Kraus operators are checked for completeness,
    sum_k K_k^dagger K_k = I (trace preservation on every input);
    an incomplete set raises ValueError.

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

    return out


def _apply_matrix_to_axes(tensor, U, axes):
    """Apply a 2^k x 2^k matrix to k axes of a (2,)*m tensor (k <= m)."""
    k = len(axes)
    m = tensor.ndim
    front = list(range(k))
    t = np.moveaxis(tensor, axes, front).reshape(2 ** k, -1)
    out = (U @ t).reshape((2,) * m)
    return np.moveaxis(out, front, axes)


def apply_unitary_density(rho, unitary, qubits):
    r"""
    Conjugate a density matrix by a k-qubit unitary: rho -> U rho U^dagger.

    U acts on `qubits` (qubits[0] is the most significant bit of U's index,
    as in vqe.apply_gate).  In the (2,)*2n tensor form of rho, U is
    contracted onto the row (ket) axes q and conj(U) onto the column (bra)
    axes n+q — the same pattern apply_channel uses for single-qubit Kraus
    operators.  No 2^n x 2^n operator is built.

    A state vector may be passed instead of rho; it is converted via
    rho = |psi><psi|.

    Args:
        rho: numpy.ndarray - density matrix (2^n, 2^n) or state vector (2^n,)
        unitary: 2^k x 2^k matrix
        qubits: int or sequence of k distinct ints in [0, n)

    Returns:
        numpy.ndarray: U rho U^dagger, shape (2^n, 2^n)
    """
    # Local import: vqe is a sibling top-level module with no dependency on
    # noise_models, so this cannot create an import cycle.
    from qusim_np.vqe import _validate_qubits

    if np.asarray(rho).ndim == 1:
        rho = to_density_matrix(rho)
    rho, n = _check_density_matrix(rho)
    qubits = _validate_qubits(qubits, n)
    k = len(qubits)
    U = np.asarray(unitary, dtype=complex)
    if U.shape != (2 ** k, 2 ** k):
        raise ValueError(
            f"unitary on {k} qubit(s) must have shape ({2 ** k}, {2 ** k}), got {U.shape}"
        )
    t = rho.reshape((2,) * (2 * n))
    t = _apply_matrix_to_axes(t, U, qubits)
    t = _apply_matrix_to_axes(t, U.conj(), [n + q for q in qubits])
    return t.reshape(rho.shape)


def kraus_branch_probabilities(state, kraus_ops, qubit=0):
    r"""
    Probabilities p_k = <phi_k|phi_k> of each Kraus branch phi_k = K_k |psi>.

    Mathematically p_k = np.vdot(K_k psi, K_k psi).  Computing it that way
    needs every branch K_k|psi> (one O(2^n) application per Kraus operator),
    so it is evaluated through the single-qubit reduced density matrix
    instead:

        p_k = <psi| K_k^dagger K_k |psi> = Tr( K_k^dagger K_k rho_q ),
        rho_q = Tr_{other qubits} |psi><psi|   (2 x 2)

    rho_q costs one O(2^n) pass; each p_k is then a 2 x 2 trace.

    Args:
        state: numpy.ndarray of shape (2^n,) - state vector
        kraus_ops: sequence of 2x2 Kraus operators
        qubit: int - qubit the channel acts on

    Returns:
        numpy.ndarray: shape (len(kraus_ops),) branch probabilities
    """
    from qusim_np.vqe import _check_state, _validate_qubits

    state, n = _check_state(state)
    (qubit,) = _validate_qubits(qubit, n)
    ops = _validate_kraus_ops(kraus_ops)
    return _branch_probabilities(state, n, ops, qubit)


def _branch_probabilities(state, n, ops, qubit):
    """Unchecked core of kraus_branch_probabilities (inputs already validated)."""
    m = np.moveaxis(state.reshape((2,) * n), qubit, 0).reshape(2, -1)
    rho_q = m @ m.conj().T
    return np.array([np.real(np.trace(K.conj().T @ K @ rho_q)) for K in ops])


def _sample_branch(state, n, ops, qubit, rng):
    """
    Unchecked core of sample_kraus_branch, for callers that have already
    validated the state, qubit and Kraus operators (e.g. QuantumCircuit,
    whose NoiseModel validates each channel once when it is added).
    """
    from qusim_np.vqe import apply_gate

    probs = _branch_probabilities(state, n, ops, qubit)
    total = probs.sum()
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"branch probabilities sum to {total}, expected 1 (is the state normalized?)"
        )
    probs = np.clip(probs, 0.0, None)
    k = int(rng.choice(len(ops), p=probs / probs.sum()))
    K = ops[k]
    norm = np.sqrt(probs[k])
    if K[0, 1] == 0 and K[1, 0] == 0 and abs(K[0, 0] - K[1, 1]) <= 1e-15:
        # K = c I  ->  phi / sqrt(p) = (c / |c|) psi : a global phase only
        return state * (K[0, 0] / norm), k
    return apply_gate(state, K, [qubit]) / norm, k


def sample_kraus_branch(state, kraus_ops, qubit=0, rng=None):
    r"""
    Apply one randomly chosen Kraus branch to a state vector (quantum trajectory step).

    Instead of evolving rho -> sum_k K_k rho K_k^dagger, a trajectory keeps a
    pure state and picks ONE branch per channel application:

        phi_k = K_k |psi>,      p_k = <phi_k|phi_k> = np.vdot(phi_k, phi_k)
        choose k with probability p_k,   |psi> -> phi_k / sqrt(p_k)

    The p_k sum to <psi| sum_k K_k^dagger K_k |psi> = 1 by completeness, and
    averaging |psi><psi| over trajectories reproduces the channel exactly:

        E[ |psi'><psi'| ] = sum_k p_k (phi_k phi_k^dagger / p_k) = E(rho)

    Memory is O(2^n) instead of the O(4^n) of a density matrix.

    Efficiency: the p_k come from kraus_branch_probabilities (one pass over
    the state), and only the chosen branch is applied.  If the chosen K_k is
    a multiple of the identity (e.g. the "no error" branch of depolarizing),
    phi_k / sqrt(p_k) equals psi up to a global phase, so no gate is applied.

    Args:
        state: numpy.ndarray of shape (2^n,) - normalized state vector
        kraus_ops: sequence of 2x2 Kraus operators (validated for completeness)
        qubit: int - qubit the channel acts on (default 0)
        rng: numpy.random.Generator or None

    Returns:
        tuple: (new_state, k) - the normalized post-branch state and the
        index of the Kraus operator that was applied
    """
    from qusim_np.vqe import _check_state, _validate_qubits

    state, n = _check_state(state)
    (qubit,) = _validate_qubits(qubit, n)
    ops = _validate_kraus_ops(kraus_ops)
    if rng is None:
        rng = np.random.default_rng()
    return _sample_branch(state, n, ops, qubit, rng)
