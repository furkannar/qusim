r"""
qusim_np/vqe.py
Variational Quantum Eigensolver — Pauli-string Hamiltonians

A Hamiltonian is written as a linear combination of Pauli strings

    H = sum_i alpha_i * P_i,    P_i = \bigotimes_k \sigma_{i,k}

where each \sigma_{i,k} \in {I, X, Y, Z} acts on qubit k and alpha_i is a
real coefficient.  Equivalently, if the Pauli-string matrices P_i are stacked
into an array p of shape (m, 2^n, 2^n), the Hamiltonian is the contraction

    H = alpha @ p        (alpha: coefficients, shape (m,))

The VQE cost function is the expectation value

    E(theta) = <psi(theta)| H |psi(theta)>

which is minimized over the parameters theta of a parameterized ansatz.

Pauli-string conventions:
- A Pauli string is written left-to-right over qubits, e.g. "XZI" means
  X on qubit 0, Z on qubit 1, I on qubit 2.  Qubit 0 is the most
  significant bit of the basis-state index (consistent with the rest of
  the engine).
- Strings may be given as a string ("XZI"), a tuple/list of labels
  (("X","Z","I")), or a tuple/list of indices into "IXYZ" ((1,2,0)).
"""

import numpy as np

from qusim_np.config import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
from qusim_np.gates import ry, rz, cnot
from qusim_np.grover_search import _validate_n

_PAULI_MATRICES = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
_PAULI_LABELS = "IXYZ"


def _normalize_pauli_string(pauli_string):
    """
    Normalize a Pauli string to a tuple of labels drawn from "IXYZ".

    Accepts a string ("XZI"), a sequence of single-character labels
    (("X", "Z", "I")), or a sequence of indices into "IXYZ" ((1, 2, 0)).
    Matching is case-insensitive.
    """
    if isinstance(pauli_string, str):
        items = list(pauli_string)
    else:
        try:
            items = list(pauli_string)
        except TypeError:
            raise ValueError(
                f"pauli_string must be a string or sequence, got {pauli_string!r}"
            ) from None

    labels = []
    for item in items:
        if isinstance(item, str):
            if len(item) != 1:
                raise ValueError(f"invalid Pauli label {item!r}; expected a single character")
            labels.append(item.upper())
        elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            if not 0 <= item < len(_PAULI_LABELS):
                raise ValueError(f"Pauli index {item} out of range [0, {len(_PAULI_LABELS)})")
            labels.append(_PAULI_LABELS[item])
        else:
            raise ValueError(f"invalid Pauli label {item!r}")

    if not labels:
        raise ValueError("pauli_string must contain at least one label")
    for lab in labels:
        if lab not in _PAULI_MATRICES:
            raise ValueError(f"invalid Pauli label {lab!r}; expected one of 'I', 'X', 'Y', 'Z'")
    return tuple(labels)


# ---------------------------------------------------------------------------
# Bitmask (symplectic) representation of Pauli strings
# ---------------------------------------------------------------------------
#
# Every single-qubit Pauli is X^x Z^z up to a phase:
#     I = X^0 Z^0,  X = X^1 Z^0,  Z = X^0 Z^1,  Y = i X Z
# (check: i X Z = i [[0,-1],[1,0]] = [[0,-i],[i,0]] = Y).
# On a basis state, Z^z|c> = (-1)^{c z}|c> and X^x|c> = |c XOR x>, so an
# n-qubit Pauli string acts as a signed permutation of basis states:
#
#     P|c> = i^{n_Y} (-1)^{popcount(c AND z)} |c XOR x>
#
# where x has a 1 on every X/Y qubit, z on every Z/Y qubit, and n_Y counts
# the Y factors.  Qubit q corresponds to bit (n-1-q) (qubit 0 = MSB).
# Hence P has exactly one nonzero per column, at row c XOR x: applying P
# or adding alpha*P into a matrix costs O(2^n) instead of O(4^n)
# (matrix) or O(n 2^n) (axis-by-axis tensor contraction).

_I_POWERS = (1, 1j, -1, -1j)


def _pauli_masks(labels):
    """Return (x_mask, z_mask, n_y) for a normalized Pauli string."""
    n = len(labels)
    x_mask = z_mask = n_y = 0
    for q, lab in enumerate(labels):
        bit = 1 << (n - 1 - q)
        if lab in ("X", "Y"):
            x_mask |= bit
        if lab in ("Z", "Y"):
            z_mask |= bit
        if lab == "Y":
            n_y += 1
    return x_mask, z_mask, n_y


def _pauli_column_values(labels):
    """
    For P = labels, return (cols, rows, vals) with P[rows[c], cols[c]] = vals[c].

    cols = 0..2^n-1, rows = cols XOR x, vals = i^{n_Y} (-1)^{popcount(c AND z)}.
    """
    x_mask, z_mask, n_y = _pauli_masks(labels)
    cols = np.arange(2 ** len(labels))
    parity = np.bitwise_count(cols & z_mask) & 1
    # bitwise_count returns uint8: convert before negating to avoid wraparound
    vals = _I_POWERS[n_y % 4] * (1.0 - 2.0 * parity)
    return cols, cols ^ x_mask, vals


def _labels_to_matrix(labels):
    """Build the dense matrix of a normalized Pauli string, O(2^n) fill."""
    N = 2 ** len(labels)
    cols, rows, vals = _pauli_column_values(labels)
    M = np.zeros((N, N), dtype=complex)
    M[rows, cols] = vals
    return M


def pauli_string_matrix(pauli_string):
    r"""
    Construct the matrix of a Pauli string P = \bigotimes_k \sigma_k.

    Args:
        pauli_string: Pauli string as a string ("XZI"), a sequence of
            labels (("X", "Z", "I")), or a sequence of indices into "IXYZ"

    Returns:
        numpy.ndarray: The Pauli-string matrix of shape (2^n, 2^n)
    """
    labels = _normalize_pauli_string(pauli_string)
    return _labels_to_matrix(labels)


def _infer_n(dim):
    """Infer the qubit count n from a Hilbert-space dimension."""
    n = int(round(np.log2(dim)))
    if n < 1 or 2 ** n != dim:
        raise ValueError(f"dimension {dim} is not a power of 2 >= 2")
    return n


def _check_state(state):
    """Validate a state vector and return (state, n)."""
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError(f"state must be a 1-D vector, got shape {state.shape}")
    return state, _infer_n(state.shape[0])


def apply_single_qubit(state, gate, qubit):
    r"""
    Apply a single-qubit gate to one qubit of a state vector.

    The state is treated as a tensor of shape (2,)*n where axis k
    corresponds to qubit k (qubit 0 is the most significant bit).
    Cost is O(n * 2^n) — no 2^n x 2^n matrix is built.

    Args:
        state: numpy.ndarray of shape (2^n,) - state amplitudes
        gate: 2x2 unitary matrix (e.g. PAULI_X, hadamard().matrix)
        qubit: int - qubit index in [0, n)

    Returns:
        numpy.ndarray: The state after applying the gate, shape (2^n,)
    """
    state, n = _check_state(state)
    if not isinstance(qubit, (int, np.integer)) or isinstance(qubit, bool) or not 0 <= qubit < n:
        raise ValueError(f"qubit must be an integer in [0, {n}), got {qubit!r}")

    U = np.asarray(gate, dtype=complex)
    if U.shape != (2, 2):
        raise ValueError(f"gate must have shape (2, 2), got {U.shape}")

    t = state.reshape((2,) * n)
    out = np.tensordot(U, t, axes=(1, qubit))
    out = np.moveaxis(out, 0, qubit)
    return out.reshape(-1)


def apply_two_qubit(state, gate, q0, q1):
    r"""
    Apply a two-qubit gate to qubits q0 and q1 of a state vector.

    The gate is a 4x4 matrix whose first tensor factor acts on qubit q0
    and whose second factor acts on qubit q1; the pair index is
    2*i_{q0} + i_{q1}, i.e. q0 is the more significant bit of the pair.
    For example, cnot().matrix with (q0, q1) = (control, target) flips
    the target qubit when the control qubit is |1>, regardless of the
    numeric order of the qubit indices.

    The state is treated as a tensor of shape (2,)*n and the gate is
    applied by moving the two target axes to the front — cost is
    O(n * 2^n) and no 2^n x 2^n matrix is built.

    Args:
        state: numpy.ndarray of shape (2^n,) - state amplitudes
        gate: 4x4 unitary matrix (e.g. cnot().matrix)
        q0: int - first qubit of the pair (control for CNOT)
        q1: int - second qubit of the pair (target for CNOT)

    Returns:
        numpy.ndarray: The state after applying the gate, shape (2^n,)
    """
    state, n = _check_state(state)
    qubits = []
    for q in (q0, q1):
        if not isinstance(q, (int, np.integer)) or isinstance(q, bool) or not 0 <= q < n:
            raise ValueError(f"qubit must be an integer in [0, {n}), got {q!r}")
        qubits.append(int(q))
    if qubits[0] == qubits[1]:
        raise ValueError("q0 and q1 must be distinct qubits")

    U = np.asarray(gate, dtype=complex)
    if U.shape != (4, 4):
        raise ValueError(f"gate must have shape (4, 4), got {U.shape}")

    t = state.reshape((2,) * n)
    t = np.moveaxis(t, qubits, [0, 1])
    t = t.reshape(4, -1)
    out = U @ t
    out = out.reshape((2, 2) + (2,) * (n - 2))
    out = np.moveaxis(out, [0, 1], qubits)
    return out.reshape(-1)


def apply_pauli_string(state, pauli_string):
    r"""
    Apply a Pauli string P = \bigotimes_k \sigma_k to a state vector.

    Uses the signed-permutation form P|c> = i^{n_Y} (-1)^{c.z} |c XOR x>,
    so the cost is O(2^n) vectorized work.

    Args:
        state: numpy.ndarray of shape (2^n,) - state amplitudes
        pauli_string: Pauli string (string, labels, or indices)

    Returns:
        numpy.ndarray: P|psi> of shape (2^n,)
    """
    state, n = _check_state(state)
    labels = _normalize_pauli_string(pauli_string)
    if len(labels) != n:
        raise ValueError(
            f"pauli string length {len(labels)} does not match state dimension 2^{n}"
        )
    # (P psi)[c XOR x] = val[c] psi[c]   (see the bitmask notes above)
    cols, rows, vals = _pauli_column_values(labels)
    out = np.empty_like(state)
    out[rows] = vals * state[cols]
    return out


def pauli_string_expval(state, pauli_string):
    r"""
    Compute the expectation value <psi|P|psi> of a Pauli string.

    Computed as vdot(psi, P|psi) without building the 2^n x 2^n matrix.

    Args:
        state: numpy.ndarray of shape (2^n,) - state amplitudes
        pauli_string: Pauli string (string, labels, or indices)

    Returns:
        float: The real expectation value <psi|P|psi>
    """
    state, _ = _check_state(state)
    psi_p = apply_pauli_string(state, pauli_string)
    return float(np.real(np.vdot(state, psi_p)))


def hamiltonian(coefficients, pauli_strings):
    r"""
    Assemble a Hamiltonian H = sum_i alpha_i * P_i from Pauli strings.

    Mathematically H = alpha @ p with p the stack of Pauli matrices
    (shape (m, 2^n, 2^n)), but that stack is never built: each P_i has
    one nonzero per column, so alpha_i P_i is scattered straight into H.
    Memory is O(4^n) for H itself (instead of O(m 4^n)) and time is
    O(m 2^n) after allocating H.

    Args:
        coefficients: array-like of shape (m,) - real coefficients alpha_i
        pauli_strings: sequence of m Pauli strings, all of the same length n

    Returns:
        numpy.ndarray: The Hamiltonian matrix of shape (2^n, 2^n)
    """
    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.ndim != 1:
        raise ValueError(f"coefficients must be a 1-D array, got shape {coeffs.shape}")

    strings = list(pauli_strings)
    if not strings:
        raise ValueError("pauli_strings must contain at least one term")
    if len(strings) != coeffs.shape[0]:
        raise ValueError(
            f"got {len(strings)} Pauli strings but {coeffs.shape[0]} coefficients"
        )

    labels_list = [_normalize_pauli_string(s) for s in strings]
    n = len(labels_list[0])
    if any(len(lab) != n for lab in labels_list):
        raise ValueError("all Pauli strings in a Hamiltonian must have the same length")

    N = 2 ** n
    H = np.zeros((N, N), dtype=complex)
    for alpha, labels in zip(coeffs, labels_list):
        cols, rows, vals = _pauli_column_values(labels)
        H[rows, cols] += alpha * vals
    return H


def expectation_value(state, operator):
    r"""
    Compute the expectation value <psi|O|psi> of a state and an operator.

    Args:
        state: numpy.ndarray of shape (2^n,) - state amplitudes
        operator: numpy.ndarray of shape (2^n, 2^n) - observable

    Returns:
        float: The real expectation value <psi|O|psi>
    """
    state, _ = _check_state(state)
    op = np.asarray(operator, dtype=complex)
    N = state.shape[0]
    if op.shape != (N, N):
        raise ValueError(f"operator must have shape ({N}, {N}), got {op.shape}")
    return float(np.real(np.vdot(state, op @ state)))


# ---------------------------------------------------------------------------
# Hardware-Efficient Ansatz (HEA) and VQE driver
# ---------------------------------------------------------------------------
#
# The ansatz alternates layers of single-qubit rotations with an
# entangling layer of CNOT gates:
#
#     |psi(theta)> = prod_l [ Entangle . Rot(theta_l) ] |0...0>
#
# Each rotation layer applies RY(a) then RZ(b) to every qubit, giving
# 2 * n parameters per layer.  RZ is included so the ansatz can produce
# complex amplitudes (RY alone only reaches real states, which would make
# every Pauli-Y expectation value identically zero).
#
# Parameter vector layout (flat, C-order):
#     theta[l, q, j]  ->  theta_flat[(l * n + q) * 2 + j]
#     j = 0 -> RY angle,  j = 1 -> RZ angle


def hea_param_count(n, layers=1):
    """Number of variational parameters of the HEA ansatz."""
    _validate_n(n)
    if not isinstance(layers, (int, np.integer)) or isinstance(layers, bool) or layers < 1:
        raise ValueError(f"layers must be a positive integer, got {layers!r}")
    return int(layers) * int(n) * 2


def _validate_hea_args(n, layers, entanglement):
    _validate_n(n)
    if not isinstance(layers, (int, np.integer)) or isinstance(layers, bool) or layers < 1:
        raise ValueError(f"layers must be a positive integer, got {layers!r}")
    if entanglement not in ("linear", "ring"):
        raise ValueError(f"entanglement must be 'linear' or 'ring', got {entanglement!r}")


def hea_state(n, theta, layers=1, entanglement="linear"):
    r"""
    Build the Hardware-Efficient Ansatz state |psi(theta)>.

    Each layer applies RY then RZ to every qubit, followed by an
    entangling layer of CNOT gates:

    - "linear": CNOTs i -> i+1 for i = 0..n-2 (open chain)
    - "ring":   CNOTs i -> (i+1) mod n for i = 0..n-1 (closed loop;
                for n = 2 this gives both directions i -> j and j -> i)

    Args:
        n: int - number of qubits
        theta: array-like of shape (layers * n * 2,) - flat parameter vector
        layers: int - number of ansatz layers (default 1)
        entanglement: "linear" or "ring" (default "linear")

    Returns:
        numpy.ndarray: The ansatz state of shape (2^n,)
    """
    _validate_hea_args(n, layers, entanglement)
    theta = np.asarray(theta, dtype=float).ravel()
    expected = hea_param_count(n, layers)
    if theta.size != expected:
        raise ValueError(
            f"theta must have {expected} parameters "
            f"(layers={layers}, n={n}, 2 per qubit), got {theta.size}"
        )
    theta = theta.reshape(layers, n, 2)

    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    cnot_matrix = cnot().matrix

    for layer in range(layers):
        for q in range(n):
            state = apply_single_qubit(state, ry(theta[layer, q, 0]).matrix, q)
            state = apply_single_qubit(state, rz(theta[layer, q, 1]).matrix, q)
        if entanglement == "ring":
            pairs = [(i, (i + 1) % n) for i in range(n)]
        else:
            pairs = [(i, i + 1) for i in range(n - 1)]
        for c, t in pairs:
            if c != t:
                state = apply_two_qubit(state, cnot_matrix, c, t)

    return state


def vqe_energy(theta, n, layers, coefficients, pauli_strings, entanglement="linear"):
    r"""
    VQE cost function E(theta) = <psi(theta)| H |psi(theta)>.

    The energy is evaluated term-by-term as
    E = sum_i alpha_i <psi|P_i|psi> using the efficient Pauli-string
    expectation (no 2^n x 2^n Hamiltonian matrix is built).

    Args:
        theta: array-like - flat HEA parameter vector
        n: int - number of qubits
        layers: int - number of ansatz layers
        coefficients: array-like of shape (m,) - Hamiltonian coefficients
        pauli_strings: sequence of m Pauli strings (all length n)
        entanglement: "linear" or "ring" (default "linear")

    Returns:
        float: The energy expectation value
    """
    state = hea_state(n, theta, layers, entanglement)
    coeffs = np.asarray(coefficients, dtype=float).ravel()
    strings = [_normalize_pauli_string(s) for s in pauli_strings]
    if len(strings) != coeffs.size:
        raise ValueError(
            f"got {len(strings)} Pauli strings but {coeffs.size} coefficients"
        )
    energies = [pauli_string_expval(state, s) for s in strings]
    return float(np.dot(coeffs, energies))


def vqe_optimize(n, layers, coefficients, pauli_strings, theta0=None,
                 entanglement="linear", method="Nelder-Mead", **kwargs):
    r"""
    Minimize the VQE energy over HEA parameters with scipy.optimize.

    Args:
        n: int - number of qubits
        layers: int - number of ansatz layers
        coefficients: array-like of shape (m,) - Hamiltonian coefficients
        pauli_strings: sequence of m Pauli strings (all length n)
        theta0: array-like or None - initial parameters; if None, a
            deterministic pseudo-random start in [-pi, pi) is used
            (nonzero init avoids the barren-plateau-like saddle at theta=0)
        entanglement: "linear" or "ring" (default "linear")
        method: scipy optimizer method (default "Nelder-Mead")
        **kwargs: forwarded to scipy.optimize.minimize

    Returns:
        scipy.optimize.OptimizeResult: result object; result.x holds the
        optimal parameters and result.fun the minimal energy
    """
    from scipy.optimize import minimize

    m = hea_param_count(n, layers)
    if theta0 is None:
        theta0 = np.random.default_rng(0).uniform(-np.pi, np.pi, m)
    else:
        theta0 = np.asarray(theta0, dtype=float).ravel()
        if theta0.size != m:
            raise ValueError(f"theta0 must have {m} parameters, got {theta0.size}")

    def cost(theta):
        return vqe_energy(theta, n, layers, coefficients, pauli_strings, entanglement)

    return minimize(cost, theta0, method=method, **kwargs)
