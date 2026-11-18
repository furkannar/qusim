r"""
qusim_np/grover_search.py
Quantum Algorithms Implementation

This module implements various quantum algorithms using the custom gates.

Grover's Search Algorithm:
- n-qubit Hadamard transform: H^{\otimes n} = H \otimes H \otimes ... \otimes H (n times)
- Single-qubit Hadamard matrix entries: H_{jk} = 0.5 * sqrt(2) * (-1)^{jk} where j,k \in {0,1}
- N = 2^n is the total number of basis states
- M = number of marked solutions (states where f(x) = 1)
- Initial state: |s> = H^{\otimes n} |0>^{\otimes n} = (1/sqrt(N)) * sum_{x=0}^{N-1} |x>
- Oracle O_f: marks target states with phase flip (-1)
- Diffusion operator: D = 2|s><s| - I = H^{\otimes n}(2|0^n><0^n| - I)H^{\otimes n}
- Grover iteration: G = D * O_f (rotation by theta in the 2D subspace)
- theta = 2 * arcsin(sqrt(M/N))
- Optimal iterations: k_opt = (pi/4) * sqrt(N/M) - 1/2
- Query complexity: O(sqrt(N/M))
"""

import numpy as np
from qusim_np.gates import hadamard


def _validate_n(n):
    """Validate that n is a positive integer."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")


def _validate_marked_states(n, marked_states):
    """
    Validate n and marked_states, returning a deduplicated list.

    Raises ValueError if n is invalid, marked_states is empty,
    contains non-integers, or contains out-of-range indices.
    """
    _validate_n(n)
    N = 2 ** n
    marked = list(dict.fromkeys(marked_states))
    if not marked:
        raise ValueError("marked_states must contain at least one state")
    for x in marked:
        if not isinstance(x, (int, np.integer)) or isinstance(x, bool):
            raise ValueError(f"marked state {x!r} is not an integer")
        if not 0 <= x < N:
            raise ValueError(f"marked state {x} out of range [0, {N})")
    return marked


def hadamard_transform(n):
    r"""
    Compute the n-qubit Hadamard transform H^{\otimes n} using tensor product.
    
    H^{\otimes n} = H \otimes H \otimes ... \otimes H (n times)
    
    Args:
        n: Number of qubits
        
    Returns:
        numpy.ndarray: The n-qubit Hadamard matrix of shape (2^n, 2^n)
    """
    _validate_n(n)
    # Single-qubit Hadamard matrix
    H = hadamard().matrix
    
    # Compute tensor product H \otimes H \otimes ... \otimes H (n times)
    H_n = H.copy()
    for _ in range(n - 1):
        H_n = np.kron(H_n, H)
    
    return H_n


def hadamard_matrix_closed_form(n):
    r"""
    Compute n-qubit Hadamard matrix using closed form formula.
    
    Matrix entries: H_{jk} = (1/sqrt(2^n)) * (-1)^{j \cdot k}
    where j and k are binary representations of the row and column indices,
    and j \cdot k is the bitwise dot product.
    
    Args:
        n: Number of qubits
        
    Returns:
        numpy.ndarray: The n-qubit Hadamard matrix of shape (2^n, 2^n)
    """
    _validate_n(n)
    N = 2 ** n
    idx = np.arange(N)
    # Bitwise dot product j . k = popcount(j AND k); only its parity matters.
    parity = np.bitwise_count(np.bitwise_and.outer(idx, idx)) & 1
    # bitwise_count returns uint8: use float arithmetic to avoid wraparound
    return ((1.0 - 2.0 * parity) / np.sqrt(N)).astype(complex)


def apply_hadamard_transform(state):
    r"""
    Apply H^{\otimes n} to a state vector via the fast Walsh-Hadamard transform.

    Uses the Sylvester recursion H_{2m} = H_2 \otimes H_m: at each of the
    n stages, pairs of amplitudes (a, b) that differ in one bit are
    replaced by (a + b, a - b).  Cost is O(n 2^n) time and O(2^n) memory,
    versus O(4^n) memory for the dense matrix.

    Args:
        state: numpy.ndarray of shape (2^n,) - state amplitudes

    Returns:
        numpy.ndarray: H^{\otimes n} |state>, shape (2^n,)
    """
    x = np.array(state, dtype=complex).ravel()
    N = x.size
    n = int(round(np.log2(N))) if N > 0 else 0
    if n < 1 or 2 ** n != N:
        raise ValueError(f"state dimension {N} is not a power of 2 >= 2")
    h = 1
    while h < N:
        x = x.reshape(-1, 2, h)
        a, b = x[:, 0, :], x[:, 1, :]
        x = np.stack((a + b, a - b), axis=1)
        h *= 2
    return x.reshape(-1) / np.sqrt(N)


def initialize_uniform_superposition(n):
    r"""
    Initialize the uniform superposition state |s> = H^{\otimes n} |0>^{\otimes n}.
    
    This creates an equal superposition over all N = 2^n basis states.
    |s> = (1/sqrt(N)) * sum_{x=0}^{N-1} |x>
    
    Args:
        n: Number of qubits
        
    Returns:
        numpy.ndarray: The uniform superposition state vector of shape (2^n,)
    """
    _validate_n(n)
    N = 2 ** n
    # H^{\otimes n}|0^n> is the first column of H^{\otimes n}: every entry
    # is (-1)^{j.0} / sqrt(N) = 1 / sqrt(N).  No matrix needed.
    return np.full(N, 1.0 / np.sqrt(N), dtype=complex)


def grover_oracle(n, marked_states):
    r"""
    Construct the Grover oracle O_f that marks target states with a phase flip.
    
    O_f|x> = (-1)^{f(x)}|x>, where f(x) = 1 for marked states, 0 otherwise.
    This is equivalent to flipping the phase of marked basis states.
    
    Args:
        n: Number of qubits
        marked_states: List of marked state indices (integers from 0 to 2^n - 1)
        
    Returns:
        numpy.ndarray: The oracle matrix of shape (2^n, 2^n)
    """
    marked_states = _validate_marked_states(n, marked_states)
    N = 2 ** n
    O = np.eye(N, dtype=complex)
    
    # Flip phase of marked states
    for x in marked_states:
        O[x, x] = -1
    
    return O


def diffusion_operator(n):
    r"""
    Construct the diffusion operator D = 2|s><s| - I.
    
    This reflects the state about the uniform superposition |s>.
    Equivalent to: D = H^{\otimes n}(2|0^n><0^n| - I)H^{\otimes n}
    
    Args:
        n: Number of qubits
        
    Returns:
        numpy.ndarray: The diffusion matrix of shape (2^n, 2^n)
    """
    _validate_n(n)
    N = 2 ** n
    # |s><s| = J / N with J the all-ones matrix, so D = (2/N) J - I.
    D = np.full((N, N), 2.0 / N, dtype=complex)
    D[np.diag_indices(N)] -= 1.0
    return D


def _apply_oracle_inplace(state, marked_states):
    """O_f|psi>: flip the sign of the marked amplitudes.  O(M)."""
    state[marked_states] *= -1


def _apply_diffusion(state):
    r"""
    D|psi> = 2|s><s|psi> - |psi> = 2 * mean(psi) - psi  ("inversion about the mean").

    Since <s|psi> = sum(psi) / sqrt(N) and |s> = (1/sqrt(N)) * ones,
    2|s><s|psi> = 2 * mean(psi) * ones.  O(N).
    """
    return 2.0 * state.mean() - state


def grover_iteration(n, marked_states):
    r"""
    Construct one Grover iteration G = D * O_f.
    
    This is a rotation by theta in the 2D subspace spanned by |alpha> and |beta>.
    
    Args:
        n: Number of qubits
        marked_states: List of marked state indices
        
    Returns:
        numpy.ndarray: The Grover iteration matrix of shape (2^n, 2^n)
    """
    marked_states = _validate_marked_states(n, marked_states)
    # O_f is diagonal with entries +/-1, so D @ O_f just negates the
    # marked columns of D: O(N^2) instead of an O(N^3) matrix product.
    G = diffusion_operator(n)
    G[:, marked_states] *= -1
    return G


def calculate_theta(n, marked_states):
    r"""
    Calculate the rotation angle theta = 2 * arcsin(sqrt(M/N)).
    
    Args:
        n: Number of qubits
        marked_states: List of marked state indices
        
    Returns:
        float: The rotation angle theta
    """
    marked_states = _validate_marked_states(n, marked_states)
    N = 2 ** n
    M = len(marked_states)
    theta = 2 * np.arcsin(np.sqrt(M / N))
    return theta


def calculate_optimal_iterations(n, marked_states):
    r"""
    Calculate the optimal number of Grover iterations.
    
    After k iterations P(marked) = sin^2((2k+1) theta/2), which peaks at
    (2k+1) theta/2 = pi/2, i.e. exactly

        k_opt = pi / (2 theta) - 1/2,    theta = 2 arcsin(sqrt(M/N))

    The familiar (pi/4) sqrt(N/M) - 1/2 is this with theta ~ 2 sqrt(M/N)
    (small-angle approximation); the exact form gives an equal or higher
    success probability whenever the two round differently (common when
    M/N is not small).

    Args:
        n: Number of qubits
        marked_states: List of marked state indices

    Returns:
        int: The optimal number of iterations (rounded to nearest integer)
    """
    theta = calculate_theta(n, marked_states)
    k_opt = np.pi / (2.0 * theta) - 0.5
    return int(round(k_opt))


def grover_search(n, marked_states):
    r"""
    Perform Grover's search algorithm to find marked states.
    
    Algorithm:
    1. Initialize |s> = H^{\otimes n}|0^n>
    2. Apply k_opt Grover iterations: G^k|s>
    3. Measure to obtain the marked state with high probability

    Each iteration is applied directly to the state vector — the oracle
    as a sign flip on the marked amplitudes and the diffusion as
    inversion about the mean — so the cost is O(N) memory and
    O(N sqrt(N/M)) time; no 2^n x 2^n matrix is built.

    Args:
        n: Number of qubits
        marked_states: List of marked state indices

    Returns:
        tuple: (final_state, k_opt, success_probability)
    """
    marked_states = _validate_marked_states(n, marked_states)
    marked = np.asarray(marked_states, dtype=np.intp)

    # Step 1: Initialize uniform superposition
    state = initialize_uniform_superposition(n)

    # Calculate optimal iterations
    k_opt = calculate_optimal_iterations(n, marked_states)

    # Step 2: Apply k_opt Grover iterations G = D O_f
    for _ in range(k_opt):
        _apply_oracle_inplace(state, marked)
        state = _apply_diffusion(state)

    # Step 3: Probability of measuring a marked state
    marked_prob = float(np.sum(np.abs(state[marked]) ** 2))

    return state, k_opt, marked_prob
