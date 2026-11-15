r"""
qusim/grover_search.py
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
from qusim.gates import hadamard


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
    H = np.zeros((N, N), dtype=complex)
    
    for j in range(N):
        for k in range(N):
            # Compute bitwise dot product j \cdot k
            j_bin = format(j, f'0{n}b')
            k_bin = format(k, f'0{n}b')
            dot_product = sum(int(j_bit) * int(k_bit) for j_bit, k_bit in zip(j_bin, k_bin))
            
            # H_{jk} = (1/sqrt(2^n)) * (-1)^{j \cdot k}
            H[j, k] = (1 / np.sqrt(N)) * ((-1) ** dot_product)
    
    return H


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
    N = 2 ** n
    
    # Initial state |0>^{\otimes n} is the first basis state
    initial_state = np.zeros(N, dtype=complex)
    initial_state[0] = 1
    
    # Apply Hadamard transform
    H_n = hadamard_transform(n)
    superposition = H_n @ initial_state
    
    return superposition


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
    N = 2 ** n
    
    # Get uniform superposition |s>
    s = initialize_uniform_superposition(n)
    
    # D = 2|s><s| - I
    D = 2 * np.outer(s, s.conj()) - np.eye(N, dtype=complex)
    
    return D


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
    O = grover_oracle(n, marked_states)
    D = diffusion_operator(n)
    G = D @ O
    
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
    
    k_opt = (pi/4) * sqrt(N/M) - 1/2
    
    Args:
        n: Number of qubits
        marked_states: List of marked state indices
        
    Returns:
        int: The optimal number of iterations (rounded to nearest integer)
    """
    marked_states = _validate_marked_states(n, marked_states)
    N = 2 ** n
    M = len(marked_states)
    k_opt = (np.pi / 4) * np.sqrt(N / M) - 0.5
    return int(round(k_opt))


def grover_search(n, marked_states):
    r"""
    Perform Grover's search algorithm to find marked states.
    
    Algorithm:
    1. Initialize |s> = H^{\otimes n}|0^n>
    2. Apply k_opt Grover iterations: G^k|s>
    3. Measure to obtain the marked state with high probability
    
    Args:
        n: Number of qubits
        marked_states: List of marked state indices
        
    Returns:
        tuple: (final_state, k_opt, success_probability)
    """
    marked_states = _validate_marked_states(n, marked_states)
    N = 2 ** n
    M = len(marked_states)
    
    # Step 1: Initialize uniform superposition
    state = initialize_uniform_superposition(n)
    
    # Calculate optimal iterations
    k_opt = calculate_optimal_iterations(n, marked_states)
    
    # Step 2: Apply k_opt Grover iterations
    G = grover_iteration(n, marked_states)
    for _ in range(k_opt):
        state = G @ state
    
    # Step 3: Calculate success probability
    # Probability of measuring a marked state
    marked_prob = sum(abs(state[x])**2 for x in marked_states)
    
    return state, k_opt, marked_prob
