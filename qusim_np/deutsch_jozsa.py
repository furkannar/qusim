r"""
qusim_np/deutsch_jozsa.py
Deutsch-Jozsa Algorithm

This module implements the Deutsch-Jozsa algorithm using the custom gates.

Deutsch-Jozsa Algorithm:
- Determines whether a function f: {0,1}^n -> {0,1} is constant or balanced
  using a single quantum query (classically requires up to 2^{n-1} + 1 queries)
- n-qubit Hadamard transform: H^{\otimes n}
- Initial state: |s> = H^{\otimes n}|0>^{\otimes n} = (1/sqrt(N)) * sum_x |x>
- Oracle O_f: O_f|x> = (-1)^{f(x)}|x> (phase kickback, no ancilla needed)
- Final Hadamard transform: H^{\otimes n} again (H is self-inverse)
- If f is constant: measurement yields |0^n> with probability 1
- If f is balanced: amplitude of |0^n> is 0, so a nonzero outcome is certain
- Amplitude of |y> after final transform:
  A_y = (1/N) * sum_x (-1)^{f(x)} (-1)^{x \cdot y}  (Walsh-Hadamard transform)
"""

import numpy as np
from qusim_np.grover_search import (
    _validate_n, apply_hadamard_transform, initialize_uniform_superposition
)
from qusim_np.resources import check_memory, state_bytes


def _evaluate_f(f, N):
    """
    Evaluate f on all N basis states.

    Args:
        f: Callable f(x) -> bool/int, or a sequence of N bits
        N: Number of basis states (2^n)

    Returns:
        list: f-values for x = 0..N-1 as 0/1 ints
    """
    if callable(f):
        return [int(bool(f(x))) for x in range(N)]
    values = list(f)
    if len(values) != N:
        raise ValueError(f"f sequence must have length {N}, got {len(values)}")
    return [int(bool(v)) for v in values]


def deutsch_jozsa_oracle(n, f):
    r"""
    Construct the Deutsch-Jozsa oracle O_f that encodes f as a phase.

    O_f|x> = (-1)^{f(x)}|x>, where f(x) = 1 picks up a phase of -1.
    This is a diagonal unitary: diag((-1)^{f(0)}, ..., (-1)^{f(N-1)}).

    Args:
        n: Number of qubits
        f: Function to evaluate - either a callable f(x) -> bool/int
           or a sequence of N = 2^n bits (f(0), f(1), ..., f(N-1))

    Returns:
        numpy.ndarray: The oracle matrix of shape (2^n, 2^n)
    """
    _validate_n(n)
    N = 2 ** n
    check_memory(state_bytes(N * N))
    values = _evaluate_f(f, N)
    phases = np.array([1 - 2 * v for v in values])  # +1 if f=0, -1 if f=1
    return np.diag(phases).astype(complex)


def deutsch_jozsa(n, f):
    r"""
    Run the Deutsch-Jozsa algorithm on function f.

    Algorithm:
    1. Initialize |s> = H^{\otimes n}|0^n>
    2. Apply oracle: O_f|s>
    3. Apply H^{\otimes n} again
    4. Measure: |0^n> outcome implies f is constant

    Args:
        n: Number of qubits
        f: Callable f(x) -> bool/int, or sequence of N = 2^n bits

    Returns:
        tuple: (final_state, zero_probability)
            final_state: state vector of shape (2^n,)
            zero_probability: probability of measuring |0^n>;
                ~1.0 indicates constant, ~0.0 indicates balanced
    """
    _validate_n(n)
    N = 2 ** n
    # f(x) is evaluated into a Python list (~36 bytes per entry) next to the state
    check_memory(state_bytes(N, 64))

    # Step 1: uniform superposition
    state = initialize_uniform_superposition(n)

    # Step 2: oracle phase kickback.  O_f is diagonal, so O_f|s> is an
    # elementwise product with (-1)^{f(x)} — no N x N matrix.
    phases = 1 - 2 * np.array(_evaluate_f(f, N))
    state = phases * state

    # Step 3: H^{\otimes n} via the fast Walsh-Hadamard transform, O(n 2^n).
    state = apply_hadamard_transform(state)

    # Step 4: probability of measuring |0...0>.  Identity:
    # A_0 = (1/N) sum_x (-1)^{f(x)}, the mean of the phases.
    zero_probability = float(abs(state[0]) ** 2)

    return state, zero_probability


def deutsch_jozsa_classify(n, f, threshold=0.5):
    r"""
    Classify f as 'constant' or 'balanced' using the Deutsch-Jozsa algorithm.

    Args:
        n: Number of qubits
        f: Callable f(x) -> bool/int, or sequence of N = 2^n bits
        threshold: Decision boundary on P(|0^n>) (default 0.5)

    Returns:
        str: 'constant' if P(|0^n>) > threshold, else 'balanced'
    """
    _, p0 = deutsch_jozsa(n, f)
    return "constant" if p0 > threshold else "balanced"
