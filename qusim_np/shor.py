r"""
qusim_np/shor.py
Shor's Factoring Algorithm

This module implements Shor's algorithm as a state-vector simulation.

Notation (kept consistent with grover_search.py, where N = 2^n is the
number of basis states):
- M = the odd composite integer to factor
- n_w = ceil(log2 M) work qubits, which hold values mod M
- t = 2 n_w counting qubits, so the counting register has Q = 2^t >= M^2 values
- n = t + n_w total qubits, so the combined ket |x>|w> has N = 2^n entries,
  with basis index x * 2^{n_w} + w

Classical reduction (factoring -> order finding):
- Pick a with 1 < a < M and gcd(a, M) = 1
- Order r = the smallest r > 0 such that a^r = 1 (mod M)
- If r is even and a^{r/2} != -1 (mod M), the factors are
  gcd(a^{r/2} - 1, M) and gcd(a^{r/2} + 1, M)

Quantum order finding:
- |0>^{t}|1>  --H^{(x)t}-->  (1/sqrt(Q)) * sum_{x=0}^{Q-1} |x>|1>
- Modular multiplication unitary: U|w> = |a w mod M> for w < M (identity for w >= M)
- Controlled-U^{2^j} on counting qubit j gives
  (1/sqrt(Q)) * sum_{x=0}^{Q-1} |x>|a^x mod M>
- Eigenstates |u_s> = (1/sqrt(r)) sum_k e^{-2 pi i s k / r} |a^k mod M>,
  U|u_s> = e^{2 pi i s / r}|u_s>, and |1> = (1/sqrt(r)) sum_s |u_s>
- After the inverse QFT on the counting register, a measured y satisfies y/Q ~ s/r
- Continued fractions: the best approximation p/q of y/Q with q < M gives
  the candidate r = q
"""

import math
from fractions import Fraction

import numpy as np
from qusim_np.config import GB, MB, config
from qusim_np.gates import hadamard
from qusim_np.resources import MemoryMonitor, available_memory_gb, check_memory


# ---------------------------------------------------------------------------
# Classical number theory
# ---------------------------------------------------------------------------

def _validate_M(M):
    """Validate that M is an integer >= 2."""
    if not isinstance(M, (int, np.integer)) or isinstance(M, bool) or M < 2:
        raise ValueError(f"M must be an integer >= 2, got {M!r}")
    return int(M)


def divisors(M):
    r"""
    List all positive divisors of M with the square-root method.

    Divisors come in pairs (d, M/d) with d <= sqrt(M) <= M/d, so testing
    d = 1..isqrt(M) finds every divisor. This takes O(sqrt(M)) time.

    Args:
        M: Integer >= 1

    Returns:
        list: The divisors of M in ascending order
    """
    if not isinstance(M, (int, np.integer)) or isinstance(M, bool) or M < 1:
        raise ValueError(f"M must be a positive integer, got {M!r}")
    M = int(M)
    small, large = [], []
    for d in range(1, math.isqrt(M) + 1):
        if M % d == 0:
            small.append(d)
            if d != M // d:
                large.append(M // d)
    return small + large[::-1]


def is_prime(M):
    """Return True if M is prime, i.e. its only divisors are 1 and M."""
    return M >= 2 and len(divisors(M)) == 2


def _integer_root(M, k):
    """Largest integer b such that b^k <= M (exact, no floating point)."""
    lo, hi = 1, 1 << (M.bit_length() // k + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** k <= M:
            lo = mid
        else:
            hi = mid - 1
    return lo


def prime_power_base(M):
    r"""
    If M = b^k for some integer b >= 2 and k >= 2, return b (else None).

    Shor's reduction fails for prime powers p^k, because then the only
    square roots of 1 mod M are +-1. They are detected classically in
    O(log^2 M) root extractions.
    """
    M = _validate_M(M)
    for k in range(M.bit_length(), 1, -1):
        b = _integer_root(M, k)
        if b >= 2 and b ** k == M:
            return b
    return None


def modular_exponentiation(a, x, M):
    r"""
    Compute a^x mod M with Python's built-in three-argument pow.

    pow uses square-and-multiply with reduction at every step, so it costs
    O(log x) multiplications of numbers smaller than M. Computing a ** x
    first would build a number with about x * log2(a) bits.
    """
    return pow(int(a), int(x), int(M))


def validate_base(a, M):
    r"""
    Check that a is a valid base for order finding: 1 < a < M and gcd(a, M) = 1.

    Coprimality is what makes U|w> = |a w mod M> a permutation of
    {0, ..., M-1}, and therefore a unitary. If gcd(a, M) > 1, multiplying
    by a is not invertible mod M, and a^r = 1 (mod M) has no solution.

    Raises:
        ValueError: If a is out of range or shares a factor with M
    """
    M = _validate_M(M)
    if not isinstance(a, (int, np.integer)) or isinstance(a, bool):
        raise ValueError(f"a must be an integer, got {a!r}")
    if not 1 < a < M:
        raise ValueError(f"a must satisfy 1 < a < M = {M}, got {a}")
    g = math.gcd(int(a), M)
    if g != 1:
        raise ValueError(f"gcd(a, M) = gcd({a}, {M}) = {g} != 1; "
                         f"{g} is already a factor of M")


def classical_order(a, M):
    """Order of a mod M by brute force, O(r) steps. Used as a reference in tests."""
    validate_base(a, M)
    r, value = 1, a % M
    while value != 1:
        value = (value * a) % M
        r += 1
    return r


# ---------------------------------------------------------------------------
# Registers and the quantum state
# ---------------------------------------------------------------------------

def _validate_memory_gb(memory_gb):
    is_number = isinstance(memory_gb, (int, float, np.integer, np.floating))
    if isinstance(memory_gb, bool) or not is_number or not memory_gb > 0:
        raise ValueError(f"memory_gb must be a positive number, got {memory_gb!r}")
    return float(memory_gb)


def memory_required(M):
    r"""
    Peak memory in bytes to simulate order finding mod M.

    config.BYTES_PER_AMPLITUDE * 2^n, with n = 3 ceil(log2 M) total qubits.
    """
    return config.BYTES_PER_AMPLITUDE * 2 ** register_sizes(M)[2]


def detected_budget_gb():
    r"""
    Memory budget detected from the machine, in GiB.

    The RAM available right now (psutil), minus the config.MIN_FREE_RAM_MB
    that is always kept free. Can be <= 0 when RAM is nearly exhausted.
    """
    return available_memory_gb() - config.MIN_FREE_RAM_MB * MB / GB


def estimate_capacity(memory_gb=None):
    r"""
    Estimate what Shor's algorithm can simulate within a memory budget.

    With memory_gb=None the budget is detected: the RAM available right
    now minus the config.MIN_FREE_RAM_MB reserve (see detected_budget_gb).
    Pass memory_gb to plan for a different machine, or to cap the
    simulation below what is free. The state uses
    config.BYTES_PER_AMPLITUDE bytes per amplitude at peak, so

        max_qubits = floor(log2(memory / config.BYTES_PER_AMPLITUDE))

    The registers always use n = 3 n_w qubits (t = 2 n_w counting plus
    n_w work), so only multiples of 3 can actually be used, and the
    largest M that fits satisfies ceil(log2 M) = n_w = max_qubits // 3,
    that is M <= 2^{n_w}.

    Args:
        memory_gb: Memory budget in GiB (2^30 bytes), e.g. 8 or 0.5, or
            None to detect it

    Returns:
        dict: "memory_gb" (the budget used), "detected" (True if it was
        read from the machine), "max_qubits" (fits the budget),
        "usable_qubits" (largest multiple of 3 that fits), "work_qubits"
        (n_w), "max_M" (largest integer that can be factored),
        "peak_bytes" (memory of a run at usable_qubits)

    Raises:
        RuntimeError: If memory_gb is None and RAM is dangerously low
    """
    detected = memory_gb is None
    if detected:
        memory_gb = detected_budget_gb()
        if memory_gb <= 0:
            check_memory()          # raises the low-RAM RuntimeError
    budget = _validate_memory_gb(memory_gb) * GB
    per_amp = config.BYTES_PER_AMPLITUDE
    if budget < per_amp:
        max_qubits = 0
    else:
        max_qubits = int(budget // per_amp).bit_length() - 1
    n_w = max_qubits // 3
    if n_w < 2:
        raise ValueError(f"{memory_gb} GiB holds {max_qubits} qubits; "
                         f"the smallest case (M = 15, 12 qubits) needs "
                         f"{memory_required(15) / GB:.2g} GiB")
    return {
        "memory_gb": float(memory_gb),
        "detected": detected,
        "max_qubits": max_qubits,
        "usable_qubits": 3 * n_w,
        "work_qubits": n_w,
        "max_M": 2 ** n_w,
        "peak_bytes": per_amp * 2 ** (3 * n_w),
    }


def _check_memory(M, memory_gb):
    r"""
    Refuse to simulate M if it does not fit.

    - memory_gb given: ValueError if memory_required(M) exceeds it
    - always: RuntimeError if the allocation would leave less than
      config.MIN_FREE_RAM_MB of RAM free (resources.check_memory), so a
      budget larger than the real free RAM is still safe
    """
    need = memory_required(M)
    if memory_gb is not None:
        budget = _validate_memory_gb(memory_gb) * GB
        if need > budget:
            cap = estimate_capacity(memory_gb)
            raise ValueError(
                f"M = {M} needs {register_sizes(M)[2]} qubits and about "
                f"{need / GB:.3g} GiB, more than memory_gb = {memory_gb}; "
                f"that budget holds {cap['usable_qubits']} qubits, i.e. M <= {cap['max_M']}")
    check_memory(need)


def register_sizes(M):
    r"""
    Qubit counts for order finding mod M.

    Returns:
        tuple: (t, n_w, n). n_w = ceil(log2 M) work qubits, t = 2 n_w
        counting qubits, and n = t + n_w total qubits. The combined state
        has N = 2^n amplitudes.
    """
    M = _validate_M(M)
    n_w = (M - 1).bit_length()   # smallest n_w with 2^{n_w} >= M
    t = 2 * n_w
    return t, n_w, t + n_w


def counting_superposition(t):
    r"""
    Uniform superposition over the counting register: H^{(x)t} |0>^{t}.

    H|0> = (|0> + |1>)/sqrt(2) is the first column of the Hadamard gate
    from gates.py, so the tensor product of t copies is
    (1/sqrt(Q)) * sum_{x=0}^{Q-1} |x> with Q = 2^t. This costs O(Q).
    The 2^t x 2^t matrix H^{(x)t} is never built.
    """
    h0 = hadamard().matrix[:, 0]
    state = np.ones(1, dtype=complex)
    for _ in range(t):
        state = np.kron(state, h0)
    return state


def modular_multiplication_permutation(a, M, n_w):
    r"""
    The unitary U|w> = |a w mod M>, stored as an index map.

    perm[w] is the image of basis state w: (a * w) mod M for w < M, and
    w itself for the unused states M <= w < 2^{n_w}. Since gcd(a, M) = 1
    this is a bijection, so U is a permutation matrix. Storing perm costs
    O(2^{n_w}) memory instead of O(4^{n_w}) for the dense matrix.
    """
    W = 2 ** n_w
    w = np.arange(W)
    perm = w.copy()
    perm[:M] = (a * w[:M]) % M
    return perm


def modular_multiplication_unitary(a, M, n_w=None):
    r"""
    Dense matrix of U|w> = |a w mod M>, for inspection and tests on small M.

    U[perm[w], w] = 1, so column w is the basis vector |a w mod M>.
    """
    validate_base(a, M)
    if n_w is None:
        n_w = register_sizes(M)[1]
    perm = modular_multiplication_permutation(a, M, n_w)
    W = 2 ** n_w
    U = np.zeros((W, W), dtype=complex)
    U[perm, np.arange(W)] = 1.0
    return U


def order_finding_state(a, M, memory_gb=None):
    r"""
    State of all n = t + n_w qubits just before the inverse QFT:

        (1/sqrt(Q)) * sum_{x=0}^{Q-1} |x> |a^x mod M>

    The circuit is built step by step:
    1. |0>^{t}|1>: the work register starts at |1>, not |0>
    2. H^{(x)t} on the counting register (Hadamard from gates.py)
    3. Controlled-U^{2^j} for each counting qubit j. U^{2^j} is
       multiplication by a^{2^j} mod M, computed with pow, so each power
       costs one permutation instead of 2^j applications of U.

    Each controlled permutation moves amplitudes in O(N) time, so the
    total cost is O(t N) time and O(N) memory, with N = 2^n.

    Args:
        a: Base with 1 < a < M and gcd(a, M) = 1
        M: Integer to factor
        memory_gb: Optional memory budget in GiB. If given, refuse to run
            (ValueError) when memory_required(M) exceeds it. Either way,
            refuse (RuntimeError) if the run would leave less than
            config.MIN_FREE_RAM_MB of RAM free.

    Returns:
        numpy.ndarray: Shape (Q, W) with Q = 2^t and W = 2^{n_w}. Entry
        [x, w] is the amplitude of |x>|w>. Use .ravel() to get the flat
        N = 2^n ket.
    """
    validate_base(a, M)
    _check_memory(M, memory_gb)
    t, n_w, n = register_sizes(M)
    Q, W = 2 ** t, 2 ** n_w

    # Steps 1 and 2: |x> runs over the uniform superposition, |w> = |1>
    state = np.zeros((Q, W), dtype=complex)
    state[:, 1] = counting_superposition(t)

    # Step 3: controlled-U^{2^j}. Counting qubit j carries weight 2^j in
    # x, so the rows it controls are those where bit j of x is set.
    x = np.arange(Q)
    monitor = MemoryMonitor(state.size)
    for j in range(t):
        a_pow = modular_exponentiation(a, 2 ** j, M)
        perm = modular_multiplication_permutation(a_pow, M, n_w)
        rows = (x >> j) & 1 == 1
        # Basis state w moves to perm[w]: new[:, perm[w]] = old[:, w],
        # i.e. new[:, k] = old[:, perm^{-1}[k]]
        inv_perm = np.argsort(perm)
        state[rows] = state[rows][:, inv_perm]
        monitor.step()
    return state


def counting_distribution(a, M, memory_gb=None):
    r"""
    Probability P(y) of each counting-register outcome after the inverse QFT.

    QFT^dagger is applied to the counting register for each work value w
    separately (np.fft.fft along axis 0, the same as inverse_qft_state in
    qft.py, one column at a time). The work register is then traced out:

        P(y) = sum_{w} |<y, w| (QFT^dagger (x) I) |psi>|^2

    Returns:
        numpy.ndarray: Shape (Q,), sums to 1, peaked near y = s Q / r
    """
    state = order_finding_state(a, M, memory_gb=memory_gb)
    state = np.fft.fft(state, axis=0, norm="ortho")
    return np.sum(np.abs(state) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Classical post-processing
# ---------------------------------------------------------------------------

def order_from_measurement(y, Q, a, M):
    r"""
    Recover the order r from a measured y using continued fractions.

    y/Q ~ s/r with r < M. Fraction.limit_denominator(M - 1) returns the
    closest fraction p/q with q < M, which is the continued-fraction
    approximation Shor's analysis requires. If gcd(s, r) > 1, then q is a
    proper divisor of r, so small multiples of q are tried as well.

    Returns:
        int or None: The order r if it could be verified, else None
    """
    if y == 0:
        return None
    q = Fraction(int(y), int(Q)).limit_denominator(M - 1).denominator
    for k in range(1, M // q + 1):
        r = k * q
        if modular_exponentiation(a, r, M) == 1:
            return r
    return None


def quantum_order_finding(a, M, shots=1, rng=None, memory_gb=None):
    r"""
    Find the order of a mod M with the simulated quantum circuit.

    Samples up to `shots` measurement outcomes y from P(y) and returns
    the first order that continued fractions can verify.

    Returns:
        tuple: (r, ys). r is the order, or None if every shot failed;
        ys is the list of measured outcomes.
    """
    if rng is None:
        rng = np.random.default_rng()
    P = counting_distribution(a, M, memory_gb=memory_gb)
    Q = P.size
    ys = []
    for y in rng.choice(Q, size=shots, p=P / np.sum(P)):
        ys.append(int(y))
        r = order_from_measurement(int(y), Q, a, M)
        if r is not None:
            return r, ys
    return None, ys


def factors_from_order(a, r, M):
    r"""
    Turn an order r into a pair of nontrivial factors, if possible.

    Needs r even and a^{r/2} != -1 (mod M). Then
    (a^{r/2} - 1)(a^{r/2} + 1) = 0 (mod M) with neither factor = 0 (mod M).

    Returns:
        tuple or None: (p, q) with p * q = M and 1 < p <= q, or None
    """
    if r is None or r % 2:
        return None
    half = modular_exponentiation(a, r // 2, M)
    if half == M - 1:
        return None
    p = math.gcd(half - 1, M)
    if 1 < p < M:
        return tuple(sorted((p, M // p)))
    return None


def shor(M, a=None, max_attempts=10, shots=4, rng=None, memory_gb=None):
    r"""
    Factor M with Shor's algorithm.

    Algorithm:
    1. Classical checks. Even M returns (2, M/2). Prime M is rejected,
       since it has no factors to find (checked with divisors()). A prime
       power M = b^k returns (b, M/b).
    2. Pick a with 1 < a < M. If gcd(a, M) > 1, that gcd is already a
       factor (only possible when a is drawn at random; a given a must
       be coprime to M, see validate_base).
    3. Quantum order finding gives r.
    4. If r is even and a^{r/2} != -1 (mod M), return the factors from
       the gcds. Otherwise try another a.

    Args:
        M: Odd composite integer to factor
        a: Optional fixed base. If given, only that base is tried.
        max_attempts: Number of random bases to try
        shots: Measurements per base for order finding
        rng: numpy.random.Generator or None
        memory_gb: Optional memory budget in GiB, checked before the
            quantum step (see estimate_capacity). None means use what the
            machine has free, keeping config.MIN_FREE_RAM_MB in reserve.

    Returns:
        dict: "factors" (p, q) or None, "a", "r", "method"
        ("classical" | "gcd" | "quantum"), and "attempts"
    """
    M = _validate_M(M)
    if rng is None:
        rng = np.random.default_rng()

    if M % 2 == 0:
        return {"factors": (2, M // 2), "a": None, "r": None,
                "method": "classical", "attempts": 0}
    if is_prime(M):
        raise ValueError(f"M = {M} is prime; there are no factors to find")
    b = prime_power_base(M)
    if b is not None:
        return {"factors": (b, M // b), "a": None, "r": None,
                "method": "classical", "attempts": 0}

    if a is not None:
        validate_base(a, M)
        bases = [int(a)]
    else:
        bases = (int(v) for v in rng.integers(2, M, size=max_attempts))

    a_i, r = None, None
    for attempt, a_i in enumerate(bases, start=1):
        g = math.gcd(a_i, M)
        if g != 1:
            return {"factors": tuple(sorted((g, M // g))), "a": a_i,
                    "r": None, "method": "gcd", "attempts": attempt}
        r, _ = quantum_order_finding(a_i, M, shots=shots, rng=rng,
                                     memory_gb=memory_gb)
        factors = factors_from_order(a_i, r, M)
        if factors is not None:
            return {"factors": factors, "a": a_i, "r": r,
                    "method": "quantum", "attempts": attempt}
    return {"factors": None, "a": a_i, "r": r, "method": "quantum",
            "attempts": 1 if a is not None else max_attempts}
