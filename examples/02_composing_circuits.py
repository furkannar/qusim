"""
Composing circuits: reusable builders, inverse, compose, custom gates,
and algorithms (Grover, Shor) written as circuits.

Run:  python examples/02_composing_circuits.py
"""

import numpy as np

from qusim_np import QuantumCircuit, inverse_qft_matrix
from qusim_np.shor import (
    modular_exponentiation, modular_multiplication_permutation,
    order_from_measurement, factors_from_order, shor,
)


def ghz(n):
    """GHZ preparation: (|0...0> + |1...1>) / sqrt(2)."""
    qc = QuantumCircuit(n).h(0)
    for k in range(n - 1):
        qc.cx(k, k + 1)
    return qc


# Reusable builder -------------------------------------------------------
g = ghz(4)
print("GHZ counts:", g.sample_counts(1000, rng=np.random.default_rng(0)))

# Inverse + compose: running a circuit and then its inverse "uncomputes" it
round_trip = g.compose(g.inverse())
print("GHZ then GHZ^-1 returns |0000>:", np.allclose(round_trip.statevector(), np.eye(16)[0]))

# compose() can place a smaller circuit on chosen qubits
bell = QuantumCircuit(2).h(0).cx(0, 1)
qc = QuantumCircuit(4).compose(bell, qubits=[1, 3])   # Bell pair on qubits 1 and 3
print("Bell on (1, 3):", qc.sample_counts(1000, rng=np.random.default_rng(1)))

# Custom gates ------------------------------------------------------------
# sqrt(X): applying it twice is a NOT gate
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
sx_twice = QuantumCircuit(1).unitary(SX, 0, name="sx").unitary(SX, 0, name="sx")
print("sx . sx |0> =", sx_twice.statevector().round(6))
print(sx_twice)

# Grover search as a circuit (2 qubits, marked state |11>) ---------------
#   oracle    : CZ flips the phase of |11>
#   diffusion : H H . X X . CZ . X X . H H  (reflection about the mean)
grover = QuantumCircuit(2).h(0).h(1)
grover.cz(0, 1)                                       # oracle
grover.h(0).h(1).x(0).x(1).cz(0, 1).x(0).x(1).h(0).h(1)   # diffusion
print("Grover finds '11':", grover.sample_counts(100, rng=np.random.default_rng(2)))
print("depth:", grover.depth, "gates:", grover.size)

# Shor's algorithm as a circuit (factor M = 15 with base a = 7) ---------
#   qubits 0-7  : counting register |x>, qubit 0 is the most significant bit
#   qubits 8-11 : work register, starts at |1>
#   controlled-U^(2^j), with U|w> = |7 w mod 15>, as a custom gate whose
#   control is the counting qubit of weight 2^j; then QFT^dagger on |x>
M, a, t, n_w = 15, 7, 8, 4
W = 2 ** n_w
counting, work = list(range(t)), list(range(t, t + n_w))

shor_qc = QuantumCircuit(t + n_w).x(work[-1])        # work register = |0001> = |1>
for q in counting:
    shor_qc.h(q)
for j in range(t):
    perm = modular_multiplication_permutation(modular_exponentiation(a, 2 ** j, M), M, n_w)
    CU = np.eye(2 * W)                               # block diag(I, U^(2^j))
    CU[W:, W:] = 0
    CU[W + perm, W + np.arange(W)] = 1
    shor_qc.unitary(CU, [counting[t - 1 - j]] + work, name=f"c-U^{2 ** j}")
shor_qc.unitary(inverse_qft_matrix(t), counting, name="qft_dg")

counts = shor_qc.sample_counts(20, rng=np.random.default_rng(3))
ys = {}
for bits, c in counts.items():                       # first t bits are y
    ys[int(bits[:t], 2)] = ys.get(int(bits[:t], 2), 0) + c
print("Shor counting register y:", dict(sorted(ys.items())))   # peaks at 0, 64, 128, 192
r = max(order_from_measurement(y, 2 ** t, a, M) or 0 for y in ys)
print(f"order r = {r}, factors of {M}:", factors_from_order(a, r, M))
print("one call:", shor(M, a=a, rng=np.random.default_rng(3)))
