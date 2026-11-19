"""
Composing circuits: reusable builders, inverse, compose, custom gates,
and an algorithm (Grover) written as a circuit.

Run:  python examples/02_composing_circuits.py
"""

import numpy as np

from qusim_np import QuantumCircuit


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
