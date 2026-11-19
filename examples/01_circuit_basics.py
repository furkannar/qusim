"""
Circuit basics: build, inspect, simulate, measure.

Run:  python examples/01_circuit_basics.py
"""

import numpy as np

from qusim_np import QuantumCircuit

rng = np.random.default_rng(1)

# 1. Build -------------------------------------------------------------
# Gate methods record an instruction and return the circuit, so they chain.
bell = QuantumCircuit(2).h(0).cx(0, 1)

# 2. Inspect -----------------------------------------------------------
print(bell)
print("qubits, gates, depth:", bell.num_qubits, bell.size, bell.depth)
print("first instruction:", bell.instructions[0].name, bell.instructions[0].qubits)

# 3. Simulate ----------------------------------------------------------
# Nothing has been computed yet; statevector() runs the circuit now.
print("state:", bell.statevector().round(3))
print("probabilities:", bell.probabilities().round(3))

# 4. Measure -----------------------------------------------------------
# Bitstrings list qubit 0 FIRST (leftmost).
print("counts:", bell.sample_counts(1000, rng=rng))
print("x on qubit 0 measures:", QuantumCircuit(2).x(0).sample_counts(10))

# 5. Observables -------------------------------------------------------
# <H> for H = ZZ + 0.5 XX + 0.2 IZ: exact, then from 8000 shots
coeffs, paulis = [1.0, 0.5, 0.2], ["ZZ", "XX", "IZ"]
print("exact <H>:", round(bell.expectation(coeffs, paulis), 6))
est, err = bell.expectation(coeffs, paulis, shots=8000, rng=rng)
print(f"8000-shot <H>: {est:.3f} +/- {err:.3f}")

# A state with real shot noise: Ry(1.0) puts qubit 0 in a superposition
tilted = QuantumCircuit(1).ry(1.0, 0)
est, err = tilted.expectation([1.0], ["Z"], shots=4000, rng=rng)
print(f"<Z> after Ry(1.0): exact {np.cos(1.0):.3f}, estimate {est:.3f} +/- {err:.3f}")
