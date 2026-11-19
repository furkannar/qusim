"""
Variational circuits: a VQE written with QuantumCircuit, then evaluated
with finite shots and under noise.

Run:  python examples/03_variational_circuits.py
"""

import numpy as np
from scipy.optimize import minimize

import qusim_np
from qusim_np import QuantumCircuit, NoiseModel, depolarizing_kraus

# H = 0.3 II + 1.2 ZZ - 0.8 XI - 0.8 IX
coeffs = [0.3, 1.2, -0.8, -0.8]
paulis = ["II", "ZZ", "XI", "IX"]
n, layers = 2, 2


def energy(theta):
    """Exact energy of the hardware-efficient ansatz at parameters theta."""
    return QuantumCircuit.from_hea(n, theta, layers).expectation(coeffs, paulis)


theta0 = np.random.default_rng(0).uniform(-np.pi, np.pi, qusim_np.hea_param_count(n, layers))
res = minimize(energy, theta0, method="Nelder-Mead")
exact = np.linalg.eigvalsh(qusim_np.hamiltonian(coeffs, paulis))[0]
print(f"VQE energy: {res.fun:.6f}   exact ground state: {exact:.6f}")

# The optimized ansatz is an ordinary circuit you can inspect or extend
best = QuantumCircuit.from_hea(n, res.x, layers)
print(f"ansatz: {best.size} gates, depth {best.depth}")

# What would hardware report? Finite shots ...
est, err = best.expectation(coeffs, paulis, shots=10_000, rng=np.random.default_rng(1))
print(f"10k shots:            {est:.3f} +/- {err:.3f}")

# ... and gate noise: 1% depolarizing after every CNOT
best.noise_model = NoiseModel().add_gate_error(depolarizing_kraus(0.01), gates="cx")
print(f"with CNOT noise:      {best.expectation(coeffs, paulis):.6f}  (exact, density matrix)")
