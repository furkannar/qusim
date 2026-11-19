"""
Noise: NoiseModel rules, exact density-matrix simulation, and quantum
trajectories for circuits too large for a density matrix.

Run:  python examples/04_noise_and_trajectories.py
"""

import numpy as np

import qusim_np
from qusim_np import QuantumCircuit, NoiseModel

rng = np.random.default_rng(3)

# 1. A noise model is a list of rules: "after gate G (on qubits Q), apply this channel"
nm = (NoiseModel()
      .add_gate_error(qusim_np.depolarizing_kraus(0.02), gates="cx")       # both CNOT qubits
      .add_gate_error(qusim_np.amplitude_damping_kraus(0.01), gates="h")   # T1 decay after H
      .add_gate_error(qusim_np.phase_damping_kraus(0.05), qubits=0))       # extra dephasing on qubit 0
print(nm)

# 2. Exact noisy simulation on a density matrix (O(4^n) memory)
qc = QuantumCircuit(3, nm).h(0).cx(0, 1).cx(1, 2)
rho = qc.density_matrix()
print("purity Tr(rho^2):", round(float(np.real(np.trace(rho @ rho))), 4), "(1 = pure)")
print("noisy <ZZI>:", round(qc.expectation([1.0], ["ZZI"]), 4))
print("noisy <XXX>:", round(qc.expectation([1.0], ["XXX"]), 4))

# 3. The same circuit with quantum trajectories (O(2^n) memory) agrees
est, err = qc.expectation([1.0], ["XXX"], method="trajectory", trajectories=2000, rng=rng)
print(f"trajectory <XXX>: {est:.4f} +/- {err:.4f}")

# 4. Trajectories scale past the density-matrix limit: 16-qubit noisy GHZ
#    (a density matrix would need 69 GB; one trajectory needs 1 MB)
n = 16
big = QuantumCircuit(n, NoiseModel().add_gate_error(qusim_np.depolarizing_kraus(0.01), gates="cx")).h(0)
for k in range(n - 1):
    big.cx(k, k + 1)
counts = big.sample_counts(50, rng=rng, method="trajectory")
kept = counts.get("0" * n, 0) + counts.get("1" * n, 0)
print(f"16-qubit GHZ: {kept}/50 shots stayed in the GHZ subspace, "
      f"{len(counts)} distinct outcomes")
