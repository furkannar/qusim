# qusim-np

**A lightweight quantum computing simulator in pure NumPy.**

State-vector gates, textbook quantum algorithms, a working
Variational Quantum Eigensolver, and noise channels — in ~1,500 lines
of well-documented, well-tested code. Built for learning, teaching,
and prototyping.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/furkannar/qusim-np/actions/workflows/ci.yml/badge.svg)](https://github.com/furkannar/qusim-np/actions/workflows/ci.yml)
[![Test count](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/furkannar/qusim-np/master/badge.json)](https://github.com/furkannar/qusim-np/actions/workflows/ci.yml)

---

## Features

| Category | What you get |
|---|---|
| **Gates** | Pauli X/Y/Z, Hadamard, S, T, CNOT, SWAP, CZ, Toffoli, `rx`/`ry`/`rz` rotations |
| **Algorithms** | Grover search, Deutsch–Jozsa, Quantum Fourier Transform |
| **VQE** | Pauli-string Hamiltonians, hardware-efficient ansatz, `scipy` optimizer |
| **Noise models** | Amplitude damping, phase damping/flip, thermal relaxation, depolarizing, random Pauli channels |
| **Performance** | 1- and 2-qubit gates applied to state vectors in O(n·2ⁿ) — no dense 2ⁿ×2ⁿ matrices in the hot path |

## Install

```bash
pip install qusim-np
```

From source:

```bash
git clone https://github.com/furkannar/qusim-np.git
cd qusim-np
pip install -e .
```

Requires Python ≥ 3.10, NumPy ≥ 2.0, SciPy ≥ 1.11.

## Quick start

```python
import qusim_np

# Grover's search — find state |101> among 8 possibilities
state, k, p = qusim_np.grover_search(3, marked_states=[5])
print(f"{k} iterations -> P(marked) = {p:.3f}")

# Deutsch-Jozsa — constant or balanced with ONE query
print(qusim_np.deutsch_jozsa_classify(3, lambda x: x % 2))   # 'balanced'

# Quantum Fourier Transform
F = qusim_np.qft_matrix(3)          # 8x8 unitary
```

### The math, briefly

Grover's algorithm rotates the state vector in the plane spanned by the
marked and unmarked subspaces. With $N = 2^n$ basis states and $M$ marked
states, each iteration rotates by

$$\theta = 2\arcsin\sqrt{\frac{M}{N}}, \qquad
k_{\mathrm{opt}} \approx \frac{\pi}{4}\sqrt{\frac{N}{M}}$$

so the marked states are amplified after $O(\sqrt{N/M})$ oracle queries.

The Quantum Fourier Transform is the unitary

$$F_{jk} = \frac{1}{\sqrt{N}}\,\omega^{jk}, \qquad
\omega = e^{2\pi i / N}, \qquad j,k = 0,\dots,N-1$$

## Variational Quantum Eigensolver

The flagship feature: find the ground-state energy of a Hamiltonian
written as a linear combination of Pauli strings,

$$H = \sum_{i=0}^{m-1} \alpha_i P_i, \qquad
P_i = \bigotimes_{k=0}^{n-1} \sigma_{i,k}, \qquad
\sigma_{i,k} \in \{I, X, Y, Z\}$$

by minimizing the variational energy

$$E(\theta) = \langle \psi(\theta) \,|\, H \,|\, \psi(\theta) \rangle$$

over the parameters of a hardware-efficient ansatz (HEA).

```python
import numpy as np
import qusim_np

# H = 0.3·II + 1.2·ZZ - 0.8·XI - 0.8·IX   (2 qubits)
coeffs  = [0.3, 1.2, -0.8, -0.8]
strings = ["II", "ZZ", "XI", "IX"]

# Minimize <psi(theta)|H|psi(theta)> over a hardware-efficient ansatz
res = qusim_np.vqe_optimize(2, layers=2,
                         coefficients=coeffs, pauli_strings=strings)
print(f"VQE energy:  {res.fun:.6f}")

# Exact answer for comparison (dense diagonalization)
H = qusim_np.hamiltonian(coeffs, strings)
print(f"Exact:       {np.linalg.eigvalsh(H)[0]:.6f}")
# -> VQE energy:  -1.700000   Exact:  -1.700000
```

### VQE building blocks

```python
# Pauli strings: "XZI", ("X","Z","I"), or index tuples (1,3,0) — all equivalent
P = qusim_np.pauli_string_matrix("XZI")       # 8x8 matrix, 3 qubits

# Expectation values WITHOUT building the 2^n x 2^n matrix
qusim_np.pauli_string_expval(state, "ZZ")

# Apply gates straight to state vectors (O(n·2^n), no big matrices)
state = qusim_np.initialize_uniform_superposition(2)
state = qusim_np.apply_single_qubit(state, qusim_np.gates.hadamard().matrix, 0)
state = qusim_np.apply_two_qubit(state, qusim_np.gates.cnot().matrix, 0, 1)

# Inspect the ansatz before optimizing
theta = np.zeros(qusim_np.hea_param_count(2, layers=2))
psi   = qusim_np.hea_state(2, theta, layers=2)
```

## Noise models

Standalone quantum channels in `qusim_np.noise_models`, built on the
general Kraus form

$$\mathcal{E}(\rho) = \sum_k K_k \,\rho\, K_k^\dagger, \qquad
\sum_k K_k^\dagger K_k = I$$

Each channel accepts a state vector or density matrix and returns a
density matrix — apply them as you wish, e.g. to model idle time or
noisy gates in your own experiments.

```python
import numpy as np
import qusim_np

# |1> subject to amplitude damping with gamma = 0.3
rho = qusim_np.to_density_matrix(np.array([0, 1]))
rho = qusim_np.amplitude_damping(rho, 0.3)

# E(rho) = [[0.3, 0], [0, 0.7]] — excited population decayed by gamma
print(rho)

# Depolarizing channel: E(rho) = (1-p) rho + (p/2) I
rho = qusim_np.depolarizing(rho, 0.1, qubit=0)

# Random Pauli channel, probs = (q_I, q_X, q_Y, q_Z)
rho = qusim_np.random_pauli(rho, (0.9, 0.05, 0.03, 0.02))

# Thermal relaxation over an idle time t (needs T2 <= 2*T1)
rho = qusim_np.thermal_relaxation(rho, t=1e-6, T1=50e-6, T2=30e-6)
```

## Qubit convention

Qubit 0 is the **most significant bit** of the basis-state index:

$$|q_0 q_1 \dots q_{n-1}\rangle \;\longleftrightarrow\;
\sum_{k=0}^{n-1} q_k\, 2^{\,n-1-k}$$

A Pauli string `"XZI"` means X on qubit 0, Z on qubit 1, I on qubit 2 —
same convention as Qiskit and most textbooks.

## Project layout

```
qusim_np/
├── config.py                # constants + Pauli matrices
├── gates.py                 # QuantumGate + gate constructors
├── grover_search.py         # Hadamard transforms, oracle, diffusion, Grover
├── deutsch_jozsa.py         # Deutsch-Jozsa algorithm
├── qft.py                   # QFT / inverse QFT
├── vqe.py                   # Pauli strings, Hamiltonians, HEA ansatz, VQE driver
└── noise_models/            # Kraus channels: damping, relaxation, Pauli errors
tests/                       # 260 tests
```

## Limitations

Honest scope notes — this is a lightweight quantum simulator:

- **Dense state vectors**: memory is O(2ⁿ); practical limit ~15–20 qubits
- **Standalone noise only**: channels operate on density matrices you manage yourself — no automatic per-gate noise injection or shot-noise sampling in the engine
- **No circuit API**: you compose gates programmatically, not via a `QuantumCircuit` builder
- **Classical only**: no real-device or cloud backend integration

For production quantum computing, see [Qiskit](https://www.ibm.com/quantum/qiskit),
[PennyLane](https://pennylane.ai/), or [Cirq](https://quantumai.google/cirq).

## Development

```bash
pip install -e .[dev]
pytest          # 260 tests, ~1.5s
```

Contributions welcome — see the test suite for conventions.

## AI assistance

Parts of this codebase were written with assistance from **SWE-2 by
Cognition**, accessed through the **Devin** platform, for coding tasks
such as implementing algorithms, tests, and documentation. All
architectural and design decisions were made by the author.

## License

[MIT](LICENSE) © 2026 Furkan Nar
