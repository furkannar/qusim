# qusim-np

**A lightweight quantum simulation package in pure NumPy.**

Matrix-free state-vector gates and algorithms, a working Variational
Quantum Eigensolver, Kraus noise channels, and finite-shot measurement
statistics — in ~3,000 lines of documented code backed by 354 tests.
Only NumPy and SciPy are required.

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
| **Noise models** | Amplitude damping, phase damping/flip, thermal relaxation, depolarizing, random Pauli channels — Kraus sets validated for completeness |
| **Shot noise** | Finite-shot estimators, CLT / Hoeffding / Bernstein shot bounds, optimal shot allocation, grouped (commuting) measurements, bitstring sampling |
| **Performance** | Matrix-free throughout: gates, Pauli strings, Grover, Deutsch–Jozsa and QFT act directly on the state vector (O(2ⁿ) – O(n·2ⁿ) per step) — see [Performance](#performance) |

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

# Apply gates straight to state vectors (O(n·2^n), no big matrices)
state = qusim_np.initialize_uniform_superposition(2)
state = qusim_np.apply_single_qubit(state, qusim_np.gates.hadamard().matrix, 0)
state = qusim_np.apply_two_qubit(state, qusim_np.gates.cnot().matrix, 0, 1)

# Expectation values WITHOUT building the 2^n x 2^n matrix (O(2^n))
qusim_np.pauli_string_expval(state, "ZZ")

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

## Shot noise

Real hardware returns samples, not amplitudes. `qusim_np.shot_noise`
turns exact states into finite-shot data and tells you how many shots
an estimate needs. For a single-shot variance $\sigma^2$, the sample
mean of $N$ shots has standard error $\sigma/\sqrt{N}$, and for a
Hamiltonian $H = \sum_i c_i P_i$ measured term-by-term the optimal
allocation is

$$N_i \propto |c_i|\,\sigma_i, \qquad
\mathrm{Var}_{\min}(\hat H) = \frac{\left(\sum_i |c_i|\,\sigma_i\right)^2}{N}$$

```python
import numpy as np
import qusim_np

a = 0.4
psi = np.array([np.cos(a), 0, 0, np.sin(a)])      # cos(a)|00> + sin(a)|11>
XX = qusim_np.pauli_string_matrix("XX")

# Measure XX on 4096 shots and estimate <XX> with its standard error
rng = np.random.default_rng(7)
outcomes = qusim_np.sample_observable_outcomes(psi, XX, shots=4096, rng=rng)
mean, stderr = qusim_np.estimate_expval(outcomes)
# -> <XX> = 0.721 +/- 0.011   (exact 0.717)

# Shots for a +/-0.01 half-width at 95% confidence on a Pauli term
qusim_np.shots_clt(1.0, 0.01, delta=0.05)          # 38415  (asymptotic)
qusim_np.shots_hoeffding(0.01, 0.05)               # 73778  (rigorous)

# Split 10,000 shots across H = 0.3 II + 1.2 ZZ - 0.8 XI - 0.8 IX
# (sigma = 0 for II: the identity is deterministic and needs no shots)
qusim_np.allocate_shots([0.3, 1.2, -0.8, -0.8], [0, 1, 1, 1], 10_000)
# -> [0, 4286, 2857, 2857]

# Computational-basis counts
qusim_np.sample_bitstrings(psi, shots=1000, rng=rng)   # -> [840, 0, 0, 160]
```

Grouped measurements of commuting terms (`group_variance`,
`allocate_group_shots`) account for the covariances between terms
measured on the same shots.

## Performance

Every algorithm runs on the state vector itself. Wherever a large
operator has known structure, a mathematical identity replaces the
dense 2ⁿ×2ⁿ matrix:

| File | Optimization | Identity used | Cost before → after |
|---|---|---|---|
| `grover_search.py` | Uniform superposition built directly | $H^{\otimes n}\lvert 0^n\rangle = \tfrac{1}{\sqrt N}(1,\dots,1)$ | O(4ⁿ) → O(2ⁿ) |
| | Oracle applied as a sign flip | $O_f$ is diagonal with entries $\pm 1$ | O(4ⁿ) → O(M) per iteration |
| | Diffusion as inversion about the mean | $D\lvert\psi\rangle = 2\,\overline{\psi} - \psi$ since $\lvert s\rangle\langle s\rvert = J/N$ | O(4ⁿ) → O(2ⁿ) per iteration |
| | `grover_iteration` matrix | $D\,O_f$ = $D$ with marked columns negated | O(8ⁿ) → O(4ⁿ) |
| | `diffusion_operator` matrix | $D = \tfrac{2}{N}J - I$ in closed form | no Hadamard matrix needed |
| | Exact optimal iteration count | $P(k) = \sin^2\!\big(\tfrac{(2k+1)\theta}{2}\big)$ peaks at $k = \tfrac{\pi}{2\theta} - \tfrac12$ | equal or higher success probability than the small-angle formula in every case checked (n ≤ 14) |
| | New `apply_hadamard_transform` | Fast Walsh–Hadamard butterfly $(a, b) \to (a+b,\,a-b)$ | O(4ⁿ) → O(n·2ⁿ) |
| | `hadamard_matrix_closed_form` vectorized | $H_{jk} = (-1)^{\mathrm{popcount}(j \,\&\, k)}/\sqrt N$ | Python double loop → NumPy |
| `deutsch_jozsa.py` | Oracle as elementwise phase, then fast Walsh–Hadamard | $A_0 = \tfrac1N\sum_x (-1)^{f(x)}$ | O(4ⁿ) → O(n·2ⁿ) |
| `qft.py` | `qft_state` / `inverse_qft_state` via FFT | QFT $=$ `ifft(x, norm="ortho")`, inverse $=$ `fft` | O(4ⁿ) → O(n·2ⁿ) |
| | `qft_matrix` reduces the exponent first | $\omega^{jk} = \omega^{jk \bmod N}$ | better precision for large $jk$ |
| `vqe.py` | Pauli strings as signed permutations | $Y = iXZ \Rightarrow P\lvert c\rangle = i^{n_Y}(-1)^{\mathrm{popcount}(c \,\&\, z)}\lvert c \oplus x\rangle$ | apply / expectation: O(n·2ⁿ) → O(2ⁿ) |
| | `hamiltonian()` scatters each term into $H$ | one nonzero per column of each $P_i$ | memory O(m·4ⁿ) → O(4ⁿ) |
| | HEA reuses the CNOT matrix across layers | — | fewer allocations per energy call |
| `noise_models/channels.py` | Kraus sets checked once for completeness | $\sum_k K_k^\dagger K_k = I \iff$ trace-preserving on every input | replaces a per-state trace check that could miss invalid channels |
| `shot_noise/bounds.py` | SciPy normal quantile imported lazily | `scipy.special.ndtri` | `import qusim_np`: ~655 ms → ~86 ms |
| `shot_noise/hamiltonian.py` | `hamiltonian_variance` accepts 0 shots on terms with $c_i\sigma_i = 0$ | such terms contribute no variance | output of `allocate_shots` can be passed straight in |
| `shot_noise/pauli_stats.py` | Shot count floored at 1 | at $\lvert\langle P\rangle\rvert = 1$ one shot fixes the outcome | no more "0 shots needed" |

`gates.py` and `config.py` are unchanged; the remaining noise-channel
files already used tensor contraction on density matrices.

Measured on a laptop CPU, before → after:

| Task | Before | After |
|---|---|---|
| `grover_search`, n = 11 | 0.46 s | < 1 ms |
| `deutsch_jozsa`, n = 11 | 0.15 s | < 1 ms |
| `qft_state`, n = 12 | 0.68 s | 2 ms |
| `hamiltonian`, n = 10, 50 terms | 0.93 s | 7 ms |
| `vqe_optimize`, n = 4, 2 layers | 1.30 s | 0.88 s |
| `grover_search`, n = 20 | out of memory | 4.1 s |
| `deutsch_jozsa`, n = 22 | out of memory | 1.7 s |

Every fast path is cross-checked against its dense reference in
`tests/test_fast_paths.py` (agreement to ~1e-15).

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
├── qft.py                   # QFT / inverse QFT (FFT-based on states)
├── vqe.py                   # Pauli strings, Hamiltonians, HEA ansatz, VQE driver
├── noise_models/            # Kraus channels: damping, relaxation, Pauli errors
└── shot_noise/              # finite-shot estimation, shot bounds, allocation, sampling
tests/                       # 354 tests, incl. test_fast_paths.py (fast vs. dense cross-checks)
```

## Limitations

Scope notes for this lightweight quantum simulation package:

- **State-vector memory**: all algorithms are now matrix-free, so the
  only O(2ⁿ) cost is the state vector itself (16 bytes per amplitude:
  ~0.5 GB at 25 qubits, ~4 GB at 28). Earlier versions also built
  2ⁿ×2ⁿ operators for Grover, Deutsch–Jozsa and the QFT, which capped
  them near 11–12 qubits; that limit is gone
- **Density matrices are O(4ⁿ)**: noise channels act on density
  matrices, so noisy simulation is practical up to ~12–13 qubits
- **Explicit matrix builders stay dense by definition**: `hamiltonian`,
  `pauli_string_matrix`, `qft_matrix`, `grover_oracle`,
  `diffusion_operator` and `hadamard_transform` return full 2ⁿ×2ⁿ
  matrices — use the state-vector functions for large n
- **Standalone noise**: channels operate on density matrices you manage
  yourself — no automatic per-gate noise injection. Shot noise is
  available via `qusim_np.shot_noise` but is not wired into the
  algorithms automatically
- **No circuit API**: you compose gates programmatically, not via a `QuantumCircuit` builder
- **Classical only**: no real-device or cloud backend integration

For production quantum computing, see [Qiskit](https://www.ibm.com/quantum/qiskit),
[PennyLane](https://pennylane.ai/), or [Cirq](https://quantumai.google/cirq).

## Development

```bash
pip install -e .[dev]
pytest          # 354 tests, ~1.5s
```

Contributions welcome — see the test suite for conventions.

## AI assistance

Parts of this codebase were written with assistance from **SWE-2 by
Cognition**, accessed through the **Devin** platform, for coding tasks
such as implementing algorithms, tests, and documentation.

Version 1.2 was developed with **Claude Opus 5.5 by Anthropic**, used
through **Claude Code**. That work covered analysing the codebase, the
matrix-free rewrites of Grover, Deutsch–Jozsa, the QFT and the
Pauli-string routines, Kraus completeness validation, fixes to the
shot-noise module, the dense-reference cross-check tests, benchmarking,
and this documentation.

All architectural and design decisions were made by the author.

## License

[MIT](LICENSE) © 2026 Furkan Nar
