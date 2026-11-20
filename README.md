# qusim-np

**A lightweight quantum simulation package in pure NumPy.**

A chainable `QuantumCircuit` builder, matrix-free state-vector algorithms
(including Shor's factoring algorithm), a working Variational Quantum
Eigensolver, Kraus noise channels with exact and quantum-trajectory noisy
simulation, finite-shot measurement statistics, and RAM guards that stop a
simulation before it exhausts the machine — in ~4,700 lines of documented
code backed by 555 tests.
Only NumPy, SciPy and psutil are required.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/furkannar/qusim-np/actions/workflows/ci.yml/badge.svg)](https://github.com/furkannar/qusim-np/actions/workflows/ci.yml)
[![Test count](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/furkannar/qusim-np/master/badge.json)](https://github.com/furkannar/qusim-np/actions/workflows/ci.yml)

---

## Features

| Category | What you get |
|---|---|
| **Circuits** | `QuantumCircuit` builder (`qc.h(0).cx(0, 1)`) with lazy simulation, counts, exact and finite-shot expectation values, compose / inverse, custom unitaries — see the [Circuit tutorial](#circuit-tutorial) |
| **Noisy circuits** | Gate-level `NoiseModel`; exact density-matrix simulation, or quantum trajectories in O(2ⁿ) memory for 20+ noisy qubits |
| **Algorithms** | Grover search, Deutsch–Jozsa, Quantum Fourier Transform, Shor factoring |
| **VQE** | Pauli-string Hamiltonians, hardware-efficient ansatz, `scipy` optimizer |
| **Noise channels** | Amplitude damping, phase damping/flip, thermal relaxation, depolarizing, random Pauli — Kraus sets validated for completeness |
| **Shot noise** | Finite-shot estimators, CLT / Hoeffding / Bernstein shot bounds, optimal shot allocation, grouped (commuting) measurements, bitstring sampling |
| **Performance** | Matrix-free throughout: gates, Pauli strings, Grover, Deutsch–Jozsa, the QFT and Shor act directly on the state vector — see [Performance](#performance) |
| **Resource guards** | Free RAM read with `psutil`; runs that would leave < 128 MB are refused before allocating, and Shor sizes itself to the machine — see [Resource guards](#resource-guards) |

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

Requires Python ≥ 3.10, NumPy ≥ 2.0, SciPy ≥ 1.11, psutil ≥ 5.9.

## Quick start

```python
import numpy as np
import qusim_np
from qusim_np import QuantumCircuit

# A Bell state, measured 1000 times
bell = QuantumCircuit(2).h(0).cx(0, 1)
print(bell.sample_counts(1000, rng=np.random.default_rng(1)))   # {'00': 507, '11': 493}
print(bell.expectation([1.0], ["ZZ"]))                          # 1.0

# Ready-made algorithms
state, k, p = qusim_np.grover_search(3, marked_states=[5])      # find |101> among 8 states
print(f"{k} iterations -> P(marked) = {p:.3f}")                 # 2 iterations -> 0.945
print(qusim_np.deutsch_jozsa_classify(3, lambda x: x % 2))      # 'balanced'
print(qusim_np.shor(15, a=7)["factors"])                        # (3, 5)
```

## Circuit tutorial

`QuantumCircuit` records gates and simulates them only when you ask for a
result. Every snippet below is taken from the runnable scripts in
[`examples/`](examples/):

| Script | Covers |
|---|---|
| [`01_circuit_basics.py`](examples/01_circuit_basics.py) | building, inspecting, simulating, measuring, observables |
| [`02_composing_circuits.py`](examples/02_composing_circuits.py) | reusable builders, `inverse`, `compose`, custom gates, Grover and Shor as circuits |
| [`03_variational_circuits.py`](examples/03_variational_circuits.py) | a VQE written with circuits, evaluated with shots and noise |
| [`04_noise_and_trajectories.py`](examples/04_noise_and_trajectories.py) | `NoiseModel`, density matrices, quantum trajectories at 16 qubits |

### 1. Build

Gate methods add an instruction and return the circuit, so calls chain:

```python
import numpy as np
import qusim_np
from qusim_np import QuantumCircuit

rng = np.random.default_rng(1)          # seeded, so the outputs below are reproducible
bell = QuantumCircuit(2).h(0).cx(0, 1)
```

The snippets in steps 1–10 build on each other; run them in order.

| Kind | Methods |
|---|---|
| 1 qubit | `x(q)` `y(q)` `z(q)` `h(q)` `s(q)` `sdg(q)` `t(q)` `tdg(q)` `id(q)` |
| Rotations | `rx(theta, q)` `ry(theta, q)` `rz(theta, q)` |
| 2 qubits | `cx(control, target)` (alias `cnot`), `cz(q0, q1)`, `swap(q0, q1)` |
| 3 qubits | `ccx(c0, c1, target)` (alias `toffoli`) |
| Any | `unitary(matrix, qubits, name="unitary")` — any 2ᵏ×2ᵏ unitary on k qubits |

### 2. Inspect

Facts about the circuit are properties:

```python
print(bell)
# QuantumCircuit(2 qubit(s), 2 gate(s), depth 2)
#   h q[0]
#   cx q[0], q[1]

bell.num_qubits, bell.size, bell.depth        # (2, 2, 2)
bell.instructions[0].name                     # 'h'   (read-only tuple of Instruction)
```

### 3. Simulate

Results are methods, because they take arguments and do real work:

```python
bell.statevector()       # [0.707, 0, 0, 0.707]
bell.probabilities()     # [0.5, 0, 0, 0.5]
bell.density_matrix()    # 4x4 |psi><psi|
```

### 4. Measure

```python
bell.sample_counts(1000, rng=rng)             # {'00': 507, '11': 493}
QuantumCircuit(2).x(0).sample_counts(10)      # {'10': 10}
```

> **Bitstrings list qubit 0 first** (leftmost), matching the package's
> qubit-0 = most-significant-bit convention. This is the **reverse of
> Qiskit**, which prints qubit 0 rightmost.

### 5. Observables

`expectation(coefficients, pauli_strings)` evaluates H = Σᵢ cᵢ Pᵢ exactly,
or from finite shots when you pass `shots`:

```python
coeffs, paulis = [1.0, 0.5, 0.2], ["ZZ", "XX", "IZ"]
bell.expectation(coeffs, paulis)                          # 1.5   (exact)
bell.expectation(coeffs, paulis, shots=8000, rng=rng)     # (1.500, 0.007)   estimate, stderr

tilted = QuantumCircuit(1).ry(1.0, 0)
tilted.expectation([1.0], ["Z"], shots=4000, rng=rng)     # (0.556, 0.013)   exact cos(1) = 0.540
```

With shots, identity terms are added exactly and the other terms share
the shots (`allocate_shots`). Each term is measured by rotating X and Y
into the Z basis and reading the parity of its qubits.

### 6. Compose, invert, and add custom gates

```python
def ghz(n):
    qc = QuantumCircuit(n).h(0)
    for k in range(n - 1):
        qc.cx(k, k + 1)
    return qc

g = ghz(4)
g.sample_counts(1000)                          # {'0000': 473, '1111': 527}
g.compose(g.inverse()).statevector()           # back to |0000>

# Place a smaller circuit on chosen qubits
QuantumCircuit(4).compose(QuantumCircuit(2).h(0).cx(0, 1), qubits=[1, 3])
#  -> counts {'0000': 507, '0101': 493}

# Custom gate: sqrt(X), applied twice, is a NOT
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
QuantumCircuit(1).unitary(SX, 0, name="sx").unitary(SX, 0, name="sx").statevector()   # [0, 1]
```

`compose` and `inverse` return new circuits and leave the original
unchanged. `inverse()` keeps gate names meaningful (`s` ↔ `sdg`,
`rx(θ)` → `rx(−θ)`), so a noise model still matches them.

### 7. Write an algorithm as a circuit

Two-qubit Grover search for |11⟩. CZ is the oracle, and
H·X·CZ·X·H is the reflection about the mean:

```python
grover = QuantumCircuit(2).h(0).h(1)
grover.cz(0, 1)                                              # oracle
grover.h(0).h(1).x(0).x(1).cz(0, 1).x(0).x(1).h(0).h(1)      # diffusion
grover.sample_counts(100)                                    # {'11': 100}
```

Shor's algorithm for M = 15, a = 7 on 12 qubits: 8 counting qubits in
superposition, a 4-qubit work register starting at |1⟩, controlled
U^(2ʲ) as custom gates (U|w⟩ = |7w mod 15⟩), then the inverse QFT. See
[Shor's algorithm](#shors-algorithm) for the math:

```python
from qusim_np import inverse_qft_matrix
from qusim_np.shor import modular_exponentiation, modular_multiplication_permutation

M, a, t, n_w = 15, 7, 8, 4
W = 2 ** n_w
counting, work = list(range(t)), list(range(t, t + n_w))

shor_qc = QuantumCircuit(t + n_w).x(work[-1])        # work register = |1>
for q in counting:
    shor_qc.h(q)
for j in range(t):                                   # controlled-U^(2^j)
    perm = modular_multiplication_permutation(modular_exponentiation(a, 2 ** j, M), M, n_w)
    CU = np.eye(2 * W)
    CU[W:, W:] = 0
    CU[W + perm, W + np.arange(W)] = 1
    shor_qc.unitary(CU, [counting[t - 1 - j]] + work, name=f"c-U^{2 ** j}")
shor_qc.unitary(inverse_qft_matrix(t), counting, name="qft_dg")
# counting register y over 20 shots: {0: 5, 64: 6, 128: 7, 192: 2} -> r = 4 -> (3, 5)
```

### 8. Variational circuits (VQE)

`QuantumCircuit.from_hea(n, theta, layers)` builds the hardware-efficient
ansatz as a circuit, so the energy is a one-line function you can hand to
any optimizer:

```python
from scipy.optimize import minimize

coeffs, paulis = [0.3, 1.2, -0.8, -0.8], ["II", "ZZ", "XI", "IX"]

def energy(theta):
    return QuantumCircuit.from_hea(2, theta, layers=2).expectation(coeffs, paulis)

theta0 = np.random.default_rng(0).uniform(-np.pi, np.pi, qusim_np.hea_param_count(2, 2))
res = minimize(energy, theta0, method="Nelder-Mead")
res.fun                                                      # -1.700000 (exact: -1.700000)

best = QuantumCircuit.from_hea(2, res.x, layers=2)
best.expectation(coeffs, paulis, shots=10_000)               # (-1.667, 0.020)  what hardware would see
```

### 9. Noise

A `NoiseModel` is a list of rules of the form "after gate G (optionally
only on qubits Q), apply this channel". The channel is applied to every
qubit the gate touched:

```python
from qusim_np import NoiseModel, depolarizing_kraus, amplitude_damping_kraus, phase_damping_kraus

nm = (NoiseModel()
      .add_gate_error(depolarizing_kraus(0.02), gates="cx")        # both CNOT qubits
      .add_gate_error(amplitude_damping_kraus(0.01), gates="h")    # T1 decay after H
      .add_gate_error(phase_damping_kraus(0.05), qubits=0))        # dephasing on qubit 0

qc = QuantumCircuit(3, nm).h(0).cx(0, 1).cx(1, 2)    # or: qc.noise_model = nm
rho = qc.density_matrix()                            # purity Tr(rho^2) = 0.8419
qc.expectation([1.0], ["XXX"])                       # 0.8719  (ideal GHZ: 1.0)
```

Noisy circuits are simulated on a density matrix: each gate is applied as
ρ → UρU†, followed by its channels. `statevector()` raises an error on a
noisy circuit, because the output is a mixed state.

### 10. Large noisy circuits: quantum trajectories

A density matrix needs O(4ⁿ) memory and runs out around 13 qubits. With
`method="trajectory"`, the circuit instead keeps a **pure state**, and each
noise channel picks one Kraus branch at random:

$$p_k = \langle\phi_k|\phi_k\rangle, \qquad \phi_k = K_k|\psi\rangle,
\qquad |\psi\rangle \to \frac{\phi_k}{\sqrt{p_k}}$$

Averaged over trajectories this reproduces the density matrix exactly,
using O(2ⁿ) memory:

```python
qc.expectation([1.0], ["XXX"], method="trajectory", trajectories=2000)
# (0.8781, 0.0093)  — agrees with the exact 0.8719

big = QuantumCircuit(16, NoiseModel().add_gate_error(depolarizing_kraus(0.01), gates="cx")).h(0)
for k in range(15):
    big.cx(k, k + 1)
big.sample_counts(50, method="trajectory")
# 44/50 shots stay in the GHZ subspace; a density matrix here would need 69 GB
```

- **`sample_counts(..., method="trajectory")`** gives each shot its own
  trajectory by default. A shot drawn from a random trajectory *is* a
  sample of the noisy state, so the counts follow exactly the same
  distribution as the density method. Passing `trajectories=T` reuses T
  trajectories across the shots: faster, still unbiased, but with extra
  variance.
- **`expectation(..., method="trajectory")`** without `shots` averages
  exact pure-state values and returns `(estimate, stderr)`. With `shots`
  it draws each shot from a trajectory.
- **Building blocks:** `kraus_branch_probabilities` and
  `sample_kraus_branch` handle a single channel, and
  `QuantumCircuit.trajectories(count)` yields the trajectory states.

### Reference

| | |
|---|---|
| **Properties** | `num_qubits`, `size`, `depth`, `instructions` (read-only), `noise_model` (type-checked setter) |
| **Results** | `statevector()`, `density_matrix()`, `probabilities()`, `sample_counts(shots, method=...)`, `expectation(coeffs, paulis, shots=None, method=...)`, `trajectories(count)` |
| **Manipulation** | `copy()`, `compose(other, qubits=None)`, `inverse()`, `QuantumCircuit.from_hea(n, theta, layers)` |
| **Not supported** | mid-circuit measurement, classical bits, symbolic parameters |

## Algorithms

Grover search, Deutsch–Jozsa, the QFT and Shor's algorithm are also available as single
functions that run directly on state vectors:

```python
import numpy as np
import qusim_np

state, k, p = qusim_np.grover_search(10, marked_states=[3, 700])
qusim_np.deutsch_jozsa_classify(8, lambda x: 0)              # 'constant'
qusim_np.qft_state(np.eye(16)[3])                            # QFT|3>, computed by FFT
qusim_np.shor(55)["factors"]                                 # (5, 11)
```

### The math, briefly

Grover's algorithm rotates the state vector in the plane spanned by the
marked and unmarked subspaces. With $N = 2^n$ basis states and $M$ marked
states, after $k$ iterations

$$P_{\text{marked}}(k) = \sin^2\!\left(\frac{(2k+1)\,\theta}{2}\right), \qquad
\theta = 2\arcsin\sqrt{\frac{M}{N}}, \qquad
k_{\text{opt}} = \frac{\pi}{2\theta} - \frac{1}{2} \approx \frac{\pi}{4}\sqrt{\frac{N}{M}}$$

so the marked states are amplified after $O(\sqrt{N/M})$ oracle queries.
`grover_search` uses the exact $k_{\text{opt}}$ (rounded).

The Quantum Fourier Transform is the unitary

$$F_{jk} = \frac{1}{\sqrt{N}}\,\omega^{jk}, \qquad
\omega = e^{2\pi i / N}, \qquad j,k = 0,\dots,N-1$$

### Shor's algorithm

Shor's algorithm factors an odd composite $M$ by finding the **order**
$r$ of a base $a$: the smallest $r > 0$ with $a^r \equiv 1 \pmod M$.
Here $M$ is the number to factor, and $N = 2^n$ keeps its meaning from
above, the number of basis states.

1. **Classical checks.** Even $M$ and prime powers $M = b^k$ are split
   directly, and primes are rejected using a square-root divisor search.
   The base must satisfy $1 < a < M$ and $\gcd(a, M) = 1$.
2. **Registers.** $n_w = \lceil \log_2 M \rceil$ work qubits and
   $t = 2n_w$ counting qubits, so $Q = 2^t \ge M^2$ and $n = t + n_w$.
3. **Superposition and modular exponentiation.** Hadamards on the
   counting register, then controlled $U^{2^j}$ with
   $U|w\rangle = |a w \bmod M\rangle$, give

$$\frac{1}{\sqrt{Q}} \sum_{x=0}^{Q-1} |x\rangle\,|a^x \bmod M\rangle$$

4. **Inverse QFT and measurement.** The outcome $y$ satisfies
   $y/Q \approx s/r$ for a random $s$. Continued fractions give $r$.
5. **Factors.** If $r$ is even and $a^{r/2} \not\equiv -1 \pmod M$, then
   $\gcd(a^{r/2} \pm 1,\, M)$ are nontrivial factors.

```python
import qusim_np

qusim_np.shor(15, a=7)
# {'factors': (3, 5), 'a': 7, 'r': 4, 'method': 'quantum', 'attempts': 1}

P = qusim_np.counting_distribution(7, 15)    # P(y) over Q = 256 outcomes
P.nonzero()[0]                               # [0, 64, 128, 192]: y/Q = s/4
qusim_np.order_from_measurement(64, 256, 7, 15)   # 4
```

`method` reports how the factors were found: `"classical"` (even or prime
power), `"gcd"` (a random base already shared a factor with $M$) or
`"quantum"` (order finding). Pass `a=` to force the quantum path. $U$ is
stored as a permutation, and $U^{2^j}$ is multiplication by
`pow(a, 2**j, M)`, so a run costs $O(t \cdot 2^n)$ time and $O(2^n)$
memory: 32 bytes per amplitude at peak (`config.BYTES_PER_AMPLITUDE`).

How far you can go depends on your machine. By default the budget is
read from it: the RAM free right now, minus the 128 MB that is always
kept in reserve (see [Resource guards](#resource-guards)). Pass
`memory_gb=` to plan for another machine or to cap the run lower:

```python
qusim_np.estimate_capacity()         # detected: 13.3 GiB free on this laptop
# {'memory_gb': 13.29, 'detected': True, 'max_qubits': 28, 'usable_qubits': 27,
#  'work_qubits': 9, 'max_M': 512, 'peak_bytes': 4294967296}
qusim_np.estimate_capacity(64)       # a 64 GiB budget: 30 qubits, M <= 1024
qusim_np.memory_required(15)         # 131072 bytes (12 qubits)
qusim_np.shor(1021 * 3, memory_gb=8) # ValueError: ... that budget holds 27 qubits, i.e. M <= 512
```

The registers use $n = 3n_w$ qubits, so only multiples of 3 are usable,
and each extra work qubit costs 8× the memory.

## Variational Quantum Eigensolver

Find the ground-state energy of a Hamiltonian written as a linear
combination of Pauli strings,

$$H = \sum_{i=0}^{m-1} \alpha_i P_i, \qquad
P_i = \bigotimes_{k=0}^{n-1} \sigma_{i,k}, \qquad
\sigma_{i,k} \in \{I, X, Y, Z\}$$

by minimizing the variational energy

$$E(\theta) = \langle \psi(\theta) \,|\, H \,|\, \psi(\theta) \rangle$$

over the parameters of a hardware-efficient ansatz (HEA). `vqe_optimize`
is the one-call driver; [tutorial step 8](#8-variational-circuits-vqe)
shows the same thing built from circuits.

```python
import numpy as np
import qusim_np

coeffs  = [0.3, 1.2, -0.8, -0.8]          # H = 0.3·II + 1.2·ZZ − 0.8·XI − 0.8·IX
strings = ["II", "ZZ", "XI", "IX"]

res = qusim_np.vqe_optimize(2, layers=2, coefficients=coeffs, pauli_strings=strings)
H = qusim_np.hamiltonian(coeffs, strings)
print(f"VQE {res.fun:.6f}   exact {np.linalg.eigvalsh(H)[0]:.6f}")   # VQE -1.700000   exact -1.700000
```

### Pauli strings

```python
import qusim_np
from qusim_np import QuantumCircuit

# "XZI", ("X","Z","I") and index tuples (1, 3, 0) are equivalent
P = qusim_np.pauli_string_matrix("XZI")                    # 8x8 matrix, 3 qubits

# Expectation values without building any 2^n x 2^n matrix (O(2^n))
psi = QuantumCircuit(2).h(0).cx(0, 1).statevector()
qusim_np.pauli_string_expval(psi, "ZZ")                    # 1.0

# Low-level: apply any k-qubit gate straight to a state vector
psi = qusim_np.apply_gate(psi, qusim_np.gates.hadamard().matrix, [1])
```

## Noise channels

The channels behind `NoiseModel` can also be used on their own. They
are built on the general Kraus form

$$\mathcal{E}(\rho) = \sum_k K_k \,\rho\, K_k^\dagger, \qquad
\sum_k K_k^\dagger K_k = I$$

Each channel accepts a state vector or density matrix and returns a
density matrix. Every `*_kraus` constructor can be passed to
`NoiseModel.add_gate_error`.

```python
import numpy as np
import qusim_np

# |1> subject to amplitude damping with gamma = 0.3
rho = qusim_np.amplitude_damping(np.array([0, 1]), 0.3)
print(rho.real)                      # [[0.3, 0], [0, 0.7]]: excited population decayed by gamma

rho = qusim_np.depolarizing(rho, 0.1, qubit=0)                     # (1-p) rho + (p/2) I
rho = qusim_np.random_pauli(rho, (0.9, 0.05, 0.03, 0.02))          # probs (q_I, q_X, q_Y, q_Z)
rho = qusim_np.thermal_relaxation(rho, t=1e-6, T1=50e-6, T2=30e-6) # needs T2 <= 2*T1
```

## Shot noise

Real hardware returns samples, not amplitudes. `qusim_np.shot_noise`
turns exact states into finite-shot data and tells you how many shots an
estimate needs. For a single-shot variance $\sigma^2$, the mean of $N$
shots has standard error $\sigma/\sqrt{N}$. For a Hamiltonian
$H = \sum_i c_i P_i$ measured term by term, the optimal allocation is

$$N_i \propto |c_i|\,\sigma_i, \qquad
\mathrm{Var}_{\min}(\hat H) = \frac{\left(\sum_i |c_i|\,\sigma_i\right)^2}{N}$$

```python
import numpy as np
import qusim_np
from qusim_np import QuantumCircuit

# cos(0.4)|00> + sin(0.4)|11>, prepared by a circuit
psi = QuantumCircuit(2).ry(0.8, 0).cx(0, 1).statevector()

# Measure XX on 4096 shots and estimate <XX> with its standard error
rng = np.random.default_rng(7)
XX = qusim_np.pauli_string_matrix("XX")
outcomes = qusim_np.sample_observable_outcomes(psi, XX, shots=4096, rng=rng)
mean, stderr = qusim_np.estimate_expval(outcomes)          # 0.721 +/- 0.011  (exact 0.717)

# Shots for a +/-0.01 half-width at 95% confidence on a Pauli term
qusim_np.shots_clt(1.0, 0.01, delta=0.05)                  # 38415  (asymptotic)
qusim_np.shots_hoeffding(0.01, 0.05)                       # 73778  (rigorous)

# Split 10,000 shots across H = 0.3 II + 1.2 ZZ - 0.8 XI - 0.8 IX
# (sigma = 0 for II: the identity is deterministic and needs no shots)
qusim_np.allocate_shots([0.3, 1.2, -0.8, -0.8], [0, 1, 1, 1], 10_000)   # [0, 4286, 2857, 2857]
```

Grouped measurements of commuting terms (`group_variance`,
`allocate_group_shots`) account for the covariances between terms
measured on the same shots. For circuits, `QuantumCircuit.expectation(...,
shots=N)` does the sampling and allocation for you.

## Performance

Wherever a large operator has known structure, a mathematical identity
replaces the dense 2ⁿ×2ⁿ matrix. Notation: N = 2ⁿ, ψ is the state vector,
s is the uniform superposition, J is the all-ones matrix, and
popcount(·) counts set bits.

**Algorithms** (`grover_search.py`, `deutsch_jozsa.py`, `qft.py`)

| Optimization | Identity used | Cost before → after |
|---|---|---|
| Uniform superposition built directly | every entry of H⊗ⁿ applied to the zero state is 1/√N | O(4ⁿ) → O(2ⁿ) |
| Grover oracle as a sign flip | the oracle is diagonal with entries ±1 | O(4ⁿ) → O(M) per iteration |
| Grover diffusion as inversion about the mean | D·ψ = 2·mean(ψ) − ψ, because the projector onto s is J/N | O(4ⁿ) → O(2ⁿ) per iteration |
| `grover_iteration` matrix | D·O = D with the marked columns negated | O(8ⁿ) → O(4ⁿ) |
| `diffusion_operator` matrix | D = (2/N)·J − I | no Hadamard matrix needed |
| Exact optimal iteration count | P(k) = sin²((2k+1)θ/2) peaks at k = π/(2θ) − ½ | never worse than the small-angle formula; better in 4,154 cases (n ≤ 14) |
| `apply_hadamard_transform` (new) | fast Walsh–Hadamard butterfly (a, b) → (a + b, a − b) | O(4ⁿ) → O(n·2ⁿ) |
| `hadamard_matrix_closed_form` vectorized | H[j, k] = (−1)^popcount(j AND k) / √N | Python double loop → NumPy |
| Deutsch–Jozsa | oracle as an elementwise phase, then fast Walsh–Hadamard; amplitude of the zero state = mean of (−1)^f(x) | O(4ⁿ) → O(n·2ⁿ) |
| `qft_state` / `inverse_qft_state` | QFT = `np.fft.ifft(x, norm="ortho")`, inverse = `np.fft.fft` | O(4ⁿ) → O(n·2ⁿ) |
| `qft_matrix` | ω^(jk) = ω^(jk mod N), reduced before `exp` | better precision for large jk |
| Shor's modular multiplication | U is a permutation when gcd(a, M) = 1; U^(2ʲ) = multiplication by `pow(a, 2**j, M)` | dense matrix O(4^n_w) → index array O(2^n_w); 2ʲ applications → 1 |

**Pauli strings and VQE** (`vqe.py`)

| Optimization | Identity used | Cost before → after |
|---|---|---|
| Pauli strings as signed permutations | Y = iXZ, so P maps basis state c to i^(n_Y) · (−1)^popcount(c AND z) times basis state (c XOR x) | apply / expectation: O(n·2ⁿ) → O(2ⁿ) |
| `hamiltonian()` writes each term straight into H | each Pauli string has one nonzero per column | memory O(m·4ⁿ) → O(4ⁿ) |
| `apply_gate` for any k-qubit gate (new) | move the k target axes to the front, one (2ᵏ × 2ᵏ) matrix product | no 2ⁿ×2ⁿ matrix for 3+ qubit gates |
| HEA reuses the CNOT matrix | — | fewer allocations per energy call |

**Noise** (`noise_models/channels.py`)

| Optimization | Identity used | Effect |
|---|---|---|
| Kraus completeness checked once | Σ K†K = I ⇔ trace-preserving on every input | replaces a per-state trace check that could accept invalid channels |
| `apply_unitary_density` (new) | U on the row axes, conj(U) on the column axes of ρ | gates on density matrices without 2ⁿ×2ⁿ operators |
| Trajectory branch probabilities | ⟨φₖ, φₖ⟩ = Tr(Kₖ†Kₖ ρ_q), with ρ_q the qubit's 2×2 reduced state | one pass over ψ instead of one per Kraus operator |
| Identity branch skipped | if Kₖ = c·I, then φₖ/√pₖ is ψ times a phase | the common "no error" branch costs nothing |
| No re-validation inside trajectories | channels are validated once, when added to the `NoiseModel` | ~3× faster on small circuits |

**Shot noise** (`shot_noise/`)

| Change | Why | Effect |
|---|---|---|
| SciPy normal quantile imported lazily | `scipy.stats` costs ~0.5 s at import | `import qusim_np`: ~655 ms → ~86 ms |
| `hamiltonian_variance` accepts 0 shots on terms with cᵢσᵢ = 0 | such terms contribute no variance | output of `allocate_shots` can be passed straight in |
| `shots_needed_pauli_std_error` returns at least 1 | at ⟨P⟩ = ±1 one shot fixes the outcome | no more "0 shots needed" |

**Measured on a laptop CPU**

| Task | Before | After |
|---|---|---|
| `grover_search`, n = 11 | 0.46 s | < 1 ms |
| `deutsch_jozsa`, n = 11 | 0.15 s | < 1 ms |
| `qft_state`, n = 12 | 0.68 s | 2 ms |
| `hamiltonian`, n = 10, 50 terms | 0.93 s | 7 ms |
| `vqe_optimize`, n = 4, 2 layers | 1.30 s | 0.88 s |
| `grover_search`, n = 20 | out of memory | 4.1 s |
| `deutsch_jozsa`, n = 22 | out of memory | 1.7 s |
| Noisy 20-qubit GHZ, one trajectory | impossible (16 TB density matrix) | 0.8 s, 16 MB |

Every fast path is cross-checked against a dense reference in
`tests/test_algorithms/test_fast_paths.py` and `tests/test_circuit/test_circuit.py`.

## Resource guards

Simulations allocate 2ⁿ (state vectors) or 4ⁿ (density matrices, dense
operators) entries, so `qusim_np/resources.py` watches the machine:

- **RAM floor.** Before a large allocation, each entry point (circuits,
  Grover, Deutsch–Jozsa, QFT, Hamiltonians, the HEA ansatz, density
  matrices, Shor) estimates its size and raises `RuntimeError` if it
  would leave less than **128 MB** free. It does not wait for the system
  to start swapping.
- **Re-checks during long runs.** Loops over large states (2²⁰
  amplitudes and up) read the RAM again every 0.5 s, in case other
  programs fill it mid-run. Between checks a step costs one clock read
  (~40 ns). The loops never pause or sleep, so large runs go at full
  speed.

```python
qusim_np.available_memory_gb()       # 13.4  (psutil: RAM free right now)
qusim_np.check_memory(2 ** 30)       # raises if 1 GiB more would leave < 128 MB
# RuntimeError: Dangerously low RAM: this step needs about 1,024 MB, which would
# leave 90 MB (minimum free: 128 MB; ...). Close other programs or reduce the
# number of qubits.
```

All thresholds live in `config` and can be changed at runtime:
`MIN_FREE_RAM_MB`, `MONITOR_MIN_AMPLITUDES`, `MONITOR_INTERVAL_S`,
`MEMORY_CHECK_CACHE_S`.
Measured overhead: none (Grover n = 19 and the circuit/VQE test suites
take the same time with the guards on).

## Conventions

### Qubit order

Qubit 0 is the **most significant bit** of the basis-state index:

$$|q_0 q_1 \dots q_{n-1}\rangle \;\longleftrightarrow\;
\sum_{k=0}^{n-1} q_k\, 2^{\,n-1-k}$$

A Pauli string `"XZI"` means X on qubit 0, Z on qubit 1 and I on qubit 2,
and a measured bitstring `"10"` means qubit 0 = 1, qubit 1 = 0. This is
the reverse of Qiskit, which prints qubit 0 rightmost.

### Notation

| Symbol | Meaning |
|---|---|
| $n$ | number of qubits |
| $N = 2^n$ | number of basis states (the length of a state vector), in every module |
| $M$ | Grover: number of marked states. Shor: the integer being factored |
| $t,\ n_w$ | Shor: counting and work qubits, $t = 2n_w$, so $n = 3n_w$ |
| $Q = 2^t$ | Shor: number of values of the counting register |

Shor's algorithm is usually written with $N$ for the number to factor.
Here $N$ always means $2^n$, so the number to factor is $M$.

### Memory units

Sizes use the **binary** convention, the same one psutil and the OS use
to report RAM. The constants live in `qusim_np.config`:

```python
from qusim_np.config import KB, MB, GB     # 2**10, 2**20, 2**30 bytes

config.MIN_FREE_RAM_MB * MB                # the RAM floor, in bytes
qusim_np.estimate_capacity(8)              # memory_gb=8 means 8 * GB bytes
```

A complex128 amplitude takes 16 bytes, so a state vector of $n$ qubits
takes $16 \cdot 2^n$ bytes and a density matrix $16 \cdot 4^n$.

### Configuration

- **`qusim_np.config`** (an instance of `QuantumConfig`) holds settings
  you may want to change, such as `MIN_FREE_RAM_MB`,
  `BYTES_PER_AMPLITUDE` and `SU2_TOLERANCE`. They are read every time a function runs, so
  `config.MIN_FREE_RAM_MB = 256` takes effect immediately.
- **Module-level constants** in `qusim_np/config.py` are fixed facts
  that are not meant to change: the Pauli matrices and `KB` / `MB` / `GB`.

### Errors and randomness

- **`ValueError`**: invalid input, such as a wrong shape, a non-unitary
  matrix, a non-coprime Shor base, or a run larger than the `memory_gb`
  you gave.
- **`RuntimeError`**: the machine itself is the problem. RAM is, or
  would become, dangerously low.
- **`rng=`**: every sampling function takes an optional
  `numpy.random.Generator`. Pass `np.random.default_rng(seed)` for
  reproducible results; `None` creates a fresh generator.

### Tests and examples

- **`tests/` mirrors the package:** algorithm tests go in
  `tests/test_algorithms/`, noise channels in `tests/test_noise_models/`,
  and so on. Fast paths are cross-checked against dense references.
- **`examples/` is a tutorial**, grouped by topic, not one file per
  algorithm. New algorithms appear as circuit sections in an existing
  script, as Grover and Shor do in `02_composing_circuits.py`. Every
  script runs as a test.

## Project layout

```
qusim_np/
├── circuit.py               # QuantumCircuit builder + Instruction
├── config.py                # constants, Pauli matrices, resource thresholds
├── gates.py                 # QuantumGate + gate constructors
├── grover_search.py         # Hadamard transforms, oracle, diffusion, Grover
├── deutsch_jozsa.py         # Deutsch-Jozsa algorithm
├── qft.py                   # QFT / inverse QFT (FFT-based on states)
├── shor.py                  # Shor's factoring: order finding, continued fractions
├── vqe.py                   # Pauli strings, apply_gate, Hamiltonians, HEA ansatz, VQE driver
├── noise_models/            # Kraus channels, trajectory sampling, NoiseModel
├── shot_noise/              # finite-shot estimation, shot bounds, allocation, sampling
└── resources.py             # RAM floor and in-loop RAM re-checks (psutil)
examples/                    # runnable circuit tutorial scripts (01-04)
tests/                       # 555 tests, grouped like the package:
├── test_algorithms/         # Grover, Deutsch-Jozsa, QFT, Shor, VQE, fast-path cross-checks
├── test_circuit/            # QuantumCircuit builder
├── test_core/               # gates, resource guards
├── test_noise_models/       # Kraus channels
├── test_shot_noise/         # finite-shot estimation and sampling
└── test_examples/           # runs every script in examples/
```

## Limitations

Scope notes for this lightweight quantum simulation package:

- **State-vector memory**: the only O(2ⁿ) cost is the state vector itself
  (16 bytes per amplitude: ~0.5 GB at 25 qubits, ~4 GB at 28)
- **Exact noisy simulation is O(4ⁿ)**: density matrices are practical up
  to ~12–13 qubits. Beyond that, use `method="trajectory"`, which needs
  O(2ⁿ) memory but returns statistical estimates. It takes ~0.8 s per
  trajectory at 20 qubits and ~16 s at 24
- **Explicit matrix builders stay dense by definition**: `hamiltonian`,
  `pauli_string_matrix`, `qft_matrix`, `grover_oracle`,
  `diffusion_operator` and `hadamard_transform` return full 2ⁿ×2ⁿ
  matrices — use the state-vector functions for large n
- **Noise is local and gate-triggered**: `NoiseModel` applies
  single-qubit channels after gates. Correlated multi-qubit channels and
  readout errors are not modelled. The standalone algorithm functions
  (`grover_search`, `deutsch_jozsa`, …) run noiselessly; build them as a
  `QuantumCircuit` to add noise
- **End-of-circuit measurement only**: no mid-circuit measurement,
  classical bits or conditional gates, and gate angles are plain numbers
  (no symbolic parameters)
- **Classical only**: no real-device or cloud backend integration

For production quantum computing, see [Qiskit](https://www.ibm.com/quantum/qiskit),
[PennyLane](https://pennylane.ai/), or [Cirq](https://quantumai.google/cirq).

## Development

```bash
pip install -e .[dev]
pytest                                   # 555 tests, ~25 s
pytest tests/test_algorithms             # or just one group
python examples/01_circuit_basics.py     # or any other tutorial script
```

Contributions welcome — see the test suite for conventions.

## AI assistance

Parts of this codebase were written with assistance from **SWE-2 by
Cognition**, accessed through the **Devin** platform, for coding tasks
such as implementing algorithms, tests, and documentation.

Versions 1.2 to 1.4 were developed with **Claude Opus 5.5 by
Anthropic**, used through **Claude Code**:

- **1.2:** codebase analysis, the matrix-free rewrites of Grover,
  Deutsch–Jozsa, the QFT and the Pauli-string routines, Kraus
  completeness validation, shot-noise fixes, the dense-reference
  cross-check tests, and benchmarking
- **1.3:** the `QuantumCircuit` API and `NoiseModel`, k-qubit
  `apply_gate` / `apply_unitary_density`, the quantum-trajectory
  simulator (built on the author's idea of evolving state vectors with
  inner-product branch probabilities instead of density matrices), the
  tutorial examples, and this documentation
- **1.4:** Shor's factoring algorithm, built to the
  author's specification: the square-root divisor search, the
  `gcd(a, M) = 1` base check, Python's `pow` for modular
  exponentiation, the Hadamard gate from `gates.py` for the
  superposition, and a cheap permutation unitary. Also the
  memory-budget estimate (peak bytes per amplitude measured with
  `tracemalloc`), the `psutil` RAM guards across all modules (the
  author's idea), the test-suite reorganization, and Python 3.13 / 3.14
  support. A CPU-throttling pause was prototyped and then removed at
  the author's request, so large runs keep full speed

All architectural and design decisions were made by the author.

## License

[MIT](LICENSE) © 2026 Furkan Nar
