r"""
qusim_np/circuit.py
Circuit builder

QuantumCircuit records a list of gate instructions and simulates them only
when a result is requested (lazy evaluation):

    qc = QuantumCircuit(2).h(0).cx(0, 1)      # build (methods chain)
    qc.statevector()                          # simulate: (|00> + |11>)/sqrt(2)
    qc.sample_counts(1000)                    # {"00": ~500, "11": ~500}
    qc.expectation([1.0], ["ZZ"], shots=4096) # finite-shot estimate

Simulation reuses the matrix-free engine: gates act on the state vector via
vqe.apply_gate (no 2^n x 2^n matrices).  With a NoiseModel attached, the
circuit is simulated on a density matrix: each gate is applied as
rho -> U rho U^dagger, followed by the model's channels.

Conventions:
- Qubit 0 is the most significant bit of the basis index, as everywhere
  in qusim_np.  Bitstrings returned by sample_counts therefore list qubit 0
  FIRST (leftmost): QuantumCircuit(2).x(0) measures "10".
  NOTE: this is the reverse of Qiskit, which prints qubit 0 rightmost.
- Multi-qubit gate arguments are ordered as in the gate matrix: for
  cx(c, t) the first argument is the control.
- Measurement happens only at the end of the circuit (no mid-circuit
  measurement or classical bits).
"""

from dataclasses import dataclass, field

import numpy as np

from qusim_np import gates as _gates
from qusim_np.grover_search import _validate_n
from qusim_np.noise_models.channels import (
    _sample_branch, apply_channel, apply_unitary_density,
)
from qusim_np.noise_models.noise_model import NoiseModel
from qusim_np.shot_noise.hamiltonian import allocate_shots
from qusim_np.shot_noise.sampling import measurement_probabilities, sample_bitstrings
from qusim_np.vqe import (
    _normalize_pauli_string, _pauli_column_values, _validate_hea_args,
    _validate_qubits, apply_gate, hea_param_count, pauli_string_expval,
)

_UNITARY_TOL = 1e-8

# Fixed matrices, built once.
_SDG = np.diag([1, -1j]).astype(complex)
_TDG = np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex)
_H = _gates.hadamard().matrix

# Names whose inverse is the same gate; and pairs that invert each other.
_SELF_INVERSE = {"id", "x", "y", "z", "h", "cx", "cz", "swap", "ccx"}
_INVERSE_PAIRS = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
_ROTATIONS = {"rx", "ry", "rz"}


@dataclass(frozen=True, eq=False)
class Instruction:
    """One recorded gate: its name, target qubits, parameters, and matrix."""

    name: str
    qubits: tuple
    params: tuple = ()
    matrix: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        m = np.array(self.matrix, dtype=complex)    # private copy
        m.setflags(write=False)
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        object.__setattr__(self, "params", tuple(self.params))

    def __str__(self):
        p = f"({', '.join(f'{v:.6g}' for v in self.params)})" if self.params else ""
        qs = ", ".join(f"q[{q}]" for q in self.qubits)
        return f"{self.name}{p} {qs}"


_METHODS = ("density", "trajectory")


def _check_method(method):
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    return method


def _check_positive_int(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _check_angle(theta):
    theta = float(theta)
    if not np.isfinite(theta):
        raise ValueError(f"rotation angle must be finite, got {theta!r}")
    return theta


class QuantumCircuit:
    r"""
    A gate-sequence builder over n qubits with lazy simulation.

    Gate methods (h, x, cx, rx, ...) record an Instruction and return self,
    so they can be chained.  Results are computed on demand by
    statevector(), density_matrix(), probabilities(), sample_counts() and
    expectation().

    Args:
        num_qubits: int >= 1
        noise_model: NoiseModel or None
    """

    def __init__(self, num_qubits, noise_model=None):
        _validate_n(num_qubits)
        self._n = int(num_qubits)
        self._instructions = []
        self.noise_model = noise_model      # goes through the validating setter

    # ------------------------------------------------------------------
    # Properties: facts about the circuit (cheap, deterministic)
    # ------------------------------------------------------------------
    @property
    def num_qubits(self):
        """Number of qubits."""
        return self._n

    @property
    def instructions(self):
        """Recorded instructions, as a read-only tuple."""
        return tuple(self._instructions)

    @property
    def size(self):
        """Total number of gates."""
        return len(self._instructions)

    @property
    def depth(self):
        """Circuit depth: length of the longest chain of gates sharing qubits."""
        level = [0] * self._n
        for inst in self._instructions:
            d = max(level[q] for q in inst.qubits) + 1
            for q in inst.qubits:
                level[q] = d
        return max(level, default=0)

    @property
    def noise_model(self):
        """The attached NoiseModel, or None for noiseless simulation."""
        return self._noise_model

    @noise_model.setter
    def noise_model(self, model):
        if model is not None and not isinstance(model, NoiseModel):
            raise TypeError(
                f"noise_model must be a NoiseModel or None, got {type(model).__name__}"
            )
        self._noise_model = model

    def _is_noisy(self):
        return self._noise_model is not None and len(self._noise_model) > 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _append(self, name, matrix, qubits, params=()):
        qubits = _validate_qubits(qubits, self._n)
        self._instructions.append(Instruction(name, tuple(qubits), params, matrix))
        return self

    # Single-qubit gates
    def id(self, q):
        """Identity (a no-op that still counts toward size, depth and noise)."""
        return self._append("id", np.eye(2), q)

    def x(self, q):
        """Pauli-X (NOT)."""
        return self._append("x", _gates.pauli_x().matrix, q)

    def y(self, q):
        """Pauli-Y."""
        return self._append("y", _gates.pauli_y().matrix, q)

    def z(self, q):
        """Pauli-Z."""
        return self._append("z", _gates.pauli_z().matrix, q)

    def h(self, q):
        """Hadamard."""
        return self._append("h", _H, q)

    def s(self, q):
        """Phase gate S = diag(1, i)."""
        return self._append("s", _gates.phase_gate().matrix, q)

    def sdg(self, q):
        """S^dagger = diag(1, -i)."""
        return self._append("sdg", _SDG, q)

    def t(self, q):
        """T gate = diag(1, e^{i pi/4})."""
        return self._append("t", _gates.t_gate().matrix, q)

    def tdg(self, q):
        """T^dagger = diag(1, e^{-i pi/4})."""
        return self._append("tdg", _TDG, q)

    def rx(self, theta, q):
        """Rotation about X by theta."""
        theta = _check_angle(theta)
        return self._append("rx", _gates.rx(theta).matrix, q, (theta,))

    def ry(self, theta, q):
        """Rotation about Y by theta."""
        theta = _check_angle(theta)
        return self._append("ry", _gates.ry(theta).matrix, q, (theta,))

    def rz(self, theta, q):
        """Rotation about Z by theta."""
        theta = _check_angle(theta)
        return self._append("rz", _gates.rz(theta).matrix, q, (theta,))

    # Two-qubit gates
    def cx(self, control, target):
        """Controlled-NOT."""
        return self._append("cx", _gates.cnot().matrix, (control, target))

    cnot = cx

    def cz(self, q0, q1):
        """Controlled-Z (symmetric in its qubits)."""
        return self._append("cz", _gates.controlled_phase().matrix, (q0, q1))

    def swap(self, q0, q1):
        """SWAP."""
        return self._append("swap", _gates.swap().matrix, (q0, q1))

    # Three-qubit gates
    def ccx(self, control0, control1, target):
        """Toffoli (CCNOT)."""
        return self._append("ccx", _gates.toffoli().matrix, (control0, control1, target))

    toffoli = ccx

    # Arbitrary unitary
    def unitary(self, matrix, qubits, name="unitary"):
        r"""
        Append an arbitrary k-qubit unitary acting on `qubits`.

        qubits[0] is the most significant bit of the matrix index.

        Raises:
            ValueError: if the shape is not 2^k x 2^k or U^dagger U != I
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty string, got {name!r}")
        qubits = _validate_qubits(qubits, self._n)
        k = len(qubits)
        U = np.asarray(matrix, dtype=complex)
        if U.shape != (2 ** k, 2 ** k):
            raise ValueError(
                f"unitary on {k} qubit(s) must have shape ({2 ** k}, {2 ** k}), got {U.shape}"
            )
        if not np.allclose(U.conj().T @ U, np.eye(2 ** k), atol=_UNITARY_TOL):
            raise ValueError("matrix is not unitary: U^dagger U != I")
        return self._append(name, U, qubits)

    # ------------------------------------------------------------------
    # Circuit manipulation
    # ------------------------------------------------------------------
    def copy(self):
        """Shallow copy (instructions are immutable; the noise model is shared)."""
        qc = QuantumCircuit(self._n, self._noise_model)
        qc._instructions = list(self._instructions)
        return qc

    def compose(self, other, qubits=None):
        r"""
        Return a new circuit: this circuit followed by `other`.

        Args:
            other: QuantumCircuit with at most num_qubits qubits
            qubits: None (other's qubit i -> qubit i), or a sequence mapping
                other's qubit i -> qubits[i]

        Returns:
            QuantumCircuit: a new circuit; self is unchanged
        """
        if not isinstance(other, QuantumCircuit):
            raise TypeError(f"can only compose a QuantumCircuit, got {type(other).__name__}")
        if qubits is None:
            if other.num_qubits > self._n:
                raise ValueError(
                    f"cannot compose a {other.num_qubits}-qubit circuit onto {self._n} qubits"
                )
            mapping = list(range(other.num_qubits))
        else:
            mapping = _validate_qubits(qubits, self._n)
            if len(mapping) != other.num_qubits:
                raise ValueError(
                    f"qubits must map all {other.num_qubits} qubits of other, got {len(mapping)}"
                )
        qc = self.copy()
        for inst in other.instructions:
            qc._instructions.append(Instruction(
                inst.name, tuple(mapping[q] for q in inst.qubits), inst.params, inst.matrix,
            ))
        return qc

    def inverse(self):
        r"""
        Return the inverse circuit U^dagger: instructions reversed, each
        matrix conjugate-transposed.  Gate names are kept meaningful
        (s <-> sdg, rx(theta) -> rx(-theta), self-inverse gates unchanged)
        so a noise model still matches them.
        """
        qc = QuantumCircuit(self._n, self._noise_model)
        for inst in reversed(self._instructions):
            name, params = inst.name, inst.params
            if name in _INVERSE_PAIRS:
                name = _INVERSE_PAIRS[name]
            elif name in _ROTATIONS:
                params = tuple(-p for p in params)
            elif name not in _SELF_INVERSE:
                name = name[:-3] if name.endswith("_dg") else name + "_dg"
            qc._instructions.append(Instruction(name, inst.qubits, params, inst.matrix.conj().T))
        return qc

    @classmethod
    def from_hea(cls, n, theta, layers=1, entanglement="linear"):
        r"""
        Build the hardware-efficient ansatz of vqe.hea_state as a circuit.

        Each layer: RY then RZ on every qubit, then CNOTs i -> i+1
        ("linear") or i -> (i+1) mod n ("ring").  Parameter layout matches
        vqe.hea_state, so QuantumCircuit.from_hea(n, theta, L).statevector()
        equals vqe.hea_state(n, theta, L).
        """
        _validate_hea_args(n, layers, entanglement)
        theta = np.asarray(theta, dtype=float).ravel()
        expected = hea_param_count(n, layers)
        if theta.size != expected:
            raise ValueError(f"theta must have {expected} parameters, got {theta.size}")
        theta = theta.reshape(layers, n, 2)
        qc = cls(n)
        if entanglement == "ring":
            pairs = [(i, (i + 1) % n) for i in range(n)]
        else:
            pairs = [(i, i + 1) for i in range(n - 1)]
        for layer in range(layers):
            for q in range(n):
                qc.ry(theta[layer, q, 0], q).rz(theta[layer, q, 1], q)
            for c, t in pairs:
                if c != t:
                    qc.cx(c, t)
        return qc

    def __len__(self):
        return len(self._instructions)

    def __repr__(self):
        noise = ", noisy" if self._is_noisy() else ""
        head = (f"QuantumCircuit({self._n} qubit(s), {self.size} gate(s), "
                f"depth {self.depth}{noise})")
        return "\n".join([head] + [f"  {inst}" for inst in self._instructions])

    # ------------------------------------------------------------------
    # Simulation: methods (they take arguments and/or do real work)
    # ------------------------------------------------------------------
    def _initial_vector(self, initial_state):
        N = 2 ** self._n
        if initial_state is None:
            psi = np.zeros(N, dtype=complex)
            psi[0] = 1.0
            return psi
        psi = np.array(initial_state, dtype=complex)
        if psi.shape != (N,):
            raise ValueError(f"initial_state must have shape ({N},), got {psi.shape}")
        return psi

    def statevector(self, initial_state=None):
        r"""
        Simulate the circuit and return the final state vector.

        Args:
            initial_state: state vector (2^n,) or None for |0...0>

        Raises:
            ValueError: if a non-empty noise model is attached (a noisy
                circuit has no pure state — use density_matrix())
        """
        if self._is_noisy():
            raise ValueError(
                "circuit has a noise model; the output is a mixed state — "
                "use density_matrix() instead of statevector()"
            )
        psi = self._initial_vector(initial_state)
        for inst in self._instructions:
            psi = apply_gate(psi, inst.matrix, inst.qubits)
        return psi

    def density_matrix(self, initial_state=None):
        r"""
        Simulate the circuit on a density matrix, applying the noise model.

        Each gate is applied as rho -> U rho U^dagger, then every matching
        noise-model channel is applied to each of the gate's qubits.

        Args:
            initial_state: state vector (2^n,), density matrix (2^n, 2^n),
                or None for |0...0><0...0|

        Returns:
            numpy.ndarray: final density matrix, shape (2^n, 2^n)
        """
        N = 2 ** self._n
        if initial_state is not None and np.ndim(initial_state) == 2:
            rho = np.array(initial_state, dtype=complex)
            if rho.shape != (N, N):
                raise ValueError(f"initial density matrix must have shape ({N}, {N}), got {rho.shape}")
        else:
            psi = self._initial_vector(initial_state)
            if not self._is_noisy():
                psi = self.statevector(psi)
                return np.outer(psi, psi.conj())
            rho = np.outer(psi, psi.conj())

        for inst in self._instructions:
            rho = apply_unitary_density(rho, inst.matrix, inst.qubits)
            if self._noise_model is not None:
                for ops, q in self._noise_model.channels_for(inst.name, inst.qubits):
                    rho = apply_channel(rho, ops, qubit=q)
        return rho

    def _trajectory(self, rng):
        r"""
        Simulate ONE quantum trajectory: a pure state in which every noise
        channel picks a single Kraus branch (see sample_kraus_branch).
        Averaging |psi><psi| over trajectories gives density_matrix().
        Memory O(2^n).  Kraus sets were validated when added to the
        NoiseModel, so the unchecked branch sampler is used here.
        """
        psi = self._initial_vector(None)
        for inst in self._instructions:
            psi = apply_gate(psi, inst.matrix, inst.qubits)
            if self._noise_model is not None:
                for ops, q in self._noise_model.channels_for(inst.name, inst.qubits):
                    psi, _ = _sample_branch(psi, self._n, ops, q, rng)
        return psi

    def trajectories(self, count, rng=None):
        r"""
        Generate `count` independent trajectory state vectors.

        Each is a pure state of shape (2^n,); their average outer product
        converges to density_matrix() at rate 1/sqrt(count).  For a
        noiseless circuit every trajectory equals statevector().

        Yields:
            numpy.ndarray: one trajectory state per iteration
        """
        count = _check_positive_int(count, "count")
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(count):
            yield self._trajectory(rng)

    def _final_state(self):
        """State vector if noiseless, else density matrix."""
        return self.density_matrix() if self._is_noisy() else self.statevector()

    def probabilities(self):
        """Exact computational-basis probabilities of the final state, shape (2^n,)."""
        return measurement_probabilities(self._final_state())

    def sample_counts(self, shots, rng=None, method="density", trajectories=None):
        r"""
        Measure all qubits `shots` times in the computational basis.

        Methods (they only differ when a noise model is attached):

        - "density" (default): simulate the exact density matrix, then draw
          `shots` samples.  Memory O(4^n) — practical to ~12-13 qubits.
        - "trajectory": simulate pure-state quantum trajectories, memory
          O(2^n) — practical to ~25 qubits.  With trajectories=None (the
          default) every shot gets its own trajectory, which makes the
          counts EXACTLY distributed like the density method (a draw from
          a uniformly random trajectory is a draw from the mixed state).
          Passing trajectories=T < shots reuses each trajectory for
          ~shots/T shots: faster, still unbiased, but with extra variance.

        Args:
            shots: int >= 1
            rng: numpy.random.Generator or None
            method: "density" or "trajectory"
            trajectories: None (= shots) or int in [1, shots]; trajectory method only

        Returns:
            dict[str, int]: bitstring -> count, qubit 0 leftmost, sorted by
            bitstring; outcomes never observed are omitted
        """
        _check_method(method)
        shots = _check_positive_int(shots, "shots")
        if rng is None:
            rng = np.random.default_rng()

        if method == "density" or not self._is_noisy():
            counts = sample_bitstrings(self._final_state(), shots, rng=rng)
        else:
            T = shots if trajectories is None else _check_positive_int(trajectories, "trajectories")
            if T > shots:
                raise ValueError(f"trajectories ({T}) cannot exceed shots ({shots})")
            base, extra = divmod(shots, T)
            counts = np.zeros(2 ** self._n, dtype=int)
            for t, psi in enumerate(self.trajectories(T, rng=rng)):
                counts += sample_bitstrings(psi, base + (t < extra), rng=rng)

        return {format(k, f"0{self._n}b"): int(c) for k, c in enumerate(counts) if c}

    def _normalize_terms(self, coefficients, pauli_strings):
        coeffs = np.asarray(coefficients, dtype=float).ravel()
        labels = [_normalize_pauli_string(s) for s in pauli_strings]
        if len(labels) != coeffs.size:
            raise ValueError(f"got {len(labels)} Pauli strings but {coeffs.size} coefficients")
        if not labels:
            raise ValueError("at least one Pauli term is required")
        for lab in labels:
            if len(lab) != self._n:
                raise ValueError(
                    f"Pauli string {''.join(lab)!r} has length {len(lab)}, "
                    f"circuit has {self._n} qubits"
                )
        return coeffs, labels

    @staticmethod
    def _exact_pauli(state, labels):
        if state.ndim == 1:
            return pauli_string_expval(state, labels)
        # Tr(rho P) = sum_c rho[c, c XOR x] * val[c]   (P[row(c), c] = val[c])
        cols, rows, vals = _pauli_column_values(labels)
        return float(np.real(np.sum(state[cols, rows] * vals)))

    def _rotate_to_z_basis(self, state, labels):
        """Rotate so that measuring Z on each non-I qubit measures that Pauli."""
        for q, lab in enumerate(labels):
            if lab == "I" or lab == "Z":
                continue
            ops = [_H] if lab == "X" else [_SDG, _H]   # Y: S^dagger then H
            for U in ops:
                if state.ndim == 1:
                    state = apply_gate(state, U, [q])
                else:
                    state = apply_unitary_density(state, U, [q])
        return state

    def _term_counts(self, labels, n_i, rng, state=None, trajectories=None):
        """
        Computational-basis counts for n_i shots after rotating into the
        measurement basis of Pauli term `labels`.  Uses `state` (vector or
        density matrix) if given, else fresh trajectories.
        """
        if state is not None:
            return sample_bitstrings(self._rotate_to_z_basis(state, labels), n_i, rng=rng)
        T = n_i if trajectories is None else min(trajectories, n_i)
        base, extra = divmod(n_i, T)
        counts = np.zeros(2 ** self._n, dtype=int)
        for t, psi in enumerate(self.trajectories(T, rng=rng)):
            counts += sample_bitstrings(self._rotate_to_z_basis(psi, labels), base + (t < extra), rng=rng)
        return counts

    def expectation(self, coefficients, pauli_strings, shots=None, rng=None,
                    method="density", trajectories=None):
        r"""
        Expectation value of H = sum_i c_i P_i on the circuit's output.

        Exact (shots=None, method="density"):
            returns float  sum_i c_i <P_i>   (Tr(rho P_i) when noisy)

        Trajectory average (shots=None, method="trajectory", noisy circuit):
            averages the exact pure-state value sum_i c_i <psi_t|P_i|psi_t>
            over `trajectories` trajectories (default 1000) and returns
            (estimate, stderr) with stderr = std_t / sqrt(T).  Memory O(2^n).

        Finite shots:
            With method="trajectory" each shot is drawn from its own
            trajectory (or from `trajectories` shared ones, as in
            sample_counts), so no density matrix is ever built.  The
            reported stderr assumes independent shots, i.e. the default
            trajectories=None; shared trajectories add variance it does
            not include.  Otherwise as below.
            Identity terms are added exactly (they need no measurement).
            The remaining shots are split with shot_noise.allocate_shots
            (N_i proportional to |c_i|, the worst-case sigma_i = 1 rule).
            Each term is measured by rotating X -> Z with H and Y -> Z with
            H S^dagger, sampling computational-basis outcomes, and reading
            the parity of the term's qubits as the +/-1 eigenvalue.
            Per-term standard error is the binomial plug-in
            sqrt((1 - P_hat^2) / N_i); errors combine as
            Var(H_hat) = sum_i c_i^2 stderr_i^2.

        Args:
            coefficients: array-like of real c_i
            pauli_strings: sequence of Pauli strings of length num_qubits
            shots: None for the exact value, or total shots (int)
            rng: numpy.random.Generator or None
            method: "density" (default) or "trajectory"
            trajectories: int or None - number of trajectories
                (trajectory method only; see above for defaults)

        Returns:
            float for shots=None with method="density"; otherwise tuple
            (estimate, stderr).  (A noiseless circuit with
            method="trajectory" returns (exact value, 0.0).)
        """
        _check_method(method)
        coeffs, labels = self._normalize_terms(coefficients, pauli_strings)
        use_trajectories = method == "trajectory" and self._is_noisy()
        if trajectories is not None:
            trajectories = _check_positive_int(trajectories, "trajectories")
        if rng is None:
            rng = np.random.default_rng()

        def energy(state):
            return float(sum(c * self._exact_pauli(state, lab) for c, lab in zip(coeffs, labels)))

        if shots is None:
            if not use_trajectories:
                value = energy(self._final_state())
                return (value, 0.0) if method == "trajectory" else value
            T = 1000 if trajectories is None else trajectories
            if T < 2:
                raise ValueError("need at least 2 trajectories to estimate a standard error")
            values = np.array([energy(psi) for psi in self.trajectories(T, rng=rng)])
            return float(values.mean()), float(values.std(ddof=1) / np.sqrt(T))

        # Finite shots: one exact state for the density method / noiseless
        # circuits; fresh trajectories per term otherwise.
        state = None if use_trajectories else self._final_state()
        is_identity = [all(ch == "I" for ch in lab) for lab in labels]
        constant = float(sum(c for c, ident in zip(coeffs, is_identity) if ident))
        idx = [i for i, ident in enumerate(is_identity) if not ident]
        if not idx:
            return constant, 0.0

        alloc = allocate_shots(coeffs[idx], None, int(shots))
        basis = np.arange(2 ** self._n)
        estimate, variance = constant, 0.0
        for i, n_i in zip(idx, alloc):
            if n_i == 0:          # zero coefficient
                continue
            labs = labels[i]
            support = 0
            for q, lab in enumerate(labs):
                if lab != "I":
                    support |= 1 << (self._n - 1 - q)
            counts = self._term_counts(labs, n_i, rng, state=state, trajectories=trajectories)
            # +/-1 eigenvalue of each basis outcome = parity of the measured support bits
            signs = 1.0 - 2.0 * (np.bitwise_count(basis & support) & 1)
            p_hat = float(counts @ signs) / n_i
            estimate += coeffs[i] * p_hat
            variance += coeffs[i] ** 2 * max(1.0 - p_hat ** 2, 0.0) / n_i
        return float(estimate), float(np.sqrt(variance))
