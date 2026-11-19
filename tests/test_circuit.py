"""Tests for the QuantumCircuit builder, apply_gate, apply_unitary_density and NoiseModel."""

import itertools

import numpy as np
import pytest

import qusim_np as q
from qusim_np import QuantumCircuit, NoiseModel, Instruction
from qusim_np.gates import toffoli, hadamard, cnot


# ----------------------------------------------------------------------
# Independent reference: explicit full operator from index bits
# ----------------------------------------------------------------------
def _bit(i, q, n):
    return (i >> (n - 1 - q)) & 1


def _embed(U, qubits, n):
    """Full 2^n x 2^n operator of U on `qubits`, built element by element."""
    N = 2 ** n
    rest = [r for r in range(n) if r not in qubits]
    sub = lambda i: sum(_bit(i, qb, n) << (len(qubits) - 1 - k) for k, qb in enumerate(qubits))
    F = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if all(_bit(i, r, n) == _bit(j, r, n) for r in rest):
                F[i, j] = U[sub(i), sub(j)]
    return F


def _rand_unitary(rng, d):
    Z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def _rand_state(rng, n):
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return psi / np.linalg.norm(psi)


def _rand_circuit(rng, n, size=12):
    qc = QuantumCircuit(n)
    for _ in range(size):
        g = rng.integers(6)
        a, b = (int(v) for v in rng.choice(n, 2, replace=False))
        if g == 0:
            qc.h(a)
        elif g == 1:
            qc.rx(rng.uniform(-3, 3), a)
        elif g == 2:
            qc.rz(rng.uniform(-3, 3), a)
        elif g == 3:
            qc.t(a)
        elif g == 4:
            qc.cx(a, b)
        else:
            qc.unitary(_rand_unitary(rng, 4), [a, b])
    return qc


# ----------------------------------------------------------------------
# apply_gate / apply_unitary_density
# ----------------------------------------------------------------------
class TestApplyGate:

    @pytest.mark.parametrize("n", [3, 4])
    def test_matches_explicit_embedding(self, n):
        rng = np.random.default_rng(n)
        for k in (1, 2, 3):
            for qubits in itertools.permutations(range(n), k):
                U = _rand_unitary(rng, 2 ** k)
                psi = _rand_state(rng, n)
                np.testing.assert_allclose(
                    q.apply_gate(psi, U, list(qubits)), _embed(U, list(qubits), n) @ psi,
                    atol=1e-12,
                )

    def test_agrees_with_one_and_two_qubit_appliers(self):
        rng = np.random.default_rng(0)
        psi = _rand_state(rng, 3)
        H = hadamard().matrix
        np.testing.assert_allclose(q.apply_gate(psi, H, 2), q.apply_single_qubit(psi, H, 2))
        C = cnot().matrix
        np.testing.assert_allclose(q.apply_gate(psi, C, [2, 0]), q.apply_two_qubit(psi, C, 2, 0))

    @pytest.mark.parametrize("bad, match", [
        ([0, 0], "distinct"), ([3], r"\[0, 3\)"), ([], "at least one"), ([True], r"\[0, 3\)"),
    ])
    def test_bad_qubits(self, bad, match):
        with pytest.raises(ValueError, match=match):
            q.apply_gate(np.ones(8) / np.sqrt(8), np.eye(2 ** max(len(bad), 1)), bad)

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            q.apply_gate(np.ones(8) / np.sqrt(8), np.eye(2), [0, 1])


class TestApplyUnitaryDensity:

    def test_matches_dense_conjugation(self):
        rng = np.random.default_rng(1)
        n = 3
        psi, phi = _rand_state(rng, n), _rand_state(rng, n)
        rho = 0.3 * np.outer(psi, psi.conj()) + 0.7 * np.outer(phi, phi.conj())
        for qubits in ([1], [2, 0], [0, 2, 1]):
            U = _rand_unitary(rng, 2 ** len(qubits))
            F = _embed(U, qubits, n)
            np.testing.assert_allclose(
                q.apply_unitary_density(rho, U, qubits), F @ rho @ F.conj().T, atol=1e-12,
            )

    def test_state_vector_input(self):
        psi = _rand_state(np.random.default_rng(2), 2)
        U = _rand_unitary(np.random.default_rng(3), 4)
        out = q.apply_unitary_density(psi, U, [0, 1])
        ref = U @ psi
        np.testing.assert_allclose(out, np.outer(ref, ref.conj()), atol=1e-12)


# ----------------------------------------------------------------------
# Building and properties
# ----------------------------------------------------------------------
class TestBuilding:

    def test_methods_chain_and_record(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        assert isinstance(qc, QuantumCircuit)
        assert [i.name for i in qc.instructions] == ["h", "cx"]
        assert qc.instructions[1].qubits == (0, 1)
        assert len(qc) == qc.size == 2

    def test_aliases_record_canonical_names(self):
        qc = QuantumCircuit(3).cnot(0, 1).toffoli(0, 1, 2)
        assert [i.name for i in qc.instructions] == ["cx", "ccx"]

    def test_rotation_params_recorded(self):
        qc = QuantumCircuit(1).rx(0.5, 0)
        assert qc.instructions[0].params == (0.5,)
        with pytest.raises(ValueError, match="finite"):
            qc.ry(np.inf, 0)

    def test_instruction_matrix_is_immutable(self):
        inst = QuantumCircuit(1).h(0).instructions[0]
        assert isinstance(inst, Instruction)
        with pytest.raises(ValueError):
            inst.matrix[0, 0] = 5

    def test_instructions_tuple_is_read_only_view(self):
        qc = QuantumCircuit(1).x(0)
        assert isinstance(qc.instructions, tuple)
        with pytest.raises(AttributeError):
            qc.num_qubits = 4

    def test_depth(self):
        assert QuantumCircuit(3).depth == 0
        # h0 and h1 are parallel; cx(0,1) after them; h2 parallel to all
        qc = QuantumCircuit(3).h(0).h(1).h(2).cx(0, 1).x(1)
        assert qc.depth == 3

    def test_invalid_qubits(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match=r"\[0, 2\)"):
            qc.h(2)
        with pytest.raises(ValueError, match="distinct"):
            qc.cx(1, 1)

    def test_unitary_validation(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="not unitary"):
            qc.unitary(np.array([[1, 1], [0, 1]]), 0)
        with pytest.raises(ValueError, match="shape"):
            qc.unitary(np.eye(2), [0, 1])
        qc.unitary(np.eye(4), [1, 0], name="my_gate")
        assert qc.instructions[0].name == "my_gate"

    def test_invalid_num_qubits(self):
        with pytest.raises(ValueError):
            QuantumCircuit(0)

    def test_repr_lists_instructions(self):
        text = repr(QuantumCircuit(2).h(0).rx(0.25, 1).cx(0, 1))
        assert "3 gate(s)" in text and "rx(0.25) q[1]" in text and "cx q[0], q[1]" in text


# ----------------------------------------------------------------------
# Statevector simulation
# ----------------------------------------------------------------------
class TestStatevector:

    def test_empty_circuit_is_zero_state(self):
        np.testing.assert_allclose(QuantumCircuit(2).statevector(), [1, 0, 0, 0])

    def test_bell_state(self):
        psi = QuantumCircuit(2).h(0).cx(0, 1).statevector()
        np.testing.assert_allclose(psi, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-12)

    def test_ghz_state(self):
        n = 5
        qc = QuantumCircuit(n).h(0)
        for k in range(n - 1):
            qc.cx(k, k + 1)
        psi = qc.statevector()
        ref = np.zeros(2 ** n, complex)
        ref[0] = ref[-1] = 1 / np.sqrt(2)
        np.testing.assert_allclose(psi, ref, atol=1e-12)

    def test_qubit_zero_is_most_significant(self):
        np.testing.assert_allclose(QuantumCircuit(3).x(0).statevector(), np.eye(8)[4])

    def test_toffoli_matches_gate_matrix(self):
        qc = QuantumCircuit(3).ccx(0, 1, 2)
        U = np.column_stack([qc.statevector(e) for e in np.eye(8)])
        np.testing.assert_allclose(U, toffoli().matrix, atol=1e-12)

    def test_initial_state(self):
        psi = QuantumCircuit(1).x(0).statevector([0, 1])
        np.testing.assert_allclose(psi, [1, 0])
        with pytest.raises(ValueError, match="shape"):
            QuantumCircuit(1).statevector([1, 0, 0, 0])

    def test_matches_dense_product(self):
        rng = np.random.default_rng(4)
        n = 3
        qc = _rand_circuit(rng, n)
        F = np.eye(2 ** n, dtype=complex)
        for inst in qc.instructions:
            F = _embed(inst.matrix, list(inst.qubits), n) @ F
        np.testing.assert_allclose(qc.statevector(), F[:, 0], atol=1e-12)

    @pytest.mark.parametrize("n,layers,ent", [(1, 2, "linear"), (3, 2, "linear"), (2, 1, "ring"), (4, 3, "ring")])
    def test_from_hea_matches_hea_state(self, n, layers, ent):
        theta = np.random.default_rng(n).uniform(-np.pi, np.pi, q.hea_param_count(n, layers))
        np.testing.assert_allclose(
            QuantumCircuit.from_hea(n, theta, layers, ent).statevector(),
            q.hea_state(n, theta, layers, ent), atol=1e-12,
        )

    def test_from_hea_wrong_param_count(self):
        with pytest.raises(ValueError, match="parameters"):
            QuantumCircuit.from_hea(2, np.zeros(3))


# ----------------------------------------------------------------------
# copy / compose / inverse
# ----------------------------------------------------------------------
class TestManipulation:

    @pytest.mark.parametrize("seed", range(5))
    def test_inverse_undoes_circuit(self, seed):
        rng = np.random.default_rng(seed)
        qc = _rand_circuit(rng, 3)
        full = qc.compose(qc.inverse())
        psi = _rand_state(rng, 3)
        np.testing.assert_allclose(full.statevector(psi), psi, atol=1e-12)

    def test_inverse_names(self):
        inv = QuantumCircuit(2).s(0).t(1).rx(0.3, 0).cx(0, 1).unitary(np.eye(2), 0, "u").inverse()
        assert [(i.name, i.params) for i in inv.instructions] == [
            ("u_dg", ()), ("cx", ()), ("rx", (-0.3,)), ("tdg", ()), ("sdg", ()),
        ]
        assert inv.inverse().instructions[0].name == "s"

    def test_compose_with_mapping(self):
        bell = QuantumCircuit(2).h(0).cx(0, 1)
        qc = QuantumCircuit(3).compose(bell, qubits=[2, 0])
        assert [i.qubits for i in qc.instructions] == [(2,), (2, 0)]
        psi = qc.statevector()
        ref = np.zeros(8, complex)
        ref[0b000] = ref[0b101] = 1 / np.sqrt(2)
        np.testing.assert_allclose(psi, ref, atol=1e-12)

    def test_compose_does_not_mutate(self):
        a = QuantumCircuit(2).h(0)
        a.compose(QuantumCircuit(2).x(1))
        assert a.size == 1

    def test_compose_errors(self):
        with pytest.raises(ValueError, match="cannot compose"):
            QuantumCircuit(1).compose(QuantumCircuit(2))
        with pytest.raises(TypeError):
            QuantumCircuit(1).compose("x")
        with pytest.raises(ValueError, match="map all"):
            QuantumCircuit(3).compose(QuantumCircuit(2), qubits=[0])

    def test_copy_is_independent(self):
        a = QuantumCircuit(1).h(0)
        b = a.copy().x(0)
        assert (a.size, b.size) == (1, 2)


# ----------------------------------------------------------------------
# Measurement: probabilities, counts, expectation
# ----------------------------------------------------------------------
class TestMeasurement:

    def test_probabilities(self):
        np.testing.assert_allclose(QuantumCircuit(2).h(0).cx(0, 1).probabilities(),
                                   [0.5, 0, 0, 0.5], atol=1e-12)

    def test_counts_bitstring_order(self):
        assert QuantumCircuit(2).x(0).sample_counts(50) == {"10": 50}
        assert QuantumCircuit(3).x(2).sample_counts(7) == {"001": 7}

    def test_counts_bell(self):
        counts = QuantumCircuit(2).h(0).cx(0, 1).sample_counts(2000, rng=np.random.default_rng(0))
        assert set(counts) == {"00", "11"}
        assert sum(counts.values()) == 2000
        assert abs(counts["00"] - 1000) < 4 * np.sqrt(500)

    def test_exact_expectation(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        assert qc.expectation([1.0, 0.5, 2.0], ["ZZ", "XX", "ZI"]) == pytest.approx(1.5)

    def test_expectation_term_validation(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="length"):
            qc.expectation([1.0], ["Z"])
        with pytest.raises(ValueError, match="coefficients"):
            qc.expectation([1.0, 2.0], ["ZZ"])

    def test_deterministic_term_has_zero_stderr(self):
        est, err = QuantumCircuit(2).h(0).cx(0, 1).expectation([1.0], ["ZZ"], shots=500)
        assert (est, err) == (1.0, 0.0)

    def test_identity_terms_are_exact(self):
        est, err = QuantumCircuit(1).expectation([2.5], ["I"], shots=10)
        assert (est, err) == (2.5, 0.0)

    @pytest.mark.parametrize("label", ["X", "Y", "Z"])
    def test_basis_rotation_single_pauli(self, label):
        """Shot estimate of each single-qubit Pauli converges to the exact value."""
        qc = QuantumCircuit(1).ry(0.7, 0).rz(0.4, 0)
        exact = q.pauli_string_expval(qc.statevector(), label)
        est, err = qc.expectation([1.0], [label], shots=20000, rng=np.random.default_rng(5))
        assert abs(est - exact) < 4 * err
        assert err == pytest.approx(np.sqrt((1 - est ** 2) / 20000))

    def test_hamiltonian_estimate_within_error(self):
        rng = np.random.default_rng(6)
        theta = rng.uniform(-np.pi, np.pi, q.hea_param_count(3, 2))
        qc = QuantumCircuit.from_hea(3, theta, 2)
        coeffs, strings = [0.4, -1.1, 0.7, 0.25, 0.9], ["III", "ZZI", "XIX", "YYZ", "IXY"]
        exact = qc.expectation(coeffs, strings)
        assert exact == pytest.approx(
            sum(c * q.pauli_string_expval(q.hea_state(3, theta, 2), s) for c, s in zip(coeffs, strings))
        )
        est, err = qc.expectation(coeffs, strings, shots=40000, rng=rng)
        assert 0 < err < 0.05
        assert abs(est - exact) < 4 * err


# ----------------------------------------------------------------------
# Noise model
# ----------------------------------------------------------------------
class TestNoise:

    def test_noise_model_property_type_checked(self):
        qc = QuantumCircuit(1)
        with pytest.raises(TypeError, match="NoiseModel"):
            qc.noise_model = "foo"
        nm = NoiseModel()
        qc.noise_model = nm
        assert qc.noise_model is nm

    def test_bad_kraus_rejected(self):
        with pytest.raises(ValueError, match="not trace-preserving"):
            NoiseModel().add_gate_error([np.diag([1.0, 0.5])])

    def test_amplitude_damping_matches_direct_channel(self):
        nm = NoiseModel().add_gate_error(q.amplitude_damping_kraus(0.3), gates="x")
        rho = QuantumCircuit(1, nm).x(0).density_matrix()
        np.testing.assert_allclose(rho, q.amplitude_damping(np.array([0, 1]), 0.3), atol=1e-12)

    def test_statevector_refuses_noisy_circuit(self):
        nm = NoiseModel().add_gate_error(q.depolarizing_kraus(0.1))
        with pytest.raises(ValueError, match="density_matrix"):
            QuantumCircuit(1, nm).h(0).statevector()

    def test_empty_noise_model_is_noiseless(self):
        qc = QuantumCircuit(2, NoiseModel()).h(0).cx(0, 1)
        psi = qc.statevector()
        np.testing.assert_allclose(qc.density_matrix(), np.outer(psi, psi.conj()), atol=1e-12)

    def test_noiseless_density_matches_dense_path(self):
        """density_matrix() via the U rho U^dagger path equals |psi><psi|."""
        rng = np.random.default_rng(7)
        qc = _rand_circuit(rng, 3)
        psi = qc.statevector()
        # force the density-matrix path with an initial density matrix
        rho0 = np.zeros((8, 8), complex)
        rho0[0, 0] = 1
        np.testing.assert_allclose(qc.density_matrix(rho0), np.outer(psi, psi.conj()), atol=1e-12)

    def test_gate_and_qubit_filters(self):
        p = 0.2
        nm = NoiseModel().add_gate_error(q.depolarizing_kraus(p), gates=["cnot"], qubits=[1])
        rho = QuantumCircuit(2, nm).x(0).cx(0, 1).density_matrix()
        # ideal output |11>; depolarizing on qubit 1 only, after the CNOT
        ref = q.depolarizing(q.to_density_matrix(np.eye(4)[3]), p, qubit=1)
        np.testing.assert_allclose(rho, ref, atol=1e-12)

    def test_two_qubit_gate_noise_on_both_qubits(self):
        p = 0.1
        nm = NoiseModel().add_gate_error(q.depolarizing_kraus(p), gates="cx")
        rho = QuantumCircuit(2, nm).h(0).cx(0, 1).density_matrix()
        psi = QuantumCircuit(2).h(0).cx(0, 1).statevector()
        ref = q.depolarizing(q.depolarizing(psi, p, qubit=0), p, qubit=1)
        np.testing.assert_allclose(rho, ref, atol=1e-12)
        assert np.trace(rho) == pytest.approx(1.0)

    def test_noisy_probabilities_counts_and_expectation(self):
        p = 0.2
        nm = NoiseModel().add_gate_error(q.depolarizing_kraus(p), gates="cx")
        qc = QuantumCircuit(2, nm).h(0).cx(0, 1)
        # Depolarizing shrinks each qubit's Bloch vector by (1-p): <ZZ> = (1-p)^2
        assert qc.expectation([1.0], ["ZZ"]) == pytest.approx((1 - p) ** 2)
        probs = qc.probabilities()
        assert probs.sum() == pytest.approx(1.0) and probs[1] > 0
        est, err = qc.expectation([1.0], ["ZZ"], shots=20000, rng=np.random.default_rng(8))
        assert abs(est - (1 - p) ** 2) < 4 * err
        assert sum(qc.sample_counts(300, rng=np.random.default_rng(9)).values()) == 300

    def test_noise_model_repr_and_rules(self):
        nm = NoiseModel().add_gate_error(q.depolarizing_kraus(0.1), gates="toffoli", qubits=2)
        assert len(nm) == 1 and nm.rules[0][1] == frozenset({"ccx"})
        assert "ccx" in repr(nm)
        with pytest.raises(ValueError):
            NoiseModel().add_gate_error(q.depolarizing_kraus(0.1), gates=[])


# ----------------------------------------------------------------------
# Quantum trajectories
# ----------------------------------------------------------------------
def _noisy_ghz(n, p=0.1, gamma=0.15):
    nm = (NoiseModel()
          .add_gate_error(q.depolarizing_kraus(p), gates="cx")
          .add_gate_error(q.amplitude_damping_kraus(gamma), gates="h"))
    qc = QuantumCircuit(n, nm).h(0)
    for k in range(n - 1):
        qc.cx(k, k + 1)
    return qc


class TestSampleKrausBranch:

    def test_single_unitary_branch_is_deterministic(self):
        psi = _rand_state(np.random.default_rng(0), 2)
        out, k = q.sample_kraus_branch(psi, [q.pauli_x().matrix], qubit=1)
        assert k == 0
        np.testing.assert_allclose(out, q.apply_gate(psi, q.pauli_x().matrix, [1]))

    def test_branch_probabilities_equal_vdot_definition(self):
        """Reduced-density shortcut gives exactly p_k = vdot(K_k psi, K_k psi)."""
        rng = np.random.default_rng(11)
        channels = [q.depolarizing_kraus(0.3), q.amplitude_damping_kraus(0.45),
                    q.phase_damping_kraus(0.2), q.random_pauli_kraus((0.6, 0.1, 0.2, 0.1))]
        for n in (1, 3, 4):
            psi = _rand_state(rng, n)
            for ops in channels:
                for qb in range(n):
                    ref = [np.vdot(q.apply_gate(psi, K, [qb]), q.apply_gate(psi, K, [qb])).real
                           for K in ops]
                    np.testing.assert_allclose(q.kraus_branch_probabilities(psi, ops, qb), ref,
                                               atol=1e-14)

    def test_identity_branch_skips_gate(self):
        """p = 0 depolarizing: always the c*I branch; state returned unchanged."""
        psi = _rand_state(np.random.default_rng(12), 3)
        out, k = q.sample_kraus_branch(psi, q.depolarizing_kraus(0.0), qubit=1)
        assert k == 0
        np.testing.assert_allclose(out, psi, atol=1e-15)

    def test_branch_frequencies_are_vdot_probabilities(self):
        """|1> under amplitude damping: decay branch K1 has p = <phi_1|phi_1> = gamma."""
        rng = np.random.default_rng(1)
        gamma, n = 0.3, 4000
        hits = sum(q.sample_kraus_branch(np.array([0, 1]), q.amplitude_damping_kraus(gamma), rng=rng)[1]
                   for _ in range(n))
        assert abs(hits / n - gamma) < 4 * np.sqrt(gamma * (1 - gamma) / n)

    def test_output_is_normalized(self):
        rng = np.random.default_rng(2)
        psi = _rand_state(rng, 3)
        for _ in range(20):
            psi, _ = q.sample_kraus_branch(psi, q.depolarizing_kraus(0.4), qubit=2, rng=rng)
            assert np.linalg.norm(psi) == pytest.approx(1.0)

    def test_average_reproduces_channel(self):
        rng = np.random.default_rng(3)
        psi = _rand_state(rng, 2)
        ops = q.amplitude_damping_kraus(0.4)
        T = 4000
        avg = sum(np.outer(s, s.conj()) for s, _ in
                  (q.sample_kraus_branch(psi, ops, qubit=0, rng=rng) for _ in range(T))) / T
        np.testing.assert_allclose(avg, q.apply_channel(psi, ops, qubit=0), atol=0.04)

    def test_rejects_unnormalized_state_and_bad_kraus(self):
        with pytest.raises(ValueError, match="normalized"):
            q.sample_kraus_branch(np.array([2.0, 0.0]), q.depolarizing_kraus(0.1))
        with pytest.raises(ValueError, match="not trace-preserving"):
            q.sample_kraus_branch(np.array([1.0, 0.0]), [np.diag([1.0, 0.5])])


class TestTrajectories:

    def test_average_trajectory_matches_density_matrix(self):
        qc = _noisy_ghz(2)
        T = 4000
        avg = sum(np.outer(s, s.conj()) for s in qc.trajectories(T, rng=np.random.default_rng(4))) / T
        np.testing.assert_allclose(avg, qc.density_matrix(), atol=0.04)

    def test_trajectory_counts_match_density_distribution(self):
        qc = _noisy_ghz(3)
        probs = qc.probabilities()
        shots = 6000
        counts = qc.sample_counts(shots, rng=np.random.default_rng(5), method="trajectory")
        assert sum(counts.values()) == shots
        for k, p in enumerate(probs):
            freq = counts.get(format(k, "03b"), 0) / shots
            assert abs(freq - p) < 4 * np.sqrt(p * (1 - p) / shots) + 1e-3

    def test_shared_trajectories(self):
        qc = _noisy_ghz(3)
        counts = qc.sample_counts(1001, rng=np.random.default_rng(6), method="trajectory", trajectories=10)
        assert sum(counts.values()) == 1001
        with pytest.raises(ValueError, match="cannot exceed"):
            qc.sample_counts(5, method="trajectory", trajectories=6)

    def test_expectation_trajectory_average(self):
        qc = _noisy_ghz(3)
        terms = ([1.0, 0.5, -0.7], ["ZZI", "XXX", "IZZ"])
        exact = qc.expectation(*terms)
        est, err = qc.expectation(*terms, method="trajectory", trajectories=3000,
                                  rng=np.random.default_rng(7))
        assert err > 0
        assert abs(est - exact) < 4 * err

    def test_expectation_trajectory_with_shots(self):
        qc = _noisy_ghz(3)
        exact = qc.expectation([1.0, 0.5], ["ZZI", "XXX"])
        est, err = qc.expectation([1.0, 0.5], ["ZZI", "XXX"], shots=6000, method="trajectory",
                                  rng=np.random.default_rng(8))
        assert abs(est - exact) < 4 * err

    def test_noiseless_trajectory_is_exact(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        assert qc.expectation([1.0], ["ZZ"], method="trajectory") == (pytest.approx(1.0), 0.0)
        counts = qc.sample_counts(100, rng=np.random.default_rng(9), method="trajectory")
        assert set(counts) <= {"00", "11"}

    def test_trajectory_never_builds_density_matrix(self, monkeypatch):
        """The trajectory paths must not fall back to the O(4^n) simulation."""
        qc = _noisy_ghz(3)
        def boom(*a, **k):
            raise AssertionError("density_matrix() called")
        monkeypatch.setattr(qc, "density_matrix", boom)
        qc.sample_counts(50, method="trajectory")
        qc.expectation([1.0], ["ZZZ"], method="trajectory", trajectories=20)
        qc.expectation([1.0], ["ZZZ"], shots=50, method="trajectory")

    def test_large_noisy_circuit_beyond_density_limit(self):
        """16 qubits: a density matrix would need 69 GB; a trajectory needs 1 MB."""
        n = 16
        qc = _noisy_ghz(n, p=0.02, gamma=0.0)
        counts = qc.sample_counts(40, rng=np.random.default_rng(10), method="trajectory")
        assert sum(counts.values()) == 40
        ghz = counts.get("0" * n, 0) + counts.get("1" * n, 0)
        # Only X/Y errors leave the GHZ subspace: P(none) = (1 - p/2)^(2*15) ~ 0.74
        assert ghz >= 20

    def test_invalid_arguments(self):
        qc = _noisy_ghz(2)
        with pytest.raises(ValueError, match="method"):
            qc.sample_counts(10, method="exact")
        with pytest.raises(ValueError, match="method"):
            qc.expectation([1.0], ["ZZ"], method="mc")
        with pytest.raises(ValueError, match="at least 2"):
            qc.expectation([1.0], ["ZZ"], method="trajectory", trajectories=1)
        with pytest.raises(ValueError, match="positive integer"):
            list(qc.trajectories(0))
