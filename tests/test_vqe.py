"""Tests for VQE Pauli-string Hamiltonian machinery."""

import numpy as np
import pytest
from qusim.config import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
from qusim.gates import pauli_x, pauli_y, pauli_z, hadamard
from qusim.vqe import (
    pauli_string_matrix, apply_single_qubit, apply_two_qubit,
    apply_pauli_string, pauli_string_expval, hamiltonian, expectation_value,
    hea_param_count, hea_state, vqe_energy, vqe_optimize,
)
from qusim.gates import cnot


class TestConfigPauliMatrices:
    """Test Pauli matrices defined in config."""

    def test_pauli_i(self):
        np.testing.assert_array_equal(PAULI_I, np.eye(2))

    def test_pauli_x_matches_gate(self):
        np.testing.assert_array_equal(PAULI_X, pauli_x().matrix)

    def test_pauli_y_matches_gate(self):
        np.testing.assert_array_equal(PAULI_Y, pauli_y().matrix)

    def test_pauli_z_matches_gate(self):
        np.testing.assert_array_equal(PAULI_Z, pauli_z().matrix)

    @pytest.mark.parametrize("P", [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z])
    def test_pauli_hermitian_and_unitary(self, P):
        np.testing.assert_allclose(P.conj().T, P, atol=1e-12)
        np.testing.assert_allclose(P.conj().T @ P, np.eye(2), atol=1e-12)


class TestPauliStringMatrix:
    """Test Pauli string matrix construction."""

    def test_single_pauli(self):
        np.testing.assert_array_equal(pauli_string_matrix("X"), PAULI_X)
        np.testing.assert_array_equal(pauli_string_matrix("Y"), PAULI_Y)
        np.testing.assert_array_equal(pauli_string_matrix("Z"), PAULI_Z)
        np.testing.assert_array_equal(pauli_string_matrix("I"), PAULI_I)

    def test_two_qubit_kron(self):
        expected = np.kron(PAULI_X, PAULI_Z)
        np.testing.assert_array_equal(pauli_string_matrix("XZ"), expected)

    def test_three_qubit_kron(self):
        expected = np.kron(np.kron(PAULI_X, PAULI_Y), PAULI_Z)
        np.testing.assert_array_equal(pauli_string_matrix("XYZ"), expected)

    def test_identity_factors(self):
        expected = np.kron(PAULI_I, np.kron(PAULI_X, PAULI_I))
        np.testing.assert_array_equal(pauli_string_matrix("IXI"), expected)

    def test_tuple_input(self):
        np.testing.assert_array_equal(
            pauli_string_matrix(("X", "Z", "I")), pauli_string_matrix("XZI")
        )

    def test_index_input(self):
        # Indices into "IXYZ": 0=I, 1=X, 2=Y, 3=Z
        np.testing.assert_array_equal(pauli_string_matrix((1, 3, 0)), pauli_string_matrix("XZI"))

    def test_case_insensitive(self):
        np.testing.assert_array_equal(pauli_string_matrix("xzi"), pauli_string_matrix("XZI"))

    def test_invalid_label(self):
        with pytest.raises(ValueError, match="invalid Pauli label"):
            pauli_string_matrix("XW")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="at least one label"):
            pauli_string_matrix("")

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="string or sequence"):
            pauli_string_matrix(5)

    @pytest.mark.parametrize("s", ["XYZ", "IXZI", "ZYX"])
    def test_pauli_string_hermitian_and_unitary(self, s):
        P = pauli_string_matrix(s)
        np.testing.assert_allclose(P.conj().T, P, atol=1e-12)
        np.testing.assert_allclose(P.conj().T @ P, np.eye(2 ** len(s)), atol=1e-12)


class TestApplySingleQubit:
    """Test applying a 2x2 gate to one qubit of a state vector."""

    def test_qubit0_is_most_significant(self):
        """Gate on qubit 0 equals kron(U, I) on a 2-qubit state."""
        n = 2
        rng = np.random.default_rng(0)
        state = rng.random(2 ** n) + 1j * rng.random(2 ** n)
        state /= np.linalg.norm(state)
        U = hadamard().matrix
        out = apply_single_qubit(state, U, 0)
        expected = np.kron(U, PAULI_I) @ state
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_qubit1(self):
        """Gate on qubit 1 equals kron(I, U)."""
        n = 2
        rng = np.random.default_rng(1)
        state = rng.random(2 ** n) + 1j * rng.random(2 ** n)
        state /= np.linalg.norm(state)
        U = hadamard().matrix
        out = apply_single_qubit(state, U, 1)
        expected = np.kron(PAULI_I, U) @ state
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_three_qubits_last_qubit(self):
        n = 3
        rng = np.random.default_rng(2)
        state = rng.random(2 ** n) + 1j * rng.random(2 ** n)
        state /= np.linalg.norm(state)
        U = hadamard().matrix
        out = apply_single_qubit(state, U, 2)
        expected = np.kron(np.kron(PAULI_I, PAULI_I), U) @ state
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_hadamard_on_zero_state(self):
        """H on qubit 0 of |00> gives (|00> + |10>)/sqrt(2)."""
        state = np.array([1, 0, 0, 0], dtype=complex)
        out = apply_single_qubit(state, hadamard().matrix, 0)
        expected = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_invalid_qubit(self):
        state = np.array([1, 0], dtype=complex)
        with pytest.raises(ValueError, match="qubit must be"):
            apply_single_qubit(state, PAULI_X, 1)
        with pytest.raises(ValueError, match="qubit must be"):
            apply_single_qubit(state, PAULI_X, -1)

    def test_invalid_gate_shape(self):
        state = np.array([1, 0], dtype=complex)
        with pytest.raises(ValueError, match="shape .2, 2."):
            apply_single_qubit(state, np.eye(4), 0)

    def test_non_power_of_two_state(self):
        with pytest.raises(ValueError, match="power of 2"):
            apply_single_qubit(np.ones(3), PAULI_X, 0)


class TestApplyPauliString:
    """Test applying a full Pauli string to a state."""

    def test_matches_matrix(self):
        n = 3
        rng = np.random.default_rng(3)
        state = rng.random(2 ** n) + 1j * rng.random(2 ** n)
        state /= np.linalg.norm(state)
        for s in ["XYZ", "IZI", "ZZZ"]:
            out = apply_pauli_string(state, s)
            expected = pauli_string_matrix(s) @ state
            np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_phase_on_basis_state(self):
        """Z on qubit 0 of |10> gives -|10>."""
        state = np.array([0, 0, 1, 0], dtype=complex)  # |10>
        out = apply_pauli_string(state, "ZI")
        np.testing.assert_allclose(out, -state, atol=1e-10)

    def test_length_mismatch(self):
        state = np.array([1, 0, 0, 0], dtype=complex)
        with pytest.raises(ValueError, match="does not match"):
            apply_pauli_string(state, "XYZ")


class TestPauliStringExpval:
    """Test Pauli string expectation values."""

    def test_product_state(self):
        """|+>|0>: <XI>=1, <IZ>=1, <ZI>=0."""
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        state = np.kron(plus, np.array([1, 0], dtype=complex))
        assert pauli_string_expval(state, "XI") == pytest.approx(1.0)
        assert pauli_string_expval(state, "IZ") == pytest.approx(1.0)
        assert pauli_string_expval(state, "ZI") == pytest.approx(0.0)

    def test_bell_state(self):
        """|Phi+> = (|00>+|11>)/sqrt(2): XX, ZZ give +1; YY gives -1; XI, ZI give 0."""
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert pauli_string_expval(state, "XX") == pytest.approx(1.0)
        assert pauli_string_expval(state, "ZZ") == pytest.approx(1.0)
        assert pauli_string_expval(state, "YY") == pytest.approx(-1.0)
        assert pauli_string_expval(state, "XI") == pytest.approx(0.0)
        assert pauli_string_expval(state, "ZI") == pytest.approx(0.0)

    def test_expval_matches_matrix(self):
        n = 3
        rng = np.random.default_rng(4)
        state = rng.random(2 ** n) + 1j * rng.random(2 ** n)
        state /= np.linalg.norm(state)
        s = "XZY"
        expected = np.real(np.vdot(state, pauli_string_matrix(s) @ state))
        assert pauli_string_expval(state, s) == pytest.approx(expected)


class TestHamiltonian:
    """Test Hamiltonian assembly H = alpha @ p."""

    def test_single_identity_term(self):
        H = hamiltonian([2.0], ["II"])
        np.testing.assert_allclose(H, 2.0 * np.eye(4), atol=1e-12)

    def test_two_terms(self):
        H = hamiltonian([0.5, -1.0], ["IX", "ZI"])
        expected = 0.5 * np.kron(PAULI_I, PAULI_X) - 1.0 * np.kron(PAULI_Z, PAULI_I)
        np.testing.assert_allclose(H, expected, atol=1e-12)

    def test_sum_of_terms(self):
        coeffs = [1.0, -0.7, 0.3]
        strings = ["XX", "YY", "ZZ"]
        H = hamiltonian(coeffs, strings)
        expected = sum(
            c * pauli_string_matrix(s) for c, s in zip(coeffs, strings)
        )
        np.testing.assert_allclose(H, expected, atol=1e-12)

    def test_hermitian_for_real_coeffs(self):
        H = hamiltonian([0.4, -1.2, 0.9], ["XYZ", "IZX", "ZZI"])
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="coefficients"):
            hamiltonian([1.0, 2.0], ["XI"])

    def test_mixed_string_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            hamiltonian([1.0, 2.0], ["XI", "IZI"])

    def test_empty_terms_raise(self):
        with pytest.raises(ValueError, match="at least one term"):
            hamiltonian([], [])


class TestExpectationValue:
    """Test general expectation value <psi|O|psi>."""

    def test_diagonal_observable(self):
        state = np.array([1, 0], dtype=complex)  # |0>
        assert expectation_value(state, PAULI_Z) == pytest.approx(1.0)
        state1 = np.array([0, 1], dtype=complex)  # |1>
        assert expectation_value(state1, PAULI_Z) == pytest.approx(-1.0)

    def test_hadamard_state(self):
        state = hadamard().matrix @ np.array([1, 0], dtype=complex)
        assert expectation_value(state, PAULI_X) == pytest.approx(1.0)
        assert expectation_value(state, PAULI_Z) == pytest.approx(0.0)

    def test_operator_shape_mismatch(self):
        state = np.array([1, 0], dtype=complex)
        with pytest.raises(ValueError, match="operator must have shape"):
            expectation_value(state, np.eye(4))


def _embedded_two_qubit_matrix(U, n, q0, q1):
    """Reference: dense 2^n x 2^n matrix for gate U on qubits (q0, q1)."""
    N = 2 ** n
    E = np.zeros((N, N), dtype=complex)
    for i in range(N):
        a = (i >> (n - 1 - q0)) & 1
        b = (i >> (n - 1 - q1)) & 1
        p = 2 * a + b
        for op in range(4):
            a2, b2 = (op >> 1) & 1, op & 1
            j = 0
            for k in range(n):
                v = a2 if k == q0 else (b2 if k == q1 else (i >> (n - 1 - k)) & 1)
                j = (j << 1) | v
            E[j, i] = U[op, p]
    return E


class TestApplyTwoQubit:
    """Test applying a 4x4 gate to two qubits of a state vector."""

    def test_n2_equals_plain_matmul(self):
        rng = np.random.default_rng(10)
        state = rng.random(4) + 1j * rng.random(4)
        U = cnot().matrix
        out = apply_two_qubit(state, U, 0, 1)
        np.testing.assert_allclose(out, U @ state, atol=1e-10)

    def test_adjacent_pair_n3(self):
        n = 3
        rng = np.random.default_rng(11)
        state = rng.random(2 ** n) + 1j * rng.random(2 ** n)
        U = cnot().matrix
        out = apply_two_qubit(state, U, 0, 1)
        np.testing.assert_allclose(out, np.kron(U, PAULI_I) @ state, atol=1e-10)
        out = apply_two_qubit(state, U, 1, 2)
        np.testing.assert_allclose(out, np.kron(PAULI_I, U) @ state, atol=1e-10)

    def test_nonadjacent_pair_n3(self):
        n = 3
        rng = np.random.default_rng(12)
        state = rng.random(2 ** n) + 1j * rng.random(2 ** n)
        U = cnot().matrix
        for q0, q1 in [(0, 2), (2, 0)]:
            out = apply_two_qubit(state, U, q0, q1)
            expected = _embedded_two_qubit_matrix(U, n, q0, q1) @ state
            np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_random_unitary_pair(self):
        n = 3
        rng = np.random.default_rng(13)
        A = rng.random((4, 4)) + 1j * rng.random((4, 4))
        state = rng.random(2 ** n) + 1j * rng.random(2 ** n)
        out = apply_two_qubit(state, A, 0, 2)
        expected = _embedded_two_qubit_matrix(A, n, 0, 2) @ state
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_cnot_flips_target(self):
        # |10> (q0=1, q1=0): control q0=1 flips q1 -> |11>
        state = np.array([0, 0, 1, 0], dtype=complex)
        out = apply_two_qubit(state, cnot().matrix, 0, 1)
        np.testing.assert_allclose(out, np.array([0, 0, 0, 1], dtype=complex), atol=1e-10)

    def test_cnot_reversed_qubit_order(self):
        # |01> (q0=0, q1=1): control is qubit 1 -> flips qubit 0 -> |11>
        state = np.array([0, 1, 0, 0], dtype=complex)
        out = apply_two_qubit(state, cnot().matrix, 1, 0)
        np.testing.assert_allclose(out, np.array([0, 0, 0, 1], dtype=complex), atol=1e-10)

    def test_cnot_control_zero_no_flip(self):
        state = np.array([1, 0, 0, 0], dtype=complex)
        out = apply_two_qubit(state, cnot().matrix, 0, 1)
        np.testing.assert_allclose(out, state, atol=1e-10)

    def test_same_qubit_raises(self):
        state = np.ones(4, dtype=complex)
        with pytest.raises(ValueError, match="distinct"):
            apply_two_qubit(state, cnot().matrix, 1, 1)

    def test_qubit_out_of_range(self):
        state = np.ones(4, dtype=complex)
        with pytest.raises(ValueError, match="qubit must be"):
            apply_two_qubit(state, cnot().matrix, 0, 2)

    def test_bad_gate_shape(self):
        state = np.ones(4, dtype=complex)
        with pytest.raises(ValueError, match="shape .4, 4."):
            apply_two_qubit(state, np.eye(2), 0, 1)


class TestHEAState:
    """Test the hardware-efficient ansatz state preparation."""

    def test_param_count(self):
        assert hea_param_count(2, 3) == 12
        assert hea_param_count(3, 1) == 6

    def test_zero_params_give_zero_state(self):
        state = hea_state(2, np.zeros(8), layers=2)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_allclose(state, expected, atol=1e-10)

    def test_single_qubit_rotation(self):
        # n=1, layers=1: RY(pi)|0> = |1> (RZ(0) is identity)
        state = hea_state(1, [np.pi, 0.0], layers=1)
        np.testing.assert_allclose(state, np.array([0, 1], dtype=complex), atol=1e-10)

    def test_normalized(self):
        rng = np.random.default_rng(20)
        theta = rng.uniform(-np.pi, np.pi, hea_param_count(3, 2))
        state = hea_state(3, theta, layers=2)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-10)

    def test_complex_amplitudes(self):
        # RZ introduces complex phases; HEA states are generically complex
        state = hea_state(1, [0.3, 0.7], layers=1)
        assert np.iscomplexobj(state)
        assert np.any(np.abs(np.imag(state)) > 1e-12)

    def test_ring_entanglement(self):
        rng = np.random.default_rng(21)
        theta = rng.uniform(-1, 1, hea_param_count(3, 2))
        state = hea_state(3, theta, layers=2, entanglement="ring")
        assert state.shape == (8,)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-10)

    def test_ring_single_qubit(self):
        # ring with n=1 has no valid pairs; should behave like linear
        state = hea_state(1, [0.5, 0.2], layers=1, entanglement="ring")
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-10)

    def test_wrong_theta_size(self):
        with pytest.raises(ValueError, match="parameters"):
            hea_state(2, np.zeros(3), layers=1)  # needs 4 params

    def test_wrong_theta_size2(self):
        with pytest.raises(ValueError, match="parameters"):
            hea_state(2, np.zeros(5), layers=1)

    def test_invalid_layers(self):
        with pytest.raises(ValueError, match="layers"):
            hea_state(2, np.zeros(4), layers=0)

    def test_invalid_entanglement(self):
        with pytest.raises(ValueError, match="entanglement"):
            hea_state(2, np.zeros(4), layers=1, entanglement="bogus")

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="positive integer"):
            hea_state(0, np.zeros(2), layers=1)


class TestVQEEnergy:
    """Test the VQE cost function."""

    def test_matches_dense_expectation(self):
        n, layers = 2, 2
        rng = np.random.default_rng(30)
        theta = rng.uniform(-np.pi, np.pi, hea_param_count(n, layers))
        coeffs = [1.3, -0.7, 0.4]
        strings = ["ZI", "IX", "ZZ"]
        e_cheap = vqe_energy(theta, n, layers, coeffs, strings)
        state = hea_state(n, theta, layers)
        H = hamiltonian(coeffs, strings)
        e_dense = expectation_value(state, H)
        assert e_cheap == pytest.approx(e_dense, abs=1e-10)

    def test_zero_state_energy(self):
        # |00> with H = -ZI gives -<Z_0> = -1
        e = vqe_energy(np.zeros(8), 2, 2, [-1.0], ["ZI"])
        assert e == pytest.approx(-1.0)

    def test_coefficient_string_mismatch(self):
        with pytest.raises(ValueError, match="Pauli strings"):
            vqe_energy(np.zeros(8), 2, 2, [1.0, 2.0], ["ZI"])


class TestVQEOptimize:
    """Test VQE optimization loop."""

    def test_n1_z_field_ground_state(self):
        # H = -Z: ground state |1> with energy -1
        res = vqe_optimize(1, 1, [-1.0], ["Z"], method="Nelder-Mead")
        assert res.fun == pytest.approx(-1.0, abs=1e-4)
        assert res.x.size == hea_param_count(1, 1)

    def test_n2_ising_improves_on_init(self):
        # Transverse-field Ising: H = -ZZ - 0.5 XI - 0.5 IX
        coeffs = [-1.0, -0.5, -0.5]
        strings = ["ZZ", "XI", "IX"]
        theta0 = np.zeros(hea_param_count(2, 1))
        e0 = vqe_energy(theta0, 2, 1, coeffs, strings)
        res = vqe_optimize(2, 1, coeffs, strings, theta0=theta0,
                           method="Nelder-Mead", options={"maxiter": 3000})
        assert res.fun < e0

        # Exact ground state energy for comparison
        H = hamiltonian(coeffs, strings)
        e_exact = np.linalg.eigvalsh(H)[0]
        assert res.fun >= e_exact - 1e-8
        assert res.fun < e_exact + 0.25  # 1-layer HEA may not reach exact GS

    def test_bad_theta0_size(self):
        with pytest.raises(ValueError, match="theta0"):
            vqe_optimize(1, 1, [1.0], ["Z"], theta0=np.zeros(5))
