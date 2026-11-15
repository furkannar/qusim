"""Tests for Quantum Fourier Transform implementation."""

import numpy as np
import pytest
from qusim.qft import qft_matrix, inverse_qft_matrix, qft_state, inverse_qft_state


class TestQFTMatrix:
    """Test QFT matrix construction."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_qft_matrix_shape(self, n):
        """Test QFT matrix has correct shape."""
        F = qft_matrix(n)
        expected_shape = (2**n, 2**n)
        assert F.shape == expected_shape

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_qft_matrix_unitarity(self, n):
        """Test QFT matrix is unitary."""
        F = qft_matrix(n)
        F_dag_F = F.conj().T @ F
        I = np.eye(2**n)
        np.testing.assert_allclose(F_dag_F, I, atol=1e-10)

    def test_qft_equals_hadamard_n1(self):
        """Test QFT for n=1 equals Hadamard gate."""
        F1 = qft_matrix(1)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(F1, H, atol=1e-10)

    def test_qft_matches_f4_n2(self):
        """Test QFT for n=2 matches known F_4 matrix."""
        i = 1j
        F4_expected = 0.5 * np.array([
            [1, 1, 1, 1],
            [1, i, -1, -i],
            [1, -1, 1, -1],
            [1, -i, -1, i],
        ], dtype=complex)
        F4 = qft_matrix(2)
        np.testing.assert_allclose(F4, F4_expected, atol=1e-10)


class TestInverseQFTMatrix:
    """Test inverse QFT matrix construction."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_inverse_qft_matrix_shape(self, n):
        """Test inverse QFT matrix has correct shape."""
        F_inv = inverse_qft_matrix(n)
        expected_shape = (2**n, 2**n)
        assert F_inv.shape == expected_shape

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_inverse_qft_matrix_unitarity(self, n):
        """Test inverse QFT matrix is unitary."""
        F_inv = inverse_qft_matrix(n)
        F_inv_dag_F_inv = F_inv.conj().T @ F_inv
        I = np.eye(2**n)
        np.testing.assert_allclose(F_inv_dag_F_inv, I, atol=1e-10)

    def test_inverse_is_conjugate_transpose(self):
        """Test inverse QFT equals conjugate transpose of QFT."""
        n = 3
        F = qft_matrix(n)
        F_inv = inverse_qft_matrix(n)
        np.testing.assert_allclose(F_inv, F.conj().T, atol=1e-10)


class TestQFTState:
    """Test QFT applied to state vectors."""

    def test_qft_state_power_of_two(self):
        """Test QFT state requires power of 2 dimension."""
        state = np.array([1.0, 2.0, 3.0])  # Not power of 2
        with pytest.raises(ValueError, match="is not a power of 2"):
            qft_state(state)

    def test_qft_state_shape(self):
        """Test QFT state preserves shape."""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        result = qft_state(state)
        assert result.shape == state.shape

    def test_qft_state_basis_transform(self):
        """Test QFT on basis state matches formula."""
        n = 3
        N = 2**n
        x = 5
        basis = np.zeros(N, dtype=complex)
        basis[x] = 1
        out = qft_state(basis)
        omega = np.exp(2j * np.pi / N)
        expected = np.array([omega ** (x * k) / np.sqrt(N) for k in range(N)])
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_qft_state_uniform_probabilities(self):
        """Test QFT on basis state produces uniform probabilities."""
        n = 3
        N = 2**n
        x = 5
        basis = np.zeros(N, dtype=complex)
        basis[x] = 1
        out = qft_state(basis)
        probs = np.abs(out)**2
        expected_prob = 1 / N
        np.testing.assert_allclose(probs, expected_prob, atol=1e-10)


class TestInverseQFTState:
    """Test inverse QFT applied to state vectors."""

    def test_inverse_qft_state_power_of_two(self):
        """Test inverse QFT state requires power of 2 dimension."""
        state = np.array([1.0, 2.0, 3.0])  # Not power of 2
        with pytest.raises(ValueError, match="is not a power of 2"):
            inverse_qft_state(state)

    def test_inverse_qft_state_shape(self):
        """Test inverse QFT state preserves shape."""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        result = inverse_qft_state(state)
        assert result.shape == state.shape


class TestQFTRoundTrip:
    """Test QFT and inverse QFT round-trip."""

    def test_round_trip_random_state(self):
        """Test inverse QFT undoes QFT on random state."""
        n = 3
        N = 2**n
        rng = np.random.default_rng(42)
        state = rng.random(N) + 1j * rng.random(N)
        state /= np.linalg.norm(state)
        recovered = inverse_qft_state(qft_state(state))
        np.testing.assert_allclose(recovered, state, atol=1e-10)

    def test_round_trip_basis_states(self):
        """Test round-trip on all basis states."""
        n = 3
        N = 2**n
        for x in range(N):
            basis = np.zeros(N, dtype=complex)
            basis[x] = 1
            recovered = inverse_qft_state(qft_state(basis))
            np.testing.assert_allclose(recovered, basis, atol=1e-10)


class TestQFTRelationToClassicalDFT:
    """Test relationship between QFT and classical DFT."""

    def test_qft_equals_scaled_ifft(self):
        """Test QFT equals sqrt(N) * numpy.fft.ifft."""
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        qft_out = qft_state(vec)
        ifft_out = np.fft.ifft(vec) * np.sqrt(len(vec))
        np.testing.assert_allclose(qft_out, ifft_out, atol=1e-10)
