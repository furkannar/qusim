"""Tests for Grover's search algorithm implementation."""

import numpy as np
import pytest
from qusim_np.grover_search import (
    hadamard_transform, hadamard_matrix_closed_form,
    initialize_uniform_superposition, grover_oracle, diffusion_operator,
    grover_iteration, calculate_theta, calculate_optimal_iterations,
    grover_search, _validate_n, _validate_marked_states
)


class TestValidation:
    """Test validation functions."""

    def test_validate_n_valid(self):
        """Test valid n values."""
        assert _validate_n(1) is None
        assert _validate_n(5) is None

    def test_validate_n_invalid(self):
        """Test invalid n values."""
        with pytest.raises(ValueError, match="n must be a positive integer"):
            _validate_n(0)
        with pytest.raises(ValueError, match="n must be a positive integer"):
            _validate_n(-1)
        with pytest.raises(ValueError, match="n must be a positive integer"):
            _validate_n(1.5)
        with pytest.raises(ValueError, match="n must be a positive integer"):
            _validate_n(True)

    def test_validate_marked_states_empty(self):
        """Test empty marked states raises error."""
        with pytest.raises(ValueError, match="marked_states must contain at least one state"):
            _validate_marked_states(2, [])

    def test_validate_marked_states_invalid_type(self):
        """Test non-integer marked states raises error."""
        with pytest.raises(ValueError, match="is not an integer"):
            _validate_marked_states(2, [1.5])
        with pytest.raises(ValueError, match="is not an integer"):
            _validate_marked_states(2, ["1"])

    def test_validate_marked_states_out_of_range(self):
        """Test out-of-range marked states raises error."""
        with pytest.raises(ValueError, match="out of range"):
            _validate_marked_states(2, [4])
        with pytest.raises(ValueError, match="out of range"):
            _validate_marked_states(2, [-1])

    def test_validate_marked_states_deduplication(self):
        """Test that duplicate states are removed."""
        result = _validate_marked_states(2, [1, 1, 2, 2])
        assert result == [1, 2]


class TestHadamardTransform:
    """Test n-qubit Hadamard transform."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_hadamard_transform_shape(self, n):
        """Test Hadamard transform has correct shape."""
        H = hadamard_transform(n)
        expected_shape = (2**n, 2**n)
        assert H.shape == expected_shape

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_hadamard_transform_unitarity(self, n):
        """Test Hadamard transform is unitary."""
        H = hadamard_transform(n)
        H_dag_H = H.conj().T @ H
        I = np.eye(2**n)
        np.testing.assert_allclose(H_dag_H, I, atol=1e-10)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tensor_vs_closed_form(self, n):
        """Test tensor product and closed form produce same result."""
        H_tensor = hadamard_transform(n)
        H_closed = hadamard_matrix_closed_form(n)
        np.testing.assert_allclose(H_tensor, H_closed, atol=1e-10)


class TestUniformSuperposition:
    """Test uniform superposition initialization."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_superposition_shape(self, n):
        """Test superposition has correct shape."""
        state = initialize_uniform_superposition(n)
        expected_shape = (2**n,)
        assert state.shape == expected_shape

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_superposition_normalized(self, n):
        """Test superposition is normalized."""
        state = initialize_uniform_superposition(n)
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_superposition_equal_amplitudes(self):
        """Test superposition has equal amplitudes."""
        n = 3
        state = initialize_uniform_superposition(n)
        N = 2**n
        expected_amplitude = 1 / np.sqrt(N)
        for amp in state:
            np.testing.assert_allclose(abs(amp), expected_amplitude, atol=1e-10)


class TestGroverOracle:
    """Test Grover oracle construction."""

    def test_oracle_shape(self):
        """Test oracle has correct shape."""
        n = 3
        marked_states = [5]
        O = grover_oracle(n, marked_states)
        assert O.shape == (2**n, 2**n)

    def test_oracle_unitarity(self):
        """Test oracle is unitary."""
        n = 3
        marked_states = [5]
        O = grover_oracle(n, marked_states)
        O_dag_O = O.conj().T @ O
        I = np.eye(2**n)
        np.testing.assert_allclose(O_dag_O, I, atol=1e-10)

    def test_oracle_phase_flip(self):
        """Test oracle flips phase of marked states."""
        n = 3
        marked_states = [5]
        O = grover_oracle(n, marked_states)
        # Check that marked state gets -1 phase
        assert np.isclose(O[5, 5], -1)
        # Check that unmarked states keep +1 phase
        for i in range(2**n):
            if i not in marked_states:
                assert np.isclose(O[i, i], 1)


class TestDiffusionOperator:
    """Test diffusion operator."""

    def test_diffusion_shape(self):
        """Test diffusion operator has correct shape."""
        n = 3
        D = diffusion_operator(n)
        assert D.shape == (2**n, 2**n)

    def test_diffusion_unitarity(self):
        """Test diffusion operator is unitary."""
        n = 3
        D = diffusion_operator(n)
        D_dag_D = D.conj().T @ D
        I = np.eye(2**n)
        np.testing.assert_allclose(D_dag_D, I, atol=1e-10)


class TestGroverIteration:
    """Test single Grover iteration."""

    def test_iteration_shape(self):
        """Test Grover iteration has correct shape."""
        n = 3
        marked_states = [5]
        G = grover_iteration(n, marked_states)
        assert G.shape == (2**n, 2**n)

    def test_iteration_unitarity(self):
        """Test Grover iteration is unitary."""
        n = 3
        marked_states = [5]
        G = grover_iteration(n, marked_states)
        G_dag_G = G.conj().T @ G
        I = np.eye(2**n)
        np.testing.assert_allclose(G_dag_G, I, atol=1e-10)


class TestThetaCalculation:
    """Test rotation angle theta calculation."""

    def test_theta_formula(self):
        """Test theta matches expected formula."""
        n = 3
        marked_states = [5]
        N = 2**n
        M = len(marked_states)
        theta = calculate_theta(n, marked_states)
        expected = 2 * np.arcsin(np.sqrt(M / N))
        np.testing.assert_allclose(theta, expected, atol=1e-10)


class TestOptimalIterations:
    """Test optimal iteration calculation."""

    def test_optimal_iterations_formula(self):
        """Test optimal iterations matches formula."""
        n = 3
        marked_states = [5]
        N = 2**n
        M = len(marked_states)
        k_opt = calculate_optimal_iterations(n, marked_states)
        expected = (np.pi / 4) * np.sqrt(N / M) - 0.5
        np.testing.assert_allclose(k_opt, round(expected), atol=1e-10)


class TestGroverSearch:
    """Test full Grover search algorithm."""

    def test_search_shape(self):
        """Test search returns correct state shape."""
        n = 3
        marked_states = [5]
        final_state, k_opt, success_prob = grover_search(n, marked_states)
        assert final_state.shape == (2**n,)
        assert isinstance(k_opt, int)
        assert 0 <= success_prob <= 1

    def test_search_normalized(self):
        """Test final state is normalized."""
        n = 3
        marked_states = [5]
        final_state, _, _ = grover_search(n, marked_states)
        norm = np.linalg.norm(final_state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_search_success_probability(self):
        """Test search amplifies marked state probability."""
        n = 3
        marked_states = [5]
        final_state, _, success_prob = grover_search(n, marked_states)
        # Success probability should be significantly higher than classical 1/N
        N = 2**n
        classical_prob = 1 / N
        assert success_prob > classical_prob

    def test_search_single_marked_state(self):
        """Test search with single marked state."""
        n = 3
        marked_states = [5]
        final_state, k_opt, success_prob = grover_search(n, marked_states)
        # Check that marked state has highest probability
        marked_prob = abs(final_state[5])**2
        for i in range(2**n):
            if i != 5:
                assert marked_prob >= abs(final_state[i])**2

    def test_search_multiple_marked_states(self):
        """Test search with multiple marked states."""
        n = 3
        marked_states = [3, 5, 7]
        final_state, _, success_prob = grover_search(n, marked_states)
        # Check that marked states have higher total probability
        marked_total = sum(abs(final_state[x])**2 for x in marked_states)
        unmarked_total = sum(abs(final_state[x])**2 for x in range(2**n) if x not in marked_states)
        assert marked_total > unmarked_total
