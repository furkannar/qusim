"""Tests for Deutsch-Jozsa algorithm implementation."""

import numpy as np
import pytest
from qusim.deutsch_jozsa import (
    deutsch_jozsa_oracle, deutsch_jozsa, deutsch_jozsa_classify, _evaluate_f
)


class TestEvaluateF:
    """Test function evaluation helper."""

    def test_evaluate_callable(self):
        """Test evaluating a callable function."""
        N = 8
        f = lambda x: x % 2
        result = _evaluate_f(f, N)
        expected = [int(x % 2) for x in range(N)]
        assert result == expected

    def test_evaluate_sequence(self):
        """Test evaluating a sequence of bits."""
        N = 8
        f = [0, 1, 0, 1, 0, 1, 0, 1]
        result = _evaluate_f(f, N)
        expected = [0, 1, 0, 1, 0, 1, 0, 1]
        assert result == expected

    def test_evaluate_sequence_wrong_length(self):
        """Test sequence with wrong length raises error."""
        N = 8
        f = [0, 1, 0]
        with pytest.raises(ValueError, match="must have length"):
            _evaluate_f(f, N)

    def test_evaluate_bool_conversion(self):
        """Test boolean values are converted to 0/1."""
        N = 4
        f = [True, False, True, False]
        result = _evaluate_f(f, N)
        expected = [1, 0, 1, 0]
        assert result == expected


class TestDeutschJozsaOracle:
    """Test Deutsch-Jozsa oracle construction."""

    def test_oracle_shape(self):
        """Test oracle has correct shape."""
        n = 3
        f = lambda x: x % 2
        O = deutsch_jozsa_oracle(n, f)
        assert O.shape == (2**n, 2**n)

    def test_oracle_unitarity(self):
        """Test oracle is unitary."""
        n = 3
        f = lambda x: x % 2
        O = deutsch_jozsa_oracle(n, f)
        O_dag_O = O.conj().T @ O
        I = np.eye(2**n)
        np.testing.assert_allclose(O_dag_O, I, atol=1e-10)

    def test_oracle_phase_kickback(self):
        """Test oracle applies phase kickback correctly."""
        n = 3
        f = lambda x: 1 if x == 5 else 0
        O = deutsch_jozsa_oracle(n, f)
        # Check that f(x)=1 gets -1 phase
        assert np.isclose(O[5, 5], -1)
        # Check that f(x)=0 keeps +1 phase
        for i in range(2**n):
            if i != 5:
                assert np.isclose(O[i, i], 1)


class TestDeutschJozsaAlgorithm:
    """Test Deutsch-Jozsa algorithm."""

    def test_algorithm_shape(self):
        """Test algorithm returns correct state shape."""
        n = 3
        f = lambda x: x % 2
        final_state, zero_prob = deutsch_jozsa(n, f)
        assert final_state.shape == (2**n,)
        assert 0 <= zero_prob <= 1

    def test_algorithm_normalized(self):
        """Test final state is normalized."""
        n = 3
        f = lambda x: x % 2
        final_state, _ = deutsch_jozsa(n, f)
        norm = np.linalg.norm(final_state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_constant_zero(self):
        """Test constant function f(x) = 0."""
        n = 3
        f = lambda x: 0
        final_state, zero_prob = deutsch_jozsa(n, f)
        # For constant function, P(|0^n>) should be ~1
        np.testing.assert_allclose(zero_prob, 1.0, atol=1e-10)

    def test_constant_one(self):
        """Test constant function f(x) = 1."""
        n = 3
        f = lambda x: 1
        final_state, zero_prob = deutsch_jozsa(n, f)
        # For constant function, P(|0^n>) should be ~1
        np.testing.assert_allclose(zero_prob, 1.0, atol=1e-10)

    def test_balanced_parity(self):
        """Test balanced function f(x) = parity(x)."""
        n = 3
        f = lambda x: bin(x).count('1') % 2
        final_state, zero_prob = deutsch_jozsa(n, f)
        # For balanced function, P(|0^n>) should be ~0
        np.testing.assert_allclose(zero_prob, 0.0, atol=1e-10)

    def test_balanced_threshold(self):
        """Test balanced function f(x) = [x >= N/2]."""
        n = 3
        N = 2**n
        f = lambda x: int(x >= N // 2)
        final_state, zero_prob = deutsch_jozsa(n, f)
        # For balanced function, P(|0^n>) should be ~0
        np.testing.assert_allclose(zero_prob, 0.0, atol=1e-10)

    def test_balanced_sequence(self):
        """Test balanced function given as bit sequence."""
        n = 3
        N = 2**n
        f = [x % 2 for x in range(N)]
        final_state, zero_prob = deutsch_jozsa(n, f)
        # For balanced function, P(|0^n>) should be ~0
        np.testing.assert_allclose(zero_prob, 0.0, atol=1e-10)


class TestDeutschJozsaClassify:
    """Test Deutsch-Jozsa classification."""

    def test_classify_constant_zero(self):
        """Test classification of constant f(x) = 0."""
        n = 3
        f = lambda x: 0
        result = deutsch_jozsa_classify(n, f)
        assert result == "constant"

    def test_classify_constant_one(self):
        """Test classification of constant f(x) = 1."""
        n = 3
        f = lambda x: 1
        result = deutsch_jozsa_classify(n, f)
        assert result == "constant"

    def test_classify_balanced_parity(self):
        """Test classification of balanced f(x) = parity(x)."""
        n = 3
        f = lambda x: bin(x).count('1') % 2
        result = deutsch_jozsa_classify(n, f)
        assert result == "balanced"

    def test_classify_balanced_threshold(self):
        """Test classification of balanced f(x) = [x >= N/2]."""
        n = 3
        N = 2**n
        f = lambda x: int(x >= N // 2)
        result = deutsch_jozsa_classify(n, f)
        assert result == "balanced"

    def test_classify_custom_threshold(self):
        """Test classification with custom threshold."""
        n = 3
        f = lambda x: 0
        result = deutsch_jozsa_classify(n, f, threshold=0.9)
        assert result == "constant"

    def test_classify_sequence_input(self):
        """Test classification with sequence input."""
        n = 3
        N = 2**n
        f = [x % 2 for x in range(N)]
        result = deutsch_jozsa_classify(n, f)
        assert result == "balanced"
