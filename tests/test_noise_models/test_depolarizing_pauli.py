"""Tests for depolarizing and random Pauli noise models."""

import numpy as np
import pytest
from qusim_np.config import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
from qusim_np.noise_models import (
    to_density_matrix, apply_channel,
    depolarizing_kraus, depolarizing,
    random_pauli_kraus, random_pauli, sample_pauli, apply_sampled_pauli,
)

PAULIS = (PAULI_I, PAULI_X, PAULI_Y, PAULI_Z)


def _random_rho(n, seed):
    rng = np.random.default_rng(seed)
    N = 2 ** n
    A = rng.random((N, N)) + 1j * rng.random((N, N))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


class TestDepolarizingKraus:
    """Kraus operators of the depolarizing channel."""

    def test_completeness(self):
        for p in [0.0, 0.3, 0.7, 1.0]:
            ops = depolarizing_kraus(p)
            M = sum(K.conj().T @ K for K in ops)
            np.testing.assert_allclose(M, np.eye(2), atol=1e-12)

    def test_kraus_coefficients(self):
        p = 0.4
        ops = depolarizing_kraus(p)
        expected = [np.sqrt(1 - 3 * p / 4), np.sqrt(p / 4)]
        np.testing.assert_allclose(ops[0], expected[0] * PAULI_I)
        for i, P in enumerate(PAULIS[1:], start=1):
            np.testing.assert_allclose(ops[i], expected[1] * P)

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="p must be"):
            depolarizing_kraus(-0.1)
        with pytest.raises(ValueError, match="p must be"):
            depolarizing_kraus(1.5)


class TestDepolarizingChannel:
    """Depolarizing channel action."""

    def test_matches_definition(self):
        """E(rho) = (1-p) rho + (p/2) I on a random state."""
        rho = _random_rho(1, 0)
        p = 0.3
        out = depolarizing(rho, p)
        expected = (1 - p) * rho + (p / 2) * np.eye(2)
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_p_zero_is_identity(self):
        rho = _random_rho(1, 1)
        np.testing.assert_allclose(depolarizing(rho, 0.0), rho, atol=1e-10)

    def test_p_one_gives_maximally_mixed(self):
        for seed in range(3):
            rho = _random_rho(1, seed)
            out = depolarizing(rho, 1.0)
            np.testing.assert_allclose(out, np.eye(2) / 2, atol=1e-10)

    def test_bloch_shrink(self):
        """|+> coherence shrinks by (1-p)."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = to_density_matrix(psi)
        p = 0.25
        out = depolarizing(rho, p)
        assert out[0, 1] == pytest.approx((1 - p) * 0.5, abs=1e-10)

    def test_unital(self):
        rho = np.eye(2, dtype=complex) / 2
        np.testing.assert_allclose(depolarizing(rho, 0.6), rho, atol=1e-10)

    def test_trace_preservation(self):
        rho = _random_rho(2, 2)
        out = depolarizing(rho, 0.5, qubit=1)
        assert np.trace(out) == pytest.approx(1.0, abs=1e-10)

    def test_equivalent_to_random_pauli_probs(self):
        """Depolarizing == random Pauli channel with probs (1-3p/4, p/4 x3)."""
        rho = _random_rho(1, 3)
        p = 0.4
        probs = (1 - 3 * p / 4, p / 4, p / 4, p / 4)
        np.testing.assert_allclose(
            depolarizing(rho, p), random_pauli(rho, probs), atol=1e-10
        )


class TestRandomPauliKraus:
    """Kraus operators of the random Pauli channel."""

    def test_completeness(self):
        probs = (0.7, 0.1, 0.1, 0.1)
        ops = random_pauli_kraus(probs)
        M = sum(K.conj().T @ K for K in ops)
        np.testing.assert_allclose(M, np.eye(2), atol=1e-12)

    def test_kraus_values(self):
        probs = (0.5, 0.2, 0.2, 0.1)
        ops = random_pauli_kraus(probs)
        for K, q, P in zip(ops, probs, PAULIS):
            np.testing.assert_allclose(K, np.sqrt(q) * P)

    def test_invalid_probs(self):
        with pytest.raises(ValueError, match="shape"):
            random_pauli_kraus((0.5, 0.5))
        with pytest.raises(ValueError, match="sum to 1"):
            random_pauli_kraus((0.5, 0.4, 0.4, 0.4))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            random_pauli_kraus((1.2, -0.1, 0.0, -0.1))


class TestRandomPauliChannel:
    """Random Pauli channel action."""

    def test_matches_mixture_definition(self):
        """E(rho) = sum_i q_i P_i rho P_i on random rho."""
        rho = _random_rho(1, 4)
        probs = (0.6, 0.2, 0.1, 0.1)
        out = random_pauli(rho, probs)
        expected = sum(
            q * (P @ rho @ P.conj().T) for q, P in zip(probs, PAULIS)
        )
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_named_channels(self):
        """Bit flip and phase flip as special cases of random Pauli."""
        rho = _random_rho(1, 5)
        # bit flip: X with prob p
        out_bf = random_pauli(rho, (0.7, 0.3, 0.0, 0.0))
        expected_bf = 0.7 * rho + 0.3 * (PAULI_X @ rho @ PAULI_X)
        np.testing.assert_allclose(out_bf, expected_bf, atol=1e-10)
        # phase flip: Z with prob p
        out_pf = random_pauli(rho, (0.7, 0.0, 0.0, 0.3))
        expected_pf = 0.7 * rho + 0.3 * (PAULI_Z @ rho @ PAULI_Z)
        np.testing.assert_allclose(out_pf, expected_pf, atol=1e-10)

    def test_identity_probs_is_identity(self):
        rho = _random_rho(1, 6)
        np.testing.assert_allclose(
            random_pauli(rho, (1.0, 0.0, 0.0, 0.0)), rho, atol=1e-10
        )

    def test_state_vector_input(self):
        psi = np.array([0, 1], dtype=complex)
        out = random_pauli(psi, (0.5, 0.5, 0.0, 0.0))
        assert out.shape == (2, 2)
        assert np.trace(out) == pytest.approx(1.0, abs=1e-10)


class TestSamplePauli:
    """Shot-level Pauli sampling."""

    def test_deterministic_probs(self):
        rng = np.random.default_rng(42)
        # p=1 on X -> always index 1
        assert sample_pauli((0.0, 1.0, 0.0, 0.0), rng) == 1
        # p=1 on Z -> always index 3
        assert sample_pauli((0.0, 0.0, 0.0, 1.0), rng) == 3

    def test_distribution(self):
        rng = np.random.default_rng(0)
        probs = (0.1, 0.2, 0.3, 0.4)
        counts = np.bincount(
            [sample_pauli(probs, rng) for _ in range(20000)], minlength=4
        )
        freq = counts / counts.sum()
        np.testing.assert_allclose(freq, probs, atol=0.02)

    def test_invalid_probs(self):
        with pytest.raises(ValueError):
            sample_pauli((0.5, 0.5))


class TestApplySampledPauli:
    """Applying a single definite Pauli."""

    def test_bit_flip_action(self):
        """X on |0> gives |1>."""
        psi = np.array([1, 0], dtype=complex)
        rho_in = to_density_matrix(psi)
        out = apply_sampled_pauli(rho_in, "X")
        expected = to_density_matrix(np.array([0, 1], dtype=complex))
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_z_phase(self):
        """Z on |+> flips coherence sign (Z|+> = |->)."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho_in = to_density_matrix(psi)
        out = apply_sampled_pauli(rho_in, 3)  # Z index
        np.testing.assert_allclose(out[0, 1], -0.5, atol=1e-10)

    def test_identity_is_noop(self):
        rho = _random_rho(1, 7)
        np.testing.assert_allclose(
            apply_sampled_pauli(rho, "I"), rho, atol=1e-10
        )

    def test_invalid_pauli(self):
        rho = np.eye(2, dtype=complex) / 2
        with pytest.raises(ValueError, match="pauli"):
            apply_sampled_pauli(rho, "W")
        with pytest.raises(ValueError, match="index"):
            apply_sampled_pauli(rho, 4)

    def test_trace_preserved(self):
        rho = _random_rho(2, 8)
        out = apply_sampled_pauli(rho, "Y", qubit=1)
        assert np.trace(out) == pytest.approx(1.0, abs=1e-10)
