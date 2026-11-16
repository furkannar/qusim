"""Tests for noise models: channels and amplitude damping."""

import warnings

import numpy as np
import pytest
from qusim_np.noise_models import (
    to_density_matrix, apply_channel,
    amplitude_damping_kraus, gamma_from_relaxation, amplitude_damping,
)


class TestToDensityMatrix:
    """Test state vector -> density matrix conversion."""

    def test_pure_state_rho(self):
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = to_density_matrix(psi)
        expected = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho, expected, atol=1e-12)

    def test_purity_one(self):
        rng = np.random.default_rng(0)
        psi = rng.random(4) + 1j * rng.random(4)
        psi /= np.linalg.norm(psi)
        rho = to_density_matrix(psi)
        np.testing.assert_allclose(np.trace(rho @ rho).real, 1.0, atol=1e-10)

    def test_rejects_matrix_input(self):
        with pytest.raises(ValueError, match="1-D"):
            to_density_matrix(np.eye(2))


class TestAmplitudeDampingKraus:
    """Test Kraus operator construction."""

    def test_shapes(self):
        K0, K1 = amplitude_damping_kraus(0.3)
        assert K0.shape == (2, 2)
        assert K1.shape == (2, 2)

    def test_kraus_completeness(self):
        """K0^dagger K0 + K1^dagger K1 = I (trace-preserving)."""
        for gamma in [0.0, 0.25, 0.5, 0.9, 1.0]:
            K0, K1 = amplitude_damping_kraus(gamma)
            M = K0.conj().T @ K0 + K1.conj().T @ K1
            np.testing.assert_allclose(M, np.eye(2), atol=1e-12)

    def test_k0_values(self):
        gamma = 0.36
        K0, _ = amplitude_damping_kraus(gamma)
        np.testing.assert_allclose(K0, np.diag([1.0, np.sqrt(1 - gamma)]))

    def test_k1_values(self):
        gamma = 0.36
        _, K1 = amplitude_damping_kraus(gamma)
        expected = np.array([[0, np.sqrt(gamma)], [0, 0]])
        np.testing.assert_allclose(K1, expected)

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            amplitude_damping_kraus(-0.1)
        with pytest.raises(ValueError, match="gamma"):
            amplitude_damping_kraus(1.5)


class TestGammaFromRelaxation:
    """Test gamma = 1 - exp(-t/T1)."""

    def test_formula(self):
        g = gamma_from_relaxation(1.0, 2.0)
        assert g == pytest.approx(1.0 - np.exp(-0.5))

    def test_long_time_saturates(self):
        g = gamma_from_relaxation(100.0, 1.0)
        assert g == pytest.approx(1.0, abs=1e-30)

    def test_zero_time(self):
        # t -> 0 gives gamma -> 0; t=0 itself is rejected, use tiny t
        g = gamma_from_relaxation(1e-12, 1.0)
        assert g == pytest.approx(0.0, abs=1e-12)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="t must be positive"):
            gamma_from_relaxation(0.0, 1.0)
        with pytest.raises(ValueError, match="T1 must be positive"):
            gamma_from_relaxation(1.0, 0.0)
        with pytest.raises(ValueError, match="T1 must be positive"):
            gamma_from_relaxation(1.0, -1.0)


class TestApplyChannel:
    """Test generic channel application."""

    def test_trace_preserved(self):
        rng = np.random.default_rng(1)
        rho = rng.random((4, 4)) + 1j * rng.random((4, 4))
        rho = rho @ rho.conj().T  # positive semidefinite
        rho /= np.trace(rho)
        K0, K1 = amplitude_damping_kraus(0.4)
        out = apply_channel(rho, [K0, K1], qubit=1)
        assert np.trace(out) == pytest.approx(1.0, abs=1e-10)

    def test_non_tp_channel_raises_and_warns(self):
        """A lone K1 is not trace-preserving -> warning + ValueError."""
        _, K1 = amplitude_damping_kraus(0.5)
        rho = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        with pytest.warns(UserWarning, match="not trace-preserving"):
            with pytest.raises(ValueError, match="not trace-preserving"):
                apply_channel(rho, [K1], qubit=0)

    def test_state_vector_input(self):
        psi = np.array([0, 1], dtype=complex)  # |1>
        K0, K1 = amplitude_damping_kraus(0.5)
        out = apply_channel(psi, [K0, K1], qubit=0)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(np.diag(out), [0.5, 0.5], atol=1e-12)

    def test_invalid_qubit(self):
        rho = np.eye(4, dtype=complex) / 4
        K0, _ = amplitude_damping_kraus(0.5)
        with pytest.raises(ValueError, match="qubit"):
            apply_channel(rho, [K0], qubit=2)

    def test_bad_rho_shape(self):
        K0, _ = amplitude_damping_kraus(0.5)
        with pytest.raises(ValueError, match="square"):
            apply_channel(np.ones((2, 3)), [K0])
        with pytest.raises(ValueError, match="power of 2"):
            apply_channel(np.eye(3, dtype=complex), [K0])

    def test_empty_kraus_ops(self):
        rho = np.eye(2, dtype=complex) / 2
        with pytest.raises(ValueError, match="at least one"):
            apply_channel(rho, [])


class TestAmplitudeDamping:
    """Test the amplitude damping channel end to end."""

    def test_single_qubit_closed_form(self):
        """E(rho) matches the analytic formula for random rho."""
        rng = np.random.default_rng(2)
        A = rng.random((2, 2)) + 1j * rng.random((2, 2))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        gamma = 0.3
        s = np.sqrt(1 - gamma)
        p00, p01, p10, p11 = rho[0, 0], rho[0, 1], rho[1, 0], rho[1, 1]
        expected = np.array([
            [p00 + gamma * p11, s * p01],
            [s * p10, (1 - gamma) * p11],
        ])
        out = amplitude_damping(rho, gamma, qubit=0)
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_excited_state_decays(self):
        """|1><1| -> (1-gamma)|1><1| + gamma|0><0|."""
        rho = np.diag([0.0, 1.0]).astype(complex)
        out = amplitude_damping(rho, 0.4)
        np.testing.assert_allclose(np.diag(out), [0.4, 0.6], atol=1e-12)

    def test_coherence_decay(self):
        """Off-diagonals shrink by sqrt(1-gamma)."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = to_density_matrix(psi)
        gamma = 0.5
        out = amplitude_damping(rho, gamma)
        assert out[0, 1] == pytest.approx(np.sqrt(1 - gamma) * 0.5, abs=1e-10)

    def test_state_vector_input_mixed_output(self):
        """Damping |1> gives a mixed state: purity < 1."""
        psi = np.array([0, 1], dtype=complex)
        out = amplitude_damping(psi, 0.5)
        purity = np.trace(out @ out).real
        assert purity == pytest.approx(0.5, abs=1e-10)  # maximally mixed: 0.5^2 + 0.5^2

    def test_zero_gamma_identity(self):
        rng = np.random.default_rng(3)
        rho = rng.random((2, 2)) + 1j * rng.random((2, 2))
        rho = (rho + rho.conj().T) / 2
        rho /= np.trace(rho)
        out = amplitude_damping(rho, 0.0)
        np.testing.assert_allclose(out, rho, atol=1e-10)

    def test_two_qubit_qubit_selection(self):
        """Damping qubit 1 of |01> equals kron(I, K) action."""
        # rho = |01><01| on 2 qubits (index 1)
        psi = np.array([0, 1, 0, 0], dtype=complex)
        rho = to_density_matrix(psi)
        gamma = 0.3
        out = amplitude_damping(rho, gamma, qubit=1)

        # Reference: embed K on qubit 1 -> kron(I, K); qubit 0 -> kron(K, I)
        K0, K1 = amplitude_damping_kraus(gamma)
        I2 = np.eye(2)
        expected = np.zeros((4, 4), dtype=complex)
        for K in (K0, K1):
            A = np.kron(I2, K)  # qubit 1 is the second tensor factor
            expected += A @ rho @ A.conj().T
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_damping_qubit0_vs_qubit1_differ(self):
        """Product state |10>: damping qubit 0 hits the excitation."""
        psi = np.array([0, 0, 1, 0], dtype=complex)  # |10>
        rho = to_density_matrix(psi)
        gamma = 0.5
        out_q0 = amplitude_damping(rho, gamma, qubit=0)
        out_q1 = amplitude_damping(rho, gamma, qubit=1)
        # |10>: qubit0=1 decays -> population moves to |00> (index 0)
        assert out_q0[0, 0] == pytest.approx(0.5, abs=1e-10)
        # qubit1=0 unaffected -> stays |10>
        np.testing.assert_allclose(out_q1, rho, atol=1e-10)

    def test_output_is_density_matrix(self):
        """Output has unit trace and is Hermitian."""
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = to_density_matrix(psi)
        out = amplitude_damping(rho, 0.4, qubit=0)
        assert np.trace(out) == pytest.approx(1.0, abs=1e-10)
        np.testing.assert_allclose(out, out.conj().T, atol=1e-10)
