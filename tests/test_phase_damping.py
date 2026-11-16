"""Tests for the phase damping noise model."""

import numpy as np
import pytest
from qusim_np.noise_models import (
    to_density_matrix, apply_channel,
    amplitude_damping, amplitude_damping_kraus,
    phase_damping_kraus, phase_flip_kraus,
    lambda_from_p, p_from_lambda, phase_damping_from_times,
    phase_damping, phase_flip, thermal_relaxation,
)


def _random_rho(n, seed):
    """Random n-qubit density matrix (via A A^dagger)."""
    rng = np.random.default_rng(seed)
    N = 2 ** n
    A = rng.random((N, N)) + 1j * rng.random((N, N))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def _choi_eigenvalues(channel_fn):
    """Eigenvalues of the Choi matrix of a single-qubit channel."""
    basis = [np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]),
             np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]])]
    J = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            E = channel_fn(basis[2 * i + j])
            J += np.kron(E, basis[2 * i + j])
    return np.linalg.eigvalsh(J)


class TestPhaseDampingKraus:
    """Kraus completeness and values."""

    @pytest.mark.parametrize("lam", [0.0, 0.1, 0.5, 1.0])
    def test_completeness(self, lam):
        K0, K1 = phase_damping_kraus(lam)
        M = K0.conj().T @ K0 + K1.conj().T @ K1
        np.testing.assert_allclose(M, np.eye(2), atol=1e-12)

    def test_kraus_values(self):
        lam = 0.36
        K0, K1 = phase_damping_kraus(lam)
        np.testing.assert_allclose(K0, np.diag([1.0, np.sqrt(1 - lam)]))
        np.testing.assert_allclose(K1, np.diag([0.0, np.sqrt(lam)]))

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda"):
            phase_damping_kraus(-0.1)
        with pytest.raises(ValueError, match="lambda"):
            phase_damping_kraus(1.5)


class TestPhaseFlipKraus:
    """Pauli-form Kraus operators."""

    def test_completeness(self):
        for p in [0.0, 0.2, 0.5]:
            K0, K1 = phase_flip_kraus(p)
            M = K0.conj().T @ K0 + K1.conj().T @ K1
            np.testing.assert_allclose(M, np.eye(2), atol=1e-12)

    def test_values(self):
        p = 0.2
        K0, K1 = phase_flip_kraus(p)
        np.testing.assert_allclose(K0, np.sqrt(1 - p) * np.eye(2))
        np.testing.assert_allclose(K1, np.sqrt(p) * np.diag([1, -1]))


class TestParameterConversion:
    """lambda <-> p converters."""

    def test_round_trip(self):
        for lam in [0.0, 0.1, 0.5, 1.0]:
            p = p_from_lambda(lam)
            assert lambda_from_p(p) == pytest.approx(lam)

    def test_p_from_lambda_formula(self):
        lam = 0.36
        p = p_from_lambda(lam)
        assert p == pytest.approx((1 - np.sqrt(1 - lam)) / 2)
        assert lambda_from_p(p) == pytest.approx(lam)

    def test_lambda_from_p_formula(self):
        p = 0.1
        lam = lambda_from_p(p)
        assert lam == pytest.approx(1 - (1 - 2 * p) ** 2)
        assert p_from_lambda(lam) == pytest.approx(p)

    def test_p_above_half_rejected(self):
        with pytest.raises(ValueError, match="1/2"):
            lambda_from_p(0.75)

    def test_worked_example(self):
        """T_phi=50us, t=1us -> lam ~ 0.0392, p ~ 0.0099."""
        lam = phase_damping_from_times(1.0, 50.0)
        p = p_from_lambda(lam)
        assert lam == pytest.approx(0.0392, abs=1e-4)
        assert p == pytest.approx(0.0099, abs=1e-4)

    def test_from_times_invalid(self):
        with pytest.raises(ValueError, match="t must be positive"):
            phase_damping_from_times(0.0, 50.0)
        with pytest.raises(ValueError, match="T_phi"):
            phase_damping_from_times(1.0, 0.0)


class TestPhaseDampingChannel:
    """Channel action on density matrices."""

    def test_trace_preservation(self):
        for seed in range(3):
            rho = _random_rho(1, seed)
            out = phase_damping(rho, 0.4)
            assert np.trace(out) == pytest.approx(1.0, abs=1e-10)

    def test_populations_unchanged(self):
        rho = _random_rho(1, 10)
        diag_before = np.diag(rho).copy()
        out = phase_damping(rho, 0.6)
        np.testing.assert_allclose(np.diag(out), diag_before, atol=1e-10)

    def test_coherence_factor(self):
        rho = _random_rho(1, 11)
        lam = 0.5
        out = phase_damping(rho, lam)
        s = np.sqrt(1 - lam)
        assert out[0, 1] == pytest.approx(s * rho[0, 1], abs=1e-10)
        assert out[1, 0] == pytest.approx(s * rho[1, 0], abs=1e-10)

    def test_unital(self):
        """E(I/2) = I/2 for phase damping."""
        rho = np.eye(2, dtype=complex) / 2
        out = phase_damping(rho, 0.7)
        np.testing.assert_allclose(out, rho, atol=1e-10)

    def test_not_unital_amplitude_damping(self):
        """Sanity: amplitude damping is non-unital (contrast)."""
        rho = np.eye(2, dtype=complex) / 2
        out = amplitude_damping(rho, 0.5)
        assert not np.allclose(out, rho, atol=1e-6)

    def test_complete_positivity(self):
        for lam in [0.0, 0.3, 0.7, 1.0]:
            eigs = _choi_eigenvalues(lambda r: phase_damping(r, lam))
            assert eigs.min() >= -1e-10

    def test_kraus_vs_pauli_form_equivalence(self):
        """Kraus lam and Pauli p give identical output on random rho."""
        rho = _random_rho(1, 12)
        lam = 0.4
        p = p_from_lambda(lam)
        out_kraus = phase_damping(rho, lam)
        out_pauli = phase_flip(rho, p)
        np.testing.assert_allclose(out_kraus, out_pauli, atol=1e-10)

    def test_composition_law(self):
        """PD(lam1) o PD(lam2) = PD(lam_eff), (1-lam_eff)=(1-lam1)(1-lam2)."""
        rho = _random_rho(1, 13)
        lam1, lam2 = 0.3, 0.5
        out_seq = phase_damping(phase_damping(rho, lam1), lam2)
        lam_eff = 1 - (1 - lam1) * (1 - lam2)
        out_eff = phase_damping(rho, lam_eff)
        np.testing.assert_allclose(out_seq, out_eff, atol=1e-10)

    def test_limits(self):
        rho = _random_rho(1, 14)
        # lam = 0 -> identity
        np.testing.assert_allclose(phase_damping(rho, 0.0), rho, atol=1e-10)
        # lam = 1 -> full dephasing: diagonal part only
        out = phase_damping(rho, 1.0)
        np.testing.assert_allclose(out, np.diag(np.diag(rho)), atol=1e-10)

    def test_plus_state(self):
        """|+> -> 0.5 [[1, sqrt(1-lam)], [sqrt(1-lam), 1]]."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = to_density_matrix(psi)
        lam = 0.36
        out = phase_damping(rho, lam)
        s = np.sqrt(1 - lam)
        expected = 0.5 * np.array([[1, s], [s, 1]])
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_phase_flip_shrinks_x(self):
        """<X> after phase flip = (1-2p) <X>_before."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)  # <X>=1
        rho = to_density_matrix(psi)
        p = 0.2
        out = phase_flip(rho, p)
        exp_x = np.trace(out @ X).real
        assert exp_x == pytest.approx(1 - 2 * p, abs=1e-10)

    def test_multi_qubit_qubit_selection(self):
        """Phase damping on qubit 1 of a 2-qubit product state."""
        # rho = |+>_0 (x) |0>_1
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        psi = np.kron(plus, np.array([1, 0], dtype=complex))
        rho = to_density_matrix(psi)
        lam = 0.5
        out = phase_damping(rho, lam, qubit=0)

        # Reference: coherence between states differing in qubit 0 shrinks
        s = np.sqrt(1 - lam)
        # rho has coherences 0.5 between |00>-|10> and |01>-|11> pairs
        np.testing.assert_allclose(out[0, 2], s * rho[0, 2], atol=1e-10)
        np.testing.assert_allclose(out[0, 3], s * rho[0, 3], atol=1e-10)


class TestThermalRelaxation:
    """Combined amplitude + phase damping channel."""

    def test_coherence_decays_as_t2(self):
        """|+> coherence after thermal relaxation = 0.5 exp(-t/T2)."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = to_density_matrix(psi)
        t, T1, T2 = 1.0, 10.0, 5.0
        out = thermal_relaxation(rho, t, T1, T2)
        assert out[0, 1] == pytest.approx(0.5 * np.exp(-t / T2), abs=1e-10)

    def test_population_decays_as_t1(self):
        """Excited population survives as exp(-t/T1)."""
        rho = np.diag([0.0, 1.0]).astype(complex)  # |1><1|
        t, T1, T2 = 2.0, 10.0, 5.0
        out = thermal_relaxation(rho, t, T1, T2)
        assert out[1, 1] == pytest.approx(np.exp(-t / T1), abs=1e-10)
        assert out[0, 0] == pytest.approx(1 - np.exp(-t / T1), abs=1e-10)

    def test_t2_equals_2t1_t1_limited(self):
        """T2 = 2T1 -> no pure dephasing; only amplitude damping acts."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = to_density_matrix(psi)
        t, T1 = 1.0, 10.0
        out_tr = thermal_relaxation(rho, t, T1, 2 * T1)
        gamma = 1 - np.exp(-t / T1)
        out_ad = amplitude_damping(rho, gamma)
        np.testing.assert_allclose(out_tr, out_ad, atol=1e-10)

    def test_unphysical_t2_raises(self):
        rho = np.eye(2, dtype=complex) / 2
        with pytest.raises(ValueError, match="T2 <= 2"):
            thermal_relaxation(rho, 1.0, 10.0, 25.0)

    def test_invalid_times(self):
        rho = np.eye(2, dtype=complex) / 2
        with pytest.raises(ValueError, match="t must be positive"):
            thermal_relaxation(rho, 0.0, 10.0, 5.0)
        with pytest.raises(ValueError, match="T1"):
            thermal_relaxation(rho, 1.0, 0.0, 5.0)
        with pytest.raises(ValueError, match="T2"):
            thermal_relaxation(rho, 1.0, 10.0, 0.0)
