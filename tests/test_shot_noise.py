"""Tests for shot-noise estimation, bounds, and sampling (spec sections 1-5)."""

import warnings

import numpy as np
import pytest

from qusim_np.config import PAULI_X, PAULI_Z, config
from qusim_np.shot_noise import (
    estimate_expval, observable_expval, sample_observable_outcomes,
    pauli_expval_from_counts, pauli_outcome_probs,
    shots_needed_pauli_std_error,
    gaussian_z, shots_clt, shots_hoeffding, shots_bernstein,
    hamiltonian_variance, uniform_allocation_variance,
    optimal_allocation_variance, allocate_shots, shots_for_hamiltonian,
    pauli_covariance, group_shot_values, group_variance,
    covariance_matrix_from_samples, allocate_group_shots,
    measurement_probabilities, sample_bitstrings, estimate_distribution,
)


class TestEstimateExpval:
    """Section 1: sample mean and standard error."""

    def test_mean_and_stderr(self):
        outcomes = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=float)
        mean, stderr = estimate_expval(outcomes)
        assert mean == pytest.approx(np.mean(outcomes))
        expected = np.std(outcomes, ddof=1) / np.sqrt(8)
        assert stderr == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one shot"):
            estimate_expval([])

    def test_single_shot_warns_nan(self):
        with pytest.warns(RuntimeWarning, match="undefined"):
            mean, stderr = estimate_expval([1.0])
        assert mean == 1.0
        assert np.isnan(stderr)

    def test_stderr_scales_as_inverse_sqrt_n(self):
        """stderr ~ sigma/sqrt(N): quadrupling shots halves the error."""
        rng = np.random.default_rng(0)
        base = rng.choice([-1.0, 1.0], size=4000)
        _, s1 = estimate_expval(base[:1000])
        _, s4 = estimate_expval(base)
        assert s4 == pytest.approx(s1 / 2, rel=0.15)


class TestObservableExpval:
    """Exact <psi|O|psi> and Tr(rho O) ground truth."""

    def test_state_vector(self):
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert observable_expval(psi, PAULI_X) == pytest.approx(1.0)
        assert observable_expval(psi, PAULI_Z) == pytest.approx(0.0)

    def test_density_matrix(self):
        rho = np.diag([0.25, 0.75]).astype(complex)
        assert observable_expval(rho, PAULI_Z) == pytest.approx(-0.5)

    def test_matches_vqe_expectation_value(self):
        from qusim_np import expectation_value
        psi = np.array([0.6, 0.8], dtype=complex)
        O = np.array([[1, 0.2j], [-0.2j, -1]], dtype=complex)
        assert observable_expval(psi, O) == pytest.approx(
            expectation_value(psi, O)
        )


class TestSampleObservableOutcomes:
    """Projective sampling in the observable's eigenbasis."""

    def test_pauli_z_on_zero_state(self):
        """|0> measured in Z basis always gives eigenvalue +1."""
        psi = np.array([1, 0], dtype=complex)
        rng = np.random.default_rng(0)
        out = sample_observable_outcomes(psi, PAULI_Z, 100, rng=rng)
        assert np.all(out == 1.0)

    def test_mean_converges_to_expval(self):
        """Sample mean of Z outcomes on |+> approaches <Z> = 0."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rng = np.random.default_rng(1)
        out = sample_observable_outcomes(psi, PAULI_Z, 20000, rng=rng)
        mean, stderr = estimate_expval(out)
        assert mean == pytest.approx(0.0, abs=0.02)
        assert stderr < 0.02

    def test_density_matrix_input(self):
        rho = np.diag([0.5, 0.5]).astype(complex)  # maximally mixed
        rng = np.random.default_rng(2)
        out = sample_observable_outcomes(rho, PAULI_X, 5000, rng=rng)
        mean, _ = estimate_expval(out)
        assert mean == pytest.approx(0.0, abs=0.05)

    def test_invalid_shots(self):
        psi = np.array([1, 0], dtype=complex)
        with pytest.raises(ValueError, match="shots"):
            sample_observable_outcomes(psi, PAULI_Z, 0)


class TestPauliStats:
    """Section 2: Pauli outcome statistics."""

    def test_p_hat_formula(self):
        p_hat, _ = pauli_expval_from_counts(75, 25)
        assert p_hat == pytest.approx(0.5)

    def test_stderr_formula(self):
        p_hat, stderr = pauli_expval_from_counts(60, 40)
        n = 100
        expected = np.sqrt((1 - p_hat ** 2) / n)
        assert stderr == pytest.approx(expected)

    def test_unanimous_counts_warn_and_floor(self):
        with pytest.warns(RuntimeWarning, match="conservative"):
            p_hat, stderr = pauli_expval_from_counts(100, 0)
        assert p_hat == 1.0
        assert stderr == pytest.approx(1 / np.sqrt(100))

    def test_zero_shots_raises(self):
        with pytest.raises(ValueError, match="positive"):
            pauli_expval_from_counts(0, 0)

    def test_outcome_probs(self):
        p_plus, p_minus = pauli_outcome_probs(0.6)
        assert p_plus == pytest.approx(0.8)
        assert p_minus == pytest.approx(0.2)

    def test_outcome_probs_invalid(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            pauli_outcome_probs(1.5)

    def test_shots_needed_std_error(self):
        """<P>=0 -> sigma2=1 -> N = 1/eps^2; <P>=1 -> N = 0."""
        assert shots_needed_pauli_std_error(0.0, 0.1) == 100
        assert shots_needed_pauli_std_error(1.0, 0.1) == 0
        # <P>=0.6 -> sigma2 = 1-0.36 = 0.64
        assert shots_needed_pauli_std_error(0.6, 0.1) == 64


class TestGaussianZ:
    """z quantiles from the spec table."""

    @pytest.mark.parametrize("delta, approx", [
        (0.10, 1.64), (0.05, 1.96), (0.01, 2.58),
    ])
    def test_spec_values(self, delta, approx):
        assert gaussian_z(delta) == pytest.approx(approx, abs=0.01)

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            gaussian_z(0.0)
        with pytest.raises(ValueError, match="delta"):
            gaussian_z(1.5)


class TestShotBounds:
    """Section 3: CLT / Hoeffding / Bernstein bounds + test vectors."""

    def test_clt_spec_vector(self):
        """eps=0.01, delta=0.05, sigma=1 -> ~38,416 shots (spec ~3.8e4).

        The spec's 38,416 uses z = 1.96; the exact quantile
        norm.ppf(0.975) = 1.95996 gives 38,415 - assert the range.
        """
        n = shots_clt(1.0, 0.01, delta=0.05)
        assert 38400 <= n <= 38420

    def test_clt_standard_error_mode(self):
        """delta=None -> N = sigma2/eps^2 (no z^2 factor)."""
        assert shots_clt(1.0, 0.1) == 100
        assert shots_clt(0.25, 0.5) == 1

    def test_hoeffding_spec_vector(self):
        """eps=0.01, delta=0.05, R=2 -> ~73,778 shots."""
        n = shots_hoeffding(0.01, 0.05)
        assert n == 73778

    def test_hoeffding_general_range(self):
        """N scales as R^2."""
        n2 = shots_hoeffding(0.01, 0.05, outcome_range=2.0)
        n4 = shots_hoeffding(0.01, 0.05, outcome_range=4.0)
        assert n4 == pytest.approx(4 * n2, abs=1)

    def test_bernstein_small_variance_beats_hoeffding(self):
        """Small sigma2 -> Bernstein needs fewer shots than Hoeffding."""
        n_b = shots_bernstein(0.01, 0.01, 0.05)
        n_h = shots_hoeffding(0.01, 0.05)
        assert n_b < n_h

    def test_bernstein_worst_case_near_hoeffding(self):
        """sigma2=1, b=2 -> Bernstein ~ Hoeffding (within ~5%)."""
        n_b = shots_bernstein(1.0, 0.01, 0.05)
        n_h = shots_hoeffding(0.01, 0.05)
        assert n_b == pytest.approx(n_h, rel=0.05)

    @pytest.mark.parametrize("fn,args", [
        (shots_clt, (1.0, -0.01)),
        (shots_clt, (1.0, 0.01, 0.0)),
        (shots_clt, (-1.0, 0.01)),
        (shots_hoeffding, (0.0, 0.05)),
        (shots_hoeffding, (0.01, 1.0)),
        (shots_hoeffding, (0.01, 0.05, -2.0)),
        (shots_bernstein, (-0.5, 0.01, 0.05)),
        (shots_bernstein, (1.0, 0.01, 0.05, 0.0)),
    ])
    def test_validation(self, fn, args):
        with pytest.raises(ValueError):
            fn(*args)

    def test_clt_small_n_warns(self):
        with pytest.warns(RuntimeWarning, match="small"):
            shots_clt(1.0, 1.0)  # N = 1 -> warn

    def test_monotone_in_inverse_eps_squared(self):
        """Halving eps quadruples the shot count."""
        n1 = shots_hoeffding(0.02, 0.05)
        n2 = shots_hoeffding(0.01, 0.05)
        assert n2 == pytest.approx(4 * n1, rel=1e-3)


class TestHamiltonianShots:
    """Section 4: Hamiltonian shot allocation."""

    COEFFS = [1.2, -0.8, 0.5]
    SIGMAS = [1.0, 0.5, 0.25]

    def test_uniform_variance_formula(self):
        m, n = 3, 900
        var = uniform_allocation_variance(self.COEFFS, self.SIGMAS, n)
        expected = m / n * sum(
            c * c * s * s for c, s in zip(self.COEFFS, self.SIGMAS)
        )
        assert var == pytest.approx(expected)

    def test_optimal_variance_formula(self):
        n = 900
        var = optimal_allocation_variance(self.COEFFS, self.SIGMAS, n)
        s = sum(abs(c) * sg for c, sg in zip(self.COEFFS, self.SIGMAS))
        assert var == pytest.approx(s ** 2 / n)

    def test_optimal_never_worse_than_uniform(self):
        """Cauchy-Schwarz: optimal <= uniform for same total shots."""
        for n in [300, 900, 5000]:
            v_opt = optimal_allocation_variance(self.COEFFS, self.SIGMAS, n)
            v_uni = uniform_allocation_variance(self.COEFFS, self.SIGMAS, n)
            assert v_opt <= v_uni + 1e-12

    def test_allocate_shots_sums_to_total(self):
        alloc = allocate_shots(self.COEFFS, self.SIGMAS, 1000)
        assert sum(alloc) == 1000
        assert len(alloc) == 3

    def test_allocate_shots_proportional(self):
        alloc = allocate_shots([1.0, 1.0], [1.0, 1.0], 100)
        assert alloc == [50, 50]

    def test_allocate_shots_min_one_and_zero_coeff(self):
        alloc = allocate_shots([1.0, 0.0, 2.0], None, 30)
        assert alloc[1] == 0          # zero coefficient -> 0 shots
        assert alloc[0] >= 1 and alloc[2] >= 1
        assert sum(alloc) == 30

    def test_allocate_shots_too_few_raises(self):
        with pytest.raises(ValueError, match="cannot give"):
            allocate_shots([1.0, 1.0, 1.0], None, 2)

    def test_hamiltonian_variance_matches_allocation(self):
        alloc = allocate_shots(self.COEFFS, self.SIGMAS, 900)
        var = hamiltonian_variance(self.COEFFS, self.SIGMAS, alloc)
        assert var == pytest.approx(
            optimal_allocation_variance(self.COEFFS, self.SIGMAS, 900),
            rel=0.05,
        )

    def test_shots_for_hamiltonian(self):
        """N = (sum |c_i| sigma_i)^2 / eps^2."""
        coeffs = [1.0, 1.0]
        sigmas = [1.0, 1.0]
        n = shots_for_hamiltonian(coeffs, sigmas, eps=0.1)
        assert n == int(np.ceil(4.0 / 0.01))

    def test_shots_for_hamiltonian_confidence(self):
        n_se = shots_for_hamiltonian([1.0], eps=0.1)
        n_cf = shots_for_hamiltonian([1.0], eps=0.1, delta=0.05)
        assert n_cf == int(np.ceil(gaussian_z(0.05) ** 2 * n_se))

    def test_shots_for_hamiltonian_validation(self):
        with pytest.raises(ValueError, match="eps"):
            shots_for_hamiltonian([1.0], eps=0.0)
        with pytest.raises(ValueError, match="delta"):
            shots_for_hamiltonian([1.0], eps=0.1, delta=1.5)


class TestGroupedMeasurements:
    """Section 5: grouped commuting measurements."""

    def test_pauli_covariance_formula(self):
        # <Z>=0.5, <X>=0.5 on some state; <ZX> product expectation
        cov = pauli_covariance(0.25, 0.5, 0.5)
        assert cov == pytest.approx(0.25 - 0.25)

    def test_group_shot_values(self):
        coeffs = [1.0, -0.5]
        outcomes = np.array([[1, 1], [1, -1], [-1, 1]], dtype=float)
        h = group_shot_values(coeffs, outcomes)
        np.testing.assert_allclose(h, [0.5, 1.5, -1.5])

    def test_group_variance_includes_covariance(self):
        """Perfectly correlated outcomes -> variance of sum, not sum of vars."""
        coeffs = [1.0, 1.0]
        # o1 == o2 on every shot (perfectly correlated)
        o = np.random.default_rng(0).choice([-1.0, 1.0], size=(2000, 1))
        outcomes = np.hstack([o, o])
        var_h, stderr = group_variance(coeffs, outcomes)
        # h = 2*o -> var(h) = 4 * var(o) = 4
        assert var_h == pytest.approx(4.0, abs=0.2)
        assert stderr == pytest.approx(np.sqrt(var_h / 2000))

    def test_covariance_matrix(self):
        rng = np.random.default_rng(0)
        outcomes = rng.choice([-1.0, 1.0], size=(500, 2))
        cov = covariance_matrix_from_samples(outcomes)
        assert cov.shape == (2, 2)
        assert cov[0, 0] == pytest.approx(np.var(outcomes[:, 0], ddof=1))

    def test_allocate_group_shots_sums(self):
        alloc = allocate_group_shots([4.0, 1.0, 0.25], 1000)
        assert sum(alloc) == 1000
        assert len(alloc) == 3
        # larger variance -> more shots
        assert alloc[0] > alloc[2]

    def test_allocate_group_shots_validation(self):
        with pytest.raises(ValueError, match="non-negative"):
            allocate_group_shots([1.0, -0.5], 100)
        with pytest.raises(ValueError, match="cannot give"):
            allocate_group_shots([1.0, 1.0], 1)


class TestSampling:
    """Measurement sampling in the computational basis."""

    def test_probabilities_state_vector(self):
        psi = np.array([0.6, 0.8], dtype=complex)
        probs = measurement_probabilities(psi)
        np.testing.assert_allclose(probs, [0.36, 0.64])

    def test_probabilities_density_matrix(self):
        rho = np.diag([0.3, 0.7]).astype(complex)
        probs = measurement_probabilities(rho)
        np.testing.assert_allclose(probs, [0.3, 0.7])

    def test_sample_counts_sum_to_shots(self):
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rng = np.random.default_rng(0)
        counts = sample_bitstrings(psi, 1000, rng=rng)
        assert counts.sum() == 1000
        assert counts.shape == (2,)

    def test_sample_frequencies_match_probs(self):
        psi = np.array([0.6, 0.8], dtype=complex)
        rng = np.random.default_rng(1)
        probs_hat, counts = estimate_distribution(psi, 20000, rng=rng)
        np.testing.assert_allclose(probs_hat, [0.36, 0.64], atol=0.02)

    def test_superposition_two_qubits(self):
        """Uniform superposition over 4 states -> ~25% each."""
        psi = np.ones(4, dtype=complex) / 2.0
        rng = np.random.default_rng(2)
        counts = sample_bitstrings(psi, 40000, rng=rng)
        np.testing.assert_allclose(counts / 40000, [0.25] * 4, atol=0.01)

    def test_invalid_shots(self):
        psi = np.array([1, 0], dtype=complex)
        with pytest.raises(ValueError, match="shots"):
            sample_bitstrings(psi, 0)

    def test_unnormalized_state_renormalizes(self):
        """measurement_probabilities normalizes; sampling still works."""
        psi = np.array([2.0, 0.0], dtype=complex)  # norm 2
        probs = measurement_probabilities(psi)
        np.testing.assert_allclose(probs, [1.0, 0.0])


class TestConfig:
    """New config helpers."""

    def test_total_shots_default(self):
        assert config.TOTAL_SHOTS > 0
        assert isinstance(config.TOTAL_SHOTS, int)

    def test_ln_2_over_delta(self):
        from qusim_np.config import ln_2_over_delta, two_ln_2_over_delta
        assert ln_2_over_delta(0.05) == pytest.approx(np.log(40.0))
        assert two_ln_2_over_delta(0.05) == pytest.approx(2 * np.log(40.0))
        with pytest.raises(ValueError, match="delta"):
            ln_2_over_delta(0.0)
