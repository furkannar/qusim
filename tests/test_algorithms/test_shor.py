"""Tests for Shor's algorithm implementation."""

import numpy as np
import pytest
import qusim_np
from qusim_np.config import GB, KB, MB, config
from qusim_np.shor import (
    divisors, is_prime, prime_power_base, modular_exponentiation,
    validate_base, classical_order, register_sizes, counting_superposition,
    modular_multiplication_permutation, modular_multiplication_unitary,
    order_finding_state, counting_distribution, order_from_measurement,
    quantum_order_finding, factors_from_order, shor,
    memory_required, estimate_capacity,
)


class TestNumberTheory:
    """Classical helpers."""

    @pytest.mark.parametrize("M", [1, 2, 12, 15, 16, 36, 97, 221, 1001])
    def test_divisors_match_brute_force(self, M):
        assert divisors(M) == [d for d in range(1, M + 1) if M % d == 0]

    def test_divisors_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            divisors(0)

    def test_is_prime(self):
        primes = [p for p in range(2, 60) if is_prime(p)]
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                          43, 47, 53, 59]

    @pytest.mark.parametrize("M,b", [(9, 3), (27, 3), (25, 5), (49, 7),
                                     (1 << 20, 2), (3 ** 13, 3)])
    def test_prime_power_detected(self, M, b):
        # Exponents are tried largest first, so the smallest base is returned
        assert prime_power_base(M) == b

    @pytest.mark.parametrize("M", [15, 21, 35, 77, 97])
    def test_prime_power_none(self, M):
        assert prime_power_base(M) is None

    @pytest.mark.parametrize("a,x,M", [(7, 0, 15), (7, 4, 15), (2, 100, 21),
                                       (13, 12345, 1001)])
    def test_modular_exponentiation(self, a, x, M):
        assert modular_exponentiation(a, x, M) == (a ** x) % M

    @pytest.mark.parametrize("a", [0, 1, 15, 16, -2])
    def test_validate_base_range(self, a):
        with pytest.raises(ValueError, match="1 < a < M"):
            validate_base(a, 15)

    @pytest.mark.parametrize("a", [3, 5, 6, 9, 10, 12])
    def test_validate_base_gcd(self, a):
        with pytest.raises(ValueError, match="gcd"):
            validate_base(a, 15)

    def test_validate_base_accepts_coprime(self):
        for a in [2, 4, 7, 8, 11, 13, 14]:
            validate_base(a, 15)

    @pytest.mark.parametrize("a,M,r", [(7, 15, 4), (2, 15, 4), (4, 15, 2),
                                       (14, 15, 2), (2, 21, 6), (5, 21, 6)])
    def test_classical_order(self, a, M, r):
        assert classical_order(a, M) == r


class TestRegistersAndState:
    """Register sizes, superposition and the modular unitary."""

    @pytest.mark.parametrize("M,n_w", [(3, 2), (15, 4), (16, 4), (17, 5), (21, 5)])
    def test_register_sizes(self, M, n_w):
        t, nw, n = register_sizes(M)
        assert nw == n_w and t == 2 * n_w and n == 3 * n_w
        assert 2 ** nw >= M and 2 ** t >= M ** 2

    @pytest.mark.parametrize("t", [1, 3, 6])
    def test_counting_superposition_uniform(self, t):
        s = counting_superposition(t)
        np.testing.assert_allclose(s, np.full(2 ** t, 1 / np.sqrt(2 ** t)), atol=1e-12)

    @pytest.mark.parametrize("a,M", [(7, 15), (2, 21), (8, 21)])
    def test_unitary_is_permutation(self, a, M):
        U = modular_multiplication_unitary(a, M)
        W = U.shape[0]
        np.testing.assert_allclose(U.conj().T @ U, np.eye(W), atol=1e-12)
        for w in range(W):
            expected = (a * w) % M if w < M else w
            assert U[expected, w] == 1

    def test_unitary_power_matches_pow(self):
        a, M = 7, 15
        U = modular_multiplication_unitary(a, M)
        for j in range(4):
            # a^{2^j} mod M can be 1 (7^8 = 1 mod 15), so use the raw permutation
            perm = modular_multiplication_permutation(modular_exponentiation(a, 2 ** j, M), M, 4)
            Uj = np.zeros((16, 16))
            Uj[perm, np.arange(16)] = 1
            np.testing.assert_allclose(np.linalg.matrix_power(U, 2 ** j), Uj, atol=1e-12)

    def test_unitary_order_is_identity(self):
        U = modular_multiplication_unitary(7, 15)
        np.testing.assert_allclose(np.linalg.matrix_power(U, 4), np.eye(16), atol=1e-12)

    @pytest.mark.parametrize("a,M", [(7, 15), (2, 15), (2, 21)])
    def test_order_finding_state(self, a, M):
        """Controlled powers produce sum_x |x>|a^x mod M> / sqrt(Q)."""
        state = order_finding_state(a, M)
        t, n_w, n = register_sizes(M)
        Q = 2 ** t
        assert state.shape == (Q, 2 ** n_w) and state.size == 2 ** n
        expected = np.zeros_like(state)
        x = np.arange(Q)
        expected[x, [modular_exponentiation(a, int(v), M) for v in x]] = 1 / np.sqrt(Q)
        np.testing.assert_allclose(state, expected, atol=1e-12)

    def test_memory_guard(self):
        # M = 15 needs 12 qubits * 32 B = 128 KiB
        with pytest.raises(ValueError, match="memory_gb"):
            order_finding_state(7, 15, memory_gb=100 * KB / GB)
        with pytest.raises(ValueError, match="memory_gb"):
            shor(15, a=7, memory_gb=100 * KB / GB)
        order_finding_state(7, 15, memory_gb=128 * KB / GB)   # exactly enough


class TestMemoryEstimate:
    """User-supplied memory budgets."""

    def test_memory_required(self):
        assert memory_required(15) == config.BYTES_PER_AMPLITUDE * 2 ** 12
        assert memory_required(33) == config.BYTES_PER_AMPLITUDE * 2 ** 18

    @pytest.mark.parametrize("gb,max_q,usable,max_M", [
        (0.5, 24, 24, 256), (1, 25, 24, 256), (8, 28, 27, 512),
        (16, 29, 27, 512), (64, 31, 30, 1024),
    ])
    def test_estimate_capacity(self, gb, max_q, usable, max_M):
        cap = estimate_capacity(gb)
        assert cap["max_qubits"] == max_q
        assert cap["usable_qubits"] == usable == 3 * cap["work_qubits"]
        assert cap["max_M"] == max_M
        assert cap["peak_bytes"] <= gb * GB

    def test_capacity_consistent_with_guard(self):
        """The largest M the estimate allows passes the guard; the next size up does not."""
        gb = MB / GB                            # 1 MiB -> 15 qubits -> n_w = 5
        cap = estimate_capacity(gb)
        assert cap["max_M"] == 32
        order_finding_state(2, 31, memory_gb=gb)
        with pytest.raises(ValueError, match="M <= 32"):
            order_finding_state(2, 33, memory_gb=gb)

    @pytest.mark.parametrize("gb", [0, -1, "8", True])
    def test_invalid_budget(self, gb):
        with pytest.raises(ValueError, match="memory_gb"):
            estimate_capacity(gb)

    def test_config_override(self, monkeypatch):
        """Changing config.BYTES_PER_AMPLITUDE changes the estimate at call time."""
        monkeypatch.setattr(config, "BYTES_PER_AMPLITUDE", 64)
        assert memory_required(15) == 64 * 2 ** 12
        assert estimate_capacity(1)["max_qubits"] == 24

    def test_detected_budget(self):
        """memory_gb=None reads the free RAM and keeps the reserve."""
        cap = estimate_capacity()
        assert cap["detected"] is True
        assert cap["memory_gb"] < qusim_np.available_memory_gb()
        assert cap["peak_bytes"] <= cap["memory_gb"] * GB
        assert estimate_capacity(8)["detected"] is False

    def test_budget_too_small(self):
        with pytest.raises(ValueError, match="smallest case"):
            estimate_capacity(1e-6)

    def test_peak_memory_matches_constant(self):
        """config.BYTES_PER_AMPLITUDE is a real upper bound on traced NumPy memory."""
        import tracemalloc
        M, a = 33, 5                            # 18 qubits
        tracemalloc.start()
        try:
            counting_distribution(a, M)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        per_amp = peak / 2 ** register_sizes(M)[2]
        assert 16 <= per_amp <= config.BYTES_PER_AMPLITUDE * 1.05


class TestDistributionAndPostProcessing:
    """Inverse QFT peaks and continued fractions."""

    def test_distribution_15_exact_peaks(self):
        """For r = 4 dividing Q = 256 the peaks are exactly at s * 64."""
        P = counting_distribution(7, 15)
        assert np.isclose(P.sum(), 1.0)
        np.testing.assert_allclose(P[[0, 64, 128, 192]], 0.25, atol=1e-12)

    def test_distribution_21_peaks_near_multiples(self):
        """r = 6 does not divide Q = 1024: most weight is near s Q / r."""
        P = counting_distribution(2, 21)
        Q, r = P.size, 6
        peaks = {round(s * Q / r) % Q for s in range(r)}
        near = sum(P[(p + d) % Q] for p in peaks for d in (-1, 0, 1))
        assert near > 0.8

    @pytest.mark.parametrize("y,r", [(64, 4), (192, 4), (128, 4)])
    def test_order_from_measurement_15(self, y, r):
        # y = 128 gives 1/2, i.e. q = 2; the multiples search recovers r = 4.
        assert order_from_measurement(y, 256, 7, 15) == r

    def test_order_from_measurement_zero(self):
        assert order_from_measurement(0, 256, 7, 15) is None

    def test_factors_from_order(self):
        assert factors_from_order(7, 4, 15) == (3, 5)
        assert factors_from_order(7, 3, 15) is None       # odd r
        assert factors_from_order(14, 2, 15) is None      # 14 = -1 mod 15
        assert factors_from_order(7, None, 15) is None

    def test_quantum_order_finding(self):
        rng = np.random.default_rng(0)
        r, ys = quantum_order_finding(7, 15, shots=10, rng=rng)
        assert r == 4 and ys


class TestShor:
    """End-to-end factoring."""

    @pytest.mark.parametrize("M", [15, 21, 33, 35])
    def test_factors_semiprimes(self, M):
        result = shor(M, rng=np.random.default_rng(1))
        p, q = result["factors"]
        assert p * q == M and 1 < p <= q
        assert p in divisors(M)

    def test_fixed_base_uses_quantum(self):
        result = shor(15, a=7, rng=np.random.default_rng(2))
        assert result == {"factors": (3, 5), "a": 7, "r": 4,
                          "method": "quantum", "attempts": 1}

    def test_fixed_base_must_be_coprime(self):
        with pytest.raises(ValueError, match="gcd"):
            shor(15, a=5)

    def test_even(self):
        assert shor(22)["factors"] == (2, 11)

    def test_prime_rejected(self):
        with pytest.raises(ValueError, match="prime"):
            shor(13)

    def test_prime_power(self):
        result = shor(27)
        assert result["method"] == "classical"
        assert np.prod(result["factors"]) == 27
