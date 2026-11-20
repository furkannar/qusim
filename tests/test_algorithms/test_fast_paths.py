"""Cross-check the matrix-free fast paths against dense reference constructions."""

import functools
import itertools

import numpy as np
import pytest

import qusim_np as q
from qusim_np.config import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
from qusim_np.grover_search import (
    apply_hadamard_transform, hadamard_transform, hadamard_matrix_closed_form,
)
from qusim_np.qft import qft_matrix, qft_state, inverse_qft_state
from qusim_np.vqe import apply_pauli_string

_M = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}


def _kron_pauli(s):
    return functools.reduce(np.kron, [_M[c] for c in s])


def _rand_state(rng, n):
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_every_pauli_string_matches_kron(n):
    rng = np.random.default_rng(n)
    for labels in itertools.product("IXYZ", repeat=n):
        s = "".join(labels)
        ref = _kron_pauli(s)
        psi = _rand_state(rng, n)
        np.testing.assert_allclose(q.pauli_string_matrix(s), ref, atol=1e-14)
        np.testing.assert_allclose(apply_pauli_string(psi, s), ref @ psi, atol=1e-14)
        assert q.pauli_string_expval(psi, s) == pytest.approx(
            np.real(np.vdot(psi, ref @ psi)), abs=1e-14)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_hamiltonian_matches_dense_sum(n):
    rng = np.random.default_rng(10 + n)
    strings = ["".join(rng.choice(list("IXYZ"), n)) for _ in range(15)]
    strings.append(strings[0])                      # repeated term must accumulate
    coeffs = rng.normal(size=len(strings))
    ref = sum(c * _kron_pauli(s) for c, s in zip(coeffs, strings))
    np.testing.assert_allclose(q.hamiltonian(coeffs, strings), ref, atol=1e-13)


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_fast_walsh_hadamard_matches_matrix(n):
    psi = _rand_state(np.random.default_rng(n), n)
    np.testing.assert_allclose(apply_hadamard_transform(psi),
                               hadamard_transform(n) @ psi, atol=1e-13)
    np.testing.assert_allclose(hadamard_matrix_closed_form(n),
                               hadamard_transform(n), atol=1e-14)


def test_fast_walsh_hadamard_rejects_bad_dimension():
    with pytest.raises(ValueError, match="power of 2"):
        apply_hadamard_transform(np.ones(3))


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_qft_fft_matches_matrix(n):
    psi = _rand_state(np.random.default_rng(n), n)
    F = qft_matrix(n)
    np.testing.assert_allclose(qft_state(psi), F @ psi, atol=1e-13)
    np.testing.assert_allclose(inverse_qft_state(psi), F.conj().T @ psi, atol=1e-13)


@pytest.mark.parametrize("n,marked", [(2, [3]), (4, [1, 7, 12]), (5, list(range(11)))])
def test_grover_vector_path_matches_dense_iteration(n, marked):
    state, k, p = q.grover_search(n, marked)
    G_ref = q.diffusion_operator(n) @ q.grover_oracle(n, marked)
    np.testing.assert_allclose(q.grover_iteration(n, marked), G_ref, atol=1e-14)
    ref = q.initialize_uniform_superposition(n)
    for _ in range(k):
        ref = G_ref @ ref
    np.testing.assert_allclose(state, ref, atol=1e-13)
    # Closed form: P(marked) = sin^2((2k+1) theta / 2)
    theta = q.calculate_theta(n, marked)
    assert p == pytest.approx(np.sin((2 * k + 1) * theta / 2) ** 2, abs=1e-12)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_deutsch_jozsa_vector_path_matches_dense(n):
    f = np.random.default_rng(n).integers(0, 2, 2 ** n)
    state, p0 = q.deutsch_jozsa(n, f)
    H = hadamard_transform(n)
    ref = H @ q.deutsch_jozsa_oracle(n, f) @ H[:, 0]
    np.testing.assert_allclose(state, ref, atol=1e-13)
    # Identity: amplitude of |0^n> is the mean of (-1)^{f(x)}
    assert p0 == pytest.approx(np.mean(1 - 2 * f) ** 2, abs=1e-14)
