# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`qusim-np` (PyPI name; package `qusim_np`) is a lightweight quantum simulation package in pure NumPy + SciPy: a `QuantumCircuit` API, matrix-free state-vector algorithms, VQE, Kraus noise channels with exact and quantum-trajectory simulation, and finite-shot statistics. Python ≥ 3.10, NumPy ≥ 2.0 (required: `np.bitwise_count` is used).

## Commands

```bash
pip install -e .[dev]                       # install with pytest
pytest -q                                   # full suite (~20-25 s; trajectory + example tests dominate)
pytest tests/test_circuit.py -q             # one file
pytest tests/test_circuit.py::TestTrajectories::test_average_trajectory_matches_density_matrix
pytest -k "fast_paths or grover" -q         # by keyword
python examples/01_circuit_basics.py        # tutorial scripts 01-04; tests/test_examples.py runs them all
```

There is no linter or formatter configured. CI (`.github/workflows/ci.yml`) runs pytest on Python 3.10–3.12; `publish.yml` publishes to PyPI when a GitHub release is published.

## Conventions that must not change

- **Qubit 0 is the most significant bit** of the basis index: `|q0 q1 … q_{n-1}⟩ ↔ Σ q_k 2^{n-1-k}`. Pauli string `"XZI"` = X on qubit 0. In code, qubit `q` is bit `1 << (n-1-q)`.
- **Bitstrings from `sample_counts` list qubit 0 leftmost** (`QuantumCircuit(2).x(0)` → `"10"`). This is the reverse of Qiskit, and it is intentional and documented.
- **Multi-qubit gate matrices:** `qubits[0]` is the most significant bit of the gate's own index (`cx(control, target)`), regardless of numeric qubit order.
- **Depolarizing convention:** `E(ρ) = (1−p)ρ + (p/2)I` (Kraus `√(1−3p/4) I, √(p/4) X/Y/Z`), not the `q/3` Pauli form. The parameter names `lam` (phase damping) and `p` (phase flip) are distinct; see the module docstrings.
- **Validation style:** functions raise `ValueError` with the valid range in the message (tests match on these messages). Integer arguments reject `bool` explicitly.

## Architecture

Simulation is **matrix-free** throughout. No 2ⁿ×2ⁿ operator is built on a hot path. Functions that *return* a dense matrix (`hamiltonian`, `qft_matrix`, `grover_oracle`, `diffusion_operator`, `hadamard_transform`, `pauli_string_matrix`) exist as explicit builders only. New code should follow the same pattern: find the identity that avoids the dense operator. The README "Performance" section lists the identities in use.

Core primitives, which everything else reuses:
- `vqe.apply_gate(state, U, qubits)` applies a k-qubit gate by moving the target axes of the `(2,)*n` state tensor to the front and doing one matmul. `apply_single_qubit` / `apply_two_qubit` are older special cases.
- `vqe._pauli_column_values(labels)` represents a Pauli string as a signed permutation, `P|c⟩ = i^{n_Y} (−1)^{popcount(c&z)} |c⊕x⟩`, using x/z bitmasks. It underlies `apply_pauli_string`, `pauli_string_expval`, `hamiltonian` and density-matrix expectations.
- `noise_models/channels.py`: `apply_channel` (single-qubit Kraus channel on ρ) and `apply_unitary_density` (U ρ U† for k qubits) contract onto the row axes `q` and the column axes `n+q` of ρ's `(2,)*2n` tensor. `_validate_kraus_ops` enforces completeness `Σ K†K = I`.
- **Trajectories:** `kraus_branch_probabilities` computes `p_k = Tr(K†K ρ_q)` from the 2×2 reduced state, which equals `vdot(Kψ, Kψ)` without building every branch. `_sample_branch` is the unchecked fast path used by `QuantumCircuit`, and it skips identity branches as a global phase. The public `sample_kraus_branch` validates its inputs.

`circuit.QuantumCircuit` is a thin layer over these primitives. Gate methods append an immutable `Instruction` and return `self`. Simulation is lazy:
- `statevector()` is pure only, and raises if a non-empty `NoiseModel` is attached.
- `density_matrix()` applies U ρ U†, then `NoiseModel.channels_for(...)` on each qubit the gate touched.
- The private `_trajectory(rng)` runs the pure-state version.
- `sample_counts` / `expectation` take `method="density" | "trajectory"`.
- Finite-shot `expectation` rotates X→Z with H and Y→Z with S†H, then reads the parity of the term's support bits. Shots are split with `shot_noise.allocate_shots`.

Gate names recorded by the circuit are canonical (`cx`, `ccx`). `NoiseModel` maps the aliases `cnot`/`toffoli` to them, and `inverse()` keeps names meaningful (`s`↔`sdg`, rotations negate `params`) so that noise rules still match.

**Import graph (keep it acyclic):** `grover_search` (owns the shared `_validate_n`) ← `qft`, `vqe`, `noise_models.channels`, `circuit`. `vqe` must not import `noise_models` at module level. `channels.py` imports from `vqe` *inside functions* for exactly this reason. `shot_noise.estimation` imports `vqe`. `circuit` imports everything and is imported last in `qusim_np/__init__.py`. SciPy is imported lazily (`scipy.optimize` in `vqe_optimize`, `scipy.special.ndtri` in `gaussian_z`) to keep `import qusim_np` fast (~90 ms). Do not add module-level `scipy.stats` imports.

Public API is re-exported explicitly. When adding a public function, update both the subpackage `__init__.py` (import + `__all__`) and `qusim_np/__init__.py` (import + `__all__`).

## Testing approach

- **Fast paths are verified against independent dense references:**
  - `tests/test_fast_paths.py`: every 4ⁿ Pauli string for n ≤ 3, plus Grover / Deutsch–Jozsa / QFT against dense matrices.
  - `tests/test_circuit.py`: `_embed` builds the full operator element-by-element from index bits.

  Any new optimization needs a test of this kind.
- **Statistical tests** (shots, trajectories) use a seeded `np.random.default_rng(...)` and assert `|estimate − exact| < 4·stderr`. Keep shot and trajectory counts modest, because these tests dominate suite runtime.
- **Pitfall:** `np.bitwise_count` returns `uint8`, so `1 - 2*parity` wraps to 255. Convert to float first (`1.0 - 2.0 * parity`).

## Documentation

- **README code blocks must execute.** Tutorial blocks (steps 1–10 under "Circuit tutorial") build on each other and run in one shared namespace. All other blocks must be self-contained. Quoted outputs come from seeded runs, so re-run the examples after behaviour changes.
- **Never put inline LaTeX (`$...$`) inside Markdown table cells.** GitHub strips escapes like `\,` `\!` `\&` and the formulas render broken. Use plain text or Unicode in tables, and keep display math (`$$...$$`) outside them.
- Keep the test counts in the README (intro, layout, Development) and the version in `pyproject.toml` in sync with releases.
- The README's "AI assistance" section discloses which AI tools contributed to which version. Update it when AI-assisted work lands.

## Git / CI

- **Never commit `badge.json` by hand.** CI regenerates and commits it on pushes to `master` only (`chore: update test badge [skip ci]`), so local `master` is often one commit behind `origin/master`. Sync before branching.
- If a PR conflicts in `badge.json`, it is the bot's file: keep the count that matches the merged code.
- Work on feature branches from an up-to-date `origin/master` and open PRs into `master`.
