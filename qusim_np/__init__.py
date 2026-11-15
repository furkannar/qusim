"""Quantum computing engine: gates, state-vector simulation, and algorithms."""

from qusim_np.config import QuantumConfig, config
from qusim_np.gates import (
    QuantumGate,
    pauli_x, pauli_y, pauli_z, hadamard, phase_gate, t_gate,
    identity, cnot, swap, controlled_phase, toffoli, rx, ry, rz,
)
from qusim_np.grover_search import (
    hadamard_transform, hadamard_matrix_closed_form,
    initialize_uniform_superposition, grover_oracle, diffusion_operator,
    grover_iteration, calculate_theta, calculate_optimal_iterations,
    grover_search,
)
from qusim_np.deutsch_jozsa import (
    deutsch_jozsa_oracle, deutsch_jozsa, deutsch_jozsa_classify,
)
from qusim_np.qft import (
    qft_matrix, inverse_qft_matrix, qft_state, inverse_qft_state,
)
from qusim_np.vqe import (
    pauli_string_matrix, apply_single_qubit, apply_two_qubit,
    apply_pauli_string, pauli_string_expval, hamiltonian, expectation_value,
    hea_param_count, hea_state, vqe_energy, vqe_optimize,
)

__all__ = [
    "QuantumConfig", "config", "QuantumGate",
    "pauli_x", "pauli_y", "pauli_z", "hadamard", "phase_gate", "t_gate",
    "identity", "cnot", "swap", "controlled_phase", "toffoli", "rx", "ry", "rz",
    "hadamard_transform", "hadamard_matrix_closed_form",
    "initialize_uniform_superposition", "grover_oracle", "diffusion_operator",
    "grover_iteration", "calculate_theta", "calculate_optimal_iterations",
    "grover_search",
    "deutsch_jozsa_oracle", "deutsch_jozsa", "deutsch_jozsa_classify",
    "qft_matrix", "inverse_qft_matrix", "qft_state", "inverse_qft_state",
    "pauli_string_matrix", "apply_single_qubit", "apply_two_qubit",
    "apply_pauli_string", "pauli_string_expval", "hamiltonian", "expectation_value",
    "hea_param_count", "hea_state", "vqe_energy", "vqe_optimize",
]
