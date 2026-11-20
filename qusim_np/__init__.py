"""Lightweight quantum simulation package: circuits, gates, state vectors, algorithms, VQE, noise and shot noise."""

from qusim_np.config import QuantumConfig, config
from qusim_np.gates import (
    QuantumGate,
    pauli_x, pauli_y, pauli_z, hadamard, phase_gate, t_gate,
    identity, cnot, swap, controlled_phase, toffoli, rx, ry, rz,
)
from qusim_np.grover_search import (
    hadamard_transform, hadamard_matrix_closed_form, apply_hadamard_transform,
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
from qusim_np.resources import (
    available_memory_bytes, available_memory_gb, check_memory, MemoryMonitor,
)
from qusim_np.shor import (
    divisors, is_prime, prime_power_base, modular_exponentiation,
    validate_base, register_sizes, modular_multiplication_unitary,
    order_finding_state, counting_distribution, order_from_measurement,
    quantum_order_finding, factors_from_order, shor,
    memory_required, estimate_capacity, detected_budget_gb,
)
from qusim_np.vqe import (
    pauli_string_matrix, apply_single_qubit, apply_two_qubit, apply_gate,
    apply_pauli_string, pauli_string_expval, hamiltonian, expectation_value,
    hea_param_count, hea_state, vqe_energy, vqe_optimize,
)
from qusim_np.noise_models import (
    to_density_matrix, apply_channel, apply_unitary_density,
    kraus_branch_probabilities, sample_kraus_branch, NoiseModel,
    amplitude_damping_kraus, gamma_from_relaxation, amplitude_damping,
    phase_damping_kraus, phase_flip_kraus,
    lambda_from_p, p_from_lambda,
    phase_damping_from_times, phase_damping, phase_flip, thermal_relaxation,
    depolarizing_kraus, depolarizing,
    random_pauli_kraus, random_pauli, sample_pauli, apply_sampled_pauli,
)
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
from qusim_np.circuit import QuantumCircuit, Instruction

__all__ = [
    "QuantumCircuit", "Instruction", "NoiseModel",
    "QuantumConfig", "config", "QuantumGate",
    "pauli_x", "pauli_y", "pauli_z", "hadamard", "phase_gate", "t_gate",
    "identity", "cnot", "swap", "controlled_phase", "toffoli", "rx", "ry", "rz",
    "hadamard_transform", "hadamard_matrix_closed_form", "apply_hadamard_transform",
    "initialize_uniform_superposition", "grover_oracle", "diffusion_operator",
    "grover_iteration", "calculate_theta", "calculate_optimal_iterations",
    "grover_search",
    "deutsch_jozsa_oracle", "deutsch_jozsa", "deutsch_jozsa_classify",
    "qft_matrix", "inverse_qft_matrix", "qft_state", "inverse_qft_state",
    "divisors", "is_prime", "prime_power_base", "modular_exponentiation",
    "validate_base", "register_sizes", "modular_multiplication_unitary",
    "order_finding_state", "counting_distribution", "order_from_measurement",
    "quantum_order_finding", "factors_from_order", "shor",
    "memory_required", "estimate_capacity", "detected_budget_gb",
    "available_memory_bytes", "available_memory_gb", "check_memory", "MemoryMonitor",
    "pauli_string_matrix", "apply_single_qubit", "apply_two_qubit", "apply_gate",
    "apply_pauli_string", "pauli_string_expval", "hamiltonian", "expectation_value",
    "hea_param_count", "hea_state", "vqe_energy", "vqe_optimize",
    "to_density_matrix", "apply_channel", "apply_unitary_density",
    "kraus_branch_probabilities", "sample_kraus_branch",
    "amplitude_damping_kraus", "gamma_from_relaxation", "amplitude_damping",
    "phase_damping_kraus", "phase_flip_kraus",
    "lambda_from_p", "p_from_lambda",
    "phase_damping_from_times", "phase_damping", "phase_flip",
    "thermal_relaxation",
    "depolarizing_kraus", "depolarizing",
    "random_pauli_kraus", "random_pauli", "sample_pauli", "apply_sampled_pauli",
    "estimate_expval", "observable_expval", "sample_observable_outcomes",
    "pauli_expval_from_counts", "pauli_outcome_probs",
    "shots_needed_pauli_std_error",
    "gaussian_z", "shots_clt", "shots_hoeffding", "shots_bernstein",
    "hamiltonian_variance", "uniform_allocation_variance",
    "optimal_allocation_variance", "allocate_shots", "shots_for_hamiltonian",
    "pauli_covariance", "group_shot_values", "group_variance",
    "covariance_matrix_from_samples", "allocate_group_shots",
    "measurement_probabilities", "sample_bitstrings", "estimate_distribution",
]
