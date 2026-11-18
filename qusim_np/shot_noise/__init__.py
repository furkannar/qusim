"""Shot noise and measurement sampling for finite-shot estimation."""

from qusim_np.shot_noise.estimation import (
    estimate_expval, observable_expval, sample_observable_outcomes,
)
from qusim_np.shot_noise.pauli_stats import (
    pauli_expval_from_counts, pauli_outcome_probs,
    shots_needed_pauli_std_error,
)
from qusim_np.shot_noise.bounds import (
    gaussian_z, shots_clt, shots_hoeffding, shots_bernstein,
)
from qusim_np.shot_noise.hamiltonian import (
    hamiltonian_variance, uniform_allocation_variance,
    optimal_allocation_variance, allocate_shots,
    shots_for_hamiltonian,
)
from qusim_np.shot_noise.grouped import (
    pauli_covariance, group_shot_values, group_variance,
    covariance_matrix_from_samples, allocate_group_shots,
)
from qusim_np.shot_noise.sampling import (
    measurement_probabilities, sample_bitstrings, estimate_distribution,
)

__all__ = [
    "estimate_expval", "observable_expval", "sample_observable_outcomes",
    "pauli_expval_from_counts", "pauli_outcome_probs",
    "shots_needed_pauli_std_error",
    "gaussian_z", "shots_clt", "shots_hoeffding", "shots_bernstein",
    "hamiltonian_variance", "uniform_allocation_variance",
    "optimal_allocation_variance", "allocate_shots",
    "shots_for_hamiltonian",
    "pauli_covariance", "group_shot_values", "group_variance",
    "covariance_matrix_from_samples", "allocate_group_shots",
    "measurement_probabilities", "sample_bitstrings",
    "estimate_distribution",
]
