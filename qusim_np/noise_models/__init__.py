"""Noise models: quantum channels applied to density matrices."""

from qusim_np.noise_models.channels import (
    to_density_matrix, apply_channel, apply_unitary_density,
    kraus_branch_probabilities, sample_kraus_branch,
)
from qusim_np.noise_models.noise_model import NoiseModel
from qusim_np.noise_models.amplitude_damping_error import (
    amplitude_damping_kraus, gamma_from_relaxation, amplitude_damping,
)
from qusim_np.noise_models.phase_damping_error import (
    phase_damping_kraus, phase_flip_kraus,
    lambda_from_p, p_from_lambda,
    phase_damping_from_times, phase_damping, phase_flip, thermal_relaxation,
)
from qusim_np.noise_models.depolarizing_error import (
    depolarizing_kraus, depolarizing,
)
from qusim_np.noise_models.random_pauli_error import (
    random_pauli_kraus, random_pauli, sample_pauli, apply_sampled_pauli,
)

__all__ = [
    "to_density_matrix", "apply_channel", "apply_unitary_density",
    "kraus_branch_probabilities", "sample_kraus_branch", "NoiseModel",
    "amplitude_damping_kraus", "gamma_from_relaxation", "amplitude_damping",
    "phase_damping_kraus", "phase_flip_kraus",
    "lambda_from_p", "p_from_lambda",
    "phase_damping_from_times", "phase_damping", "phase_flip",
    "thermal_relaxation",
    "depolarizing_kraus", "depolarizing",
    "random_pauli_kraus", "random_pauli", "sample_pauli", "apply_sampled_pauli",
]
