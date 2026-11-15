"""Tests for quantum gates implementation."""

import numpy as np
import pytest
from qusim_np.gates import (
    QuantumGate,
    pauli_x, pauli_y, pauli_z, hadamard, phase_gate, t_gate,
    identity, cnot, swap, controlled_phase, toffoli, rx, ry, rz
)
from qusim_np.config import config


class TestRotationGatesValidation:
    """Test rotation gates SU(2) validation."""

    def test_rx_valid(self):
        """Test valid Rx gate."""
        rx_gate = rx(np.pi/4)
        assert rx_gate is not None

    def test_ry_valid(self):
        """Test valid Ry gate."""
        ry_gate = ry(np.pi/4)
        assert ry_gate is not None

    def test_rz_valid(self):
        """Test valid Rz gate."""
        rz_gate = rz(np.pi/4)
        assert rz_gate is not None

    def test_invalid_gate_raises_error(self):
        """Test that invalid gate raises ValueError."""
        with pytest.raises(ValueError, match="Gate is not unitary"):
            QuantumGate([[1, 0], [0, 2]], validate_su2=True)


class TestSingleQubitGates:
    """Test single-qubit gate implementations."""

    def test_pauli_x(self):
        """Test Pauli-X gate."""
        gate = pauli_x()
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(gate.matrix, expected)

    def test_pauli_y(self):
        """Test Pauli-Y gate."""
        gate = pauli_y()
        expected = np.array([[0, -1j], [1j, 0]])
        np.testing.assert_array_equal(gate.matrix, expected)

    def test_pauli_z(self):
        """Test Pauli-Z gate."""
        gate = pauli_z()
        expected = np.diag([1, -1])
        np.testing.assert_array_equal(gate.matrix, expected)

    def test_hadamard(self):
        """Test Hadamard gate."""
        gate = hadamard()
        expected = np.array([[config.INV_SQRT_2, config.INV_SQRT_2],
                            [config.INV_SQRT_2, -config.INV_SQRT_2]])
        np.testing.assert_array_almost_equal(gate.matrix, expected)

    def test_phase_gate(self):
        """Test Phase (S) gate."""
        gate = phase_gate()
        expected = np.diag([1, 1j])
        np.testing.assert_array_equal(gate.matrix, expected)

    def test_t_gate(self):
        """Test T gate."""
        gate = t_gate()
        expected = np.diag([1, np.exp(1j * np.pi / 4)])
        np.testing.assert_array_almost_equal(gate.matrix, expected)

    def test_identity(self):
        """Test Identity gate."""
        gate = identity(n=1)
        expected = np.eye(2)
        np.testing.assert_array_equal(gate.matrix, expected)

    def test_identity_multi_qubit(self):
        """Test Identity gate for multiple qubits."""
        gate = identity(n=2)
        expected = np.eye(4)
        np.testing.assert_array_equal(gate.matrix, expected)


class TestTwoQubitGates:
    """Test two-qubit gate implementations."""

    def test_cnot(self):
        """Test CNOT gate."""
        gate = cnot()
        expected = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])
        np.testing.assert_array_equal(gate.matrix, expected)

    def test_swap(self):
        """Test SWAP gate."""
        gate = swap()
        expected = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(gate.matrix, expected)

    def test_controlled_phase(self):
        """Test Controlled Phase gate."""
        gate = controlled_phase()
        expected = np.diag([1, 1, 1, -1])
        np.testing.assert_array_equal(gate.matrix, expected)


class TestThreeQubitGates:
    """Test three-qubit gate implementations."""

    def test_toffoli(self):
        """Test Toffoli (CCNOT) gate."""
        gate = toffoli()
        # Toffoli should swap |110> (index 6) and |111> (index 7)
        expected = np.eye(8)
        expected[[6, 7]] = expected[[7, 6]]
        np.testing.assert_array_equal(gate.matrix, expected)


class TestRotationGates:
    """Test rotation gate implementations."""

    def test_rx_matrix(self):
        """Test Rx gate matrix form."""
        theta = np.pi / 4
        gate = rx(theta)
        expected = np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                            [-1j * np.sin(theta/2), np.cos(theta/2)]])
        np.testing.assert_array_almost_equal(gate.matrix, expected)

    def test_ry_matrix(self):
        """Test Ry gate matrix form."""
        theta = np.pi / 4
        gate = ry(theta)
        expected = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                            [np.sin(theta/2), np.cos(theta/2)]])
        np.testing.assert_array_almost_equal(gate.matrix, expected)

    def test_rz_matrix(self):
        """Test Rz gate matrix form."""
        theta = np.pi / 4
        gate = rz(theta)
        expected = np.diag([np.exp(-1j * theta/2), np.exp(1j * theta/2)])
        np.testing.assert_array_almost_equal(gate.matrix, expected)


class TestQuantumGateClass:
    """Test QuantumGate class functionality."""

    def test_gate_multiplication(self):
        """Test gate composition via multiplication."""
        gate1 = pauli_x()
        gate2 = pauli_x()
        result = gate1 * gate2
        expected = np.eye(2)
        np.testing.assert_array_almost_equal(result.matrix, expected)

    def test_gate_state_multiplication(self):
        """Test gate applied to state vector."""
        gate = pauli_x()
        state = np.array([1, 0])  # |0>
        result = gate * state
        expected = np.array([0, 1])  # |1>
        np.testing.assert_array_almost_equal(result, expected)

    def test_gate_repr(self):
        """Test gate string representation."""
        gate = pauli_x()
        repr_str = repr(gate)
        assert "QuantumGate" in repr_str
