"""
qusim/gates.py
Quantum Gates Implementation

This module implements common quantum gates using numpy matrices.

Single-Qubit Gates:
- Pauli-X (NOT): X = [[0, 1], [1, 0]]
- Pauli-Y: Y = [[0, -i], [i, 0]]
- Pauli-Z: Z = [[1, 0], [0, -1]]
- Hadamard: H = 1/sqrt(2) * [[1, 1], [1, -1]]
- Phase (S): S = [[1, 0], [0, i]]
- T gate: T = [[1, 0], [0, exp(i*pi/4)]]

Two-Qubit Gates:
- CNOT: Controlled-NOT gate
- SWAP: Swaps two qubits
- Controlled Phase: Adds phase when control is |1>

Three-Qubit Gates:
- Toffoli (CCNOT): Flips target when both controls are |1>

Rotation Gates (SU(2) matrices):
- Rx(theta) = [[cos(theta/2), -i*sin(theta/2)], [-i*sin(theta/2), cos(theta/2)]]
- Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
- Rz(theta) = [[exp(-i*theta/2), 0], [0, exp(i*theta/2)]]

Unitary Constraint:
For rotation gates, we enforce:
- U^dagger * U = I (unitary condition)
- det(U) = 1 (special unitary group SU(2))
"""

import numpy as np
from qusim.config import config

class QuantumGate:
    """Base class for quantum gates"""
    
    def __init__(self, matrix, validate_su2=False):
        self.matrix = np.array(matrix, dtype=complex)
        if validate_su2:
            self._validate_unitary()
    
    def _validate_unitary(self):
        """Validate that the gate is unitary with determinant 1"""
        n = self.matrix.shape[0]
        # Check U^dagger U = I
        identity_check = self.matrix.conj().T @ self.matrix
        expected_identity = np.eye(n)
        if not np.allclose(identity_check, expected_identity, atol=config.SU2_TOLERANCE):
            raise ValueError(f"Gate is not unitary: U^dagger U != I")
        # Check det(U) = 1
        det = np.linalg.det(self.matrix)
        if not np.isclose(det, 1.0, atol=config.SU2_TOLERANCE):
            raise ValueError(f"Gate determinant is not 1: det(U) = {det}")
    
    def __mul__(self, other):
        """Compose gates (matrix multiplication)"""
        if isinstance(other, QuantumGate):
            return QuantumGate(self.matrix @ other.matrix)
        elif isinstance(other, np.ndarray):
            return self.matrix @ other
        return NotImplemented
    
    def __repr__(self):
        return f"QuantumGate:\n{self.matrix}"


# Single-qubit gates
def pauli_x():
    """Pauli-X gate (NOT gate)"""
    return QuantumGate([[0, 1], [1, 0]])


def pauli_y():
    """Pauli-Y gate"""
    return QuantumGate([[0, -1j], [1j, 0]])


def pauli_z():
    """Pauli-Z gate"""
    return QuantumGate(np.diag([1, -1]))


def hadamard():
    """Hadamard gate - creates superposition"""
    return QuantumGate([[config.INV_SQRT_2, config.INV_SQRT_2], [config.INV_SQRT_2, -config.INV_SQRT_2]])


def phase_gate():
    """Phase gate (S gate)"""
    return QuantumGate(np.diag([1, 1j]))


def t_gate():
    """T gate (pi/8 gate)"""
    return QuantumGate(np.diag([1, np.exp(1j * np.pi / 4)]))


def identity(n=1):
    """Identity gate"""
    return QuantumGate(np.eye(2**n))


# Two-qubit gates
def cnot():
    """CNOT gate - controlled NOT"""
    return QuantumGate([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def swap():
    """SWAP gate - swaps two qubits"""
    return QuantumGate([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def controlled_phase():
    """Controlled phase gate"""
    return QuantumGate(np.diag([1, 1, 1, -1]))


# Three-qubit gates
def toffoli():
    """Toffoli (CCNOT) gate - flips target qubit when both controls are |1>"""
    # Basis order |q0 q1 q2>, index = 4*q0 + 2*q1 + q2
    # Swaps |110> (index 6) and |111> (index 7)
    matrix = np.eye(8)
    matrix[[6, 7]] = matrix[[7, 6]]
    return QuantumGate(matrix)


# Rotation gates
def rx(theta):
    """Rotation around X-axis"""
    return QuantumGate([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]], validate_su2=True)


def ry(theta):
    """Rotation around Y-axis"""
    return QuantumGate([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], validate_su2=True)


def rz(theta):
    """Rotation around Z-axis"""
    return QuantumGate(np.diag([np.exp(-1j * theta/2), np.exp(1j * theta/2)]), validate_su2=True)
