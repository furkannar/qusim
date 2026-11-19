r"""
qusim_np/noise_models/noise_model.py
Gate-level noise model for QuantumCircuit

A NoiseModel is a list of rules "after gate G (optionally only on qubits Q),
apply the single-qubit channel with Kraus operators {K_k}".  When a circuit
with a noise model is simulated (QuantumCircuit.density_matrix), each gate
U is applied as rho -> U rho U^dagger and then every matching rule applies
its channel to EACH qubit the gate acted on:

    rho -> E_q(rho) = sum_k K_k^{(q)} rho K_k^{(q) dagger}

This is the standard local (uncorrelated) noise model: a two-qubit gate
followed by depolarizing noise means independent depolarizing on both of
its qubits.  Correlated multi-qubit channels are not modelled.

Kraus operators come from the existing constructors, e.g.
depolarizing_kraus(p), amplitude_damping_kraus(gamma),
phase_damping_kraus(lam), random_pauli_kraus(probs), and are checked for
completeness (sum_k K_k^dagger K_k = I) when the rule is added.
"""

import numpy as np

from qusim_np.noise_models.channels import _validate_kraus_ops

# Alternate spellings of gate names, mapped to the names QuantumCircuit records.
GATE_ALIASES = {"cnot": "cx", "toffoli": "ccx"}


def _canonical_gate_name(name):
    if not isinstance(name, str) or not name:
        raise ValueError(f"gate name must be a non-empty string, got {name!r}")
    name = name.lower()
    return GATE_ALIASES.get(name, name)


class NoiseModel:
    """Collection of gate-triggered single-qubit noise channels."""

    def __init__(self):
        self._rules = []   # list of (kraus_ops tuple, gates frozenset|None, qubits frozenset|None)

    def add_gate_error(self, kraus_ops, gates=None, qubits=None):
        r"""
        Add a channel applied after matching gates.

        Args:
            kraus_ops: sequence of 2x2 Kraus operators (single-qubit channel)
            gates: None (after every gate), a gate name ("cx"), or a
                sequence of names.  Aliases "cnot" and "toffoli" are accepted.
            qubits: None (any qubit), an int, or a sequence of ints.  The
                channel is applied only to the gate's qubits in this set.

        Returns:
            NoiseModel: self, so calls can be chained
        """
        ops = tuple(_validate_kraus_ops(kraus_ops))

        if gates is None:
            gate_set = None
        else:
            if isinstance(gates, str):
                gates = [gates]
            gate_set = frozenset(_canonical_gate_name(g) for g in gates)
            if not gate_set:
                raise ValueError("gates must be None or a non-empty collection")

        if qubits is None:
            qubit_set = None
        else:
            if isinstance(qubits, (int, np.integer)) and not isinstance(qubits, bool):
                qubits = [qubits]
            qubit_set = set()
            for q in qubits:
                if not isinstance(q, (int, np.integer)) or isinstance(q, bool) or q < 0:
                    raise ValueError(f"qubit must be a non-negative integer, got {q!r}")
                qubit_set.add(int(q))
            if not qubit_set:
                raise ValueError("qubits must be None or a non-empty collection")
            qubit_set = frozenset(qubit_set)

        self._rules.append((ops, gate_set, qubit_set))
        return self

    def channels_for(self, gate_name, gate_qubits):
        """
        Yield (kraus_ops, qubit) pairs to apply after `gate_name` on `gate_qubits`,
        in rule order, then qubit order.
        """
        name = _canonical_gate_name(gate_name)
        for ops, gate_set, qubit_set in self._rules:
            if gate_set is not None and name not in gate_set:
                continue
            for q in gate_qubits:
                if qubit_set is None or q in qubit_set:
                    yield ops, q

    @property
    def rules(self):
        """Read-only view of the rules as (kraus_ops, gates, qubits) tuples."""
        return tuple(self._rules)

    def __len__(self):
        return len(self._rules)

    def __repr__(self):
        lines = [f"NoiseModel({len(self._rules)} rule(s))"]
        for ops, gate_set, qubit_set in self._rules:
            g = "all gates" if gate_set is None else ", ".join(sorted(gate_set))
            q = "all qubits" if qubit_set is None else ", ".join(map(str, sorted(qubit_set)))
            lines.append(f"  {len(ops)} Kraus op(s) after [{g}] on [{q}]")
        return "\n".join(lines)
