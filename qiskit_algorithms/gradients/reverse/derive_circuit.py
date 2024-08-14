# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Split a circuit into subcircuits, each containing a single parameterized gate."""

from __future__ import annotations
import itertools
from collections.abc import Sequence

from qiskit.circuit import QuantumCircuit, Parameter, Gate, ParameterExpression
from qiskit.circuit.library import RXGate, RYGate, RZGate, CRXGate, CRYGate, CRZGate


def gradient_lookup(gate: Gate) -> list[tuple[complex, QuantumCircuit]]:
    """Returns a circuit implementing the gradient of the input gate.

    Args:
        gate: The gate whose derivative is returned.

    Returns:
        The derivative of the input gate as list of ``(coeff, circuit)`` pairs,
        where the sum of all ``coeff * circuit`` elements describes the full derivative.
        The circuit is the unitary part of the derivative with a potential separate ``coeff``.
        The output is a list as derivatives of e.g. controlled gates can only be described
        as a sum of ``coeff * circuit`` pairs.

    Raises:
        NotImplementedError: If the derivative of ``gate`` is not implemented.
    """

    param = gate.params[0]
    if isinstance(gate, RXGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rx(param, 0)
        derivative.x(0)
        return [(-0.5j, derivative)]
    if isinstance(gate, RYGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.ry(param, 0)
        derivative.y(0)
        return [(-0.5j, derivative)]
    if isinstance(gate, RZGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rz(param, 0)
        derivative.z(0)
        return [(-0.5j, derivative)]
    if isinstance(gate, CRXGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rx(param, 1)
        proj1.x(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rx(param, 1)
        proj2.x(1)

        return [(-0.25j, proj1), (0.25j, proj2)]
    if isinstance(gate, CRYGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.ry(param, 1)
        proj1.y(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.ry(param, 1)
        proj2.y(1)

        return [(-0.25j, proj1), (0.25j, proj2)]
    if isinstance(gate, CRZGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rz(param, 1)
        proj1.z(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rz(param, 1)
        proj2.z(1)

        return [(-0.25j, proj1), (0.25j, proj2)]
    raise NotImplementedError("Cannot implement gradient for", gate)


def derive_circuit(
    circuit: QuantumCircuit, parameter: Parameter, check: bool = True
) -> Sequence[tuple[complex, QuantumCircuit]]:
    """Return the analytic gradient expression of the input circuit wrt. a single parameter.

    Returns a list of ``(coeff, gradient_circuit)`` tuples, where the derivative of the circuit is
    given by the sum of the gradient circuits multiplied by their coefficient.

    For example, the circuit::

           ┌───┐┌───────┐┌─────┐
        q: ┤ H ├┤ Rx(x) ├┤ Sdg ├
           └───┘└───────┘└─────┘

    returns the coefficient `-0.5j` and the circuit equivalent to::

           ┌───┐┌───────┐┌───┐┌─────┐
        q: ┤ H ├┤ Rx(x) ├┤ X ├┤ Sdg ├
           └───┘└───────┘└───┘└─────┘

    as the derivative of `Rx(x)` is `-0.5j Rx(x) X`.

    Args:
        circuit: The quantum circuit to derive.
        parameter: The parameter with respect to which we derive.
        check: If ``True`` (default) check that the parameter is valid and that no product
            rule is required.

    Returns:
        A list of ``(coeff, gradient_circuit)`` tuples.

    Raises:
        ValueError: If ``parameter`` is of the wrong type.
        ValueError: If ``parameter`` is not in this circuit.
        NotImplementedError: If a non-unique parameter is added, as the product rule is not yet
            supported in this function.
    """
    if check:
        # this is added as useful user-warning, since sometimes ``ParameterExpression``s are
        # passed around instead of ``Parameter``s
        if not isinstance(parameter, Parameter):
            raise ValueError(f"parameter must be of type Parameter, not {type(parameter)}.")

        if parameter not in circuit.parameters:
            raise ValueError(f"The parameter {parameter} is not in this circuit.")

        # check uniqueness
        seen_parameters: set[Parameter] = set()
        for instruction in circuit.data:
            # get parameters in the current operation
            new_parameters = set()
            for p in instruction.operation.params:
                if isinstance(p, ParameterExpression):
                    new_parameters.update(p.parameters)

            if duplicates := seen_parameters.intersection(new_parameters):
                raise NotImplementedError(
                    "Product rule is not supported, circuit parameters must be unique, but "
                    f"{duplicates} are duplicated."
                )

            seen_parameters.update(new_parameters)

    summands, op_context = [], []
    for i, op in enumerate(circuit.data):
        gate = op.operation
        op_context.append((op.qubits, op.clbits))
        if parameter in gate.params:
            coeffs_and_grads = gradient_lookup(gate)
            summands += [coeffs_and_grads]
        else:
            summands += [[(1, gate)]]

    gradient = []
    for product_rule_term in itertools.product(*summands):
        summand_circuit = QuantumCircuit(*circuit.qregs)
        c = complex(1)
        for i, term in enumerate(product_rule_term):
            c *= term[0]
            # Qiskit changed the format of the stored value. The newer Qiskit has this internal
            # method to go from the older (legacy) format to new. This logic may need updating
            # at some point if this internal method goes away.
            if hasattr(summand_circuit.data, "_resolve_legacy_value"):
                value = summand_circuit.data._resolve_legacy_value(term[1], *op_context[i])
                summand_circuit.data.append(value)
            else:
                summand_circuit.data.append([term[1], *op_context[i]])
        gradient += [(c, summand_circuit.copy())]

    return gradient
