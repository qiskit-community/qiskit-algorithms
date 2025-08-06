# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
_circuit_key function from Qiskit 1.4

This file is to be deleted once all interfaces such as the BaseStateFidelity's and gradients' accept
PUB-like inputs instead of separate arguments.
"""
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Bit


def _bits_key(bits: tuple[Bit, ...], circuit: QuantumCircuit) -> tuple:
    return tuple(
        (
            circuit.find_bit(bit).index,
            tuple((reg[0].size, reg[0].name, reg[1]) for reg in circuit.find_bit(bit).registers),
        )
        for bit in bits
    )


def _format_params(param):
    if isinstance(param, np.ndarray):
        return param.data.tobytes()
    elif isinstance(param, QuantumCircuit):
        return _circuit_key(param)
    elif isinstance(param, Iterable):
        return tuple(param)
    return param


def _circuit_key(circuit: QuantumCircuit, functional: bool = True) -> tuple:
    """Private key function for QuantumCircuit.

    This is the workaround until :meth:`QuantumCircuit.__hash__` will be introduced.
    If key collision is found, please add elements to avoid it.

    Args:
        circuit: Input quantum circuit.
        functional: If True, the returned key only includes functional data (i.e. execution related).

    Returns:
        Composite key for circuit.
    """
    functional_key: tuple = (
        circuit.num_qubits,
        circuit.num_clbits,
        circuit.num_parameters,
        tuple(  # circuit.data
            (
                _bits_key(data.qubits, circuit),  # qubits
                _bits_key(data.clbits, circuit),  # classical bits
                data.operation.name,  # operation.name
                tuple(_format_params(param) for param in data.operation.params),  # operation.params
            )
            for data in circuit.data
        ),
        None if not hasattr(circuit, "op_start_times") else tuple(circuit.op_start_times),
    )
    if functional:
        return functional_key
    return (
        circuit.name,
        *functional_key,
    )
