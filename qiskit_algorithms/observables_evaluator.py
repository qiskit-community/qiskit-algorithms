# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Evaluator of observables for algorithms."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .exceptions import AlgorithmError
from .list_or_dict import ListOrDict


def estimate_observables(
    estimator: BaseEstimatorV2,
    quantum_state: QuantumCircuit,
    observables: ListOrDict[BaseOperator],
    parameter_values: Sequence[float] | None = None,
    threshold: float = 1e-12,
) -> ListOrDict[tuple[float, dict[str, Any]]]:
    """
    Accepts a sequence of operators and calculates their expectation values - means
    and metadata. They are calculated with respect to a quantum state provided. A user
    can optionally provide a threshold value which filters mean values falling below the threshold.

    Args:
        estimator: An estimator primitive used for calculations.
        quantum_state: A (parameterized) quantum circuit preparing a quantum state that expectation
            values are computed against. It is expected to be an ISA circuit.
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.
        parameter_values: Optional list of parameters values to evaluate the quantum circuit on.
        threshold: A threshold value that defines which mean values should be neglected (helpful for
            ignoring numerical instabilities close to 0).

    Returns:
        A list or a dictionary of tuples (mean, metadata).

    Raises:
        AlgorithmError: If a primitive job is not successful.
    """

    if isinstance(observables, dict):
        observables_list = list(observables.values())
    else:
        observables_list = observables

    if len(observables_list) > 0:
        observables_list = _handle_zero_ops(observables_list)
        parameter_values_: Sequence[float] | Sequence[Sequence[float]] | None = parameter_values
        try:
            estimator_job = estimator.run([(quantum_state, observables_list, parameter_values_)])
            estimator_result = estimator_job.result()[0]
            expectation_values = estimator_result.data.evs
        except Exception as exc:
            raise AlgorithmError("The primitive job failed!") from exc

        metadata = estimator_result.metadata
        # Discard values below threshold
        observables_means = expectation_values * (np.abs(expectation_values) > threshold)
        # zip means and metadata into tuples
        observables_results = list(zip(observables_means, [metadata] * len(observables_means)))
    else:
        observables_results = []

    return _prepare_result(observables_results, observables)


def _handle_zero_ops(
    observables_list: list[BaseOperator],
) -> list[BaseOperator]:
    """Replaces all occurrence of operators equal to 0 in the list with an equivalent ``SparsePauliOp``
    operator."""
    if observables_list:
        zero_op = SparsePauliOp.from_list([("I" * observables_list[0].num_qubits, 0)])
        for ind, observable in enumerate(observables_list):
            if observable == 0:
                observables_list[ind] = zero_op
    return observables_list


def _prepare_result(
    observables_results: list[tuple[float, dict]],
    observables: ListOrDict[BaseOperator],
) -> ListOrDict[tuple[float, dict[str, Any]]]:
    """
    Prepares a list of tuples of eigenvalues and metadata tuples from
    ``observables_results`` and ``observables``.

    Args:
        observables_results: A list of tuples (mean, metadata).
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.

    Returns:
        A list or a dictionary of tuples (mean, metadata).
    """

    observables_eigenvalues: ListOrDict[tuple[float, dict]]

    if isinstance(observables, list):
        observables_eigenvalues = []
        for value in observables_results:
            observables_eigenvalues.append(value)

    else:
        observables_eigenvalues = {}
        for key, value in zip(observables.keys(), observables_results):
            observables_eigenvalues[key] = value

    return observables_eigenvalues
