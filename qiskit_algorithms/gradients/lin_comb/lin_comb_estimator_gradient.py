# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Gradient of probabilities with linear combination of unitaries (LCU)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..base.base_estimator_gradient import BaseEstimatorGradient
from ..base.estimator_gradient_result import EstimatorGradientResult
from ..utils import DerivativeType, _make_lin_comb_gradient_circuit, _make_lin_comb_observables
from ...custom_types import Transpiler
from ...run_estimator_job import run_estimator_job
from ...utils.circuit_key import _circuit_key


class LinCombEstimatorGradient(BaseEstimatorGradient):
    """Compute the gradients of the expectation values.
    This method employs a linear combination of unitaries [1].

    **Reference:**
    [1] Schuld et al., Evaluating analytic gradients on quantum hardware, 2018
    `arXiv:1811.11184 <https://arxiv.org/pdf/1811.11184.pdf>`_
    """

    SUPPORTED_GATES = [
        "rx",
        "ry",
        "rz",
        "rzx",
        "rzz",
        "ryy",
        "rxx",
        "cx",
        "cy",
        "cz",
        "ccx",
        "swap",
        "iswap",
        "h",
        "t",
        "s",
        "sdg",
        "x",
        "y",
        "z",
    ]

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        precision: float | None = None,
        derivative_type: DerivativeType = DerivativeType.REAL,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the gradients.
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``. Defaults to
                ``DerivativeType.REAL``.

                    - ``DerivativeType.REAL`` computes :math:`2 \mathrm{Re}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.IMAG`` computes :math:`2 \mathrm{Im}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.COMPLEX`` computes :math:`2 ⟨ψ(ω)|O(θ)|dω ψ(ω)〉`.
            precision: Precision to be used by the underlying Estimator. If provided, this number
                takes precedence over the default precision of the primitive. If None, the default
                precision of the primitive is used.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are produced within this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        """
        self._lin_comb_cache: dict[tuple, dict[Parameter, QuantumCircuit]] = {}
        super().__init__(
            estimator,
            precision,
            derivative_type=derivative_type,
            transpiler=transpiler,
            transpiler_options=transpiler_options,
        )

    @BaseEstimatorGradient.derivative_type.setter  # type: ignore[attr-defined]
    def derivative_type(self, derivative_type: DerivativeType) -> None:
        """Set the derivative type."""
        self._derivative_type = derivative_type

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        precision: float | Sequence[float] | None,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, precision=precision
        )
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        precision: float | Sequence[float] | None,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        has_transformed_precision = False

        if isinstance(precision, float) or precision is None:
            precision = [precision] * len(circuits)
            has_transformed_precision = True

        metadata = []
        all_n = []
        pubs = []

        if not (len(circuits) == len(observables) == len(parameters) == len(parameter_values)):
            raise ValueError(
                f"circuits, observables, parameters, and parameter_values must have the same length, "
                f"but have respective lengths {len(circuits)},  {len(observables)}, {len(parameters)} "
                f"and {len(parameter_values)}."
            )

        for circuit, observable, parameter_values_, parameters_, precision_ in zip(
            circuits, observables, parameter_values, parameters, precision
        ):
            # Prepare circuits for the gradient of the specified parameters.
            meta = {"parameters": parameters_}
            circuit_key = _circuit_key(circuit)

            if circuit_key not in self._lin_comb_cache:
                # Cache the circuits for the linear combination of unitaries.
                # We only cache the circuits for the specified parameters in the future.
                self._lin_comb_cache[circuit_key] = _make_lin_comb_gradient_circuit(
                    circuit, add_measurement=False
                )

            lin_comb_circuits = self._lin_comb_cache[circuit_key]
            gradient_circuits = []

            for param in parameters_:
                gradient_circuits.append(lin_comb_circuits[param])

            n = len(gradient_circuits)
            # Make the observable as :class:`~qiskit.quantum_info.SparsePauliOp` and
            # add an ancillary operator to compute the gradient.
            observable = SparsePauliOp(observable)
            observable_1, observable_2 = _make_lin_comb_observables(
                observable, self._derivative_type
            )
            # If its derivative type is `DerivativeType.COMPLEX`, calculate the gradient
            # of the real and imaginary parts separately.
            meta["derivative_type"] = self.derivative_type
            metadata.append(meta)
            # Combine inputs into a single job to reduce overhead.
            if self._derivative_type == DerivativeType.COMPLEX:
                all_n.append(2 * n)
                pubs.extend(
                    [
                        (gradient_circuit, observable_1, parameter_values_, precision_)
                        for gradient_circuit in gradient_circuits
                    ]
                )
                pubs.extend(
                    [
                        (gradient_circuit, observable_2, parameter_values_, precision_)
                        for gradient_circuit in gradient_circuits
                    ]
                )
            else:
                all_n.append(n)
                pubs.extend(
                    [
                        (gradient_circuit, observable_1, parameter_values_, precision_)
                        for gradient_circuit in gradient_circuits
                    ]
                )

        if self._transpiler is not None:
            for index, pub in enumerate(pubs):
                new_circuit = self._transpiler.run(pub[0], **self._transpiler_options)
                new_observable = pub[1].apply_layout(new_circuit.layout)
                pubs[index] = (new_circuit, new_observable) + pub[2:]

        # Run the single job with all circuits.
        results = run_estimator_job(self._estimator, pubs)

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0

        for n in all_n:
            # this disable is needed as Pylint does not understand derivative_type is a property if
            # it is only defined in the base class and the getter is in the child
            # pylint: disable=comparison-with-callable
            if self.derivative_type == DerivativeType.COMPLEX:
                gradient = np.zeros(n // 2, dtype="complex")
                gradient.real = np.array(
                    [result.data.evs for result in results[partial_sum_n : partial_sum_n + n // 2]]
                )
                gradient.imag = np.array(
                    [
                        result.data.evs
                        for result in results[partial_sum_n + n // 2 : partial_sum_n + n]
                    ]
                )

            else:
                gradient = np.real(
                    [result.data.evs for result in results[partial_sum_n : partial_sum_n + n]]
                )

            partial_sum_n += n
            gradients.append(gradient)

        if has_transformed_precision:
            precision = precision[0]

            if precision is None:
                precision = results[0].metadata["target_precision"]
        else:
            for i, (precision_, result) in enumerate(zip(precision, results)):
                if precision_ is None:
                    precision[i] = results[i].metadata["target_precision"]

        return EstimatorGradientResult(gradients=gradients, metadata=metadata, precision=precision)
