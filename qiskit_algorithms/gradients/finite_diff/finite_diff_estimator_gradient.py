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

"""Gradient of Sampler with Finite difference method."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Any

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..base.base_estimator_gradient import BaseEstimatorGradient
from ..base.estimator_gradient_result import EstimatorGradientResult
from ...custom_types import Transpiler

from ...exceptions import AlgorithmError


class FiniteDiffEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation values by finite difference method [1].

    **Reference:**
    [1] `Finite difference method <https://en.wikipedia.org/wiki/Finite_difference_method>`_
    """

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        epsilon: float,
        precision: float | None = None,
        *,
        method: Literal["central", "forward", "backward"] = "central",
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            precision: Precision to be used by the underlying Estimator. If provided, this number
                takes precedence over the default precision of the primitive. If None, the default
                precision of the primitive is used.
            method: The computation method of the gradients.

                    - ``central`` computes :math:`\frac{f(x+e)-f(x-e)}{2e}`,
                    - ``forward`` computes :math:`\frac{f(x+e) - f(x)}{e}`,
                    - ``backward`` computes :math:`\frac{f(x)-f(x-e)}{e}`

                where :math:`e` is epsilon.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are run when using this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.

        Raises:
            ValueError: If ``epsilon`` is not positive.
            TypeError: If ``method`` is invalid.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._epsilon = epsilon
        if method not in ("central", "forward", "backward"):
            raise TypeError(
                f"The argument method should be central, forward, or backward: {method} is given."
            )
        self._method = method
        super().__init__(
            estimator, precision, transpiler=transpiler, transpiler_options=transpiler_options
        )

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
        metadata = []
        all_n = []
        has_transformed_precision = False

        if isinstance(precision, float) or precision is None:
            precision = [precision] * len(circuits)
            has_transformed_precision = True

        if self._transpiler is not None:
            circuits = self._transpiler.run(circuits, **self._transpiler_options)
            observables = [
                obs.apply_layout(circuit.layout) for (circuit, obs) in zip(circuits, observables)
            ]

        pubs = []

        for circuit, observable, parameter_values_, parameters_, precision_ in zip(
            circuits, observables, parameter_values, parameters, precision
        ):
            # Indices of parameters to be differentiated
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata.append({"parameters": parameters_})

            # Combine inputs into a single job to reduce overhead.
            offset = np.identity(circuit.num_parameters)[indices, :]
            if self._method == "central":
                plus = parameter_values_ + self._epsilon * offset
                minus = parameter_values_ - self._epsilon * offset
                n = 2 * len(indices)
                all_n.append(n)
                pubs.append((circuit, observable, plus.tolist() + minus.tolist(), precision_))
            elif self._method == "forward":
                plus = parameter_values_ + self._epsilon * offset
                n = len(indices) + 1
                all_n.append(n)
                pubs.append((circuit, observable, [parameter_values_] + plus.tolist(), precision_))
            elif self._method == "backward":
                minus = parameter_values_ - self._epsilon * offset
                n = len(indices) + 1
                all_n.append(n)
                pubs.append((circuit, observable, [parameter_values_] + minus.tolist(), precision_))

        # Run the single job with all circuits.
        job = self._estimator.run(pubs)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        # Compute the gradients
        gradients = []
        partial_sum_n = 0
        for n, result_n in zip(all_n, results):
            # Ensure gradient is always defined for the append below after the if block
            # otherwise lint errors out. I left the if block as it has been coded though
            # as the values are checked in the constructor I could have made the last elif
            # a simple else instead of defining this here.
            gradient = None
            result = result_n.data.evs
            if self._method == "central":
                gradient = (result[: n // 2] - result[n // 2 :]) / (2 * self._epsilon)
            elif self._method == "forward":
                gradient = (result[1:] - result[0]) / self._epsilon
            elif self._method == "backward":
                gradient = (result[0] - result[1:]) / self._epsilon
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
