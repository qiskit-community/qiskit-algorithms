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
from typing import Any

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..base.base_estimator_gradient import BaseEstimatorGradient
from ..base.estimator_gradient_result import EstimatorGradientResult
from ...custom_types import Transpiler

from ...exceptions import AlgorithmError


class SPSAEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation value by the Simultaneous Perturbation Stochastic
    Approximation (SPSA) [1].

    **Reference:**
    [1] J. C. Spall, Adaptive stochastic approximation by the simultaneous perturbation method in
    IEEE Transactions on Automatic Control, vol. 45, no. 10, pp. 1839-1853, Oct 2020,
    `doi: 10.1109/TAC.2000.880982 <https://ieeexplore.ieee.org/document/880982>`_
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        estimator: BaseEstimatorV2,
        epsilon: float,
        batch_size: int = 1,
        seed: int | None = None,
        precision: float | None = None,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            epsilon: The offset size for the SPSA gradients.
            batch_size: The number of gradients to average.
            seed: The seed for a random perturbation vector.
            precision: Precision to be used by the underlying Estimator. If provided, this number
                takes precedence over the default precision of the primitive. If None, the default
                precision of the primitive is used.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are run when using this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        Raises:
            ValueError: If ``epsilon`` is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._seed = np.random.default_rng(seed)

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
        offsets = []
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

        if not (
            len(circuits)
            == len(observables)
            == len(parameters)
            == len(parameter_values)
            == len(precision)
        ):
            raise ValueError(
                f"circuits, observables, parameters, parameter_values and precision must have the same "
                f"length, but have respective lengths {len(circuits)},  {len(observables)}, "
                f"{len(parameters)}, {len(parameter_values)} and {len(precision)}."
            )

        for circuit, observable, parameter_values_, parameters_, precision_ in zip(
            circuits, observables, parameter_values, parameters, precision
        ):
            metadata.append({"parameters": parameters_})
            # Make random perturbation vectors.
            offset = [
                (-1) ** (self._seed.integers(0, 2, len(circuit.parameters)))
                for _ in range(self._batch_size)
            ]
            plus = [parameter_values_ + self._epsilon * offset_ for offset_ in offset]
            minus = [parameter_values_ - self._epsilon * offset_ for offset_ in offset]
            offsets.append(offset)

            # Combine inputs into a single job to reduce overhead.
            pubs.append((circuit, observable, plus + minus, precision_))

        # Run the single job with all circuits.
        job = self._estimator.run(pubs)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        # Compute the gradients.
        gradients = []
        for i, result in enumerate(results):
            evs = result.data.evs
            n = evs.shape[0] // 2
            diffs = (evs[:n] - evs[n:]) / (2 * self._epsilon)
            # Calculate the gradient for each batch. Note that (``diff`` / ``offset``) is the gradient
            # since ``offset`` is a perturbation vector of 1s and -1s.
            batch_gradients = np.array([diff / offset for diff, offset in zip(diffs, offsets[i])])
            # Take the average of the batch gradients.
            gradient = np.mean(batch_gradients, axis=0)
            indices = [circuits[i].parameters.data.index(p) for p in metadata[i]["parameters"]]
            gradients.append(gradient[indices])

        if has_transformed_precision:
            precision = precision[0]

            if precision is None:
                precision = results[0].metadata["target_precision"]
        else:
            for i, (precision_, result) in enumerate(zip(precision, results)):
                if precision_ is None:
                    precision[i] = results[i].metadata["target_precision"]

        return EstimatorGradientResult(gradients=gradients, metadata=metadata, precision=precision)
