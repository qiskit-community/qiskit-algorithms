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

from collections import defaultdict
from typing import Literal, Sequence, Any

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSamplerV2

from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult
from ...custom_types import Transpiler

from ...exceptions import AlgorithmError


class FiniteDiffSamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by finite difference method [1].

    **Reference:**
    [1] `Finite difference method <https://en.wikipedia.org/wiki/Finite_difference_method>`_
    """

    def __init__(
        self,
        sampler: BaseSamplerV2,
        epsilon: float,
        shots: int | None = None,
        *,
        method: Literal["central", "forward", "backward"] = "central",
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ):
        r"""
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            shots: Number of shots to be used by the underlying Sampler. If provided, this number
                takes precedence over the default precision of the primitive. If None, the default
                number of shots of the primitive is used.
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
            sampler, shots, transpiler=transpiler, transpiler_options=transpiler_options
        )

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None,
        *,
        shots: int | Sequence[int] | None,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        metadata = []
        all_n = []
        has_transformed_shots = False

        if isinstance(shots, int) or shots is None:
            shots = [shots] * len(circuits)
            has_transformed_shots = True

        if self._transpiler is not None:
            circuits = self._transpiler.run(circuits, **self._transpiler_options)

        pubs = []
        for circuit, parameter_values_, parameters_, shots_ in zip(
            circuits, parameter_values, parameters, shots
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
                pubs.append((circuit, plus.tolist() + minus.tolist(), shots_))
            elif self._method == "forward":
                plus = parameter_values_ + self._epsilon * offset
                n = len(indices) + 1
                pubs.append((circuit, [parameter_values_] + plus.tolist(), shots_))
                all_n.append(n)
            elif self._method == "backward":
                minus = parameter_values_ - self._epsilon * offset
                n = len(indices) + 1
                pubs.append((circuit, [parameter_values_] + minus.tolist(), shots_))
                all_n.append(n)

        # Run the single job with all circuits.
        job = self._sampler.run(pubs)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        # Compute the gradients.
        gradients = []

        for n, result_n in zip(all_n, results):
            gradient = []
            result = [
                {label: value / res.num_shots for label, value in res.get_int_counts().items()}
                for res in getattr(result_n.data, next(iter(result_n.data)))
            ]
            if self._method == "central":
                for dist_plus, dist_minus in zip(result[: n // 2], result[n // 2 :]):
                    grad_dist: dict[int, float] = defaultdict(float)
                    for key, value in dist_plus.items():
                        grad_dist[key] += value / (2 * self._epsilon)
                    for key, value in dist_minus.items():
                        grad_dist[key] -= value / (2 * self._epsilon)
                    gradient.append(dict(grad_dist))
            elif self._method == "forward":
                dist_zero = result[0]
                for dist_plus in result[1:]:
                    grad_dist = defaultdict(float)
                    for key, value in dist_plus.items():
                        grad_dist[key] += value / self._epsilon
                    for key, value in dist_zero.items():
                        grad_dist[key] -= value / self._epsilon
                    gradient.append(dict(grad_dist))
            elif self._method == "backward":
                dist_zero = result[0]
                for dist_minus in result[1:]:
                    grad_dist = defaultdict(float)
                    for key, value in dist_zero.items():
                        grad_dist[key] += value / self._epsilon
                    for key, value in dist_minus.items():
                        grad_dist[key] -= value / self._epsilon
                    gradient.append(dict(grad_dist))

            gradients.append(gradient)

        if has_transformed_shots:
            shots = shots[0]

            if shots is None:
                shots = results[0].metadata["shots"]
        else:
            for i, (shots_, result) in enumerate(zip(shots, results)):
                if shots_ is None:
                    shots[i] = result.metadata["shots"]

        return SamplerGradientResult(gradients=gradients, metadata=metadata, shots=shots)
