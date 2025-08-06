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
from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSamplerV2

from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult
from ...custom_types import Transpiler

from ...exceptions import AlgorithmError


class SPSASamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the Simultaneous Perturbation Stochastic
    Approximation (SPSA) [1].

    **Reference:**
    [1] J. C. Spall, Adaptive stochastic approximation by the simultaneous perturbation method in
    IEEE Transactions on Automatic Control, vol. 45, no. 10, pp. 1839-1853, Oct 2020,
    `doi: 10.1109/TAC.2000.880982 <https://ieeexplore.ieee.org/document/880982>`_.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        sampler: BaseSamplerV2,
        epsilon: float,
        batch_size: int = 1,
        seed: int | None = None,
        shots: int | None = None,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the SPSA gradients.
            batch_size: number of gradients to average.
            seed: The seed for a random perturbation vector.
            shots: Number of shots to be used by the underlying sampler.
                The order of priority is: number of shots in ``run`` method > fidelity's
                number of shots > primitive's default number of shots.
                Higher priority setting overrides lower priority setting.
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
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._seed = np.random.default_rng(seed)

        super().__init__(
            sampler, shots, transpiler=transpiler, transpiler_options=transpiler_options
        )

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        shots: int | Sequence[int] | None = None,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        metadata, offsets = [], []
        all_n = []
        has_transformed_shots = False

        if isinstance(shots, int) or shots is None:
            shots = [shots] * len(circuits)
            has_transformed_shots = True

        pubs = []

        if self._transpiler is not None:
            circuits = self._transpiler.run(circuits, **self._transpiler_options)

        for circuit, parameter_values_, parameters_, shots_ in zip(
            circuits, parameter_values, parameters, shots
        ):
            # Indices of parameters to be differentiated.
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata.append({"parameters": parameters_})
            offset = np.array(
                [
                    (-1) ** (self._seed.integers(0, 2, len(circuit.parameters)))
                    for _ in range(self._batch_size)
                ]
            )
            plus = [parameter_values_ + self._epsilon * offset_ for offset_ in offset]
            minus = [parameter_values_ - self._epsilon * offset_ for offset_ in offset]
            offsets.append(offset)

            # Combine inputs into a single job to reduce overhead.
            n = 2 * self._batch_size
            all_n.append(n)
            pubs.append((circuit, plus + minus, shots_))

        # Run the single job with all circuits.
        job = self._sampler.run(pubs)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        for i, (n, result_n) in enumerate(zip(all_n, results)):
            dist_diffs = {}
            result = [
                {label: value / res.num_shots for label, value in res.get_int_counts().items()}
                for res in getattr(result_n.data, next(iter(result_n.data)))
            ]
            for j, (dist_plus, dist_minus) in enumerate(zip(result[: n // 2], result[n // 2 :])):
                dist_diff: dict[int, float] = defaultdict(float)
                for key, value in dist_plus.items():
                    dist_diff[key] += value / (2 * self._epsilon)
                for key, value in dist_minus.items():
                    dist_diff[key] -= value / (2 * self._epsilon)
                dist_diffs[j] = dist_diff
            gradient = []
            indices = [circuits[i].parameters.data.index(p) for p in metadata[i]["parameters"]]
            for j in indices:
                gradient_j: dict[int, float] = defaultdict(float)
                for k in range(self._batch_size):
                    for key, value in dist_diffs[k].items():
                        gradient_j[key] += value * offsets[i][k][j]
                gradient_j = {key: value / self._batch_size for key, value in gradient_j.items()}
                gradient.append(gradient_j)
            gradients.append(gradient)
            partial_sum_n += n

        if has_transformed_shots:
            shots = shots[0]

            if shots is None:
                shots = results[0].metadata["shots"]
        else:
            for i, (shots_, result) in enumerate(zip(shots, results)):
                if shots_ is None:
                    shots[i] = result.metadata["shots"]

        return SamplerGradientResult(gradients=gradients, metadata=metadata, shots=shots)
