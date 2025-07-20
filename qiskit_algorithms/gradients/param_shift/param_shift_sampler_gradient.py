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
Gradient of probabilities with parameter shift
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from qiskit.circuit import Parameter, QuantumCircuit

from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult
from ..utils import _make_param_shift_parameter_values

from ...exceptions import AlgorithmError


class ParamShiftSamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the parameter shift rule [1].

    **Reference:**
    [1] Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., and Killoran, N. Evaluating analytic
    gradients on quantum hardware, `DOI <https://doi.org/10.1103/PhysRevA.99.032331>`_
    """

    SUPPORTED_GATES = [
        "x",
        "y",
        "z",
        "h",
        "rx",
        "ry",
        "rz",
        "p",
        "cx",
        "cy",
        "cz",
        "ryy",
        "rxx",
        "rzz",
        "rzx",
    ]

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        shots: int | None,
    ) -> SamplerGradientResult:
        """Compute the estimator gradients on the given circuits."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )

        if self._transpiler is not None:
            g_circuits = self._transpiler.run(g_circuits, **self._transpiler_options)

        results = self._run_unique(g_circuits, g_parameter_values, g_parameters, shots)
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        shots: int | None,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        metadata = []
        all_n = []
        has_transformed_shots = False

        if isinstance(shots, int) or shots is None:
            shots = [shots] * len(circuits)
            has_transformed_shots = True

        pubs = []

        for circuit, parameter_values_, parameters_, shots_ in zip(
            circuits, parameter_values, parameters, shots
        ):
            metadata.append({"parameters": parameters_})
            # Make parameter values for the parameter shift rule.
            param_shift_parameter_values = _make_param_shift_parameter_values(
                circuit, parameter_values_, parameters_
            )
            # Combine inputs into a single job to reduce overhead.
            n = len(param_shift_parameter_values)
            pubs.append((circuit, param_shift_parameter_values, shots_))
            all_n.append(n)

        # Run the single job with all circuits.
        job = self._sampler.run(pubs)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        # Compute the gradients.
        gradients = []
        for n, result_n in zip(all_n, results):
            gradient = []
            result = [
                {label: value / res.num_shots for label, value in res.get_int_counts().items()}
                for res in getattr(result_n.data, next(iter(result_n.data)))
            ]
            for dist_plus, dist_minus in zip(result[: n // 2], result[n // 2 :]):
                grad_dist: dict[int, float] = defaultdict(float)
                for key, val in dist_plus.items():
                    grad_dist[key] += val / 2
                for key, val in dist_minus.items():
                    grad_dist[key] -= val / 2
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
