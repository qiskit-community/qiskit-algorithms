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

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSamplerV2

from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult
from ..utils import _make_lin_comb_gradient_circuit
from ...custom_types import Transpiler
from ...exceptions import AlgorithmError
from ...utils.circuit_key import _circuit_key


class LinCombSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability.
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
        sampler: BaseSamplerV2,
        shots: int | None = None,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            shots: Number of shots to be used by the underlying Sampler. If provided, this number
                takes precedence over the default precision of the primitive. If None, the default
                number of shots of the primitive is used.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are produced within this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        """
        self._lin_comb_cache: dict[tuple, dict[Parameter, QuantumCircuit]] = {}
        super().__init__(
            sampler, shots, transpiler=transpiler, transpiler_options=transpiler_options
        )

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        shots: int | None = None,
    ) -> SamplerGradientResult:
        """Compute the estimator gradients on the given circuits."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(g_circuits, g_parameter_values, g_parameters, shots=shots)
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        shots: int | None = None,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        metadata = []
        all_n = []
        has_transformed_shots = False

        if isinstance(shots, int) or shots is None:
            shots = [shots] * len(circuits)
            has_transformed_shots = True

        pubs = []

        if not (len(circuits) == len(parameters) == len(parameter_values) == len(shots)):
            raise ValueError(
                f"circuits, parameters, parameter_values and shots must have the same length, but "
                f"have respective lengths {len(circuits)},  {len(parameters)}, {len(parameter_values)} "
                f"and {len(shots)}."
            )

        for circuit, parameter_values_, parameters_, shots_ in zip(
            circuits, parameter_values, parameters, shots
        ):
            # Prepare circuits for the gradient of the specified parameters.
            # TODO: why is this not wrapped into another list level like it is done elsewhere?
            metadata.append({"parameters": parameters_})
            circuit_key = _circuit_key(circuit)
            if circuit_key not in self._lin_comb_cache:
                # Cache the circuits for the linear combination of unitaries.
                # We only cache the circuits for the specified parameters in the future.
                self._lin_comb_cache[circuit_key] = _make_lin_comb_gradient_circuit(
                    circuit, add_measurement=True
                )
            lin_comb_circuits = self._lin_comb_cache[circuit_key]
            gradient_circuits = []
            for param in parameters_:
                gradient_circuits.append(lin_comb_circuits[param])
            # Combine inputs into a single job to reduce overhead.
            n = len(gradient_circuits)
            pubs.extend([(circ, parameter_values_, shots_) for circ in gradient_circuits])
            all_n.append(n)

        if self._transpiler is not None:
            for index, pub in enumerate(pubs):
                pubs[index] = (self._transpiler.run(pub[0], **self._transpiler_options),) + pub[1:]

        # Run the single job with all circuits.
        job = self._sampler.run(pubs)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        for i, n in enumerate(all_n):
            gradient = []
            result = []

            for result_n in results[partial_sum_n : partial_sum_n + n]:
                res = result_n.data.meas
                result.append(
                    {label: value / res.num_shots for label, value in res.get_int_counts().items()}
                )

            m = 2 ** circuits[i].num_qubits
            for dist in result:
                grad_dist: dict[int, float] = defaultdict(float)
                for key, value in dist.items():
                    if key < m:
                        grad_dist[key] += value
                    else:
                        grad_dist[key - m] -= value
                gradient.append(dict(grad_dist))
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
