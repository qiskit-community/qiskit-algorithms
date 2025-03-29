# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
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

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.utils import init_observable, _circuit_key
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..base.base_estimator_gradient import BaseEstimatorGradient
from ..base.estimator_gradient_result import EstimatorGradientResult
from ..utils import DerivativeType, _make_lin_comb_gradient_circuit, _make_lin_comb_observables

from ...exceptions import AlgorithmError


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

            precision: Precision to be used by the underlying estimator.
                The order of priority is: precision in ``run`` method > fidelity's
                precision > primitive's default precision.
                Higher priority setting overrides lower priority setting.
        """
        self._lin_comb_cache: dict[tuple, dict[Parameter, QuantumCircuit]] = {}
        super().__init__(estimator, precision, derivative_type=derivative_type)

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
        precision: float | None = None,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, precision
        )
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        precision: float | None = None,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        has_transformed_precision = False

        if isinstance(precision, float):
            precision=[precision]*len(circuits)
            has_transformed_precision = True

        #(job_circuits, job_observables, job_param_values,
        metadata = []
        all_n = []
        pubs = []
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
            observable = init_observable(observable)
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
                pubs.extend( [(gradient_circuit, observable_1, parameter_values_, precision_) for gradient_circuit in gradient_circuits] )
                pubs.extend( [(gradient_circuit, observable_2, parameter_values_, precision_) for gradient_circuit in gradient_circuits] )
            else:
                all_n.append(n)
                pubs.extend( [(gradient_circuit, observable_1, parameter_values_, precision_) for gradient_circuit in gradient_circuits] )

        # Run the single job with all circuits.
        job = self._estimator.run(
            pubs
        )
        try:
            results = job.result()
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        for n, result_n in zip(all_n, results):
            # this disable is needed as Pylint does not understand derivative_type is a property if
            # it is only defined in the base class and the getter is in the child
            # pylint: disable=comparison-with-callable
            result = result_n.data.evs
            if self.derivative_type == DerivativeType.COMPLEX:
                gradient = np.zeros(n // 2, dtype="complex")
                gradient.real = result[:n // 2]
                gradient.imag = result[n // 2 : n]

            else:
                gradient = np.real(result[:n])
            partial_sum_n += n
            gradients.append(gradient)

        if has_transformed_precision:
            precision = precision[0]
        return EstimatorGradientResult(gradients=gradients, metadata=metadata, precision=precision)
