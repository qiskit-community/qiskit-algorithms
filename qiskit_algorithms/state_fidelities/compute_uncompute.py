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
Compute-uncompute fidelity interface using primitives
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob

from .base_state_fidelity import BaseStateFidelity
from .state_fidelity_result import StateFidelityResult
from ..algorithm_job import AlgorithmJob
from ..custom_types import Transpiler
from ..exceptions import AlgorithmError


class ComputeUncompute(BaseStateFidelity):
    r"""
    This class leverages the sampler primitive to calculate the state
    fidelity of two quantum circuits following the compute-uncompute
    method (see [1] for further reference).
    The fidelity can be defined as the state overlap.

    .. math::

            |\langle\psi(x)|\phi(y)\rangle|^2

    where :math:`x` and :math:`y` are optional parametrizations of the
    states :math:`\psi` and :math:`\phi` prepared by the circuits
    ``circuit_1`` and ``circuit_2``, respectively.

    **Reference:**
    [1] Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala,
    A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning
    with quantum-enhanced feature spaces. Nature, 567(7747), 209-212.
    `arXiv:1804.11326v2 [quant-ph] <https://arxiv.org/pdf/1804.11326.pdf>`_

    """

    def __init__(
        self,
        sampler: BaseSamplerV2,
        shots: int | None = None,
        local: bool = False,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Args:
            sampler: Sampler primitive instance.
            shots: Number of shots to be used by the underlying sampler.
                The order of priority is: number of shots in ``run`` method > fidelity's
                number of shots > primitive's default number of shots.
                Higher priority setting overrides lower priority setting.
            local: If set to ``True``, the fidelity is averaged over
                single-qubit projectors

                .. math::

                    \hat{O} = \frac{1}{N}\sum_{i=1}^N|0_i\rangle\langle 0_i|,

                instead of the global projector :math:`|0\rangle\langle 0|^{\otimes n}`.
                This coincides with the standard (global) fidelity in the limit of
                the fidelity approaching 1. Might be used to increase the variance
                to improve trainability in algorithms such as :class:`~.time_evolvers.PVQD`.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are produced within this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.

        Raises:
            ValueError: If the sampler is not an instance of ``BaseSamplerV2``.
        """
        if not isinstance(sampler, BaseSamplerV2):
            raise ValueError(
                f"The sampler should be an instance of BaseSamplerV2, " f"but got {type(sampler)}"
            )
        self._sampler: BaseSamplerV2 = sampler
        self._local = local
        self._shots = shots
        super().__init__(transpiler=transpiler, transpiler_options=transpiler_options)

    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Combines ``circuit_1`` and ``circuit_2`` to create the
        fidelity circuit following the compute-uncompute method.

        Args:
            circuit_1: (Parametrized) quantum circuit.
            circuit_2: (Parametrized) quantum circuit.

        Returns:
            The fidelity quantum circuit corresponding to circuit_1 and circuit_2.
        """
        if len(circuit_1.clbits) > 0:
            circuit_1.remove_final_measurements()
        if len(circuit_2.clbits) > 0:
            circuit_2.remove_final_measurements()

        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        return circuit

    def _run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        shots: int | Sequence[int] | None = None,
    ) -> AlgorithmJob:
        r"""
        Computes the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second) following the compute-uncompute method.

        Args:
            circuits_1: (Parametrized) quantum circuits preparing :math:`|\psi\rangle`.
            circuits_2: (Parametrized) quantum circuits preparing :math:`|\phi\rangle`.
            values_1: Numerical parameters to be bound to the first circuits.
            values_2: Numerical parameters to be bound to the second circuits.
            shots: Number of shots to be used by the underlying Sampler. If a single integer is
                provided, this number will be used for all circuits. If a sequence of integers is
                provided, they will be used on a per-circuit basis. If not set, the fidelity's default
                number of shots will be used for all circuits, and if that is None (not set) then the
                underlying primitive's default number of shots will be used for all circuits.

        Returns:
            An AlgorithmJob for the fidelity calculation.

        Raises:
            ValueError: At least one pair of circuits must be defined.
            AlgorithmError: If the sampler job is not completed successfully.
        """
        circuits = self._construct_circuits(circuits_1, circuits_2)
        if len(circuits) == 0:
            raise ValueError(
                "At least one pair of circuits must be defined to calculate the state overlap."
            )
        values = self._construct_value_list(circuits_1, circuits_2, values_1, values_2)

        # The priority of number of shots options is as follows:
        # number in `run` method > fidelity's default number of shots >
        # primitive's default number of shots.
        if not isinstance(shots, Sequence):
            if shots is None:
                shots = self.shots
            coerced_pubs = [
                SamplerPub.coerce((circuit, value), shots)
                for circuit, value in zip(circuits, values)
            ]
        else:
            coerced_pubs = [
                SamplerPub.coerce((circuit, value), shots_number)
                for circuit, value, shots_number in zip(circuits, values, shots)
            ]

        job = self._sampler.run(coerced_pubs)

        return AlgorithmJob(ComputeUncompute._call, job, circuits, self._local)

    @staticmethod
    def _call(
        job: PrimitiveJob, circuits: Sequence[QuantumCircuit], local: bool
    ) -> StateFidelityResult:
        try:
            result = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed!") from exc

        pub_results_data = [
            getattr(pub_result.data, circuit.cregs[0].name)
            for pub_result, circuit in zip(result, circuits)
        ]
        quasi_dists = [
            {
                label: value / prob_dist.num_shots
                for label, value in prob_dist.get_int_counts().items()
            }
            for prob_dist in pub_results_data
        ]

        if local:
            raw_fidelities = [
                ComputeUncompute._get_local_fidelity(prob_dist, circuit.num_qubits)
                for prob_dist, circuit in zip(quasi_dists, circuits)
            ]
        else:
            raw_fidelities = [
                ComputeUncompute._get_global_fidelity(prob_dist) for prob_dist in quasi_dists
            ]
        fidelities = ComputeUncompute._truncate_fidelities(raw_fidelities)
        shots = [pub_result_data.num_shots for pub_result_data in pub_results_data]

        if len(shots) == 1:
            shots = shots[0]

        return StateFidelityResult(
            fidelities=fidelities,
            raw_fidelities=raw_fidelities,
            metadata=result.metadata,
            shots=shots,
        )

    @property
    def shots(self) -> int | None:
        """Return the number of shots used by the `run` method of the Sampler primitive. If None,
        the default number of shots of the primitive is used.

        Returns:
            The default number of shots.
        """
        return self._shots

    @shots.setter
    def shots(self, shots: int | None):
        """Update the fidelity's default number of shots setting.

        Args:
            shots: The new default number of shots.
        """

        self._shots = shots

    @staticmethod
    def _get_global_fidelity(probability_distribution: dict[int, float]) -> float:
        """Process the probability distribution of a measurement to determine the
        global fidelity.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The global fidelity.
        """
        return probability_distribution.get(0, 0)

    @staticmethod
    def _get_local_fidelity(probability_distribution: dict[int, float], num_qubits: int) -> float:
        """Process the probability distribution of a measurement to determine the
        local fidelity by averaging over single-qubit projectors.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The local fidelity.
        """
        fidelity = 0.0
        for qubit in range(num_qubits):
            for bitstring, prob in probability_distribution.items():
                # Check whether the bit representing the current qubit is 0
                if not bitstring >> qubit & 1:
                    fidelity += prob / num_qubits
        return fidelity
