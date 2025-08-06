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

"""Expectation value for a diagonal observable using a sampler primitive."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from qiskit.primitives import (
    BaseSamplerV2,
    BaseEstimatorV2,
    PubResult,
    EstimatorPubLike,
    DataBin,
    SamplerPubLike,
    PrimitiveResult,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.algorithm_job import AlgorithmJob


class _DiagonalEstimatorResult(PubResult):
    """A result from an expectation of a diagonal observable."""

    def __init__(
        self,
        data: DataBin,
        metadata: dict[str, Any] | None = None,
        best_measurements: list[dict[str, Any]] | None = None,
    ):
        super().__init__(data, metadata)
        # TODO make each measurement a dataclass rather than a dict
        self.best_measurements: list[dict[str, Any]] | None = best_measurements


class _DiagonalEstimator(BaseEstimatorV2):
    """An estimator for diagonal observables."""

    def __init__(
        self,
        sampler: BaseSamplerV2,
        aggregation: float | Callable[[Iterable[tuple[float, float]]], float] | None = None,
        callback: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        r"""Evaluate the expectation of quantum state with respect to diagonal operators.

        Args:
            sampler: The sampler used to evaluate the circuits.
            aggregation: The aggregation function to aggregate the measurement outcomes. If a float
                this specified the CVaR :math:`\alpha` parameter.
            callback: A callback which is given the best measurements of all circuits in each
                evaluation.
        """
        self.sampler = sampler

        if not callable(aggregation):
            aggregation = _get_cvar_aggregation(aggregation)

        self.aggregation = aggregation
        self.callback = callback

    # If precision is set to None, the default number of shots of the Sampler will be used. It will
    # otherwise be computed as a function of the observable and the precision
    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> AlgorithmJob[PrimitiveResult[_DiagonalEstimatorResult]]:
        # Since we will convert the standalone observables to a list, this `observables` list will
        # remember the shape of the original observables, either standalone or in a list.
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        job = AlgorithmJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[_DiagonalEstimatorResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs])

    # Adapted from StatevectorEstimator, OK with the license?
    def _run_pub(self, pub: EstimatorPub) -> _DiagonalEstimatorResult:
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        bound_circuits = parameter_values.bind_all(circuit)
        bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)
        sampler_pubs: list[SamplerPubLike] = []
        evs = np.zeros_like(bc_circuits, dtype=np.float64)
        best_measurements = []

        for index in np.ndindex(*bc_circuits.shape):
            bound_circuit = bc_circuits[index]
            observable = bc_obs[index]
            paulis, coeffs = zip(*observable.items())
            obs = SparsePauliOp(paulis, coeffs)
            _check_observable_is_diagonal(obs)

            if pub.precision is None:
                sampler_pubs.append((bound_circuit.measure_all(inplace=False),))
            else:
                sampler_pubs.append(
                    (
                        bound_circuit.measure_all(inplace=False),
                        None,
                        # Ensures a standard deviation of at most pub.precision
                        round((sum(obs.coeffs) / pub.precision) ** 2),
                    )
                )

        job = self.sampler.run(sampler_pubs)
        sampler_pubs_results = job.result()

        for sampler_pub_result, index in zip(sampler_pubs_results, np.ndindex(*bc_circuits.shape)):
            observable = bc_obs[index]
            paulis, coeffs = zip(*observable.items())
            obs = SparsePauliOp(paulis, coeffs)
            sampled = {
                label: value / sampler_pub_result.data.meas.num_shots
                for label, value in sampler_pub_result.data.meas.get_int_counts().items()
            }
            evaluated = {
                state: (probability, _evaluate_sparsepauli(state, obs))
                for state, probability in sampled.items()
            }
            evs[index] = np.real_if_close(self.aggregation(evaluated.values()))
            best_result = min(evaluated.items(), key=lambda x: x[1][1])
            best_measurements.append(
                {
                    "state": best_result[0],
                    "bitstring": bin(best_result[0])[2:].zfill(pub.circuit.num_qubits),
                    "value": best_result[1][1],
                    "probability": best_result[1][0],
                }
            )

        if self.callback is not None:
            self.callback(best_measurements)

        data = DataBin(evs=evs, shape=evs.shape)

        return _DiagonalEstimatorResult(
            data,
            metadata={
                "circuit_metadata": pub.circuit.metadata,
                "target_precision": pub.precision,
            },
            best_measurements=best_measurements,
        )


def _get_cvar_aggregation(alpha: float | None) -> Callable[[Iterable[tuple[float, float]]], float]:
    """Get the aggregation function for CVaR with confidence level ``alpha``."""
    if alpha is None:
        alpha = 1
    elif not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1] but was {alpha}")

    # if alpha is close to 1 we can avoid the sorting
    if np.isclose(alpha, 1):

        def aggregate(measurements: Iterable[tuple[float, float]]) -> float:
            return sum(probability * value for probability, value in measurements)

    else:

        def aggregate(measurements: Iterable[tuple[float, float]]) -> float:
            # sort by values
            sorted_measurements = sorted(measurements, key=lambda x: x[1])

            accumulated_percent = 0.0  # once alpha is reached, stop
            cvar = 0.0
            for probability, value in sorted_measurements:
                cvar += value * min(probability, alpha - accumulated_percent)
                accumulated_percent += probability
                if accumulated_percent >= alpha:
                    break

            return cvar / alpha

    return aggregate


_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.complex128)


def _evaluate_sparsepauli(state: int, observable: SparsePauliOp) -> float:
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])


def _check_observable_is_diagonal(observable: SparsePauliOp) -> None:
    is_diagonal = not np.any(observable.paulis.x)
    if not is_diagonal:
        raise ValueError("The observable must be diagonal.")
