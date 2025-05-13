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

"""Expectation value for a diagonal observable using a sampler primitive."""

from __future__ import annotations

from collections.abc import Callable, Sequence, Mapping, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit.primitives import BaseSamplerV2, BaseEstimatorV2, PubResult, EstimatorPubLike, DataBin, \
    SamplerPubLike, PrimitiveResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.algorithm_job import AlgorithmJob


class _DiagonalEstimatorResult(PubResult):
    """A result from an expectation of a diagonal observable."""

    def __init__(self, data: DataBin, metadata: dict[str, Any] | None = None, best_measurements: Sequence[Mapping[str, Any]] | None = None):
        super().__init__(data, metadata)
        # TODO make each measurement a dataclass rather than a dict
        # Should this be a dict when a single observable is passed, to mimic the fact that evs is a
        # 0-shaped array?
        self.best_measurements: Sequence[Mapping[str, Any]] | None = best_measurements


class _DiagonalEstimator(BaseEstimatorV2):
    """An estimator for diagonal observables."""

    def __init__(
        self,
        sampler: BaseSamplerV2,
        aggregation: float | Callable[[Iterable[tuple[float, float]]], float] | None = None,
        callback: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        precision: float = 1e-2,
    ) -> None:
        r"""Evaluate the expectation of quantum state with respect to diagonal operators.

        Args:
            sampler: The sampler used to evaluate the circuits.
            aggregation: The aggregation function to aggregate the measurement outcomes. If a float
                this specified the CVaR :math:`\alpha` parameter.
            callback: A callback which is given the best measurements of all circuits in each
                evaluation.
            precision: Precision for the Estimator, determines the number of shots used by the
                Sampler. The number of shots to evaluate the expectation value of an observable
                :math:`H` is set to :math:`\frac{S^2}{\varepsilon^2}`, with :math:`\varepsilon`
                being the precision and :math:`S` being the sum of Pauli coefficients. This ensures
                that the final standard deviation of the estimator is less than ``precision``. This
                is however quite pessimistic as it assumes that the covariance between Pauli terms
                is maximal, which may lead to a large number of shots for a precision set too low.
                If the ``precision`` parameter of the ``run`` method is set, it will override this
                value.
        """
        self._precision: float = precision
        self.sampler = sampler

        if not callable(aggregation):
            aggregation = _get_cvar_aggregation(aggregation)

        self.aggregation = aggregation
        self.callback = callback

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> AlgorithmJob[PrimitiveResult[_DiagonalEstimatorResult]]:
        if precision is None:
            precision = self._precision

        # Since we will convert the standalone observables to a list, this `observables` list will
        # remember the shape of the original observables, either standalone or in a list.
        observables: list[SparsePauliOp | list[SparsePauliOp]] = []
        sampler_pubs: list[SamplerPubLike] = []

        for pub in pubs:
            coerced_pub = EstimatorPub.coerce(pub, precision)
            pauli_lists: list[dict[str, float]] | dict[str, float] = coerced_pub.observables.tolist()

            # PUB contained a single observable
            if isinstance(pauli_lists, dict):
                new_obs = SparsePauliOp.from_list(pauli_lists.items())
                _check_observable_is_diagonal(new_obs)
                observables.append(new_obs)
                # FIXME: should we put a warning when the computed number of shots is too high?
                sampler_pubs.append(
                    (
                        coerced_pub.circuit.measure_all(inplace=False),
                        coerced_pub.parameter_values,
                        int(np.round(.5 + (sum(abs(new_obs.coeffs)) / coerced_pub.precision) ** 2))
                    )
                )
            else:
                observables.append([])

                for obs in pauli_lists:
                    new_obs = SparsePauliOp.from_list(obs.items())
                    _check_observable_is_diagonal(new_obs)
                    observables[-1].append(new_obs)
                    # We could make a single PUB using the maximal amount of shots needed for all
                    # these observables, but that would mean that the results are then correlated
                    # between the observables, which is potentially undesirable. We are thus forced
                    # to use a single PUB for each observable separately.
                    sampler_pubs.append(
                        (
                            coerced_pub.circuit.measure_all(inplace=False),
                            coerced_pub.parameter_values,
                            int(np.round(.5 + (sum(abs(new_obs.coeffs)) / coerced_pub.precision) ** 2))
                        )
                    )

        job = AlgorithmJob(self._call, sampler_pubs, observables)
        job._submit()
        return job

    def _call(self, sampler_pubs: list[SamplerPubLike], observables: list[SparsePauliOp | list[SparsePauliOp]]) -> PrimitiveResult[_DiagonalEstimatorResult]:
        job = self.sampler.run(sampler_pubs)
        sampler_pub_results = job.result()
        index = 0
        diagonal_estimator_results = []

        for obs in observables:
            if isinstance(obs, SparsePauliOp):
                pubres = sampler_pub_results[index]
                sampled = {
                    label: value / pubres.data.meas.num_shots
                    for label, value in pubres.data.meas.get_int_counts().items()
                }
                evaluated = {
                    state: (probability, _evaluate_sparsepauli(state, obs))
                    for state, probability in sampled.items()
                }
                result = np.array(self.aggregation(evaluated.values()))
                best_result = min(evaluated.items(), key=lambda x: x[1][1])
                best_measurement = [{
                    "state": best_result[0],
                    "bitstring": bin(best_result[0])[2:].zfill(sampler_pubs[index][0].num_qubits),
                    "value": best_result[1][1],
                    "probability": best_result[1][0],
                }]

                if self.callback is not None:
                    self.callback(best_measurement)

                data = DataBin(evs=result, shape=result.shape)

                diagonal_estimator_results.append(
                    _DiagonalEstimatorResult(
                        data,
                        metadata={"circuit_metadata": sampler_pubs[index][0].metadata},
                        best_measurements=best_measurement
                    )
                )
                index += 1
                continue

            pubres_list = sampler_pub_results[index: index + len(obs)]
            samples = [
                {
                    label: value / pubres.data.meas.num_shots
                    for label, value in pubres.data.meas.get_int_counts().items()
                } for pubres in pubres_list
            ]
            evaluations = [
                {
                    state: (probability, _evaluate_sparsepauli(state, observable))
                    for state, probability in sampled.items()
                } for sampled, observable in zip(samples, obs)
            ]
            results = np.array([self.aggregation(evaluated.values()) for evaluated in evaluations])
            best_measurements = []

            for evaluated in evaluations:
                best_result = min(evaluated.items(), key=lambda x: x[1][1])
                best_measurements.append(
                    {
                        "state": best_result[0],
                        "bitstring": bin(best_result[0])[2:].zfill(sampler_pubs[index][0].num_qubits),
                        "value": best_result[1][1],
                        "probability": best_result[1][0],
                    }
                )

            if self.callback is not None:
                self.callback(best_measurements)

            data = DataBin(evs=results, shape=results.shape)

            diagonal_estimator_results.append(
                _DiagonalEstimatorResult(
                    data=data,
                    metadata={"circuit_metadata": sampler_pubs[index][0].metadata},
                    best_measurements=best_measurement
                )
            )

        return PrimitiveResult(diagonal_estimator_results)


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
