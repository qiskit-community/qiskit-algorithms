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
from importlib.metadata import metadata
from numbers import Real
from typing import Any, Union

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2, BaseEstimatorV2, PubResult, EstimatorPubLike, DataBin, \
    BindingsArrayLike, SamplerPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.algorithm_job import AlgorithmJob

# TODO: can't assume this, may be given as Pauli instead of SparsePauliOp... Use BaseOperator instead
# Enforce the fact that the observables are given in a SparsePauliOp way
_SparsePauliOpArrayLike = SparsePauliOp | Sequence[SparsePauliOp]
_EstimatorPubLikeSparsePauli = Union[
    EstimatorPub,
    tuple[QuantumCircuit, _SparsePauliOpArrayLike],
    tuple[QuantumCircuit, _SparsePauliOpArrayLike, BindingsArrayLike],
    tuple[QuantumCircuit, _SparsePauliOpArrayLike, BindingsArrayLike, Real],
]


@dataclass(frozen=True)
class _DiagonalEstimatorResult(PubResult):
    """A result from an expectation of a diagonal observable."""

    # TODO make each measurement a dataclass rather than a dict
    best_measurements: Sequence[Mapping[str, Any]] | None = None


class _DiagonalEstimator(BaseEstimatorV2):
    """An estimator for diagonal observables."""

    def __init__(
        self,
        sampler: BaseSamplerV2,
        aggregation: float | Callable[[Iterable[tuple[float, float]]], float] | None = None,
        callback: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        precision: float = 1e-3,
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
                :math:`H` is set to :math:`\frac{\mathrm{S}^2}{\varepsilon^2}`, with
                :math:`\varepsilon` being the precision and :math:`S` being the sum of Pauli
                coefficients. This ensures that the final standard deviation of the estimator is
                less than ``precision``. This is however quite pessimistic as it assumes that the
                covariance between Pauli terms is maximal, which may lead to a large number of shots
                for a precision set too low. If the ``precision`` parameter of the ``run`` method is
                 set, it will override this value.
        """
        self._precision: float = precision
        self.sampler = sampler

        if not callable(aggregation):
            aggregation = _get_cvar_aggregation(aggregation)

        self.aggregation = aggregation
        self.callback = callback

    def run(
        self, pubs: Iterable[_EstimatorPubLikeSparsePauli], *, precision: float | None = None
    ) -> AlgorithmJob:
        if precision is None:
            precision = self._precision

        observables = []
        sampler_pubs = []
        print("PUBS", pubs)

        for pub in pubs:
            if isinstance(pub[1], SparsePauliOp):
                _check_observable_is_diagonal(pub[1])
                n_shots = round(.5 + (sum(pubs[1].coeffs) / pub.precision) ** 2)
            else:
                for op in pub[1]:
                    _check_observable_is_diagonal(op)
                n_shots = [round(.5 + (sum(op.coeffs) / pub.precision) ** 2) for op in pub[1]]

            observables.append(pub[1])

            pub = EstimatorPub.coerce(pub, precision)

            sampler_pubs.append(
                (
                    pub.circuit.measure_all(inplace=False),
                    pub.parameter_values,
                    n_shots
                )
            )

        job = AlgorithmJob(self._call, sampler_pubs, observables)
        job._submit()
        return job

    def _call(self, sampler_pubs: Iterable[SamplerPubLike], observables: list[list[SparsePauliOp]]) -> _DiagonalEstimatorResult:
        job = self.sampler.run(sampler_pubs)
        results = job.result()
        samples: list[dict[int, float]] = []
        for i, pubres in enumerate(results):
            sample = {
                label: value / pubres.data.meas.num_shots
                for label, value in pubres.data.meas.get_int_counts().items()
            }
            samples.append(sample)

        # a list of dictionaries containing: {state: (measurement probability, value)}
        evaluations: list[dict[int, tuple[float, float]]] = [
            {
                state: (probability, _evaluate_sparsepauli(state, op))
                for op, (state, probability) in zip(ops, sampled.items())
            }
            for ops, sampled in zip(observables, samples)
        ]

        results = np.array([self.aggregation(evaluated.values()) for evaluated in evaluations])

        # get the best measurements
        best_measurements = []
        for sampler_pub, evaluated in zip(sampler_pubs, evaluations):
            best_result = min(evaluated.items(), key=lambda x: x[1][1])
            best_measurements.append(
                {
                    "state": best_result[0],
                    "bitstring": bin(best_result[0])[2:].zfill(sampler_pub[0].num_qubits),
                    "value": best_result[1][1],
                    "probability": best_result[1][0],
                }
            )

        if self.callback is not None:
            self.callback(best_measurements)

        # Add stds? Or not needed since it's only called internally?
        data = DataBin(evs=results, shape=results.shape)

        return _DiagonalEstimatorResult(
            data=data,
            metadata=[
                {"shots": sampler_pub[2], "circuit_metadata": sampler_pub[0].metadata}
                for sampler_pub in sampler_pubs
            ],
            best_measurements=best_measurements
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
