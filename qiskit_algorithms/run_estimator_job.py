# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility function to run Estimator jobs."""

from collections.abc import Iterable

import numpy as np
from qiskit.primitives import BaseEstimatorV2, EstimatorPubLike, PrimitiveResult, PubResult, DataBin
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.object_array import object_array
from qiskit.quantum_info import SparsePauliOp, PauliList


def run_estimator_job(
    estimator: BaseEstimatorV2, pubs: Iterable[EstimatorPubLike]
) -> PrimitiveResult[PubResult]:
    """
    Accepts a sequence of EstimatorPubLike and returns the result of running the given Estimator on
    these PUBs. Additionally, the observables in the PUBs that are nil are bypassed: they are not
    given to the Estimator, since their expectation value is known. As such, the function directly
    sets the expectation value to 0 for these observables. This is done in a transparent way.

    Args:
        estimator: An estimator primitive used for calculations.
        pubs: The PUBs-like to compute the expectation values of.

    Returns:
        The result of running the given Estimator on these PUBs.

    Raises:
        AlgorithmError: If a primitive job is not successful.
    """
    purged_pubs: list[EstimatorPubLike] = []
    masks: list[np.ndarray] = []
    results_shapes: list[tuple[int, ...]] = []

    for pub in pubs:
        if isinstance(pub, EstimatorPub):
            observables = pub.observables
            parameter_values = pub.parameter_values
            precision = pub.precision
        else:
            observables = object_array(pub[1], list_types=(PauliList,))
            parameter_values = None if len(pub) <= 2 or pub[2] is None else np.array(pub[2])
            precision = None if len(pub) <= 3 or pub[3] is None else pub[3]

        mask = np.empty(shape=observables.shape, dtype=np.bool)

        for nindex, obs in np.ndenumerate(observables):
            mask[nindex] = not isinstance(obs, SparsePauliOp) or obs.simplify().coeffs.any()

        if parameter_values is None or len(parameter_values.shape) <= 1:
            results_shapes.append(observables.shape)
        else:
            results_shapes.append(
                np.broadcast_shapes(observables.shape, parameter_values.shape[1:])
            )

        if mask.any():
            if parameter_values is None:
                purged_pubs.append((pub[0], observables[mask], None, precision))
            else:
                # try to apply mask, if not possible then we shouldn't have applied it because the
                # parameter values are broadcast
                try:
                    purged_pubs.append(
                        (pub[0], observables[mask], parameter_values[mask], precision)
                    )
                except IndexError:
                    purged_pubs.append((pub[0], observables[mask], parameter_values, precision))

        masks.append(mask)

    if not purged_pubs:
        return PrimitiveResult(
            [
                PubResult(
                    data=DataBin(evs=np.zeros(shape=result_shape), shape=result_shape),
                    metadata={
                        "target_precision": (
                            estimator.default_precision
                            if len(pub) <= 3 or pub[3] is None
                            else pub[3]
                        ),
                        "circuit_metadata": pub[0].metadata,
                    },
                )
                for (pub, result_shape) in zip(pubs, results_shapes)
            ],
            metadata={"version": 2},
        )

    job = estimator.run(purged_pubs)

    # import here to avoid cyclic import
    # pylint: disable=cyclic-import
    from qiskit_algorithms import AlgorithmError

    try:
        results = job.result()
    except AlgorithmError as exc:
        raise AlgorithmError("Estimator job failed.") from exc

    pub_results: list[PubResult] = []

    for index, pub in enumerate(pubs):
        mask = masks[index]
        current_result = results[index]
        evs = np.empty(shape=results_shapes[index], dtype=np.float64)
        observable_index = 0

        for nindex, mask_value in np.ndenumerate(mask):
            if mask_value:
                evs[nindex] = current_result.data.evs[observable_index]
                observable_index += 1
            else:
                evs[nindex] = 0

        pub_results.append(
            PubResult(
                data=DataBin(evs=evs, shape=evs.shape),
                metadata={
                    "target_precision": (
                        estimator.default_precision if len(pub) <= 3 or pub[3] is None else pub[3]
                    ),
                    "circuit_metadata": pub[0].metadata,
                },
            )
        )

    return PrimitiveResult(pub_results, metadata={"version": 2})
