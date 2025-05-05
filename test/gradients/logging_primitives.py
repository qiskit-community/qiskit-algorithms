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

"""Test primitives that check what kind of operations are in the circuits they execute."""
from typing import Iterable

from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub


class LoggingEstimator(StatevectorEstimator):
    """An estimator checking what operations were in the circuits it executed."""

    def __init__(self, default_precision: float = 0.0, seed: int | None = None, operations_callback=None):
        super().__init__(default_precision=default_precision, seed=seed)
        self.operations_callback = operations_callback

    def _run(self, pubs: list[EstimatorPub]):
        if self.operations_callback is not None:
            ops = [pub.circuit.count_ops() for pub in pubs]
            self.operations_callback(ops)
        return super()._run(pubs)


class LoggingSampler(StatevectorSampler):
    """A sampler checking what operations were in the circuits it executed."""

    def __init__(self, shots: int = 1024, seed: int | None = None, operations_callback=None):
        super().__init__(default_shots=shots, seed=seed)
        self.operations_callback = operations_callback

    def _run(self, pubs: Iterable[SamplerPub]):
        ops = [pub.circuit.count_ops() for pub in pubs]
        self.operations_callback(ops)
        return super()._run(pubs)
