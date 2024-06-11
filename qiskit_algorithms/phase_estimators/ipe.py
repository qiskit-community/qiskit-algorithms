# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""The Iterative Quantum Phase Estimation Algorithm."""

from __future__ import annotations

import numpy

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.primitives import BaseSampler

from qiskit_algorithms.exceptions import AlgorithmError

from .phase_estimator import PhaseEstimator
from .phase_estimator import PhaseEstimatorResult


class IterativePhaseEstimation(PhaseEstimator):
    """Run the Iterative quantum phase estimation (QPE) algorithm.

    Given a unitary circuit and a circuit preparing an eigenstate, return the phase of the
    eigenvalue as a number in :math:`[0,1)` using the iterative phase estimation algorithm.

    [1]: Dobsicek et al. (2006), Arbitrary accuracy iterative phase estimation algorithm as a two
       qubit benchmark, `arxiv/quant-ph/0610214 <https://arxiv.org/abs/quant-ph/0610214>`_
    """

    def __init__(
        self,
        num_iterations: int,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        Args:
            num_iterations: The number of iterations (rounds) of the phase estimation to run.
            sampler: The sampler primitive on which the circuit will be sampled.

        Raises:
            ValueError: if num_iterations is not greater than zero.
            AlgorithmError: If a sampler is not provided
        """
        if sampler is None:
            raise AlgorithmError("A sampler must be provided.")

        if num_iterations <= 0:
            raise ValueError("`num_iterations` must be greater than zero.")
        self._num_iterations = num_iterations
        self._sampler = sampler

    def construct_circuit(
        self,
        unitary: QuantumCircuit,
        state_preparation: QuantumCircuit,
        k: int,
        omega: float = 0.0,
        measurement: bool = False,
    ) -> QuantumCircuit:
        """Construct the kth iteration Quantum Phase Estimation circuit.

        For details of parameters, see Fig. 2 in https://arxiv.org/pdf/quant-ph/0610214.pdf.

        Args:
            unitary: The circuit representing the unitary operator whose eigenvalue (via phase)
                 will be measured.
            state_preparation: The circuit that prepares the state whose eigenphase will be
                 measured.  If this parameter is omitted, no preparation circuit
                 will be run and input state will be the all-zero state in the
                 computational basis.
            k: the iteration idx.
            omega: the feedback angle.
            measurement: Boolean flag to indicate if measurement should
                be included in the circuit.

        Returns:
            QuantumCircuit: the quantum circuit per iteration
        """
        k = self._num_iterations if k is None else k
        # The auxiliary (phase measurement) qubit
        phase_register = QuantumRegister(1, name="a")
        eigenstate_register = QuantumRegister(unitary.num_qubits, name="q")
        qc = QuantumCircuit(eigenstate_register)
        qc.add_register(phase_register)
        if isinstance(state_preparation, QuantumCircuit):
            qc.append(state_preparation, eigenstate_register)
        elif state_preparation is not None:
            qc += state_preparation.construct_circuit("circuit", eigenstate_register)
        # hadamard on phase_register[0]
        qc.h(phase_register[0])
        # controlled-U
        # TODO: We may want to allow flexibility in how the power is computed
        # For example, it may be desirable to compute the power via Trotterization, if
        # we are doing Trotterization anyway.
        unitary_power = unitary.power(2 ** (k - 1)).control()
        qc = qc.compose(unitary_power, [unitary.num_qubits] + list(range(0, unitary.num_qubits)))
        qc.p(omega, phase_register[0])
        # hadamard on phase_register[0]
        qc.h(phase_register[0])
        if measurement:
            c = ClassicalRegister(1, name="c")
            qc.add_register(c)
            qc.measure(phase_register, c)
        return qc

    def _estimate_phase_iteratively(self, unitary, state_preparation):
        """
        Main loop of iterative phase estimation.
        """
        omega_coef = 0
        # k runs from the number of iterations back to 1
        for k in range(self._num_iterations, 0, -1):
            omega_coef /= 2

            qc = self.construct_circuit(
                unitary, state_preparation, k, -2 * numpy.pi * omega_coef, True
            )
            try:
                sampler_job = self._sampler.run([qc])
                result = sampler_job.result().quasi_dists[0]
            except Exception as exc:
                raise AlgorithmError("The primitive job failed!") from exc
            x = 1 if result.get(1, 0) > result.get(0, 0) else 0

            omega_coef = omega_coef + x / 2

        return omega_coef

    # pylint: disable=signature-differs
    def estimate(
        self, unitary: QuantumCircuit, state_preparation: QuantumCircuit | None = None
    ) -> "IterativePhaseEstimationResult":
        """
        Estimate the eigenphase of the input unitary and initial-state pair.

        Args:
            unitary: The circuit representing the unitary operator whose eigenvalue (via phase)
                     will be measured.
            state_preparation: The circuit that prepares the state whose eigenphase will be
                     measured.  If this parameter is omitted, no preparation circuit
                     will be run and input state will be the all-zero state in the
                     computational basis.

        Returns:
            Estimated phase in an IterativePhaseEstimationResult object.

        Raises:
            AlgorithmError: If a sampler is not provided.
        """
        phase = self._estimate_phase_iteratively(unitary, state_preparation)

        return IterativePhaseEstimationResult(self._num_iterations, phase)


class IterativePhaseEstimationResult(PhaseEstimatorResult):
    """Phase Estimation Result."""

    def __init__(self, num_iterations: int, phase: float) -> None:
        """
        Args:
            num_iterations: number of iterations used in the phase estimation.
            phase: the estimated phase.
        """

        self._num_iterations = num_iterations
        self._phase = phase

    @property
    def phase(self) -> float:
        r"""Return the estimated phase as a number in :math:`[0.0, 1.0)`.

        1.0 corresponds to a phase of :math:`2\pi`. It is assumed that the input vector is an
        eigenvector of the unitary so that the peak of the probability density occurs at the bit
        string that most closely approximates the true phase.
        """
        return self._phase

    @property
    def num_iterations(self) -> int:
        r"""Return the number of iterations used in the estimation algorithm."""
        return self._num_iterations
