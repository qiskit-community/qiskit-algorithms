# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Phase estimation for the spectrum of a Hamiltonian"""

from __future__ import annotations


from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import BaseSampler
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit.synthesis import EvolutionSynthesis

from .phase_estimation import PhaseEstimation
from .hamiltonian_phase_estimation_result import HamiltonianPhaseEstimationResult
from .phase_estimation_scale import PhaseEstimationScale


class HamiltonianPhaseEstimation:
    r"""Run the Quantum Phase Estimation algorithm to find the eigenvalues of a Hermitian operator.

    This class is nearly the same as :class:`~qiskit_algorithms.PhaseEstimation`, differing only
    in that the input in that class is a unitary operator, whereas here the input is a Hermitian
    operator from which a unitary will be obtained by scaling and exponentiating. The scaling is
    performed in order to prevent the phases from wrapping around :math:`2\pi`.
    The problem of estimating eigenvalues :math:`\lambda_j` of the Hermitian operator
    :math:`H` is solved by running a circuit representing

    .. math::

        \exp(i b H) |\psi\rangle = \sum_j \exp(i b \lambda_j) c_j |\lambda_j\rangle,

    where the input state is

    .. math::

        |\psi\rangle = \sum_j c_j |\lambda_j\rangle,

    and :math:`\lambda_j` are the eigenvalues of :math:`H`.

    Here, :math:`b` is a scaling factor sufficiently large to map positive :math:`\lambda` to
    :math:`[0,\pi)` and negative :math:`\lambda` to :math:`[\pi,2\pi)`. Each time the circuit is
    run, one measures a phase corresponding to :math:`lambda_j` with probability :math:`|c_j|^2`.

    If :math:`H` is a Pauli sum, the bound :math:`b` is computed from the sum of the absolute
    values of the coefficients of the terms. There is no way to reliably recover eigenvalues
    from phases very near the endpoints of these intervals. Because of this you should be aware
    that for degenerate cases, such as :math:`H=Z`, the eigenvalues :math:`\pm 1` will be
    mapped to the same phase, :math:`\pi`, and so cannot be distinguished. In this case, you need
    to specify a larger bound as an argument to the method ``estimate``.

    This class uses and works together with :class:`~qiskit_algorithms.PhaseEstimationScale` to
    manage scaling the Hamiltonian and the phases that are obtained by the QPE algorithm. This
    includes setting, or computing, a bound on the eigenvalues of the operator, using this
    bound to obtain a scale factor, scaling the operator, and shifting and scaling the measured
    phases to recover the eigenvalues.

    Note that, although we speak of "evolving" the state according the Hamiltonian, in the
    present algorithm, we are not actually considering time evolution. Rather, the role of time is
    played by the scaling factor, which is chosen to best extract the eigenvalues of the
    Hamiltonian.

    A few of the ideas in the algorithm may be found in Ref. [1].

    **Reference:**

    [1]: Quantum phase estimation of multiple eigenvalues for small-scale (noisy) experiments
         T.E. O'Brien, B. Tarasinski, B.M. Terhal
         `arXiv:1809.09697 <https://arxiv.org/abs/1809.09697>`_

    """

    def __init__(
        self,
        num_evaluation_qubits: int,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        Args:
            num_evaluation_qubits: The number of qubits used in estimating the phase. The phase will
                be estimated as a binary string with this many bits.
            sampler: The sampler primitive on which the circuit will be sampled.
        """
        self._phase_estimation = PhaseEstimation(
            num_evaluation_qubits=num_evaluation_qubits,
            sampler=sampler,
        )

    def _get_scale(self, hamiltonian, bound=None) -> PhaseEstimationScale:
        if bound is None:
            return PhaseEstimationScale.from_pauli_sum(hamiltonian)

        return PhaseEstimationScale(bound)

    def _get_unitary(self, hamiltonian, pe_scale, evolution: EvolutionSynthesis) -> QuantumCircuit:
        """Evolve the Hamiltonian to obtain a unitary.

        Apply the scaling to the Hamiltonian that has been computed from an eigenvalue bound
        and compute the unitary by applying the evolution object.
        """

        evo = PauliEvolutionGate(hamiltonian, -pe_scale.scale, synthesis=evolution)
        unitary = QuantumCircuit(evo.num_qubits)
        unitary.append(evo, unitary.qubits)

        return unitary.decompose().decompose()
        # Decomposing twice allows some 1Q Hamiltonians to give correct results
        # when using MatrixEvolution(), that otherwise would give incorrect results.
        # It does not break any others that we tested.

    def estimate(
        self,
        hamiltonian: Pauli | SparsePauliOp,
        state_preparation: QuantumCircuit | Statevector | None = None,
        evolution: EvolutionSynthesis | None = None,
        bound: float | None = None,
    ) -> HamiltonianPhaseEstimationResult:
        """Run the Hamiltonian phase estimation algorithm.

        Args:
            hamiltonian: A Hermitian operator. The allowed types are ``Pauli`` and ``SparsePauliOp``.
            state_preparation: The state to be prepared, whose eigenphase will be
                measured. If this parameter is omitted, no preparation circuit will be run and
                input state will be the all-zero state in the computational basis.
            evolution: An evolution synthesis class.
            bound: An upper bound on the absolute value of the eigenvalues of
                ``hamiltonian``. If omitted, then ``hamiltonian`` must be a SparsePauliOp,
                in which case a bound will be computed. The tighter the bound,
                the higher the resolution of computed phases.

        Returns:
            ``HamiltonianPhaseEstimationResult`` instance containing the result of the estimation
            and diagnostic information.

        Raises:
            TypeError: If ``evolution`` is not of type ``EvolutionSynthesis``.
            TypeError: If ``hamiltonian`` type is not ``Pauli`` or ``SparsePauliOp``.
            ValueError: If ``bound`` is ``None`` and ``hamiltonian`` is not a Pauli sum.
        """
        if evolution is not None and not isinstance(evolution, EvolutionSynthesis):
            raise TypeError(f"Expecting type EvolutionSynthesis, got {type(evolution)}")

        if not isinstance(hamiltonian, (Pauli, SparsePauliOp)):
            raise TypeError(
                f"Expecting Hamiltonian type Pauli, SparsePauliOp, " f"got {type(hamiltonian)}."
            )

        if isinstance(state_preparation, Statevector):
            circuit = QuantumCircuit(state_preparation.num_qubits)
            circuit.prepare_state(state_preparation.data)
            state_preparation = circuit

        if isinstance(hamiltonian, SparsePauliOp):
            id_coefficient, hamiltonian_no_id = _remove_identity(hamiltonian)
        else:
            id_coefficient = 0.0
            hamiltonian_no_id = hamiltonian

        pe_scale = self._get_scale(hamiltonian_no_id, bound)
        unitary = self._get_unitary(hamiltonian_no_id, pe_scale, evolution)

        # run phase estimation
        phase_estimation_result = self._phase_estimation.estimate(
            unitary=unitary, state_preparation=state_preparation
        )
        return HamiltonianPhaseEstimationResult(
            phase_estimation_result=phase_estimation_result,
            id_coefficient=id_coefficient,
            phase_estimation_scale=pe_scale,
        )


def _remove_identity(pauli_sum: SparsePauliOp):
    """Remove any identity operators from ``pauli_sum``. Return
    the sum of the coefficients of the identities and the new operator.
    """

    def _get_identity(size):
        identity = SparsePauliOp("I")
        for _ in range(size - 1):
            identity = identity ^ SparsePauliOp("I")
        return identity

    idcoeff = 0.0
    if isinstance(pauli_sum, SparsePauliOp):
        for operator in pauli_sum:
            if operator.paulis == ["I" * pauli_sum.num_qubits]:
                idcoeff += operator.coeffs[0]
                pauli_sum = pauli_sum - operator.coeffs[0] * _get_identity(pauli_sum.num_qubits)

    return idcoeff, pauli_sum.simplify()
