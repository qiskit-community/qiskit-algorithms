# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test phase estimation"""

import unittest
from test import QiskitAlgorithmsTestCase
from ddt import ddt, data, unpack
import numpy as np
from qiskit.circuit.library import ZGate, XGate, HGate, IGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, Operator
from qiskit.synthesis import MatrixExponential, SuzukiTrotter
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit

from qiskit_algorithms import PhaseEstimationScale
from qiskit_algorithms.phase_estimators import (
    PhaseEstimation,
    HamiltonianPhaseEstimation,
    IterativePhaseEstimation,
)


@ddt
class TestHamiltonianPhaseEstimation(QiskitAlgorithmsTestCase):
    """Tests for obtaining eigenvalues from phase estimation"""

    # pylint: disable=too-many-positional-arguments
    # sampler tests
    def hamiltonian_pe_sampler(
        self,
        hamiltonian,
        state_preparation=None,
        num_evaluation_qubits=6,
        evolution=None,
        bound=None,
    ):
        """Run HamiltonianPhaseEstimation and return result with all phases."""
        sampler = Sampler()
        phase_est = HamiltonianPhaseEstimation(
            num_evaluation_qubits=num_evaluation_qubits, sampler=sampler
        )
        result = phase_est.estimate(
            hamiltonian=hamiltonian,
            state_preparation=state_preparation,
            evolution=evolution,
            bound=bound,
        )
        return result

    @data(MatrixExponential(), SuzukiTrotter(reps=4))
    def test_pauli_sum_1_sampler(self, evolution):
        """Two eigenvalues from Pauli sum with X, Z"""
        hamiltonian = SparsePauliOp.from_list([("X", 0.5), ("Z", 1)])
        state_preparation = QuantumCircuit(1).compose(HGate())

        result = self.hamiltonian_pe_sampler(hamiltonian, state_preparation, evolution=evolution)
        phase_dict = result.filter_phases(0.162, as_float=True)
        phases = list(phase_dict.keys())
        phases.sort()

        self.assertAlmostEqual(phases[0], -1.125, delta=0.001)
        self.assertAlmostEqual(phases[1], 1.125, delta=0.001)

    @data(MatrixExponential(), SuzukiTrotter(reps=3))
    def test_pauli_sum_2_sampler(self, evolution):
        """Two eigenvalues from Pauli sum with X, Y, Z"""
        hamiltonian = SparsePauliOp.from_list([("X", 0.5), ("Z", 1), ("Y", 1)])
        state_preparation = None

        result = self.hamiltonian_pe_sampler(hamiltonian, state_preparation, evolution=evolution)
        phase_dict = result.filter_phases(0.1, as_float=True)
        phases = list(phase_dict.keys())
        phases.sort()

        self.assertAlmostEqual(phases[0], -1.484, delta=0.001)
        self.assertAlmostEqual(phases[1], 1.484, delta=0.001)

    def test_single_pauli_op_sampler(self):
        """Two eigenvalues from Pauli sum with X, Y, Z"""
        hamiltonian = SparsePauliOp(Pauli("Z"))
        state_preparation = None

        result = self.hamiltonian_pe_sampler(hamiltonian, state_preparation, evolution=None)
        eigv = result.most_likely_eigenvalue
        with self.subTest("First eigenvalue"):
            self.assertAlmostEqual(eigv, 1.0, delta=0.001)

        state_preparation = QuantumCircuit(1).compose(XGate())

        result = self.hamiltonian_pe_sampler(hamiltonian, state_preparation, bound=1.05)
        eigv = result.most_likely_eigenvalue
        with self.subTest("Second eigenvalue"):
            self.assertAlmostEqual(eigv, -0.98, delta=0.01)

    @data(
        (Statevector(QuantumCircuit(2).compose(IGate()).compose(HGate()))),
        (QuantumCircuit(2).compose(IGate()).compose(HGate())),
    )
    def test_H2_hamiltonian_sampler(self, state_preparation):
        """Test H2 hamiltonian"""

        hamiltonian = SparsePauliOp.from_list(
            [
                ("II", -1.0523732457728587),
                ("IZ", 0.3979374248431802),
                ("ZI", -0.3979374248431802),
                ("ZZ", -0.011280104256235324),
                ("XX", 0.18093119978423147),
            ]
        )

        evo = SuzukiTrotter(reps=4)
        result = self.hamiltonian_pe_sampler(hamiltonian, state_preparation, evolution=evo)
        with self.subTest("Most likely eigenvalues"):
            self.assertAlmostEqual(result.most_likely_eigenvalue, -1.855, delta=0.001)
        with self.subTest("Most likely phase"):
            self.assertAlmostEqual(result.phase, 0.5937, delta=0.001)
        with self.subTest("All eigenvalues"):
            phase_dict = result.filter_phases(0.1)
            phases = sorted(phase_dict.keys())
            self.assertAlmostEqual(phases[0], -1.8551, delta=0.001)
            self.assertAlmostEqual(phases[1], -1.2376, delta=0.001)
            self.assertAlmostEqual(phases[2], -0.8979, delta=0.001)

    def test_matrix_evolution_sampler(self):
        """1Q Hamiltonian with MatrixEvolution"""
        # hamiltonian = PauliSumOp(SparsePauliOp.from_list([("X", 0.5), ("Y", 0.6), ("I", 0.7)]))

        hamiltonian = SparsePauliOp.from_list([("X", 0.5), ("Y", 0.6), ("I", 0.7)])
        state_preparation = None
        result = self.hamiltonian_pe_sampler(
            hamiltonian, state_preparation, evolution=MatrixExponential()
        )
        phase_dict = result.filter_phases(0.2, as_float=True)
        phases = sorted(phase_dict.keys())
        self.assertAlmostEqual(phases[0], -0.090, delta=0.001)
        self.assertAlmostEqual(phases[1], 1.490, delta=0.001)


@ddt
class TestPhaseEstimation(QiskitAlgorithmsTestCase):
    """Evolution tests."""

    # pylint: disable=too-many-positional-arguments
    # sampler tests
    def one_phase_sampler(
        self,
        unitary_circuit,
        state_preparation=None,
        phase_estimator=None,
        num_iterations=6,
        shots=None,
    ):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return the estimated phase as a value in :math:`[0,1)`.
        """
        if shots is not None:
            options = {"shots": shots}
        else:
            options = {}
        sampler = Sampler(options=options)
        if phase_estimator is None:
            phase_estimator = IterativePhaseEstimation
        if phase_estimator == IterativePhaseEstimation:
            p_est = IterativePhaseEstimation(num_iterations=num_iterations, sampler=sampler)
        elif phase_estimator == PhaseEstimation:
            p_est = PhaseEstimation(num_evaluation_qubits=6, sampler=sampler)
        else:
            raise ValueError("Unrecognized phase_estimator")
        result = p_est.estimate(unitary=unitary_circuit, state_preparation=state_preparation)
        phase = result.phase
        return phase

    @data(
        (QuantumCircuit(1).compose(XGate()), 0.5, None, IterativePhaseEstimation),
        (QuantumCircuit(1).compose(XGate()), 0.5, 1000, IterativePhaseEstimation),
        (None, 0.0, 1000, IterativePhaseEstimation),
        (QuantumCircuit(1).compose(XGate()), 0.5, 1000, PhaseEstimation),
        (None, 0.0, 1000, PhaseEstimation),
        (QuantumCircuit(1).compose(XGate()), 0.5, None, PhaseEstimation),
    )
    @unpack
    def test_qpe_Z_sampler(self, state_preparation, expected_phase, shots, phase_estimator):
        """eigenproblem Z, |0> and |1>"""
        unitary_circuit = QuantumCircuit(1).compose(ZGate())
        phase = self.one_phase_sampler(
            unitary_circuit,
            state_preparation=state_preparation,
            phase_estimator=phase_estimator,
            shots=shots,
        )
        self.assertEqual(phase, expected_phase)

    @data(
        (QuantumCircuit(1).compose(HGate()), 0.0, IterativePhaseEstimation),
        (QuantumCircuit(1).compose(HGate()).compose(ZGate()), 0.5, IterativePhaseEstimation),
        (QuantumCircuit(1).compose(HGate()), 0.0, PhaseEstimation),
        (QuantumCircuit(1).compose(HGate()).compose(ZGate()), 0.5, PhaseEstimation),
    )
    @unpack
    def test_qpe_X_plus_minus_sampler(self, state_preparation, expected_phase, phase_estimator):
        """eigenproblem X, (|+>, |->)"""
        unitary_circuit = QuantumCircuit(1).compose(XGate())
        phase = self.one_phase_sampler(
            unitary_circuit,
            state_preparation,
            phase_estimator,
        )
        self.assertEqual(phase, expected_phase)

    @data(
        (QuantumCircuit(1).compose(XGate()), 0.125, IterativePhaseEstimation),
        (QuantumCircuit(1).compose(IGate()), 0.875, IterativePhaseEstimation),
        (QuantumCircuit(1).compose(XGate()), 0.125, PhaseEstimation),
        (QuantumCircuit(1).compose(IGate()), 0.875, PhaseEstimation),
    )
    @unpack
    def test_qpe_RZ_sampler(self, state_preparation, expected_phase, phase_estimator):
        """eigenproblem RZ, (|0>, |1>)"""
        alpha = np.pi / 2
        unitary_circuit = QuantumCircuit(1)
        unitary_circuit.rz(alpha, 0)
        phase = self.one_phase_sampler(
            unitary_circuit,
            state_preparation,
            phase_estimator,
        )
        self.assertEqual(phase, expected_phase)

    @data(
        (
            QuantumCircuit(2).compose(XGate(), qubits=[0]).compose(XGate(), qubits=[1]),
            0.25,
            IterativePhaseEstimation,
        ),
        (
            QuantumCircuit(2).compose(IGate(), qubits=[0]).compose(XGate(), qubits=[1]),
            0.125,
            IterativePhaseEstimation,
        ),
        (
            QuantumCircuit(2).compose(XGate(), qubits=[0]).compose(XGate(), qubits=[1]),
            0.25,
            PhaseEstimation,
        ),
        (
            QuantumCircuit(2).compose(IGate(), qubits=[0]).compose(XGate(), qubits=[1]),
            0.125,
            PhaseEstimation,
        ),
    )
    @unpack
    def test_qpe_two_qubit_unitary(self, state_preparation, expected_phase, phase_estimator):
        """two qubit unitary T ^ T"""
        unitary_circuit = QuantumCircuit(2)
        unitary_circuit.t(0)
        unitary_circuit.t(1)
        phase = self.one_phase_sampler(
            unitary_circuit,
            state_preparation,
            phase_estimator,
        )
        self.assertEqual(phase, expected_phase)

    def test_check_num_iterations_sampler(self):
        """test check for num_iterations greater than zero"""
        unitary_circuit = QuantumCircuit(1).compose(XGate())
        state_preparation = None
        with self.assertRaises(ValueError):
            self.one_phase_sampler(unitary_circuit, state_preparation, num_iterations=-1)

    def test_phase_estimation_scale_from_operator(self):
        """test that PhaseEstimationScale from_pauli_sum works with Operator"""
        circ = QuantumCircuit(2)
        op = Operator(circ)
        scale = PhaseEstimationScale.from_pauli_sum(op)
        self.assertEqual(scale._bound, 4.0)

    # pylint: disable=too-many-positional-arguments
    def phase_estimation_sampler(
        self,
        unitary_circuit,
        sampler: Sampler,
        state_preparation=None,
        num_evaluation_qubits=6,
        construct_circuit=False,
    ):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return all results
        """
        phase_est = PhaseEstimation(num_evaluation_qubits=num_evaluation_qubits, sampler=sampler)
        if construct_circuit:
            pe_circuit = phase_est.construct_circuit(unitary_circuit, state_preparation)
            result = phase_est.estimate_from_pe_circuit(pe_circuit)
        else:
            result = phase_est.estimate(
                unitary=unitary_circuit, state_preparation=state_preparation
            )
        return result

    @data(True, False)
    def test_qpe_Zplus_sampler(self, construct_circuit):
        """superposition eigenproblem Z, |+>"""
        unitary_circuit = QuantumCircuit(1).compose(ZGate())
        state_preparation = QuantumCircuit(1).compose(HGate())  # prepare |+>
        sampler = Sampler()
        result = self.phase_estimation_sampler(
            unitary_circuit,
            sampler,
            state_preparation,
            construct_circuit=construct_circuit,
        )

        phases = result.filter_phases(1e-15, as_float=True)
        with self.subTest("test phases has correct values"):
            self.assertEqual(list(phases.keys()), [0.0, 0.5])

        with self.subTest("test phases has correct probabilities"):
            np.testing.assert_allclose(list(phases.values()), [0.5, 0.5])

        with self.subTest("test bitstring representation"):
            phases = result.filter_phases(1e-15, as_float=False)
            self.assertEqual(list(phases.keys()), ["000000", "100000"])


if __name__ == "__main__":
    unittest.main()
