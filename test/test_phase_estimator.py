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
from itertools import product
from test import QiskitAlgorithmsTestCase

import numpy as np
from ddt import ddt, data, unpack
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, IGate, ZGate
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector, Operator
from qiskit.synthesis import MatrixExponential, SuzukiTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_algorithms import (
    HamiltonianPhaseEstimation,
    IterativePhaseEstimation,
    PhaseEstimation,
    PhaseEstimationScale,
)


@ddt
class TestHamiltonianPhaseEstimation(QiskitAlgorithmsTestCase):
    """Tests for obtaining eigenvalues from phase estimation"""

    pm = generate_preset_pass_manager(optimization_level=1, seed_transpiler=42)

    # sampler tests
    def hamiltonian_pe_sampler(
        self,
        hamiltonian,
        state_preparation=None,
        num_evaluation_qubits=6,
        evolution=None,
        bound=None,
        pass_manager=None,
    ):
        """Run HamiltonianPhaseEstimation and return result with all phases."""
        sampler = Sampler(default_shots=10_000, seed=42)
        phase_est = HamiltonianPhaseEstimation(
            num_evaluation_qubits=num_evaluation_qubits, sampler=sampler, pass_manager=pass_manager
        )
        result = phase_est.estimate(
            hamiltonian=hamiltonian,
            state_preparation=state_preparation,
            evolution=evolution,
            bound=bound,
        )
        return result

    @data(*product((MatrixExponential(), SuzukiTrotter(reps=4)), (None, pm)))
    @unpack
    def test_pauli_sum_1_sampler(self, evolution, pass_manager):
        """Two eigenvalues from Pauli sum with X, Z"""
        hamiltonian = SparsePauliOp.from_list([("X", 0.5), ("Z", 1)])
        state_preparation = QuantumCircuit(1).compose(HGate())

        result = self.hamiltonian_pe_sampler(
            hamiltonian, state_preparation, evolution=evolution, pass_manager=pass_manager
        )
        phase_dict = result.filter_phases(0.162, as_float=True)
        phases = list(phase_dict.keys())
        phases.sort()

        self.assertAlmostEqual(phases[0], -1.125, delta=0.001)
        self.assertAlmostEqual(phases[1], 1.125, delta=0.001)

    @data(*product((MatrixExponential(), SuzukiTrotter(reps=3)), (None, pm)))
    @unpack
    def test_pauli_sum_2_sampler(self, evolution, pass_manager):
        """Two eigenvalues from Pauli sum with X, Y, Z"""
        hamiltonian = SparsePauliOp.from_list([("X", 0.5), ("Z", 1), ("Y", 1)])
        state_preparation = None

        result = self.hamiltonian_pe_sampler(
            hamiltonian, state_preparation, evolution=evolution, pass_manager=pass_manager
        )
        phase_dict = result.filter_phases(0.1, as_float=True)
        phases = list(phase_dict.keys())
        phases.sort()

        self.assertAlmostEqual(phases[0], -1.484, delta=0.001)
        self.assertAlmostEqual(phases[1], 1.484, delta=0.001)

    @data(None, pm)
    def test_single_pauli_op_sampler(self, pass_manager):
        """Two eigenvalues from Pauli sum with X, Y, Z"""
        hamiltonian = SparsePauliOp(Pauli("Z"))
        state_preparation = None

        result = self.hamiltonian_pe_sampler(
            hamiltonian, state_preparation, evolution=None, pass_manager=pass_manager
        )
        eigv = result.most_likely_eigenvalue
        with self.subTest("First eigenvalue"):
            self.assertAlmostEqual(eigv, 1.0, delta=0.001)

        state_preparation = QuantumCircuit(1).compose(XGate())

        result = self.hamiltonian_pe_sampler(hamiltonian, state_preparation, bound=1.05)
        eigv = result.most_likely_eigenvalue
        with self.subTest("Second eigenvalue"):
            self.assertAlmostEqual(eigv, -0.98, delta=0.01)

    @data(
        *product(
            (
                (Statevector(QuantumCircuit(2).compose(IGate()).compose(HGate()))),
                (QuantumCircuit(2).compose(IGate()).compose(HGate())),
            ),
            (None, pm),
        )
    )
    @unpack
    def test_H2_hamiltonian_sampler(self, state_preparation, pass_manager):
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
        result = self.hamiltonian_pe_sampler(
            hamiltonian, state_preparation, evolution=evo, pass_manager=pass_manager
        )
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

    @data(None, pm)
    def test_matrix_evolution_sampler(self, pass_manager):
        """1Q Hamiltonian with MatrixEvolution"""
        # hamiltonian = PauliSumOp(SparsePauliOp.from_list([("X", 0.5), ("Y", 0.6), ("I", 0.7)]))

        hamiltonian = SparsePauliOp.from_list([("X", 0.5), ("Y", 0.6), ("I", 0.7)])
        state_preparation = None
        result = self.hamiltonian_pe_sampler(
            hamiltonian, state_preparation, evolution=MatrixExponential(), pass_manager=pass_manager
        )
        phase_dict = result.filter_phases(0.2, as_float=True)
        phases = sorted(phase_dict.keys())
        self.assertAlmostEqual(phases[0], -0.090, delta=0.001)
        self.assertAlmostEqual(phases[1], 1.490, delta=0.001)


@ddt
class TestPhaseEstimation(QiskitAlgorithmsTestCase):
    """Evolution tests."""

    pm = generate_preset_pass_manager(optimization_level=1, seed_transpiler=42)

    # sampler tests
    def one_phase_sampler(
        self,
        unitary_circuit,
        state_preparation=None,
        phase_estimator=None,
        num_iterations=6,
        shots=None,
        pass_manager=None,
    ):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return the estimated phase as a value in :math:`[0,1)`.
        """

        if shots is not None:
            sampler = Sampler(default_shots=shots, seed=42)
        else:
            sampler = Sampler(seed=42)
        if phase_estimator is None:
            phase_estimator = IterativePhaseEstimation
        if phase_estimator == IterativePhaseEstimation:
            p_est = IterativePhaseEstimation(
                num_iterations=num_iterations, sampler=sampler, pass_manager=pass_manager
            )
        elif phase_estimator == PhaseEstimation:
            p_est = PhaseEstimation(
                num_evaluation_qubits=6, sampler=sampler, pass_manager=pass_manager
            )
        else:
            raise ValueError("Unrecognized phase_estimator")
        result = p_est.estimate(unitary=unitary_circuit, state_preparation=state_preparation)
        phase = result.phase
        return phase

    @data(
        *product(
            ((None, 0.0), (QuantumCircuit(1).compose(XGate()), 0.5)),
            (None, 1000),
            (IterativePhaseEstimation, PhaseEstimation),
            (None, pm),
        )
    )
    @unpack
    def test_qpe_Z_sampler(
        self, state_preparation_and_expected_phase, shots, phase_estimator, pass_manager
    ):
        """eigenproblem Z, |0> and |1>"""
        state_preparation, expected_phase = state_preparation_and_expected_phase
        unitary_circuit = QuantumCircuit(1).compose(ZGate())
        phase = self.one_phase_sampler(
            unitary_circuit,
            state_preparation=state_preparation,
            phase_estimator=phase_estimator,
            shots=shots,
            pass_manager=pass_manager,
        )
        self.assertEqual(phase, expected_phase)

    @data(
        *product(
            (
                (QuantumCircuit(1).compose(HGate()), 0.0),
                (QuantumCircuit(1).compose(HGate()).compose(ZGate()), 0.5),
            ),
            (IterativePhaseEstimation, PhaseEstimation),
            (None, pm),
        )
    )
    @unpack
    def test_qpe_X_plus_minus_sampler(
        self, state_preparation_and_expected_phase, phase_estimator, pass_manager
    ):
        """eigenproblem X, (|+>, |->)"""
        state_preparation, expected_phase = state_preparation_and_expected_phase
        unitary_circuit = QuantumCircuit(1).compose(XGate())
        phase = self.one_phase_sampler(
            unitary_circuit, state_preparation, phase_estimator, pass_manager=pass_manager
        )
        self.assertEqual(phase, expected_phase)

    @data(
        *product(
            (
                (QuantumCircuit(1).compose(XGate()), 0.125),
                (QuantumCircuit(1).compose(IGate()), 0.875),
            ),
            (IterativePhaseEstimation, PhaseEstimation),
            (None, pm),
        )
    )
    @unpack
    def test_qpe_RZ_sampler(
        self, state_preparation_and_expected_phase, phase_estimator, pass_manager
    ):
        """eigenproblem RZ, (|0>, |1>)"""
        state_preparation, expected_phase = state_preparation_and_expected_phase
        alpha = np.pi / 2
        unitary_circuit = QuantumCircuit(1)
        unitary_circuit.rz(alpha, 0)
        phase = self.one_phase_sampler(
            unitary_circuit, state_preparation, phase_estimator, pass_manager=pass_manager
        )
        self.assertEqual(phase, expected_phase)

    @data(
        *product(
            (
                (QuantumCircuit(2).compose(XGate(), qubits=[0]).compose(XGate(), qubits=[1]), 0.25),
                (
                    QuantumCircuit(2).compose(IGate(), qubits=[0]).compose(XGate(), qubits=[1]),
                    0.125,
                ),
            ),
            (IterativePhaseEstimation, PhaseEstimation),
            (None, pm),
        )
    )
    @unpack
    def test_qpe_two_qubit_unitary(
        self, state_preparation_and_expected_phase, phase_estimator, pass_manager
    ):
        """two qubit unitary T ^ T"""
        state_preparation, expected_phase = state_preparation_and_expected_phase
        unitary_circuit = QuantumCircuit(2)
        unitary_circuit.t(0)
        unitary_circuit.t(1)
        phase = self.one_phase_sampler(
            unitary_circuit, state_preparation, phase_estimator, pass_manager=pass_manager
        )
        self.assertEqual(phase, expected_phase)

    @data(None, pm)
    def test_check_num_iterations_sampler(self, pass_manager):
        """test check for num_iterations greater than zero"""
        unitary_circuit = QuantumCircuit(1).compose(XGate())
        state_preparation = None
        with self.assertRaises(ValueError):
            self.one_phase_sampler(
                unitary_circuit, state_preparation, num_iterations=-1, pass_manager=pass_manager
            )

    def test_phase_estimation_scale_from_operator(self):
        """test that PhaseEstimationScale from_pauli_sum works with Operator"""
        circ = QuantumCircuit(2)
        op = Operator(circ)
        scale = PhaseEstimationScale.from_pauli_sum(op)
        self.assertEqual(scale._bound, 4.0)

    def phase_estimation_sampler(
        self,
        unitary_circuit,
        sampler: Sampler,
        state_preparation=None,
        num_evaluation_qubits=6,
        construct_circuit=False,
        pass_manager=None,
    ):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return all results
        """
        phase_est = PhaseEstimation(
            num_evaluation_qubits=num_evaluation_qubits, sampler=sampler, pass_manager=pass_manager
        )
        if construct_circuit:
            pe_circuit = phase_est.construct_circuit(unitary_circuit, state_preparation)
            result = phase_est.estimate_from_pe_circuit(pe_circuit)
        else:
            result = phase_est.estimate(
                unitary=unitary_circuit, state_preparation=state_preparation
            )
        return result

    @data(*product((True, False), (None, pm)))
    @unpack
    def test_qpe_Zplus_sampler(self, construct_circuit, pass_manager):
        """superposition eigenproblem Z, |+>"""
        unitary_circuit = QuantumCircuit(1).compose(ZGate())
        state_preparation = QuantumCircuit(1).compose(HGate())  # prepare |+>
        sampler = Sampler(default_shots=10_000, seed=42)
        result = self.phase_estimation_sampler(
            unitary_circuit,
            sampler,
            state_preparation,
            construct_circuit=construct_circuit,
            pass_manager=pass_manager,
        )

        phases = result.filter_phases(1e-15, as_float=True)
        with self.subTest("test phases has correct values"):
            self.assertEqual(list(phases.keys()), [0.0, 0.5])

        with self.subTest("test phases has correct probabilities"):
            np.testing.assert_allclose(list(phases.values()), [0.5, 0.5], atol=1e-2, rtol=1e-2)

        with self.subTest("test bitstring representation"):
            phases = result.filter_phases(1e-15, as_float=False)
            self.assertEqual(list(phases.keys()), ["000000", "100000"])


if __name__ == "__main__":
    unittest.main()
