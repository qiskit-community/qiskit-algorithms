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

"""Test evolver problem class."""
import unittest
from test import QiskitAlgorithmsTestCase
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.circuit import Parameter

from qiskit_algorithms import TimeEvolutionProblem


@ddt
class TestTimeEvolutionProblem(QiskitAlgorithmsTestCase):
    """Test evolver problem class."""

    def test_init_default(self):
        """Tests that all default fields are initialized correctly."""
        hamiltonian = Pauli("Y")
        time = 2.5
        qc = QuantumCircuit(1)
        qc.x(0)
        initial_state = Statevector(qc)

        evo_problem = TimeEvolutionProblem(hamiltonian, time, initial_state)

        expected_hamiltonian = Pauli("Y")
        expected_time = 2.5
        expected_initial_state = Statevector(qc)
        expected_aux_operators = None
        expected_t_param = None
        expected_param_value_dict = None

        self.assertEqual(evo_problem.hamiltonian, expected_hamiltonian)
        self.assertEqual(evo_problem.time, expected_time)
        self.assertEqual(Statevector(evo_problem.initial_state), expected_initial_state)
        self.assertEqual(evo_problem.aux_operators, expected_aux_operators)
        self.assertEqual(evo_problem.t_param, expected_t_param)
        self.assertEqual(evo_problem.param_value_map, expected_param_value_dict)

    @data(QuantumCircuit(1), Statevector([1, 0]))
    def test_init_all(self, initial_state):
        """Tests that all fields are initialized correctly."""
        t_parameter = Parameter("t")
        hamiltonian = SparsePauliOp("Z", t_parameter) + SparsePauliOp("Y")
        time = 2
        aux_operators = [Pauli("X"), Pauli("Y")]
        param_value_dict = {t_parameter: 3.2}

        evo_problem = TimeEvolutionProblem(
            hamiltonian,
            time,
            initial_state,
            aux_operators,
            t_param=t_parameter,
            param_value_map=param_value_dict,
        )

        expected_hamiltonian = SparsePauliOp("Z", t_parameter) + SparsePauliOp("Y")
        expected_time = 2
        expected_type = QuantumCircuit
        expected_aux_operators = [Pauli("X"), Pauli("Y")]
        expected_t_param = t_parameter
        expected_param_value_dict = {t_parameter: 3.2}

        self.assertEqual(evo_problem.hamiltonian, expected_hamiltonian)
        self.assertEqual(evo_problem.time, expected_time)
        self.assertEqual(type(evo_problem.initial_state), expected_type)
        self.assertEqual(evo_problem.aux_operators, expected_aux_operators)
        self.assertEqual(evo_problem.t_param, expected_t_param)
        self.assertEqual(evo_problem.param_value_map, expected_param_value_dict)


if __name__ == "__main__":
    unittest.main()
