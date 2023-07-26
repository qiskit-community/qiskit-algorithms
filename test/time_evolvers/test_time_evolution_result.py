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
"""Class for testing evolution result."""
import unittest
from test import QiskitAlgorithmsTestCase
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_algorithms import TimeEvolutionResult


class TestTimeEvolutionResult(QiskitAlgorithmsTestCase):
    """Class for testing evolution result and relevant metadata."""

    def test_init_state(self):
        """Tests that a class is initialized correctly with an evolved_state."""
        evolved_state = Statevector(QuantumCircuit(1))
        evo_result = TimeEvolutionResult(evolved_state=evolved_state)

        expected_state = Statevector(QuantumCircuit(1))
        expected_aux_ops_evaluated = None

        self.assertEqual(evo_result.evolved_state, expected_state)
        self.assertEqual(evo_result.aux_ops_evaluated, expected_aux_ops_evaluated)

    def test_init_observable(self):
        """Tests that a class is initialized correctly with an evolved_observable."""
        evolved_state = Statevector(QuantumCircuit(1))
        evolved_aux_ops_evaluated = [(5j, 5j), (1.0, 8j), (5 + 1j, 6 + 1j)]
        evo_result = TimeEvolutionResult(evolved_state, evolved_aux_ops_evaluated)

        expected_state = Statevector(QuantumCircuit(1))
        expected_aux_ops_evaluated = [(5j, 5j), (1.0, 8j), (5 + 1j, 6 + 1j)]

        self.assertEqual(evo_result.evolved_state, expected_state)
        self.assertEqual(evo_result.aux_ops_evaluated, expected_aux_ops_evaluated)


if __name__ == "__main__":
    unittest.main()
