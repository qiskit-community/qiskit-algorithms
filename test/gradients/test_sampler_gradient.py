# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

"""Test Sampler Gradients"""

import unittest
from test import QiskitAlgorithmsTestCase

import numpy as np
from ddt import ddt, data, unpack
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2, real_amplitudes
from qiskit.circuit.library.standard_gates import RXXGate
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2

from qiskit_algorithms.gradients import (
    FiniteDiffSamplerGradient,
    LinCombSamplerGradient,
    ParamShiftSamplerGradient,
    SPSASamplerGradient,
)
from .logging_primitives import LoggingSampler

gradient_factories = [
    (
        lambda sampler: FiniteDiffSamplerGradient(sampler, epsilon=1e-2, method="central"),
        3_000_000,
        1e-1,
        1e-1,
    ),
    (
        lambda sampler: FiniteDiffSamplerGradient(sampler, epsilon=1e-2, method="forward"),
        3_000_000,
        1e-1,
        1e-1,
    ),
    (
        lambda sampler: FiniteDiffSamplerGradient(sampler, epsilon=1e-2, method="backward"),
        3_000_000,
        1e-1,
        1e-1,
    ),
    (ParamShiftSamplerGradient, 1_000_000, 1e-2, 1e-2),
    (LinCombSamplerGradient, 1_000_000, 1e-2, 1e-2),
]

THREE_QUBITS_BACKEND = GenericBackendV2(num_qubits=3, coupling_map=[[0, 1], [1, 2]], seed=54)


@ddt
class TestSamplerGradient(QiskitAlgorithmsTestCase):
    """Test Sampler Gradient"""

    def setUp(self):
        super().setUp()
        self.seed = 42

    @data(*gradient_factories)
    @unpack
    def test_single_circuit(self, grad, shots, atol, rtol):
        """Test the sampler gradient for a single circuit"""
        sampler = StatevectorSampler(
            default_shots=shots, seed=np.random.default_rng(seed=self.seed)
        )
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: 0, 1: 0}],
            [{0: -0.499999, 1: 0.499999}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, rtol=rtol, atol=atol)

    @data(*gradient_factories)
    @unpack
    def test_gradient_p(self, grad, shots, atol, rtol):
        """Test the sampler gradient for p"""
        sampler = StatevectorSampler(
            default_shots=shots, seed=np.random.default_rng(seed=self.seed)
        )
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: 0, 1: 0}],
            [{0: -0.5, 1: 0.5}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

    @data(*gradient_factories)
    @unpack
    def test_gradient_u(self, grad, shots, atol, rtol):
        """Test the sampler gradient for u"""
        sampler = StatevectorSampler(
            default_shots=shots, seed=np.random.default_rng(seed=self.seed)
        )
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}, {0: 0, 1: 0}, {0: 0, 1: 0}],
            [{0: -0.176777, 1: 0.176777}, {0: -0.426777, 1: 0.426777}, {0: -0.426777, 1: 0.426777}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

    @data(*gradient_factories)
    @unpack
    def test_gradient_efficient_su2(self, grad, shots, atol, rtol):
        """Test the sampler gradient for efficient_su2"""
        sampler = StatevectorSampler(
            default_shots=shots, seed=np.random.default_rng(seed=self.seed)
        )
        qc = efficient_su2(2, reps=1)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [
            [np.pi / 4 for _ in qc.parameters],
            [np.pi / 2 for _ in qc.parameters],
        ]
        expected = [
            [
                {
                    0: -0.11963834764831836,
                    1: -0.05713834764831845,
                    2: -0.21875,
                    3: 0.39552669529663675,
                },
                {
                    0: -0.32230339059327373,
                    1: -0.03125,
                    2: 0.2339150429449554,
                    3: 0.11963834764831843,
                },
                {
                    0: 0.012944173824159189,
                    1: -0.01294417382415923,
                    2: 0.07544417382415919,
                    3: -0.07544417382415919,
                },
                {
                    0: 0.2080266952966367,
                    1: -0.03125000000000002,
                    2: -0.11963834764831842,
                    3: -0.057138347648318405,
                },
                {
                    0: -0.11963834764831838,
                    1: 0.11963834764831838,
                    2: -0.21875,
                    3: 0.21875,
                },
                {
                    0: -0.2781092167691146,
                    1: -0.0754441738241592,
                    2: 0.27810921676911443,
                    3: 0.07544417382415924,
                },
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            ],
            [
                {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 0.0,
                },
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                {
                    0: -0.25,
                    1: 0.25,
                    2: 0.25,
                    3: -0.25,
                },
                {
                    0: 0.25,
                    1: 0.25,
                    2: -0.25,
                    3: -0.25,
                },
                {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 0.0,
                },
                {
                    0: -0.25,
                    1: 0.25,
                    2: 0.25,
                    3: -0.25,
                },
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            ],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(expected[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

    @data(*gradient_factories)
    @unpack
    def test_gradient_2qubit_gate(self, grad, shots, atol, rtol):
        """Test the sampler gradient for 2 qubit gates"""
        sampler = StatevectorSampler(
            default_shots=shots, seed=np.random.default_rng(seed=self.seed)
        )
        for gate in [RXXGate]:
            param_list = [[np.pi / 4], [np.pi / 2]]
            correct_results = [
                [{0: -0.5 / np.sqrt(2), 1: 0, 2: 0, 3: 0.5 / np.sqrt(2)}],
                [{0: -0.5, 1: 0, 2: 0, 3: 0.5}],
            ]
            for i, param in enumerate(param_list):
                a = Parameter("a")
                qc = QuantumCircuit(2)
                qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                qc.measure_all()
                gradient = grad(sampler)
                gradients = gradient.run([qc], [param]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=2)
                array2 = _quasi2array(correct_results[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

    @data(*gradient_factories)
    @unpack
    def test_gradient_parameter_coefficient(self, grad, shots, atol, rtol):
        """Test the sampler gradient for parameter variables with coefficients"""
        sampler = StatevectorSampler(
            default_shots=shots, seed=np.random.default_rng(seed=self.seed)
        )
        qc = real_amplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[0] + qc.parameters[1].sin(), 1)
        qc.u(qc.parameters[0], qc.parameters[1], qc.parameters[3], 1)
        qc.p(2 * qc.parameters[0] + 1, 0)
        qc.rxx(qc.parameters[0] + 2, 0, 1)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4 for _ in qc.parameters], [np.pi / 2 for _ in qc.parameters]]
        correct_results = [
            [
                {
                    0: 0.30014831912265927,
                    1: -0.6634809704357856,
                    2: 0.343589357193753,
                    3: 0.019743294119373426,
                },
                {
                    0: 0.16470607453981906,
                    1: -0.40996282450610577,
                    2: 0.08791803062881773,
                    3: 0.15733871933746948,
                },
                {
                    0: 0.27036068339663866,
                    1: -0.273790986018701,
                    2: 0.12752010079553433,
                    3: -0.12408979817347202,
                },
                {
                    0: -0.2098616294167757,
                    1: -0.2515823946449894,
                    2: 0.21929102305386305,
                    3: 0.24215300100790207,
                },
            ],
            [
                {
                    0: -1.844810060881004,
                    1: 0.04620532700836027,
                    2: 1.6367366426074323,
                    3: 0.16186809126521057,
                },
                {
                    0: 0.07296073407769421,
                    1: -0.021774869186331716,
                    2: 0.02177486918633173,
                    3: -0.07296073407769456,
                },
                {
                    0: -0.07794369186049102,
                    1: -0.07794369186049122,
                    2: 0.07794369186049117,
                    3: 0.07794369186049112,
                },
                {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 0.0,
                },
            ],
        ]

        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(correct_results[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

    @data(*gradient_factories)
    @unpack
    def test_gradient_parameters(self, grad, shots, atol, rtol):
        """Test the sampler gradient for parameters"""
        sampler = StatevectorSampler(
            default_shots=shots, seed=np.random.default_rng(seed=self.seed)
        )
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rz(b, 0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4, np.pi / 2]]
        expected = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param], parameters=[[a]]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=1)
            array2 = _quasi2array(expected[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            qc.measure_all()
            param_values = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            params = [[a, b, c], [c, b, a], [a, c], [c, a]]
            expected = [
                [
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                    {0: 0.3061861668168149, 1: -0.3061861668167012},
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                ],
                [
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                    {0: 0.3061861668168149, 1: -0.3061861668167012},
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                ],
                [
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                ],
                [
                    {0: -0.3061861668168149, 1: 0.30618616681678645},
                    {0: -0.17677666583387008, 1: 0.17677666583378482},
                ],
            ]
            for i, p in enumerate(params):  # pylint: disable=invalid-name
                gradients = gradient.run([qc], param_values, parameters=[p]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=1)
                array2 = _quasi2array(expected[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

    @data(*gradient_factories)
    @unpack
    def test_gradient_multi_arguments(self, grad, shots, atol, rtol):
        """Test the sampler gradient for multiple arguments"""
        sampler = StatevectorSampler(
            default_shots=shots, seed=np.random.default_rng(seed=self.seed)
        )
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        qc2.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
            [{0: -0.5, 1: 0.5}],
        ]
        gradients = gradient.run([qc, qc2], param_list).result().gradients
        for i, q_dists in enumerate(gradients):
            array1 = _quasi2array(q_dists, num_qubits=1)
            array2 = _quasi2array(correct_results[i], num_qubits=1)
            np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

        # parameters
        with self.subTest(msg="Different parameters"):
            c = Parameter("c")
            qc3 = QuantumCircuit(1)
            qc3.rx(c, 0)
            qc3.ry(a, 0)
            qc3.measure_all()
            param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
            gradients = (
                gradient.run([qc, qc3, qc3], param_list2, parameters=[[a], [c], None])
                .result()
                .gradients
            )
            correct_results = [
                [{0: -0.5 / np.sqrt(2), 1: 0.5 / np.sqrt(2)}],
                [{0: -0.25, 1: 0.25}],
                [{0: -0.25, 1: 0.25}, {0: -0.25, 1: 0.25}],
            ]
            for i, q_dists in enumerate(gradients):
                array1 = _quasi2array(q_dists, num_qubits=1)
                array2 = _quasi2array(correct_results[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=atol, rtol=rtol)

    @data(*gradient_factories)
    @unpack
    # pylint: disable=unused-argument
    def test_gradient_validation(self, grad, shots, atol, rtol):
        """Test sampler gradient's validation"""
        sampler = StatevectorSampler()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()
        gradient = grad(sampler)
        param_list = [[np.pi / 4], [np.pi / 2]]
        with self.assertRaises(ValueError):
            gradient.run([qc], param_list)
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc], [[np.pi / 4, np.pi / 4]])

    def test_spsa_gradient(self):
        """Test the SPSA sampler gradient"""
        sampler = StatevectorSampler(default_shots=3_000_000, seed=np.random.default_rng(self.seed))
        with self.assertRaises(ValueError):
            _ = SPSASamplerGradient(sampler, epsilon=-0.1)

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(2)
        qc.rx(b, 0)
        qc.rx(a, 1)
        qc.measure_all()
        param_list = [[1, 2]]
        correct_results = [
            [
                {0: 0.2273244, 1: -0.6480598, 2: 0.2273244, 3: 0.1934111},
                {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
            ],
        ]
        gradient = SPSASamplerGradient(sampler, epsilon=1e-2, seed=123)
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [param]).result().gradients[0]
            array1 = _quasi2array(gradients, num_qubits=2)
            array2 = _quasi2array(correct_results[i], num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

        # multi parameters
        with self.subTest(msg="Multiple parameters"):
            param_list2 = [[1, 2], [1, 2], [3, 4]]
            correct_results2 = [
                [
                    {0: 0.2273244, 1: -0.6480598, 2: 0.2273244, 3: 0.1934111},
                    {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
                ],
                [
                    {0: -0.2273244, 1: 0.6480598, 2: -0.2273244, 3: -0.1934111},
                ],
                [
                    {0: -0.0141129, 1: -0.0564471, 2: -0.3642884, 3: 0.4348484},
                    {0: 0.0141129, 1: 0.0564471, 2: 0.3642884, 3: -0.4348484},
                ],
            ]
            gradient = SPSASamplerGradient(sampler, epsilon=1e-2, seed=123)
            gradients = (
                gradient.run([qc] * 3, param_list2, parameters=[None, [b], None]).result().gradients
            )
            for i, result in enumerate(gradients):
                array1 = _quasi2array(result, num_qubits=2)
                array2 = _quasi2array(correct_results2[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

        # batch size
        with self.subTest(msg="Batch size"):
            param_list = [[1, 1]]
            gradient = SPSASamplerGradient(sampler, epsilon=1e-2, batch_size=4, seed=123)
            gradients = gradient.run([qc], param_list).result().gradients
            correct_results3 = [
                [
                    {
                        0: -0.1620149622932887,
                        1: -0.25872053011771756,
                        2: 0.3723827084675668,
                        3: 0.04835278392088804,
                    },
                    {
                        0: -0.1620149622932887,
                        1: 0.3723827084675668,
                        2: -0.25872053011771756,
                        3: 0.04835278392088804,
                    },
                ]
            ]
            for i, q_dists in enumerate(gradients):
                array1 = _quasi2array(q_dists, num_qubits=2)
                array2 = _quasi2array(correct_results3[i], num_qubits=2)
                np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            qc.measure_all()
            param_list = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            correct_results = [
                [
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                ],
                [
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                ],
                [
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                ],
                [
                    {0: 0.17677624757590138, 1: -0.17677624757590138},
                    {0: -0.17677624757590138, 1: 0.17677624757590138},
                ],
            ]
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = SPSASamplerGradient(sampler, epsilon=1e-2, seed=123)
                gradients = gradient.run([qc], param_list, parameters=[p]).result().gradients[0]
                array1 = _quasi2array(gradients, num_qubits=1)
                array2 = _quasi2array(correct_results[i], num_qubits=1)
                np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

    @unittest.skip(
        "Should pass in approximately 3 hours and 20 minutes, and we're not sure to see the point "
        "of this test?"
    )
    @data(ParamShiftSamplerGradient, LinCombSamplerGradient)
    def test_gradient_random_parameters(self, grad):
        """Test param shift and lin comb w/ random parameters"""
        rng = np.random.default_rng(123)
        qc = real_amplitudes(num_qubits=3, reps=1)
        params = qc.parameters
        qc.rx(3.0 * params[0] + params[1].sin(), 0)
        qc.ry(params[0].exp() + 2 * params[1], 1)
        qc.rz(params[0] * params[1] - params[2], 2)
        qc.p(2 * params[0] + 1, 0)
        qc.u(params[0].sin(), params[1] - 2, params[2] * params[3], 1)
        qc.sx(2)
        qc.rxx(params[0].sin(), 1, 2)
        qc.ryy(params[1].cos(), 2, 0)
        qc.rzz(params[2] * 2, 0, 1)
        qc.crx(params[0].exp(), 1, 2)
        qc.cry(params[1].arctan(), 2, 0)
        qc.crz(params[2] * -2, 0, 1)
        qc.dcx(0, 1)
        qc.csdg(0, 1)
        qc.ccx(0, 1, 2)
        qc.iswap(0, 2)
        qc.swap(1, 2)
        qc.global_phase = params[0] * params[1] + params[2].cos().exp()
        qc.measure_all()

        sampler = StatevectorSampler(
            default_shots=1_000_000, seed=np.random.default_rng(seed=self.seed)
        )
        findiff = FiniteDiffSamplerGradient(sampler, 1e-2)
        gradient = grad(sampler)

        num_qubits = qc.num_qubits
        num_tries = 10
        param_values = rng.normal(0, 2, (num_tries, qc.num_parameters)).tolist()
        result1 = findiff.run([qc] * num_tries, param_values).result().gradients
        result2 = gradient.run([qc] * num_tries, param_values).result().gradients
        self.assertEqual(len(result1), len(result2))
        for res1, res2 in zip(result1, result2):
            array1 = _quasi2array(res1, num_qubits)
            array2 = _quasi2array(res2, num_qubits)
            np.testing.assert_allclose(array1, array2, rtol=1e-1, atol=1e-1)

    @data(
        FiniteDiffSamplerGradient,
        ParamShiftSamplerGradient,
        LinCombSamplerGradient,
        SPSASamplerGradient,
    )
    def test_shots(self, grad):
        """Test sampler gradient's shots options"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()
        sampler = StatevectorSampler(default_shots=100)
        with self.subTest("sampler"):
            if grad is FiniteDiffSamplerGradient or grad is SPSASamplerGradient:
                gradient = grad(sampler, epsilon=1e-6)
            else:
                gradient = grad(sampler)
            shots = gradient.shots
            result = gradient.run([qc], [[1]]).result()
            self.assertEqual(result.shots, 100)
            self.assertEqual(shots, None)

        with self.subTest("gradient init"):
            if grad is FiniteDiffSamplerGradient or grad is SPSASamplerGradient:
                gradient = grad(sampler, epsilon=1e-6, shots=200)
            else:
                gradient = grad(sampler, shots=200)
            shots = gradient.shots
            result = gradient.run([qc], [[1]]).result()
            self.assertEqual(result.shots, 200)
            self.assertEqual(shots, 200)

        with self.subTest("gradient update"):
            if grad is FiniteDiffSamplerGradient or grad is SPSASamplerGradient:
                gradient = grad(sampler, epsilon=1e-6, shots=200)
            else:
                gradient = grad(sampler, shots=200)
            gradient.shots = 300
            shots = gradient.shots
            result = gradient.run([qc], [[1]]).result()
            self.assertEqual(result.shots, 300)
            self.assertEqual(shots, 300)

        with self.subTest("gradient run"):
            if grad is FiniteDiffSamplerGradient or grad is SPSASamplerGradient:
                gradient = grad(sampler, epsilon=1e-6, shots=200)
            else:
                gradient = grad(sampler, shots=200)
            shots = gradient.shots
            result = gradient.run([qc], [[1]], shots=400).result()
            self.assertEqual(result.shots, 400)
            # Only default + sampler shots. Not run.
            self.assertEqual(shots, 200)

    @data(
        FiniteDiffSamplerGradient,
        ParamShiftSamplerGradient,
        LinCombSamplerGradient,
        SPSASamplerGradient,
    )
    def test_operations_preserved(self, gradient_cls):
        """Test non-parameterized instructions are preserved and not unrolled."""
        x = Parameter("x")
        circuit = QuantumCircuit(2)
        circuit.initialize(np.array([1, 1, 0, 0]) / np.sqrt(2))  # this should remain as initialize
        circuit.crx(x, 0, 1)  # this should get unrolled
        circuit.measure_all()

        values = [np.pi / 2]
        expect = [{0: 0, 1: -0.25, 2: 0, 3: 0.25}]

        ops = []

        def operations_callback(op):
            ops.append(op)

        sampler = LoggingSampler(shots=3_000_000, operations_callback=operations_callback)

        if gradient_cls in [SPSASamplerGradient, FiniteDiffSamplerGradient]:
            gradient = gradient_cls(sampler, epsilon=1e-2)
        else:
            gradient = gradient_cls(sampler)

        job = gradient.run([circuit], [values])
        result = job.result()

        with self.subTest(msg="assert initialize is preserved"):
            self.assertTrue(all("initialize" in ops_i[0].keys() for ops_i in ops))

        with self.subTest(msg="assert result is correct"):
            array1 = _quasi2array(result.gradients[0], num_qubits=2)
            array2 = _quasi2array(expect, num_qubits=2)
            np.testing.assert_allclose(array1, array2, atol=1e-1, rtol=1e-1)

    @data(
        FiniteDiffSamplerGradient,
        ParamShiftSamplerGradient,
        LinCombSamplerGradient,
        SPSASamplerGradient,
    )
    def test_transpiler(self, gradient_cls):
        """Test that the transpiler is called"""
        pass_manager = generate_preset_pass_manager(
            backend=THREE_QUBITS_BACKEND, optimization_level=1, seed_transpiler=42
        )
        counts = [0]

        def callback(**kwargs):
            counts[0] = kwargs["count"]

        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.measure_all()
        sampler = StatevectorSampler()

        # Test transpilation without options
        if gradient_cls in [SPSASamplerGradient, FiniteDiffSamplerGradient]:
            gradient = gradient_cls(sampler, epsilon=1e-2, transpiler=pass_manager)
        else:
            gradient = gradient_cls(sampler, transpiler=pass_manager)

        gradient.run([qc], [[1]]).result()

        # Test transpiler is called using callback function
        if gradient_cls in [SPSASamplerGradient, FiniteDiffSamplerGradient]:
            gradient = gradient_cls(
                sampler,
                epsilon=1e-2,
                transpiler=pass_manager,
                transpiler_options={"callback": callback},
            )
        else:
            gradient = gradient_cls(
                sampler, transpiler=pass_manager, transpiler_options={"callback": callback}
            )

        gradient.run([qc], [[1]]).result()

        self.assertGreater(counts[0], 0)


def _quasi2array(quasis: list[dict], num_qubits: int) -> np.ndarray:
    ret = np.zeros((len(quasis), 2**num_qubits))
    for i, quasi in enumerate(quasis):
        ret[i, list(quasi.keys())] = list(quasi.values())
    return ret


if __name__ == "__main__":
    unittest.main()
