# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the quantum amplitude estimation algorithm."""

import unittest

import numpy as np
from ddt import ddt, idata, data, unpack
from qiskit import QuantumRegister, QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import QFT, GroverOperator
from qiskit.quantum_info import Operator, Statevector
from qiskit.primitives import StatevectorSampler

from qiskit_algorithms import (
    AmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
    IterativeAmplitudeEstimation,
    FasterAmplitudeEstimation,
    EstimationProblem,
)
from test import QiskitAlgorithmsTestCase  # pylint: disable=wrong-import-order


class BernoulliStateIn(QuantumCircuit):
    """A circuit preparing sqrt(1 - p)|0> + sqrt(p)|1>."""

    def __init__(self, probability):
        super().__init__(1)
        angle = 2 * np.arcsin(np.sqrt(probability))
        self.ry(angle, 0)


class BernoulliGrover(QuantumCircuit):
    """The Grover operator corresponding to the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(1, global_phase=np.pi)
        self.angle = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self.angle, 0)

    # Disable for pylint needed for Qiskit < 1.1.0 where annotated does not exist
    # pylint: disable=unused-argument
    def power(self, power, matrix_power=False, annotated: bool = False):
        if matrix_power:
            return super().power(power, True)

        powered = QuantumCircuit(1)
        powered.ry(power * 2 * self.angle, 0)
        return powered


class SineIntegral(QuantumCircuit):
    r"""Construct the A operator to approximate the integral

        \int_0^1 \sin^2(x) d x

    with a specified number of qubits.
    """

    def __init__(self, num_qubits):
        qr_state = QuantumRegister(num_qubits, "state")
        qr_objective = QuantumRegister(1, "obj")
        super().__init__(qr_state, qr_objective)

        # prepare 1/sqrt{2^n} sum_x |x>_n
        self.h(qr_state)

        # apply the sine/cosine term
        self.ry(2 * 1 / 2 / 2**num_qubits, qr_objective[0])
        for i, qubit in enumerate(qr_state):
            self.cry(2 * 2**i / 2**num_qubits, qubit, qr_objective[0])


@ddt
class TestBernoulli(QiskitAlgorithmsTestCase):
    """Tests based on the Bernoulli A operator.

    This class tests
        * the estimation result
        * the constructed circuits
    """

    def setUp(self):
        super().setUp()

        def sampler_shots(shots=10_000):
            return StatevectorSampler(default_shots=shots, seed=42)

        self._sampler_shots = sampler_shots

    @idata(
        [
            [0.2, 100_000, AmplitudeEstimation(2), {"estimation": 0.5, "mle": 0.2}],
            [0.49, 1_000_000, AmplitudeEstimation(3), {"estimation": 0.5, "mle": 0.49}],
            [0.2, 100_000, MaximumLikelihoodAmplitudeEstimation([0, 1, 2]), {"estimation": 0.2}],
            [0.49, 100_000, MaximumLikelihoodAmplitudeEstimation(3), {"estimation": 0.49}],
            [0.2, 1_000_000, IterativeAmplitudeEstimation(0.1, 0.1), {"estimation": 0.2}],
            [0.49, 100_000, IterativeAmplitudeEstimation(0.001, 0.01), {"estimation": 0.49}],
            # Number of shots for Sampler is not used in FasterAmplitudeEstimation
            [0.2, 1, FasterAmplitudeEstimation(0.01, 5, rescale=False), {"estimation": 0.2}],
            [0.12, 1, FasterAmplitudeEstimation(0.1, 3, rescale=False), {"estimation": 0.12}],
        ]
    )
    @unpack
    def test_sampler_with_shots(self, prob, shots, qae, expect):
        """sampler with shots test"""
        qae.sampler = self._sampler_shots(shots)
        problem = EstimationProblem(BernoulliStateIn(prob), [0], BernoulliGrover(prob))

        result = qae.estimate(problem)
        for key, value in expect.items():
            self.assertAlmostEqual(
                value, getattr(result, key), places=3, msg=f"estimate `{key}` failed"
            )

    @data(True, False)
    def test_qae_circuit(self, efficient_circuit):
        """Test circuits resulting from canonical amplitude estimation.

        Build the circuit manually and from the algorithm and compare the resulting unitaries.
        """
        prob = 0.5

        problem = EstimationProblem(BernoulliStateIn(prob), objective_qubits=[0])
        for m in [2, 5]:
            qae = AmplitudeEstimation(m)
            angle = 2 * np.arcsin(np.sqrt(prob))

            # manually set up the inefficient AE circuit
            qr_eval = QuantumRegister(m, "a")
            qr_objective = QuantumRegister(1, "q")
            circuit = QuantumCircuit(qr_eval, qr_objective)

            # initial Hadamard gates
            for i in range(m):
                circuit.h(qr_eval[i])

            # A operator
            circuit.ry(angle, qr_objective)

            if efficient_circuit:
                qae.grover_operator = BernoulliGrover(prob)
                for power in range(m):
                    circuit.cry(2 * 2**power * angle, qr_eval[power], qr_objective[0])
            else:
                oracle = QuantumCircuit(1)
                oracle.z(0)

                state_preparation = QuantumCircuit(1)
                state_preparation.ry(angle, 0)
                grover_op = GroverOperator(oracle, state_preparation)
                for power in range(m):
                    circuit.compose(
                        grover_op.power(2**power).control(),
                        qubits=[qr_eval[power], qr_objective[0]],
                        inplace=True,
                    )

            # fourier transform
            iqft = QFT(m, do_swaps=False).inverse().reverse_bits()
            circuit.append(iqft.to_instruction(), qr_eval)

            actual_circuit = qae.construct_circuit(problem, measurement=False)

            self.assertEqual(Operator(circuit), Operator(actual_circuit))

    @data(True, False)
    def test_iqae_circuits(self, efficient_circuit):
        """Test circuits resulting from iterative amplitude estimation.

        Build the circuit manually and from the algorithm and compare the resulting unitaries.
        """
        prob = 0.5
        problem = EstimationProblem(BernoulliStateIn(prob), objective_qubits=[0])

        for k in [2, 5]:
            qae = IterativeAmplitudeEstimation(0.01, 0.05)
            angle = 2 * np.arcsin(np.sqrt(prob))

            # manually set up the inefficient AE circuit
            q_objective = QuantumRegister(1, "q")
            circuit = QuantumCircuit(q_objective)

            # A operator
            circuit.ry(angle, q_objective)

            if efficient_circuit:
                qae.grover_operator = BernoulliGrover(prob)
                circuit.ry(2 * k * angle, q_objective[0])

            else:
                oracle = QuantumCircuit(1)
                oracle.z(0)
                state_preparation = QuantumCircuit(1)
                state_preparation.ry(angle, 0)
                grover_op = GroverOperator(oracle, state_preparation)
                for _ in range(k):
                    circuit.compose(grover_op, inplace=True)

            actual_circuit = qae.construct_circuit(problem, k, measurement=False)
            self.assertEqual(Operator(circuit), Operator(actual_circuit))

    @data(True, False)
    def test_mlae_circuits(self, efficient_circuit):
        """Test the circuits constructed for MLAE"""
        prob = 0.5
        problem = EstimationProblem(BernoulliStateIn(prob), objective_qubits=[0])

        for k in [2, 5]:
            qae = MaximumLikelihoodAmplitudeEstimation(k)
            angle = 2 * np.arcsin(np.sqrt(prob))

            # compute all the circuits used for MLAE
            circuits = []

            # 0th power
            q_objective = QuantumRegister(1, "q")
            circuit = QuantumCircuit(q_objective)
            circuit.ry(angle, q_objective)
            circuits += [circuit]

            # powers of 2
            for power in range(k):
                q_objective = QuantumRegister(1, "q")
                circuit = QuantumCircuit(q_objective)

                # A operator
                circuit.ry(angle, q_objective)

                # Q^(2^j) operator
                if efficient_circuit:
                    qae.grover_operator = BernoulliGrover(prob)
                    circuit.ry(2 * 2**power * angle, q_objective[0])

                else:
                    oracle = QuantumCircuit(1)
                    oracle.z(0)
                    state_preparation = QuantumCircuit(1)
                    state_preparation.ry(angle, 0)
                    grover_op = GroverOperator(oracle, state_preparation)
                    for _ in range(2**power):
                        circuit.compose(grover_op, inplace=True)
                circuits += [circuit]

            actual_circuits = qae.construct_circuits(problem, measurement=False)

            for actual, expected in zip(actual_circuits, circuits):
                self.assertEqual(Operator(actual), Operator(expected))


@ddt
class TestSineIntegral(QiskitAlgorithmsTestCase):
    """Tests based on the A operator to integrate sin^2(x).

    This class tests
        * the estimation result
        * the confidence intervals
    """

    def setUp(self):
        super().setUp()

        def sampler_shots(shots=100):
            return StatevectorSampler(default_shots=shots, seed=42)

        self._sampler_shots = sampler_shots

    @idata(
        [
            [2, 1_000_000, AmplitudeEstimation(2), {"estimation": 0.5, "mle": 0.2702}],
            [4, 100_000, MaximumLikelihoodAmplitudeEstimation(4), {"estimation": 0.2725}],
            [3, 1_000_000, IterativeAmplitudeEstimation(0.1, 0.1), {"estimation": 0.2721}],
            [3, 1, FasterAmplitudeEstimation(0.01, 6), {"estimation": 0.2721}],
        ]
    )
    @unpack
    def test_sampler_with_shots(self, n, shots, qae, expect):
        """Sampler with shots end-to-end test."""
        # construct factories for A and Q
        qae.sampler = self._sampler_shots(shots)
        estimation_problem = EstimationProblem(SineIntegral(n), objective_qubits=[n])

        result = qae.estimate(estimation_problem)
        for key, value in expect.items():
            self.assertAlmostEqual(
                value, getattr(result, key), places=3, msg=f"estimate `{key}` failed"
            )

    @idata(
        [
            [
                AmplitudeEstimation(3),
                "mle",
                {
                    "likelihood_ratio": (0.2494734, 0.3003771),
                    "fisher": (0.2486176, 0.2999286),
                    "observed_fisher": (0.2484562, 0.3000900),
                },
            ],
            [
                MaximumLikelihoodAmplitudeEstimation(3),
                "estimation",
                {
                    "likelihood_ratio": (0.2598794, 0.2798536),
                    "fisher": (0.2584889, 0.2797018),
                    "observed_fisher": (0.2659279, 0.2722627),
                },
            ],
        ]
    )
    @unpack
    def test_confidence_intervals(self, qae, key, expect):
        """End-to-end test for all confidence intervals."""
        n = 3

        # shots
        shots = 100
        alpha = 0.01

        estimation_problem = EstimationProblem(SineIntegral(n), objective_qubits=[n])
        qae.sampler = self._sampler_shots(shots)
        result = qae.estimate(estimation_problem)

        for method, expected_confint in expect.items():
            confint = qae.compute_confidence_interval(result, alpha, method)
            np.testing.assert_array_almost_equal(np.asarray(confint), expected_confint, decimal=1)
            self.assertTrue(confint[0] <= getattr(result, key) <= confint[1])

    def test_iqae_confidence_intervals(self):
        """End-to-end test for the IQAE confidence interval."""
        n = 3
        # Careful, changes according to the seed
        expected_confint = (
            0.26,
            0.32,
        )

        estimation_problem = EstimationProblem(SineIntegral(n), objective_qubits=[n])

        shots = 100
        qae = IterativeAmplitudeEstimation(0.1, 0.01, sampler=self._sampler_shots(shots))
        result = qae.estimate(estimation_problem)

        confint = result.confidence_interval
        np.testing.assert_array_almost_equal(confint, expected_confint, decimal=2)
        self.assertTrue(confint[0] <= result.estimation <= confint[1])


@ddt
class TestTranspiler(QiskitAlgorithmsTestCase):
    """Tests to check that the transpiler is indeed called."""

    def setUp(self):
        super().setUp()

        self.pm = generate_preset_pass_manager(optimization_level=1, seed_transpiler=42)
        self.counts = [0]

        # pylint: disable=unused-argument
        def callback(**kwargs):
            self.counts[0] += 1

        self.callback = callback

        circuit = QuantumCircuit(1)
        self.problem = EstimationProblem(
            circuit, objective_qubits=[0], is_good_state=lambda x: True
        )

    @idata(
        [
            [AmplitudeEstimation, {"num_eval_qubits": 1}],
            [IterativeAmplitudeEstimation, {"epsilon_target": 0.1, "alpha": 0.1}],
        ]
    )
    @unpack
    def test_transpiler_ae_iae(self, qae_class, kwargs):
        """Test that the transpiler is called on AE and IAE"""
        # Test transpilation without setting options
        qae = qae_class(transpiler=self.pm, **kwargs)
        qae.construct_circuit(self.problem)

        # Test transpiler is called using callback function
        qae = qae_class(
            transpiler=self.pm, transpiler_options={"callback": self.callback}, **kwargs
        )
        qae.construct_circuit(self.problem)

        self.assertGreater(self.counts[0], 0)

    @unittest.skip("Won't pass until Qiskit/qiskit#14250 is fixed")
    def test_transpiler_mlae(self):
        """Test that the transpiler is called on MLAE"""
        # Test transpilation without setting options
        mlae = MaximumLikelihoodAmplitudeEstimation([0, 1], transpiler=self.pm)
        mlae.construct_circuits(self.problem)

        # Test transpiler is called using callback function
        mlae = MaximumLikelihoodAmplitudeEstimation(
            [0, 1], transpiler=self.pm, transpiler_options={"callback": self.callback}
        )
        mlae.construct_circuits(self.problem)

        self.assertGreater(self.counts[0], 0)

    def test_transpiler_fae(self):
        """Test that the transpiler is called on FAE"""
        # Test transpilation without setting options
        fae = FasterAmplitudeEstimation(0.1, 1, transpiler=self.pm)
        fae.construct_circuit(self.problem, k=1)

        # Test transpiler is called using callback function
        fae = FasterAmplitudeEstimation(
            0.1, 1, transpiler=self.pm, transpiler_options={"callback": self.callback}
        )
        fae.construct_circuit(self.problem, k=1)

        self.assertGreater(self.counts[0], 0)


class TestAmplitudeEstimation(QiskitAlgorithmsTestCase):
    """Specific tests for canonical AE."""

    def test_warns_if_good_state_set(self):
        """Check AE warns if is_good_state is set."""
        circuit = QuantumCircuit(1)
        problem = EstimationProblem(circuit, objective_qubits=[0], is_good_state=lambda x: True)

        qae = AmplitudeEstimation(num_eval_qubits=1, sampler=StatevectorSampler())

        with self.assertWarns(Warning):
            _ = qae.estimate(problem)


class TestFasterAmplitudeEstimation(QiskitAlgorithmsTestCase):
    """Specific tests for Faster AE."""

    def setUp(self):
        super().setUp()
        self._sampler = StatevectorSampler(default_shots=10_000, seed=2)

    def test_rescaling(self):
        """Test the rescaling."""
        amplitude = 0.8
        scaling = 0.25
        circuit = QuantumCircuit(1)
        circuit.ry(2 * np.arcsin(amplitude), 0)
        problem = EstimationProblem(circuit, objective_qubits=[0])

        rescaled = problem.rescale(scaling)
        rescaled_amplitude = Statevector.from_instruction(rescaled.state_preparation).data[3]

        self.assertAlmostEqual(scaling * amplitude, rescaled_amplitude)

    def test_sampler_run_without_rescaling(self):
        """Run Faster AE without rescaling if the amplitude is in [0, 1/4]."""
        # construct estimation problem
        prob = 0.11
        a_op = QuantumCircuit(1)
        a_op.ry(2 * np.arcsin(np.sqrt(prob)), 0)
        problem = EstimationProblem(a_op, objective_qubits=[0])

        # construct algo without rescaling
        fae = FasterAmplitudeEstimation(0.1, 1, rescale=False, sampler=self._sampler)

        # run the algo
        result = fae.estimate(problem)

        # assert the result is correct
        self.assertAlmostEqual(result.estimation, prob, places=2)

        # assert no rescaling was used
        theta = np.mean(result.theta_intervals[-1])
        value_without_scaling = np.sin(theta) ** 2
        self.assertAlmostEqual(result.estimation, value_without_scaling)

    def test_rescaling_with_custom_grover_raises(self):
        """Test that the rescaling option fails if a custom Grover operator is used."""
        prob = 0.8
        a_op = BernoulliStateIn(prob)
        q_op = BernoulliGrover(prob)
        problem = EstimationProblem(a_op, objective_qubits=[0], grover_operator=q_op)

        # construct algo without rescaling
        fae = FasterAmplitudeEstimation(0.1, 1, sampler=self._sampler)

        # run the algo
        with self.assertWarns(Warning):
            _ = fae.estimate(problem)

    def test_good_state(self):
        """Test with a good state function."""

        expect = 0.2

        def is_good_state(bitstr):
            return bitstr[1] == "1"

        # construct the estimation problem where the second qubit is ignored
        a_op = QuantumCircuit(2)
        a_op.ry(2 * np.arcsin(np.sqrt(0.2)), 0)

        # oracle only affects first qubit
        oracle = QuantumCircuit(2)
        oracle.z(0)

        # reflect only on first qubit
        q_op = GroverOperator(oracle, a_op, reflection_qubits=[0])

        # but we measure both qubits (hence both are objective qubits)
        problem = EstimationProblem(
            a_op, objective_qubits=[0, 1], grover_operator=q_op, is_good_state=is_good_state
        )

        # cannot use rescaling with a custom grover operator
        fae = FasterAmplitudeEstimation(0.01, 5, rescale=False, sampler=self._sampler)

        # run the algo
        result = fae.estimate(problem)

        # assert the result is correct
        self.assertAlmostEqual(result.estimation, expect, places=3)  # reduced from 5


if __name__ == "__main__":
    unittest.main()
