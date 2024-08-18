# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ADAM optimizer."""

from test import QiskitAlgorithmsTestCase

from ddt import ddt, data
import numpy as np

from qiskit_algorithms.optimizers import ADAM, Optimizer
from qiskit_algorithms.utils import algorithm_globals


@ddt
class TestADAM(QiskitAlgorithmsTestCase):
    """Tests for the ADAM optimizer."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 52
        # Feature vector
        self.x = np.array([1, 2, 3, 4])
        # Target value
        self.y = 5

    def objective(self, w):
        """
        Objective function to minimize mean squared error.

        Parameters:
        w : numpy array
            The weights (including bias) for the linear model.

        Returns:
        float
            The mean squared error.
        """
        # Extract weights and bias from the parameter vector
        new_shape = (5, int(len(w) / 5))
        w = np.reshape(w, new_shape)

        weights = w[:-1, :]
        bias = w[-1, :]
        # Calculate the predicted values
        y_pred = np.dot(self.x, weights) + bias
        # Calculate the mean squared error
        mse = np.mean((self.y - y_pred) ** 2)
        return mse

    def run_optimizer(self, optimizer: Optimizer, weights: np.ndarray, max_nfev: int):
        """Test the optimizer.

        Args:
            optimizer: The optimizer instance to test.
            weights: The weights to optimize.
            max_nfev: The maximal allowed number of function evaluations.
        """

        # Minimize
        res = optimizer.minimize(self.objective, np.array(weights), None)
        error = res.fun
        nfev = res.nfev

        self.assertAlmostEqual(error, 0, places=3)
        self.assertLessEqual(nfev, max_nfev)

    @data(1, 5)
    def test_adam_max_evals(self, max_evals_grouped):
        """adam test"""
        # Initialize weights (including bias)
        w = np.zeros(len(self.x) + 1)
        # Initialize optimizer
        optimizer = ADAM(maxiter=10000, tol=1e-06)
        # Test one evaluation at a time
        optimizer.set_max_evals_grouped(max_evals_grouped)
        self.run_optimizer(optimizer, w, max_nfev=10000)
