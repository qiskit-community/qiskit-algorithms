# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational Quantum Imaginary Time Evolution algorithm."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from collections.abc import Callable

import numpy as np
from scipy.integrate import OdeSolver

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BaseEstimatorV2

from .solvers.ode.forward_euler_solver import ForwardEulerSolver

from .variational_principles import ImaginaryVariationalPrinciple, ImaginaryMcLachlanPrinciple
from .var_qte import VarQTE

from ..imaginary_time_evolver import ImaginaryTimeEvolver
from ...custom_types import Transpiler


class VarQITE(VarQTE, ImaginaryTimeEvolver):
    """Variational Quantum Imaginary Time Evolution algorithm.

    .. code-block::python

        import numpy as np

        from qiskit_algorithms import TimeEvolutionProblem, VarQITE
        from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
        from qiskit.circuit.library import efficient_su2
        from qiskit.quantum_info import SparsePauliOp, Pauli
        from qiskit.primitives import StatevectorEstimator

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        ansatz = efficient_su2(observable.num_qubits, reps=1)
        init_param_values = np.ones(len(ansatz.parameters)) * np.pi/2
        var_principle = ImaginaryMcLachlanPrinciple()
        time = 1

        # without evaluating auxiliary operators
        evolution_problem = TimeEvolutionProblem(observable, time)
        var_qite = VarQITE(ansatz, init_param_values, var_principle)
        evolution_result = var_qite.evolve(evolution_problem)

        # evaluating auxiliary operators
        aux_ops = [Pauli("XX"), Pauli("YZ")]
        evolution_problem = TimeEvolutionProblem(observable, time, aux_operators=aux_ops)
        var_qite = VarQITE(ansatz, init_param_values, var_principle, StatevectorEstimator())
        evolution_result = var_qite.evolve(evolution_problem)
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_parameters: Mapping[Parameter, float] | Sequence[float],
        variational_principle: ImaginaryVariationalPrinciple | None = None,
        estimator: BaseEstimatorV2 | None = None,
        ode_solver: type[OdeSolver] | str = ForwardEulerSolver,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        num_timesteps: int | None = None,
        imag_part_tol: float = 1e-7,
        num_instability_tol: float = 1e-7,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Args:
            ansatz: Ansatz to be used for variational time evolution.
            initial_parameters: Initial parameter values for the ansatz.
            variational_principle: Variational Principle to be used. Defaults to
                ``ImaginaryMcLachlanPrinciple``.
            estimator: An estimator primitive used for calculating expectation values of
                TimeEvolutionProblem.aux_operators.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
            lse_solver: Linear system of equations solver callable. It accepts ``A`` and ``b`` to
                solve ``Ax=b`` and returns ``x``. If ``None``, the default ``np.linalg.lstsq``
                solver is used.
            num_timesteps: The number of timesteps to take. If ``None``, it is
                automatically selected to achieve a timestep of approximately 0.01. Only
                relevant in case of the ``ForwardEulerSolver``.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            num_instability_tol: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be non-negative.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are run when using this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        """
        if variational_principle is None:
            variational_principle = ImaginaryMcLachlanPrinciple()
        super().__init__(
            ansatz,
            initial_parameters,
            variational_principle,
            estimator,
            ode_solver,
            lse_solver=lse_solver,
            num_timesteps=num_timesteps,
            imag_part_tol=imag_part_tol,
            num_instability_tol=num_instability_tol,
            transpiler=transpiler,
            transpiler_options=transpiler_options,
        )
