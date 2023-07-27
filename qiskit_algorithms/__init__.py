# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================
Algorithms (:mod:`qiskit_algorithms`)
=====================================
It contains a collection of quantum algorithms, for use with quantum computers, to
carry out research and investigate how to solve problems in different domains on
near-term quantum devices with short depth circuits.

Algorithms configuration includes the use of :mod:`~qiskit_algorithms.optimizers` which
were designed to be swappable sub-parts of an algorithm. Any component and may be exchanged for
a different implementation of the same component type in order to potentially alter the behavior
and outcome of the algorithm.

.. currentmodule:: qiskit_algorithms

Algorithms
==========

It contains a variety of quantum algorithms and these have been grouped by logical function such
as minimum eigensolvers and amplitude amplifiers. These algorithms are based on the Qiskit Primitives.


Amplitude Amplifiers
--------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplificationProblem
   AmplitudeAmplifier
   Grover
   GroverResult


Amplitude Estimators
--------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplitudeEstimator
   AmplitudeEstimatorResult
   AmplitudeEstimation
   AmplitudeEstimationResult
   EstimationProblem
   FasterAmplitudeEstimation
   FasterAmplitudeEstimationResult
   IterativeAmplitudeEstimation
   IterativeAmplitudeEstimationResult
   MaximumLikelihoodAmplitudeEstimation
   MaximumLikelihoodAmplitudeEstimationResult


Eigensolvers
------------

Algorithms to find eigenvalues of an operator. For chemistry these can be used to find excited
states of a molecule, and ``qiskit-nature`` has some algorithms that leverage chemistry specific
knowledge to do this in that application domain.
These algorithms are based on the Qiskit Primitives.

.. autosummary::
   :toctree: ../stubs/

   eigensolvers

Time Evolvers
-------------

Algorithms to evolve quantum states in time. Both real and imaginary time evolution is possible
with algorithms that support them. For machine learning, Quantum Imaginary Time Evolution might be
used to train Quantum Boltzmann Machine Neural Networks for example.
These algorithms are based on the Qiskit Primitives.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   RealTimeEvolver
   ImaginaryTimeEvolver
   TimeEvolutionResult
   TimeEvolutionProblem
   PVQD
   PVQDResult
   SciPyImaginaryEvolver
   SciPyRealEvolver
   VarQITE
   VarQRTE

Variational Quantum Time Evolution
++++++++++++++++++++++++++++++++++

Classes used by variational quantum time evolution algorithms - :class:`.VarQITE` and
:class:`.VarQRTE`.

.. autosummary::
   :toctree: ../stubs/

   time_evolvers.variational


Trotterization-based Quantum Real Time Evolution
++++++++++++++++++++++++++++++++++++++++++++++++

Package for primitives-enabled Trotterization-based quantum time evolution
algorithm - :class:`~.time_evolvers.TrotterQRTE`.

.. autosummary::
   :toctree: ../stubs/

   time_evolvers.trotterization


Gradients
----------

Algorithms to calculate the gradient of a quantum circuit. These algorithms are based
on the Qiskit Primitives.

.. autosummary::
   :toctree: ../stubs/

   gradients


Minimum Eigensolvers
---------------------

Algorithms that can find the minimum eigenvalue of an operator. These algorithms are
based on the Qiskit Primitives.

.. autosummary::
   :toctree: ../stubs/

   minimum_eigensolvers


Optimizers
----------

Classical optimizers for use by quantum variational algorithms. These algorithms
are based on the Qiskit Primitives.

.. autosummary::
   :toctree: ../stubs/

   optimizers


Phase Estimators
----------------

Algorithms that estimate the phases of eigenstates of a unitary. These algorithms
are based on the Qiskit Primitives.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HamiltonianPhaseEstimation
   HamiltonianPhaseEstimationResult
   PhaseEstimationScale
   PhaseEstimation
   PhaseEstimationResult
   IterativePhaseEstimation


State Fidelities
----------------

Algorithms that compute the fidelity of pairs of quantum states. These algorithms
are based on the Qiskit Primitives.

.. autosummary::
   :toctree: ../stubs/

   state_fidelities


Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   AlgorithmError


Utility classes
---------------

Utility classes used by algorithms (mainly for type-hinting purposes).

.. autosummary::
   :toctree: ../stubs/

   AlgorithmJob

"""
from .algorithm_job import AlgorithmJob
from .algorithm_result import AlgorithmResult
from .variational_algorithm import VariationalAlgorithm, VariationalResult
from .amplitude_amplifiers import Grover, GroverResult, AmplificationProblem, AmplitudeAmplifier
from .amplitude_estimators import (
    AmplitudeEstimator,
    AmplitudeEstimatorResult,
    AmplitudeEstimation,
    AmplitudeEstimationResult,
    FasterAmplitudeEstimation,
    FasterAmplitudeEstimationResult,
    IterativeAmplitudeEstimation,
    IterativeAmplitudeEstimationResult,
    MaximumLikelihoodAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimationResult,
    EstimationProblem,
)

from .phase_estimators import (
    HamiltonianPhaseEstimation,
    HamiltonianPhaseEstimationResult,
    PhaseEstimationScale,
    PhaseEstimation,
    PhaseEstimationResult,
    IterativePhaseEstimation,
)
from .exceptions import AlgorithmError
from .observables_evaluator import estimate_observables

from .time_evolvers import (
    ImaginaryTimeEvolver,
    RealTimeEvolver,
    TimeEvolutionProblem,
    TimeEvolutionResult,
    PVQD,
    PVQDResult,
    SciPyImaginaryEvolver,
    SciPyRealEvolver,
    VarQITE,
    VarQRTE,
    VarQTE,
    VarQTEResult,
)

from .eigensolvers import (
    Eigensolver,
    EigensolverResult,
    NumPyEigensolver,
    NumPyEigensolverResult,
    VQD,
    VQDResult,
)

from .minimum_eigensolvers import (
    AdaptVQE,
    AdaptVQEResult,
    MinimumEigensolver,
    MinimumEigensolverResult,
    NumPyMinimumEigensolver,
    NumPyMinimumEigensolverResult,
    QAOA,
    SamplingMinimumEigensolver,
    SamplingMinimumEigensolverResult,
    SamplingVQE,
    SamplingVQEResult,
    VQE,
    VQEResult,
)

from .version import __version__

__all__ = [
    "__version__",
    "AlgorithmJob",
    "AlgorithmResult",
    "VariationalAlgorithm",
    "VariationalResult",
    "AmplitudeAmplifier",
    "AmplificationProblem",
    "Grover",
    "GroverResult",
    "AmplitudeEstimator",
    "AmplitudeEstimatorResult",
    "AmplitudeEstimation",
    "AmplitudeEstimationResult",
    "FasterAmplitudeEstimation",
    "FasterAmplitudeEstimationResult",
    "IterativeAmplitudeEstimation",
    "IterativeAmplitudeEstimationResult",
    "MaximumLikelihoodAmplitudeEstimation",
    "MaximumLikelihoodAmplitudeEstimationResult",
    "EstimationProblem",
    "RealTimeEvolver",
    "ImaginaryTimeEvolver",
    "TimeEvolutionResult",
    "TimeEvolutionProblem",
    "HamiltonianPhaseEstimation",
    "HamiltonianPhaseEstimationResult",
    "PhaseEstimationScale",
    "PhaseEstimation",
    "PhaseEstimationResult",
    "PVQD",
    "PVQDResult",
    "SciPyRealEvolver",
    "SciPyImaginaryEvolver",
    "IterativePhaseEstimation",
    "AlgorithmError",
    "estimate_observables",
    "VarQITE",
    "VarQRTE",
    "VarQTE",
    "VarQTEResult",
    "Eigensolver",
    "EigensolverResult",
    "NumPyEigensolver",
    "NumPyEigensolverResult",
    "VQD",
    "VQDResult",
    "AdaptVQE",
    "AdaptVQEResult",
    "MinimumEigensolver",
    "MinimumEigensolverResult",
    "NumPyMinimumEigensolver",
    "NumPyMinimumEigensolverResult",
    "QAOA",
    "SamplingMinimumEigensolver",
    "SamplingMinimumEigensolverResult",
    "SamplingVQE",
    "SamplingVQEResult",
    "VQE",
    "VQEResult",
]
