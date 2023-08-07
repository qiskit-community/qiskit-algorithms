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
============================================
Qiskit Algorithms (:mod:`qiskit_algorithms`)
============================================
Qiskit Algorithms is a library of quantum algorithms for quantum computing with
`Qiskit <https://qiskit.org>`_.
These algorithms can be used to carry out research and investigate how to solve
problems in different domains on simulators and near-term real quantum devices
using short depth circuits.

Now there are some classical algorithms here, for example the :class:`.NumPyMinimumEigensolver`,
which take the same input but solve the problem classically. This has utility in the near-term,
where problems are still tractable classically, to validate and/or act as a reference.
There are also classical :mod:`.optimizers` for use with variational algorithms such as :class:`.VQE`.

While :mod:`.optimizers` were been mentioned, algorithms may also use :mod:`.gradients` and
:mod:`.state_fidelities`. Any of these may also be used directly if you have need of such function.

The quantum algorithms here all use ``Primitives`` to execute quantum circuits. This can be an
``Estimator``, which computes expectation values, or a ``Sampler`` which computes
probability distributions. Refer to the specific algorithm for more information in this regard.

.. currentmodule:: qiskit_algorithms

Algorithms
==========

The algorithms are now presented and are grouped by logical function, such
as minimum eigensolvers, amplitude amplifiers, time evolvers etc. Within each group the
algorithms comform to an interface defined there, which allows them to be used interchangeably
in an application that specifies use of an algorithm type by that interface. E.g. A Qiskit Nature
application may take a minimum eigensolver, to solve a ground state problem, and require it to
conform to the :class:`.MinimumEigensolver` interface. As long as it does, for example
:class:`.VQE` is one such algorithm conforming to that interface, then the application
 can work with it.

Amplitude Amplifiers
--------------------
Algorithms based on amplitude amplification.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplificationProblem
   AmplitudeAmplifier
   Grover
   GroverResult


Amplitude Estimators
--------------------
Algorithms based on amplitude estimation.

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

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Eigensolver
   EigensolverResult
   NumPyEigensolver
   NumPyEigensolverResult
   VQD
   VQDResult


Gradients
---------
Algorithms to calculate the gradient of a quantum circuit.

.. autosummary::
   :toctree:

   gradients


Minimum Eigensolvers
--------------------
Algorithms to find the minimum eigenvalue of an operator.

The set of these algorithms, taking an ``Estimator`` primitive, that can
solve for a general hamiltonian.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolver
   MinimumEigensolverResult
   NumPyMinimumEigensolver
   NumPyMinimumEigensolverResult
   VQE
   VQEResult
   AdaptVQE
   AdaptVQEResult

The set of these algorithms, taking an ``Sampler`` primitive, that can
solve for a diagonal hamiltonian such as an Ising Hamiltonian of an optimization problem.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SamplingMinimumEigensolver
   SamplingMinimumEigensolverResult
   SamplingVQE
   SamplingVQEResult
   QAOA


Optimizers
----------
Classical optimizers designed for use by quantum variational algorithms.

.. autosummary::
   :toctree:

   optimizers


Phase Estimators
----------------
Algorithms that estimate the phases of eigenstates of a unitary.

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
Algorithms that compute the fidelity of pairs of quantum states.

.. autosummary::
   :toctree:

   state_fidelities


Time Evolvers
-------------
Algorithms to evolve quantum states in time. Both real and imaginary time evolution is possible
with algorithms that support them. For machine learning, Quantum Imaginary Time Evolution might be
used to train Quantum Boltzmann Machine Neural Networks for example.

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
Classes used by variational quantum time evolution algorithms -
:class:`.VarQITE` and :class:`.VarQRTE`.

.. autosummary::
   :toctree:

   time_evolvers.variational


Trotterization-based Quantum Real Time Evolution
++++++++++++++++++++++++++++++++++++++++++++++++
Trotterization-based quantum time evolution algorithms -
:class:`~.time_evolvers.trotterization.TrotterQRTE`.

.. autosummary::
   :toctree:

   time_evolvers.trotterization


Miscellaneous
=============
Various classes used by qiskit-algorithms that are part of and exposed
by the public API.


Exceptions
----------

.. autosummary::
   :toctree:
   :nosignatures:

   AlgorithmError


Utility classes
---------------

Utility classes used by algorithms (mainly for type-hinting purposes).

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

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
