# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Additional optional constants.
"""

from qiskit.utils import LazyImportTester
from qiskit.version import get_version_info

HAS_NLOPT = LazyImportTester("nlopt", name="NLopt Optimizer", install="pip install nlopt")
HAS_SKQUANT = LazyImportTester(
    "skquant.opt",
    name="scikit-quant",
    install="pip install scikit-quant",
)
HAS_SQSNOBFIT = LazyImportTester("SQSnobFit", install="pip install SQSnobFit")
# From Qiskit 2.0.0 onward, tweedledum isn't required anymore to use the phase oracle
CAN_USE_PHASE_ORACLE = get_version_info() >= "2.0.0" or LazyImportTester(
    "tweedledum", install="pip install tweedledum"
)
