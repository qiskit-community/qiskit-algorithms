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
Sampler result class
"""

from __future__ import annotations

from typing import Any, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplerGradientResult:
    """Result of SamplerGradient."""

    gradients: list[list[dict[int, float]]]
    """The gradients of the sample probabilities."""
    metadata: list[dict[str, Any]] | list[list[dict[str, Any]]]
    """Additional information about the job."""
    shots: int | Sequence[int]
    """Primitive number of shots for the execution of the job."""
