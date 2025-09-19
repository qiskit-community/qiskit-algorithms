# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Types used by the qiskit-algorithms package."""
from __future__ import annotations

from typing import Any, Protocol, TypeAlias, TypeVar, overload

from qiskit import QuantumCircuit

_T = TypeVar("_T")  # Pylint does not allow single character class names.
ListOrDict = list[_T | None] | dict[str, _T]

_Circuits: TypeAlias = list[QuantumCircuit] | QuantumCircuit


class Transpiler(Protocol):
    """A Generic type to represent a transpiler."""

    @overload
    def run(self, circuits: list[QuantumCircuit], /, **options: Any) -> list[QuantumCircuit]:
        """Transpile a list of quantum circuits."""
        ...

    @overload
    def run(self, circuits: QuantumCircuit, /, **options: Any) -> QuantumCircuit:
        """Transpile a single quantum circuit."""
        ...

    def run(self, circuits: _Circuits, /, **options: Any) -> _Circuits:
        """Transpile a circuit or a list of quantum circuits."""
        pass
