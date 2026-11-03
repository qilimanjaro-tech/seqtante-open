# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment targets: qubits and couplers, named ``q14`` and ``c2_3`` (ADR-0001).

A target equals its name, so ``QubitTarget(14) == "q14"``, and hashes like it, so a target and its
name find each other as dict keys.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

_QUBIT_NAME = re.compile(r"q(0|[1-9]\d*)", re.ASCII)
_COUPLER_NAME = re.compile(r"c(0|[1-9]\d*)_(0|[1-9]\d*)", re.ASCII)


@dataclass(frozen=True, eq=False)
class ExperimentTarget:
    """Base of every target. Build one with a subclass or :meth:`from_str`.

    Args:
        kind (str): ``"qubit"`` or ``"coupler"``, set by the subclass.
    """

    kind: str = field(init=False)

    @staticmethod
    def from_str(name: str) -> "Target":
        """Parses a target name: ``q<n>`` or ``c<n>_<m>`` with ``n < m``, without leading zeros.

        Raises:
            ValueError: If ``name`` is not a string or not a valid target name.
        """
        if isinstance(name, str):
            if qubit := _QUBIT_NAME.fullmatch(name):
                return QubitTarget(int(qubit[1]))
            if (coupler := _COUPLER_NAME.fullmatch(name)) and int(coupler[1]) < int(coupler[2]):
                return CouplerTarget(int(coupler[1]), int(coupler[2]))
        raise ValueError(
            f"Invalid target {name!r}: expected a qubit 'q<n>' or a coupler 'c<n>_<m>' with n < m, without leading zeros."
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, ExperimentTarget)):
            return str(self) == str(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass(frozen=True, eq=False)
class QubitTarget(ExperimentTarget):
    """A qubit, named ``q<qubit_id>``.

    Args:
        qubit_id (int): Index of the qubit.
    """

    qubit_id: int
    kind: Literal["qubit"] = field(default="qubit", init=False, repr=False)

    def __post_init__(self) -> None:
        _check_index(self.qubit_id, "qubit_id")

    def __str__(self) -> str:
        return f"q{self.qubit_id}"


@dataclass(frozen=True, eq=False)
class CouplerTarget(ExperimentTarget):
    """A coupler between two qubits, named ``c<qubit_id_1>_<qubit_id_2>``, with ``qubit_id_1 < qubit_id_2``.

    Args:
        qubit_id_1 (int): Index of the lower qubit.
        qubit_id_2 (int): Index of the higher qubit.
    """

    qubit_id_1: int
    qubit_id_2: int
    kind: Literal["coupler"] = field(default="coupler", init=False, repr=False)

    def __post_init__(self) -> None:
        _check_index(self.qubit_id_1, "qubit_id_1")
        _check_index(self.qubit_id_2, "qubit_id_2")
        if self.qubit_id_1 >= self.qubit_id_2:
            raise ValueError(
                f"Invalid coupler between qubits {self.qubit_id_1} and {self.qubit_id_2}: qubit_id_1 must be lower "
                "than qubit_id_2. Use CouplerTarget.between to order them."
            )

    @classmethod
    def between(cls, qubit_a: int, qubit_b: int) -> "CouplerTarget":
        """The coupler between two qubits, in either order, as listed in the runcard's topology."""
        return cls(min(qubit_a, qubit_b), max(qubit_a, qubit_b))

    def __str__(self) -> str:
        return f"c{self.qubit_id_1}_{self.qubit_id_2}"


Target = QubitTarget | CouplerTarget
"""Any target, for annotations that narrow with ``isinstance``."""


def _check_index(value: object, name: str) -> None:
    """Rejects anything but a non-negative ``int`` as a qubit index."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid {name} {value!r}: must be a non-negative int.")
