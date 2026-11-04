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

"""Set platform parameters and put them back afterwards (ADR-0006).

A node sets every platform parameter it touches through a :class:`ParameterRestore` handle rather
than calling ``platform.set_parameter`` directly, so :func:`restoring_parameters` can undo exactly
those changes in its ``finally`` -- including when the body raises. QProgram resets only what it
sets itself, so a node that moves a flux bias or an LO has to restore it, and this is where that
happens for every node instead of a hand-written ``finally`` per experiment.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qililab import Platform
    from qililab.typings.enums import Parameter


class ParameterRestore:
    """A handle that sets platform parameters and remembers each one's value before the first set.

    Set through :meth:`set` instead of ``platform.set_parameter``; :meth:`restore` then puts every
    touched parameter back to the value it held when the handle first touched it. Repeated sets of
    the same parameter keep the earliest value, so restoring returns to the state before the node
    ran, not to an intermediate one.
    """

    def __init__(self, platform: Platform) -> None:
        self._platform = platform
        self._prior: dict[tuple[str, Parameter], Any] = {}

    def set(self, alias: str, parameter: Parameter, value: Any) -> None:
        """Set ``parameter`` on bus ``alias`` to ``value``, recording its prior value on first touch."""
        key = (alias, parameter)
        if key not in self._prior:
            self._prior[key] = self._platform.get_parameter(alias=alias, parameter=parameter)
        self._platform.set_parameter(alias=alias, parameter=parameter, value=value)

    def restore(self) -> None:
        """Put every parameter touched through :meth:`set` back to its prior value, newest first."""
        for (alias, parameter), value in reversed(self._prior.items()):
            self._platform.set_parameter(alias=alias, parameter=parameter, value=value)
        self._prior.clear()


@contextmanager
def restoring_parameters(platform: Platform) -> Iterator[ParameterRestore]:
    """Yield a :class:`ParameterRestore` and restore every parameter set through it on exit.

    The restore runs in a ``finally``, so parameters are put back whether the body returns or raises.
    """
    restore = ParameterRestore(platform)
    try:
        yield restore
    finally:
        restore.restore()
