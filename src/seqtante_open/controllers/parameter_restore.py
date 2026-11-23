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
from typing import TYPE_CHECKING

from qililab.typings.enums import Parameter

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qililab import Platform
    from qililab.typings import ParameterValue

#: Parameters whose read-back value cannot be set back as-is, mapped to the scalar parameters it splits
#: into. A Qblox sequencer reads ``GAIN`` back as the tuple ``(gain_i, gain_q)`` but only sets it from one
#: float, so the prior gain is restored through ``GAIN_I`` and ``GAIN_Q``. A qililab workaround: drop the
#: entry once qililab's ``GAIN`` round-trips.
_RESTORED_AS_PARTS: dict[Parameter, tuple[Parameter, ...]] = {
    Parameter.GAIN: (Parameter.GAIN_I, Parameter.GAIN_Q),
}


class ParameterRestore:
    """A handle that sets platform parameters and remembers each one's value before the first set.

    Set through :meth:`set` instead of ``platform.set_parameter``; :meth:`restore` then puts every
    touched parameter back to the value it held when the handle first touched it. Repeated sets of
    the same parameter keep the earliest value, so restoring returns to the state before the node
    ran, not to an intermediate one.
    """

    def __init__(self, platform: Platform) -> None:
        self._platform = platform
        self._touched: set[tuple[str, Parameter]] = set()
        self._prior: dict[tuple[str, Parameter], ParameterValue] = {}

    def set(self, alias: str, parameter: Parameter, value: ParameterValue) -> None:
        """Set ``parameter`` on bus ``alias`` to ``value``, recording its prior value on first touch.

        A prior value that reads back as one value per part (a Qblox ``GAIN`` as ``(gain_i, gain_q)``) is
        recorded per part, so :meth:`restore` sets each scalar back instead of the tuple.
        """
        key = (alias, parameter)
        if key not in self._touched:
            self._touched.add(key)
            prior = self._platform.get_parameter(alias=alias, parameter=parameter)
            for restore_key, restore_value in _restore_entries(alias, parameter, prior):
                self._prior.setdefault(restore_key, restore_value)
        self._platform.set_parameter(alias=alias, parameter=parameter, value=value)

    def restore(self) -> None:
        """Put every parameter touched through :meth:`set` back to its prior value, newest first.

        Every parameter is attempted even if an earlier one fails to set, so a single failing restore
        does not strand the rest at their set values. Any failures are re-raised together once every
        parameter has been attempted.
        """
        errors: list[Exception] = []
        for (alias, parameter), value in reversed(self._prior.items()):
            try:
                self._platform.set_parameter(alias=alias, parameter=parameter, value=value)
            except Exception as error:  # ruff: ignore[blind-except] -- every parameter must be attempted
                errors.append(error)
        self._prior.clear()
        self._touched.clear()
        if errors:
            raise ExceptionGroup("restoring parameters failed", errors)


def _restore_entries(
    alias: str, parameter: Parameter, prior: ParameterValue
) -> list[tuple[tuple[str, Parameter], ParameterValue]]:
    """The ``(alias, parameter)`` keys and values that put ``parameter`` back to ``prior``.

    One entry, unless ``parameter`` reads back as a tuple of its parts (see ``_RESTORED_AS_PARTS``): then
    one entry per part. The earliest value of a part wins, as for any parameter.
    """
    parts = _RESTORED_AS_PARTS.get(parameter)
    if parts is not None and isinstance(prior, tuple) and len(prior) == len(parts):
        return [((alias, part), value) for part, value in zip(parts, prior, strict=True)]
    return [((alias, parameter), prior)]


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
