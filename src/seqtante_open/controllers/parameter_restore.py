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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qililab import Platform
    from qililab.typings import ParameterValue
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
        self._prior: dict[tuple[str, Parameter], ParameterValue] = {}

    def set(self, alias: str, parameter: Parameter, value: ParameterValue) -> None:
        """Set ``parameter`` on bus ``alias`` to ``value``, recording its prior value on first touch."""
        key = (alias, parameter)
        if key not in self._prior:
            self._prior[key] = self._platform.get_parameter(alias=alias, parameter=parameter)
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
        if errors:
            raise ExceptionGroup("restoring parameters failed", errors)


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


class _RestoringPlatform:
    """A platform-facing proxy whose ``set_parameter`` records prior values for restore (ADR-0006).

    An experiment driver ported from the legacy code calls ``platform.set_parameter`` directly for the
    LO, IF and power it needs, which bypasses the restore handle and leaves those values on the fridge
    for the next node. Passing this proxy in its place routes every ``set_parameter`` through a
    :class:`ParameterRestore`, so :func:`restoring_parameters` undoes them, without the driver having to
    know about the handle. Everything else -- ``get_parameter``, ``execute_qprogram``, ``get_element``,
    the attributes :class:`~qililab.result.StreamArray` reads -- is delegated to the real platform.
    """

    def __init__(self, platform: Platform, restore: ParameterRestore) -> None:
        self._platform = platform
        self._restore = restore

    def set_parameter(self, alias: str, parameter: Parameter, value: ParameterValue, *args: Any, **kwargs: Any) -> None:
        """Set ``parameter`` on ``alias`` through the restore handle so ``run`` can undo it."""
        if args or kwargs:
            # The restore handle keys on (alias, parameter) only; a channel-scoped set cannot be undone
            # through it yet. Fail loudly rather than silently leaving the change unrestored.
            raise NotImplementedError("restoring_platform does not support extra set_parameter arguments yet")
        self._restore.set(alias, parameter, value)

    def __getattr__(self, name: str) -> object:
        # Reached only for attributes this proxy does not define, so set_parameter is never delegated.
        # The proxy is only ever used through the Platform-typed restoring_platform() below, so this
        # return type is not what callers see; object keeps it off Any without weakening that boundary.
        return getattr(self._platform, name)


def restoring_platform(platform: Platform, restore: ParameterRestore) -> Platform:
    """A :class:`~qililab.Platform`-typed proxy whose parameter writes go through ``restore`` (ADR-0006).

    Typed as a ``Platform`` so a driver that takes one accepts it; at runtime it is the thin
    :class:`_RestoringPlatform` proxy, which records each ``set_parameter`` for
    :func:`restoring_parameters` to undo and delegates everything else to ``platform``.
    """
    return cast("Platform", _RestoringPlatform(platform, restore))
