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
"""The one place that maps ``(target, line[, axis])`` to a qililab bus alias (ADR-0008).

Bus keys keep qililab's existing convention:

- drive and readout are ``<line>_<target>`` (``drive_q14``, ``readout_q14``);
- flux is target-first, axis last: ``flux_<target>[_<axis>]`` (``flux_q14_z``,
  ``flux_c2_3_z``), never ``flux_<axis>_<target>``, so crosstalk calibration, the
  bus sorter and the fit-manager target parsing keep working.

Coupler targets are normalised to ascending index order (``c3_2`` -> ``c2_3``), as
the flux helpers already do. Nothing else in the codebase should spell a bus name;
``experiments.utils.flux_buses`` builds on this module.
"""

from enum import StrEnum

#: Flux-line axes, ordered as they appear last in a flux bus alias.
AXES = ("x", "y", "z")

_AXES = frozenset(AXES)


class Role(StrEnum):
    """A control line, and equivalently the role a pulse plays on it (ADR-0008)."""

    drive = "drive"
    readout = "readout"
    flux = "flux"


def bus_alias(target: str, role: Role, axis: str | None = None) -> str:
    """The bus alias driving ``target``'s ``role`` line.

    Args:
        target: a qubit (``q14``) or coupler (``c2_3``) target.
        role: the control line.
        axis: the flux axis (``x``/``y``/``z``); only for :attr:`Role.flux`.

    Returns:
        The bus alias, e.g. ``drive_q14``, ``readout_q14``, ``flux_q14_z``.

    Raises:
        ValueError: if ``axis`` is given for a non-flux line, or is not a known axis.
    """
    if role is Role.flux:
        if axis is None:
            return f"flux_{_normalise_flux_target(target)}"
        if axis not in _AXES:
            raise ValueError(f"unknown flux axis {axis!r}: expected one of {AXES}")
        return f"flux_{_normalise_flux_target(target)}_{axis}"
    if axis is not None:
        raise ValueError(f"{role} buses take no axis (got {axis!r})")
    return f"{role.value}_{target}"


def parse_bus_alias(bus: str) -> tuple[str, Role, str | None]:
    """Split a bus alias into ``(target, role, axis)``.

    The inverse of :func:`bus_alias`. For flux, the target (not the axis) follows
    ``flux_``; the optional axis is the last ``_``-separated token (ADR-0008).

    Raises:
        ValueError: if ``bus`` is not a ``<line>_...`` alias with a known line.
    """
    line, separator, rest = bus.partition("_")
    if not separator:
        raise ValueError(f"not a bus alias: {bus!r}")
    role = Role(line)
    if role is Role.flux:
        target, _, tail = rest.rpartition("_")
        if target and tail in _AXES:
            return target, role, tail
        return rest, role, None
    return rest, role, None


def _normalise_flux_target(target: str) -> str:
    """Order a coupler target's indices ascending (``c3_2`` -> ``c2_3``); pass qubits through."""
    if target.startswith("c") and "_" in target:
        first, _, second = target[1:].partition("_")
        low, high = sorted((int(first), int(second)))
        return f"c{low}_{high}"
    return target
