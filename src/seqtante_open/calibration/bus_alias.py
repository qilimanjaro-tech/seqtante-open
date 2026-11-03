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
- flux is axis-first: ``flux_<axis>_<target>`` (``flux_z_q14``, ``flux_z_c2_3``),
  following qililab's flux bus naming, so crosstalk calibration, the bus sorter and
  the fit-manager target parsing keep working.

Target names are validated and canonicalised through :mod:`seqtante_open.targets` (``q<n>``,
``c<n>_<m>`` with ``n < m``), so a mis-ordered coupler like ``c3_2`` is rejected rather than silently
fixed — callers order couplers with :meth:`~seqtante_open.targets.CouplerTarget.between`. Nothing else
in the codebase should spell a bus name; ``experiments.utils.flux_buses`` builds on this module.
"""

from enum import StrEnum

from seqtante_open.targets import ExperimentTarget, Target

#: Flux-line axes, ordered as they appear last in a flux bus alias.
AXES = ("x", "y", "z")

_AXES = frozenset(AXES)


class Role(StrEnum):
    """A control line, and equivalently the role a pulse plays on it (ADR-0008)."""

    drive = "drive"
    readout = "readout"
    flux = "flux"


def bus_alias(target: str | Target, role: Role, axis: str | None = None) -> str:
    """The bus alias driving ``target``'s ``role`` line.

    Args:
        target: a qubit (``q14``) or coupler (``c2_3``) target, as a name or an
            :class:`~seqtante_open.targets.ExperimentTarget`. Validated and canonicalised through
            :mod:`seqtante_open.targets`, so a mis-ordered coupler (``c3_2``) is rejected.
        role: the control line.
        axis: the flux axis (``x``/``y``/``z``); only for :attr:`Role.flux`.

    Returns:
        The bus alias, e.g. ``drive_q14``, ``readout_q14``, ``flux_z_q14``.

    Raises:
        ValueError: if ``target`` is not a valid target, if ``axis`` is given for a non-flux line, or
            if ``axis`` is not a known axis.
    """
    target_name = str(ExperimentTarget.from_str(str(target)))
    if role is Role.flux:
        if axis is None:
            return f"flux_{target_name}"
        if axis not in _AXES:
            raise ValueError(f"unknown flux axis {axis!r}: expected one of {AXES}")
        return f"flux_{axis}_{target_name}"
    if axis is not None:
        raise ValueError(f"{role} buses take no axis (got {axis!r})")
    return f"{role.value}_{target_name}"


def parse_bus_alias(bus: str) -> tuple[str, Role, str | None]:
    """Split a bus alias into ``(target, role, axis)``.

    The inverse of :func:`bus_alias`. For flux, the axis (when present) follows ``flux_`` and the
    target is the remainder (ADR-0008).

    Raises:
        ValueError: if ``bus`` is not a ``<line>_...`` alias with a known line and a valid target.
    """
    line, separator, rest = bus.partition("_")
    if not separator:
        raise ValueError(f"not a bus alias: {bus!r}")
    role = Role(line)
    if role is Role.flux:
        head, _, remainder = rest.partition("_")
        target, axis = (remainder, head) if remainder and head in _AXES else (rest, None)
    else:
        target, axis = rest, None
    # validate the target, so this is a true inverse of bus_alias
    ExperimentTarget.from_str(target)
    return target, role, axis
