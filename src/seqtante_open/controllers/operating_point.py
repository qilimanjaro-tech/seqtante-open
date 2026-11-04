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

"""Place a target at its operating point before an experiment runs (ADR-0001, ADR-0006).

A node runs a target at one operating point named by the target's experiment section (ADR-0001).
:func:`apply_operating_point` sets the target's own flux and the flux of every neighbour its point
parks, each through a :class:`~seqtante_open.controllers.parameter_restore.ParameterRestore` handle
so the run undoes them. The stored frequencies (drive/readout LO and IF) are played by the
experiment itself, not set here.

:func:`park_lo_outside_sweep` moves an LO clear of a frequency sweep's band; its policy is pinned
in the lifecycle PR (task 5c), where it is wired into the sweep nodes and set on hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qililab.typings.enums import Parameter

from seqtante_open.calibration.bus_alias import Role, bus_alias

if TYPE_CHECKING:
    from seqtante_open.calibration.model import CalibrationFile, Flux, OperatingPoint
    from seqtante_open.controllers.parameter_restore import ParameterRestore
    from seqtante_open.targets import Target


def operating_point_for(file: CalibrationFile, experiment: str, target: Target | str) -> OperatingPoint:
    """The operating point ``experiment`` runs ``target`` at, resolved through its section (ADR-0001).

    Raises:
        KeyError: if the experiment has no section for the target, or the named point does not exist.
    """
    section = file.experiments[experiment][str(target)]
    return file.operating_points[str(target)][section.operating_point]


def apply_operating_point(
    file: CalibrationFile,
    experiment: str,
    target: Target | str,
    restore: ParameterRestore,
) -> None:
    """Set ``target`` and its parked neighbours to the fluxes of its operating point, through ``restore``.

    Sets each flux bus's ``FLUX`` parameter to the stored bias (ADR-0011). The ``restore`` handle
    records the prior values, so the run returns every bus to where it was. The ``x`` bias is set
    only where the point stores one (z-only loops omit it).
    """
    point = operating_point_for(file, experiment, target)
    _apply_flux(restore, target, point.flux)
    for neighbour, flux in point.parked_flux.items():
        _apply_flux(restore, neighbour, flux)


def _apply_flux(restore: ParameterRestore, target: Target | str, flux: Flux) -> None:
    restore.set(bus_alias(target, Role.flux, "z"), Parameter.FLUX, flux.z)
    if flux.x is not None:
        restore.set(bus_alias(target, Role.flux, "x"), Parameter.FLUX, flux.x)


def park_lo_outside_sweep(
    restore: ParameterRestore,
    *,
    bus: str,
    lo: float,
    if_band: tuple[float, float],
) -> None:
    """Move ``bus``'s LO clear of a frequency sweep's band, through ``restore`` (task 5c).

    A sweep covers the absolute IF band ``if_band`` (low, high) and the measured tone is ``LO + IF``,
    so with the LO at the band's centre it sits inside the measurement window. This moves the LO so
    the whole band lies to one side of it, through ``restore`` (the stored LO value is untouched),
    and the restore handle puts it back afterwards.

    The exact shift -- which side of the band, and the margin -- is pinned in the lifecycle PR
    (task 5c), where this is wired into the sweep nodes and set on hardware.

    Raises:
        NotImplementedError: always, until the policy is pinned in task 5c.
    """
    raise NotImplementedError("the LO-parking policy is pinned when this is wired into the sweep nodes (task 5c)")
