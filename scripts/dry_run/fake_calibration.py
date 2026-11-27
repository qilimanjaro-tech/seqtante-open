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

"""A made-up calibration file for a dry run: every operating point and pulse the tree's steps use, per target."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qili_qprograms.bus_alias import Role, bus_alias
from qili_qprograms.calibration import pulse_name
from qili_qprograms.calibration.io import write_calibration_file
from qili_qprograms.calibration.model import CalibrationFile, DriveFreqs, Flux, OperatingPoint, ReadoutFreqs
from qililab import IQDrag, Square
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.utils.serialization import serialize_to

if TYPE_CHECKING:
    from seqtante_open.controllers.calibration_tree_parser import Step

DRIVE_LO, DRIVE_IF = 4.5e9, 100e6
READOUT_LO, READOUT_IF, READOUT_ATTENUATION = 6.3e9, 50e6, 10.0
FLUX_AXES = ("z", "x")


def operating_points(steps: list[Step]) -> dict[str, list[str]]:
    """The operating points each target's steps run at, by target, in the order the tree first names them."""
    points: dict[str, list[str]] = {}
    for step in steps:
        for target in step.targets:
            point = step.parameters[target].get("operating_point")
            if point is not None and point not in points.setdefault(str(target), []):
                points[str(target)].append(point)
    return points


def fake_calibration(steps: list[Step], platform, path: str) -> str:
    """Write a calibration file for every target and operating point the steps use, and return its path.

    The tree's first operating point sits at zero flux and each next one 0.1 Φ0 higher on z. Every point
    has the same drive and readout frequencies, a 2000 ns square readout and a 40 ns DRAG π pulse.
    The crosstalk matrix is the identity over the targets' flux buses that ``platform`` has.
    """
    points_by_target = operating_points(steps)
    order = list(dict.fromkeys(point for points in points_by_target.values() for point in points))
    file = CalibrationFile(
        operating_points_version=1,
        operating_points={
            target: {
                point: OperatingPoint(
                    flux=Flux(z=0.1 * order.index(point)),
                    drive=DriveFreqs(lo=DRIVE_LO, if_=DRIVE_IF),
                    readout=ReadoutFreqs(lo=READOUT_LO, if_=READOUT_IF, attenuation=READOUT_ATTENUATION),
                )
                for point in points
            }
            for target, points in points_by_target.items()
        },
    )
    calibration = Calibration()
    write_calibration_file(calibration, file)
    for target, points in points_by_target.items():
        for point in points:
            readout = Square(amplitude=0.1, duration=2000)
            calibration.add_waveform(bus_alias(target, Role.readout), pulse_name("readout", point), readout)
            pi = IQDrag(amplitude=0.5, duration=40, num_sigmas=4, drag_coefficient=0.0)
            calibration.add_waveform(bus_alias(target, Role.drive), pulse_name("pi", point), pi)
    platform_buses = {bus.alias for bus in platform.buses}
    flux_buses = [
        alias
        for target in points_by_target
        for axis in FLUX_AXES
        if (alias := bus_alias(target, Role.flux, axis)) in platform_buses
    ]
    calibration.crosstalk_matrix = CrosstalkMatrix.from_buses(
        {bus: {other: 1.0 if bus == other else 0.0 for other in flux_buses} for bus in flux_buses}
    )
    serialize_to(calibration, path)
    return path
