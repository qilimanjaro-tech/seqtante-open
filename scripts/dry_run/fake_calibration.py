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

"""A made-up calibration file for a dry run: every operating point and pulse the tree names, for every target."""

from __future__ import annotations

from typing import Any

from qili_qprograms.bus_alias import Role, bus_alias
from qili_qprograms.calibration.io import write_calibration_file
from qili_qprograms.calibration.model import CalibrationFile, DriveFreqs, Flux, OperatingPoint, ReadoutFreqs
from qililab import IQDrag, Square
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.utils.serialization import serialize_to

DRIVE_LO, DRIVE_IF = 4.5e9, 100e6
READOUT_LO, READOUT_IF, READOUT_ATTENUATION = 6.3e9, 50e6, 10.0


def collect(tree: Any, key: str) -> list:
    """Every value stored under ``key`` anywhere in ``tree``, in order, lists flattened."""
    found: list = []
    if isinstance(tree, dict):
        for name, value in tree.items():
            if name == key:
                found += value if isinstance(value, list) else [value]
            else:
                found += collect(value, key)
    elif isinstance(tree, list):
        for item in tree:
            found += collect(item, key)
    return found


def fake_calibration(tree: dict, platform, path: str) -> str:
    """Write a calibration file for every target and operating point ``tree`` names, and return its path.

    The first operating point sits at zero flux and each next one 0.1 Φ0 higher on z. Every point
    has the same drive and readout frequencies, a 2000 ns square readout and a 40 ns DRAG π pulse.
    The crosstalk matrix is the identity over the targets' flux buses in ``platform``.
    """
    targets = list(dict.fromkeys(str(target) for target in collect(tree, "targets")))
    points = list(dict.fromkeys(collect(tree, "operating_point")))
    file = CalibrationFile(
        operating_points_version=1,
        operating_points={
            target: {
                point: OperatingPoint(
                    flux=Flux(z=0.1 * index),
                    drive=DriveFreqs(lo=DRIVE_LO, if_=DRIVE_IF),
                    readout=ReadoutFreqs(lo=READOUT_LO, if_=READOUT_IF, attenuation=READOUT_ATTENUATION),
                )
                for index, point in enumerate(points)
            }
            for target in targets
        },
    )
    calibration = Calibration()
    write_calibration_file(calibration, file)
    for target in targets:
        for point in points:
            calibration.add_waveform(
                bus_alias(target, Role.readout), f"readout-{point}", Square(amplitude=0.1, duration=2000)
            )
            pi = IQDrag(amplitude=0.5, duration=40, num_sigmas=4, drag_coefficient=0.0)
            calibration.add_waveform(bus_alias(target, Role.drive), f"pi-{point}", pi)
    flux_buses = [
        bus.alias
        for bus in platform.buses
        if bus.alias.startswith("flux_") and any(bus.alias.endswith(f"_{target}") for target in targets)
    ]
    calibration.crosstalk_matrix = CrosstalkMatrix.from_buses(
        {bus: {other: 1.0 if bus == other else 0.0 for other in flux_buses} for bus in flux_buses}
    )
    serialize_to(calibration, path)
    return path
