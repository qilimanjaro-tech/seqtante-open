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

"""A simulated QDAC-II that records every SCPI command qililab sends to it.

The real qcodes driver runs on pyvisa-sim, so the commands are the ones a real QDAC-II would receive.
The upstream simulation file does not answer the marker queries qililab uses to find free internal
triggers, so this driver answers them from the marker registers written so far.
"""

from __future__ import annotations

import re

import qcodes_contrib_drivers.sims as sims
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2
from qililab.instruments.qdevil.qdevil_qdac2 import QDevilQDac2

SIM_VISALIB = sims.__file__.replace("__init__.py", "QDAC2.yaml@sim")

_MARKER_WRITE = re.compile(r"SOUR(\d+):(\w+):MARK:(\w+)\s+(\d+)", re.IGNORECASE)
_MARKER_QUERY = re.compile(r"SOUR(\d+):(\w+):MARK:(\w+)\?", re.IGNORECASE)


class RecordingQDac2(QDac2):
    """QDac2 on pyvisa-sim, recording SCPI from creation and keeping the marker registers."""

    def __init__(self, name: str, address: str = "GPIB::1::INSTR") -> None:
        super().__init__(name, address=address, visalib=SIM_VISALIB)
        self._no_binary_values = True
        self.markers: dict[tuple[int, str, str], int] = {}
        self.start_recording_scpi()

    def write(self, cmd: str) -> None:
        for part in cmd.split(";"):
            match = _MARKER_WRITE.fullmatch(part.strip().lstrip(":"))
            if match:
                channel, generator, location, trigger = match.groups()
                self.markers[int(channel), generator.upper(), location.upper()] = int(trigger)
        super().write(cmd)

    def ask(self, cmd: str) -> str:
        parts = [part.strip().lstrip(":") for part in cmd.split(";")]
        matches = [_MARKER_QUERY.fullmatch(part) for part in parts]
        if not all(matches):
            return super().ask(cmd)
        if self._record_commands:
            self._scpi_sent.append(cmd)
        return ",".join(
            str(self.markers.get((int(channel), generator.upper(), location.upper()), 0))
            for channel, generator, location in (match.groups() for match in matches if match)
        )


def attach_simulated_qdacs(platform) -> dict[str, RecordingQDac2]:
    """Replace the device of every QDAC-II in the platform with a RecordingQDac2, by instrument alias."""
    devices = {}
    for instrument in platform.instruments.elements:
        if isinstance(instrument, QDevilQDac2):
            instrument.device = RecordingQDac2(f"sim_{instrument.alias}")
            devices[instrument.alias] = instrument.device
    return devices
