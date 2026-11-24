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

"""Set up a drive bus's source to play an IQ drive, for any experiment.

Which parameters arm a drive depends only on the instrument on the bus, not on the node, the experiment
or an operating point, so any node or experiment that drives a qubit sets it up here:

- Rohde & Schwarz: ``POWER`` in dBm, IQ modulation on (else it plays a bare carrier instead of the IQ
  drive) and the RF output on.
- QCM-RF: the sequencer ``GAIN`` in [-1, 1], and hardware modulation on.
- Anything else: warns, and sets nothing.

Every write goes through a :class:`~seqtante_open.controllers.parameter_restore.ParameterRestore`, so the
caller's restore scope turns the source back off and puts the power or gain back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from qililab import Parameter

if TYPE_CHECKING:
    from qililab import Platform
    from qililab.typings import ParameterValue

    from seqtante_open.controllers.parameter_restore import ParameterRestore

#: Warned when a drive bus has no instrument whose power or gain this module knows how to set.
NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."


def drive_source_writes(
    platform: Platform, drive_bus: str, *, drive_power: float | None = None, drive_gain: float | None = None
) -> list[tuple[Parameter, ParameterValue]]:
    """The ``(parameter, value)`` writes that arm ``drive_bus``'s source, in the order to make them.

    Writes nothing itself, so a caller can check the drive level before touching the hardware.

    Args:
        platform (Platform): The platform the bus belongs to.
        drive_bus (str): The drive bus alias.
        drive_power (float | None): The Rohde & Schwarz power in dBm; required for that instrument.
        drive_gain (float | None): The QCM-RF sequencer gain in [-1, 1]; required for that instrument.

    Raises:
        ValueError: if the bus's instrument needs ``drive_power``/``drive_gain`` and it was not given.
    """
    instruments = [instrument.name.name for instrument in platform.get_element(drive_bus).instruments]
    if "ROHDE_SCHWARZ" in instruments:
        if drive_power is None:
            raise ValueError(f"{drive_bus}: a Rohde & Schwarz drive source needs drive_power (dBm)")
        return [(Parameter.IQ_MODULATION, True), (Parameter.POWER, drive_power), (Parameter.RF_ON, True)]
    if "QCMRF" in instruments:
        if drive_gain is None:
            raise ValueError(f"{drive_bus}: a QCM-RF drive needs drive_gain (sequencer gain in [-1, 1])")
        return [(Parameter.HARDWARE_MODULATION, True), (Parameter.GAIN, drive_gain)]
    warn(NO_DRIVE_INSTRUMENT_MSG, stacklevel=2)
    return []


def set_drive_source(
    restore: ParameterRestore,
    platform: Platform,
    drive_bus: str,
    *,
    drive_power: float | None = None,
    drive_gain: float | None = None,
) -> None:
    """Arm ``drive_bus``'s source through ``restore`` (see :func:`drive_source_writes` for the arguments).

    Raises:
        ValueError: if the bus's instrument needs ``drive_power``/``drive_gain`` and it was not given.
    """
    for parameter, value in drive_source_writes(platform, drive_bus, drive_power=drive_power, drive_gain=drive_gain):
        restore.set(drive_bus, parameter, value)
