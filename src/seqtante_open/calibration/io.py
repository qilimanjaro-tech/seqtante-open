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
"""Bridge between the calibration-file model and qililab's ``Calibration`` (ADR-0001).

Operating points and experiment sections live in ``Calibration.parameters``; pulses are named
waveforms in ``Calibration.waveforms``, referenced by name from a section (ADR-0008). The calibration
file as a whole is a serialized ``Calibration`` (``serialize_to`` / ``deserialize_from``), the same
object the calibration nodes already read and write.
"""

from qililab.qprogram.calibration import Calibration
from qililab.utils.serialization import deserialize_from, serialize_to

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.calibration.model import CalibrationFile, ReadoutLevels

#: Keys under ``Calibration.parameters`` that hold the calibration-file content.
_CONTENT_KEYS = ("operating_points_version", "operating_points", "experiments")


def read_calibration_file(calibration: Calibration) -> CalibrationFile:
    """Parse and validate the operating points and sections from a ``Calibration``.

    Raises:
        ValueError: if the ``Calibration`` carries no calibration-file content.
        pydantic.ValidationError: if the content does not match the model.
    """
    if "operating_points_version" not in calibration.parameters:
        raise ValueError("no calibration-file content in Calibration.parameters (missing 'operating_points_version')")
    data = {key: calibration.parameters[key] for key in _CONTENT_KEYS if key in calibration.parameters}
    return CalibrationFile.model_validate(data)


def write_calibration_file(calibration: Calibration, file: CalibrationFile) -> None:
    """Store the model into ``calibration.parameters`` in place, leaving other parameters untouched.

    Uses the wire aliases (``if``) and drops absent optionals. The caller persists the ``Calibration``
    with :func:`save_calibration`.
    """
    # mode="json" renders Role enum keys and the like as plain scalars, so the Calibration
    # serializes to clean YAML.
    dumped = file.model_dump(by_alias=True, exclude_none=True, mode="json")
    for key in _CONTENT_KEYS:
        calibration.parameters[key] = dumped[key]


def load_calibration(path: str) -> Calibration:
    """Deserialize a ``Calibration`` from a calibration-file path."""
    return deserialize_from(path, Calibration)


def save_calibration(calibration: Calibration, path: str) -> None:
    """Serialize a ``Calibration`` to a calibration-file path."""
    serialize_to(calibration, path)


def check_section_pulses_exist(file: CalibrationFile, calibration: Calibration) -> None:
    """Verify every pulse a section names exists on its resolved bus in ``Calibration.waveforms``.

    This is the existence half of pulse validation that the pure model cannot do (ADR-0008).

    Raises:
        ValueError: naming the first section whose pulse is missing from its bus.
    """
    for experiment, by_target in file.experiments.items():
        for target, section in by_target.items():
            for role, pulse_name in section.pulses.items():
                bus = bus_alias(target, role)
                if not calibration.has_waveform(bus, pulse_name):
                    raise ValueError(
                        f"experiment '{experiment}' target '{target}': pulse '{pulse_name}' for role "
                        f"'{role}' is not defined on bus '{bus}' in Calibration.waveforms"
                    )


def readout_levels_for(file: CalibrationFile, experiment: str, target: str) -> ReadoutLevels | None:
    """The readout levels to use for a section, or ``None`` if its operating point has none.

    Readout levels are tagged with the readout pulse they were taken with. A section that plays a
    different readout pulse must not use them (ADR-0001, D8).

    Raises:
        ValueError: if the section plays a readout pulse other than the one the levels were taken with.
    """
    section = file.experiments[experiment][target]
    levels = file.operating_points[target][section.operating_point].readout.levels
    if levels is None:
        return None
    played = section.pulses.get(Role.readout)
    if played is not None and played != levels.pulse:
        raise ValueError(
            f"experiment '{experiment}' target '{target}': readout levels at operating point "
            f"'{section.operating_point}' were taken with pulse '{levels.pulse}', but the section plays "
            f"'{played}'; refusing to use them (ADR-0001, D8)"
        )
    return levels
