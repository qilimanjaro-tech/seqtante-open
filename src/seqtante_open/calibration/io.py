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


def read_calibration_file(calibration: Calibration, *, check_pulses: bool = True) -> CalibrationFile:
    """Parse and validate the operating points and sections from a ``Calibration``.

    When ``check_pulses`` (the default), every pulse a section names must also exist on its bus in
    ``Calibration.waveforms`` — the load-time guarantee of ADR-0008. Pass ``check_pulses=False`` to read
    a file whose waveforms are not populated yet (e.g. while seeding, where points are written first).

    Raises:
        ValueError: if the ``Calibration`` carries no content, or (when checking) a named pulse is missing.
        pydantic.ValidationError: if the content does not match the model.
    """
    if "operating_points_version" not in calibration.parameters:
        raise ValueError("no calibration-file content in Calibration.parameters (missing 'operating_points_version')")
    data = {key: calibration.parameters[key] for key in _CONTENT_KEYS if key in calibration.parameters}
    file = CalibrationFile.model_validate(data)
    if check_pulses:
        check_section_pulses_exist(file, calibration)
    return file


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
                if role is Role.flux:
                    # A flux bus carries an axis (flux_z_q14); the role->bus resolution has none, so a
                    # named flux pulse cannot be placed on a real bus. None is needed for December
                    # (ADR-0008 open question), so fail loudly rather than check a non-existent bus.
                    raise ValueError(
                        f"experiment '{experiment}' target '{target}': flux pulses are not supported "
                        f"(a flux pulse needs an axis to resolve its bus; ADR-0008 open question)"
                    )
                bus = bus_alias(target, role)
                if not calibration.has_waveform(bus, pulse_name):
                    raise ValueError(
                        f"experiment '{experiment}' target '{target}': pulse '{pulse_name}' for role "
                        f"'{role}' is not defined on bus '{bus}' in Calibration.waveforms"
                    )


def readout_levels_for(file: CalibrationFile, experiment: str, target: str, readout_pulse: str) -> ReadoutLevels | None:
    """The readout levels for ``target`` at its section's operating point, for a consumer playing ``readout_pulse``.

    Readout levels are tagged with the readout pulse they were taken with; a consumer playing a
    different readout pulse must not use them (ADR-0001, D8). Because ``pulses`` is sparse (a section
    lists only the roles it overrides, ADR-0008), the section alone may not name the readout pulse —
    so the caller passes the one it will actually play, resolved from the section's override or the
    experiment's default.

    Returns:
        The levels, or ``None`` if the operating point stores none.

    Raises:
        ValueError: if there is no section for ``experiment``/``target``, or if ``readout_pulse``
            differs from the pulse the levels were taken with.
    """
    try:
        section = file.experiments[experiment][target]
    except KeyError as error:
        raise ValueError(f"no section for experiment '{experiment}' target '{target}'") from error
    levels = file.operating_points[target][section.operating_point].readout.levels
    if levels is None:
        return None
    if readout_pulse != levels.pulse:
        raise ValueError(
            f"experiment '{experiment}' target '{target}': readout levels at operating point "
            f"'{section.operating_point}' were taken with pulse '{levels.pulse}', but the consumer plays "
            f"'{readout_pulse}'; refusing to use them (ADR-0001, D8)"
        )
    return levels
