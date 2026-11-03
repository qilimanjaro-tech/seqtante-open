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
"""Seed a fresh calibration file from the experimentalists' manual calibration (ADR-0001).

The initial operating points come from a manual calibration of the system by the experimentalists;
this assembles their numbers into a qililab ``Calibration`` — operating points and sections in
``Calibration.parameters``, named pulses in ``Calibration.waveforms`` — ready to serialize. There is
no migration from the old QHC-1608 format.
"""

from collections.abc import Mapping

from qililab.qprogram.calibration import Calibration
from qililab.waveforms import IQWaveform, Waveform

from seqtante_open.calibration.io import check_section_pulses_exist, write_calibration_file
from seqtante_open.calibration.model import CalibrationFile


def seed_calibration(
    file: CalibrationFile,
    waveforms: Mapping[str, Mapping[str, Waveform | IQWaveform]],
) -> Calibration:
    """Build a fresh ``Calibration`` from manual calibration numbers.

    Args:
        file: the operating points and experiment sections (the manual numbers), already validated.
        waveforms: the named pulses, keyed by bus then pulse name — the pulses the sections reference.

    Returns:
        A ``Calibration`` with the operating points and sections in ``parameters`` and the pulses in
        ``waveforms``, ready to persist with :func:`seqtante_open.calibration.io.save_calibration`.

    Raises:
        ValueError: if a pulse named by a section is not among ``waveforms`` on its resolved bus.
    """
    calibration = Calibration()
    write_calibration_file(calibration, file)
    for bus, named_pulses in waveforms.items():
        for name, waveform in named_pulses.items():
            calibration.add_waveform(bus, name, waveform)
    check_section_pulses_exist(file, calibration)
    return calibration
