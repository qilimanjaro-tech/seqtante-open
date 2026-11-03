"""Tests for seeding a fresh calibration file from manual calibration numbers (ADR-0001)."""

import pytest
from qililab import Square

from seqtante_open.calibration.io import read_calibration_file
from seqtante_open.calibration.model import CalibrationFile
from seqtante_open.calibration.seed import seed_calibration


def _waveforms() -> dict:
    """The named pulses the rabi section references."""
    return {
        "readout_q14": {"readout-reset": Square(amplitude=1.0, duration=1000)},
        "drive_q14": {"pi-reset": Square(amplitude=0.5, duration=40)},
    }


def test_seed_assembles_points_and_pulses(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)

    calibration = seed_calibration(model, _waveforms())

    # Operating points and sections are present (and pulses exist, since the default check passes).
    assert read_calibration_file(calibration) == model
    assert calibration.has_waveform("readout_q14", "readout-reset")
    assert calibration.has_waveform("drive_q14", "pi-reset")


def test_seed_rejects_a_pulse_not_provided(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)
    waveforms = {"readout_q14": {"readout-reset": Square(amplitude=1.0, duration=1000)}}  # 'pi-reset' missing

    with pytest.raises(ValueError, match="not defined on bus 'drive_q14'"):
        seed_calibration(model, waveforms)
