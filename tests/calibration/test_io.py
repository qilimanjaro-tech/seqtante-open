"""Tests for the qililab ``Calibration`` bridge (ADR-0001, ADR-0008, D8)."""

from pathlib import Path

import pytest
from qililab import Square
from qililab.qprogram.calibration import Calibration

from seqtante_open.calibration.io import (
    check_section_pulses_exist,
    load_calibration,
    read_calibration_file,
    readout_levels_for,
    save_calibration,
    write_calibration_file,
)
from seqtante_open.calibration.model import CalibrationFile


def _waveforms_for(calibration: Calibration) -> None:
    """Register the readout and drive pulses the rabi section names."""
    calibration.add_waveform("readout_q14", "readout-reset", Square(amplitude=1.0, duration=1000))
    calibration.add_waveform("drive_q14", "pi-reset", Square(amplitude=0.5, duration=40))


def test_round_trips_through_a_calibration(valid_file: dict, tmp_path: Path) -> None:
    model = CalibrationFile.model_validate(valid_file)
    calibration = Calibration()

    write_calibration_file(calibration, model)
    path = str(tmp_path / "calibration.yml")
    save_calibration(calibration, path)
    reloaded = load_calibration(path)

    assert read_calibration_file(reloaded) == model


def test_write_preserves_unrelated_parameters(valid_file: dict) -> None:
    calibration = Calibration()
    calibration.parameters["something_else"] = {"keep": 1}

    write_calibration_file(calibration, CalibrationFile.model_validate(valid_file))

    assert calibration.parameters["something_else"] == {"keep": 1}


def test_read_without_content_errors() -> None:
    with pytest.raises(ValueError, match="no calibration-file content"):
        read_calibration_file(Calibration())


def test_check_section_pulses_exist_passes_when_present(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)
    calibration = Calibration()
    _waveforms_for(calibration)

    check_section_pulses_exist(model, calibration)  # no raise


def test_check_section_pulses_exist_errors_when_missing(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)
    calibration = Calibration()
    calibration.add_waveform("readout_q14", "readout-reset", Square(amplitude=1.0, duration=1000))
    # drive pulse 'pi-reset' on 'drive_q14' is deliberately absent

    with pytest.raises(ValueError, match="not defined on bus 'drive_q14'"):
        check_section_pulses_exist(model, calibration)


def test_readout_levels_for_returns_levels_when_pulse_matches(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)
    levels = readout_levels_for(model, "rabi", "q14")
    assert levels is not None
    assert levels.pulse == "readout-reset"


def test_readout_levels_for_refuses_on_pulse_mismatch(valid_file: dict) -> None:
    valid_file["experiments"]["rabi"]["q14"]["pulses"]["readout"] = "readout-other"
    model = CalibrationFile.model_validate(valid_file)
    with pytest.raises(ValueError, match="D8"):
        readout_levels_for(model, "rabi", "q14")


def test_readout_levels_for_none_when_point_has_no_levels(valid_file: dict) -> None:
    valid_file["experiments"]["t1"] = {"q14": {"operating_point": "sweet_spot", "pulses": {}}}
    model = CalibrationFile.model_validate(valid_file)
    assert readout_levels_for(model, "t1", "q14") is None
