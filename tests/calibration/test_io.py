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

    # check_pulses=False: this test exercises the structure round-trip, not waveform presence.
    assert read_calibration_file(reloaded, check_pulses=False) == model


def test_read_checks_named_pulses_exist(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)
    calibration = Calibration()
    write_calibration_file(calibration, model)

    # No waveforms registered yet: the load-time pulse check fails (ADR-0008).
    with pytest.raises(ValueError, match="not defined on bus"):
        read_calibration_file(calibration)

    _waveforms_for(calibration)
    assert read_calibration_file(calibration) == model  # passes once the waveforms exist


def test_write_preserves_unrelated_parameters(valid_file: dict) -> None:
    calibration = Calibration()
    calibration.parameters["something_else"] = {"keep": 1}

    write_calibration_file(calibration, CalibrationFile.model_validate(valid_file))

    assert calibration.parameters["something_else"] == {"keep": 1}


def test_read_without_content_errors() -> None:
    empty = Calibration()
    with pytest.raises(ValueError, match="no calibration-file content"):
        read_calibration_file(empty)


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
    levels = readout_levels_for(model, "rabi", "q14", "readout-reset")
    assert levels is not None
    assert levels.pulse == "readout-reset"


def test_readout_levels_for_refuses_on_pulse_mismatch(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)
    # The consumer plays a readout pulse other than the one the levels were taken with (D8),
    # even though the sparse section does not override readout.
    with pytest.raises(ValueError, match="D8"):
        readout_levels_for(model, "rabi", "q14", "readout-other")


def test_readout_levels_for_none_when_point_has_no_levels(valid_file: dict) -> None:
    valid_file["experiments"]["t1"] = {"q14": {"operating_point": "sweet_spot", "pulses": {}}}
    model = CalibrationFile.model_validate(valid_file)
    assert readout_levels_for(model, "t1", "q14", "readout-sweet_spot") is None


def test_readout_levels_for_errors_without_section(valid_file: dict) -> None:
    # No section for this experiment/target: raise a clear ValueError, not a bare KeyError.
    model = CalibrationFile.model_validate(valid_file)
    with pytest.raises(ValueError, match="no section for experiment 't1' target 'q14'"):
        readout_levels_for(model, "t1", "q14", "readout-reset")


def test_check_section_pulses_exist_rejects_flux_pulse(valid_file: dict) -> None:
    # A flux bus carries an axis, which role->bus resolution has none of, so a named flux pulse
    # can't be placed on a real bus: reject it loudly (ADR-0008 open question).
    valid_file["experiments"]["rabi"]["q14"]["pulses"]["flux"] = "flux-reset"
    model = CalibrationFile.model_validate(valid_file)
    calibration = Calibration()
    _waveforms_for(calibration)

    with pytest.raises(ValueError, match="flux pulses are not supported"):
        check_section_pulses_exist(model, calibration)
