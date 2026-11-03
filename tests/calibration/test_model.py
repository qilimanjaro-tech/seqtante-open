"""Tests for the calibration-file pydantic model (ADR-0001, ADR-0008, ADR-0010, ADR-0011)."""

import pytest
from pydantic import ValidationError

from seqtante_open.calibration.bus_alias import Role
from seqtante_open.calibration.model import CalibrationFile


def test_parses_a_valid_file(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)

    assert model.operating_points_version == 1
    reset = model.operating_points["q14"]["reset"]
    assert reset.flux.z == pytest.approx(0.615)  # plain measured value (ADR-0011)
    assert reset.drive.if_ == pytest.approx(-100e6)  # `if` read through the alias
    assert reset.readout.levels is not None
    assert reset.readout.levels.pulse == "readout-reset"
    assert set(reset.parked_flux) == {"c13_14", "c14_15"}  # neighbour fluxes parked (ADR-0010)
    assert reset.parked_flux["c13_14"].z == pytest.approx(0.02)
    section = model.experiments["rabi"]["q14"]
    assert section.operating_point == "reset"
    assert section.pulses == {Role.readout: "readout-reset", Role.drive: "pi-reset"}


def test_if_round_trips_as_if_on_the_wire(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)
    dumped = model.model_dump(by_alias=True, exclude_none=True)
    drive = dumped["operating_points"]["q14"]["reset"]["drive"]
    assert "if" in drive
    assert "if_" not in drive


def test_sweet_spot_has_no_readout_levels(valid_file: dict) -> None:
    model = CalibrationFile.model_validate(valid_file)
    assert model.operating_points["q14"]["sweet_spot"].readout.levels is None


def test_flux_x_is_optional(valid_file: dict) -> None:
    del valid_file["operating_points"]["q14"]["sweet_spot"]["flux"]["x"]  # a z-only loop
    model = CalibrationFile.model_validate(valid_file)
    assert model.operating_points["q14"]["sweet_spot"].flux.x is None


def test_missing_required_field_errors(valid_file: dict) -> None:
    del valid_file["operating_points"]["q14"]["reset"]["drive"]["lo"]
    with pytest.raises(ValidationError):
        CalibrationFile.model_validate(valid_file)


def test_unknown_key_is_forbidden(valid_file: dict) -> None:
    valid_file["operating_points"]["q14"]["reset"]["flux"]["w"] = 1.0
    with pytest.raises(ValidationError):
        CalibrationFile.model_validate(valid_file)


def test_unknown_role_rejected(valid_file: dict) -> None:
    valid_file["experiments"]["rabi"]["q14"]["pulses"]["magic"] = "pi-reset"
    with pytest.raises(ValidationError):
        CalibrationFile.model_validate(valid_file)


def test_malformed_section_pulse_name_rejected(valid_file: dict) -> None:
    valid_file["experiments"]["rabi"]["q14"]["pulses"]["readout"] = "Readout"
    with pytest.raises(ValidationError, match="invalid pulse name"):
        CalibrationFile.model_validate(valid_file)


def test_malformed_readout_levels_pulse_rejected(valid_file: dict) -> None:
    valid_file["operating_points"]["q14"]["reset"]["readout"]["levels"]["pulse"] = "pi--reset"
    with pytest.raises(ValidationError, match="invalid pulse name"):
        CalibrationFile.model_validate(valid_file)


def test_section_referencing_unknown_operating_point_rejected(valid_file: dict) -> None:
    valid_file["experiments"]["rabi"]["q14"]["operating_point"] = "nope"
    with pytest.raises(ValidationError, match="is not defined under operating_points"):
        CalibrationFile.model_validate(valid_file)


def test_invalid_operating_point_target_rejected(valid_file: dict) -> None:
    valid_file["operating_points"]["not_a_qubit"] = valid_file["operating_points"]["q14"]
    with pytest.raises(ValidationError, match="Invalid target"):
        CalibrationFile.model_validate(valid_file)


def test_invalid_parked_flux_target_rejected(valid_file: dict) -> None:
    valid_file["operating_points"]["q14"]["reset"]["parked_flux"]["nope"] = {"z": 0.0}
    with pytest.raises(ValidationError, match="Invalid target"):
        CalibrationFile.model_validate(valid_file)
