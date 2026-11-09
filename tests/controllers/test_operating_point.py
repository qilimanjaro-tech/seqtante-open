"""Tests for the operating-point helper (ADR-0001, ADR-0006)."""

import pytest
from qililab.typings.enums import Parameter

from seqtante_open.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    ExperimentSection,
    Flux,
    OperatingPoint,
    ReadoutFreqs,
)
from seqtante_open.controllers.operating_point import (
    apply_operating_point,
    operating_point_for,
    park_lo_outside_sweep,
)
from seqtante_open.controllers.parameter_restore import ParameterRestore
from seqtante_open.targets import ExperimentTarget


class FakePlatform:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.sets = []

    def get_parameter(self, *, alias, parameter):
        return self.values.get((alias, parameter), 0.0)

    def set_parameter(self, *, alias, parameter, value):
        self.values[alias, parameter] = value
        self.sets.append((alias, parameter, value))


def _freqs() -> dict:
    return {
        "drive": DriveFreqs(lo=4e9, if_=1e8),
        "readout": ReadoutFreqs(lo=7e9, if_=1e8, attenuation=10.0),
    }


def _file() -> CalibrationFile:
    point = OperatingPoint(flux=Flux(z=0.1, x=0.02), parked_flux={"q2": Flux(z=0.3)}, **_freqs())
    return CalibrationFile(
        operating_points_version=1,
        operating_points={"q1": {"sweet": point}},
        experiments={"single_tone": {"q1": ExperimentSection(operating_point="sweet")}},
    )


def test_operating_point_for_resolves_the_section():
    point = operating_point_for(_file(), "single_tone", "q1")
    assert point.flux.z == pytest.approx(0.1)
    assert point.flux.x == pytest.approx(0.02)


def test_sets_target_and_parked_fluxes():
    platform = FakePlatform()
    apply_operating_point(_file(), "single_tone", ExperimentTarget.from_str("q1"), ParameterRestore(platform))
    assert platform.values["flux_z_q1", Parameter.FLUX] == pytest.approx(0.1)
    assert platform.values["flux_x_q1", Parameter.FLUX] == pytest.approx(0.02)
    assert platform.values["flux_z_q2", Parameter.FLUX] == pytest.approx(0.3)


def test_omits_x_flux_where_the_point_has_none():
    platform = FakePlatform()
    apply_operating_point(_file(), "single_tone", ExperimentTarget.from_str("q1"), ParameterRestore(platform))
    # q2 is parked with z only, so no x bus is touched.
    assert all(alias != "flux_x_q2" for alias, _, _ in platform.sets)


def test_fluxes_are_set_through_restore_and_can_be_undone():
    platform = FakePlatform({("flux_z_q1", Parameter.FLUX): 0.9})
    restore = ParameterRestore(platform)
    apply_operating_point(_file(), "single_tone", ExperimentTarget.from_str("q1"), restore)
    assert platform.values["flux_z_q1", Parameter.FLUX] == pytest.approx(0.1)
    restore.restore()
    assert platform.values["flux_z_q1", Parameter.FLUX] == pytest.approx(0.9)


def test_park_lo_outside_sweep_is_scaffolded_until_the_sweep_nodes():
    restore = ParameterRestore(FakePlatform())
    with pytest.raises(NotImplementedError, match="sweep nodes"):
        park_lo_outside_sweep(restore, bus="readout_q1", lo=7e9, if_band=(-1.5e6, 1.5e6))
