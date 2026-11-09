"""Tests for the crosstalk session: the one swap point for qililab's crosstalk API (QHC-1735)."""

import pytest
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.controllers.crosstalk_session import crosstalk_session

BUSES = ("flux_z_q1", "flux_z_q2")


def _crosstalk() -> CrosstalkMatrix:
    return CrosstalkMatrix.from_buses({b: {bb: (1.0 if b == bb else 0.0) for bb in BUSES} for b in BUSES})


def _calibration() -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = _crosstalk()
    return calibration


class FakePlatform:
    """Records the crosstalk/flux/bias calls in order; it drives no hardware."""

    def __init__(self):
        self.calls = []  # (name, arg)

    def set_crosstalk(self, *, crosstalk):
        self.calls.append(("set_crosstalk", crosstalk))

    def set_flux_to_zero(self):
        self.calls.append(("set_flux_to_zero", None))

    def set_bias_to_zero(self):
        self.calls.append(("set_bias_to_zero", None))


def test_installs_matrix_zeros_flux_then_zeros_bias():
    platform = FakePlatform()
    with crosstalk_session(platform, _calibration()):
        assert [name for name, _ in platform.calls] == ["set_crosstalk", "set_flux_to_zero"]
    assert [name for name, _ in platform.calls] == ["set_crosstalk", "set_flux_to_zero", "set_bias_to_zero"]


def test_installs_the_calibrations_matrix():
    platform = FakePlatform()
    calibration = _calibration()
    with crosstalk_session(platform, calibration):
        pass
    installed = next(arg for name, arg in platform.calls if name == "set_crosstalk")
    assert installed is calibration.crosstalk_matrix


def test_zeros_bias_even_when_body_raises():
    platform = FakePlatform()
    calibration = _calibration()

    def run_and_raise():
        with crosstalk_session(platform, calibration):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_and_raise()
    assert ("set_bias_to_zero", None) in platform.calls


def test_zeros_bias_even_when_zeroing_flux_raises():
    platform = FakePlatform()

    def failing_set_flux_to_zero():
        platform.calls.append(("set_flux_to_zero", None))
        raise RuntimeError("flux zeroing failed partway")

    platform.set_flux_to_zero = failing_set_flux_to_zero

    with pytest.raises(RuntimeError):
        with crosstalk_session(platform, _calibration()):
            pass
    assert ("set_bias_to_zero", None) in platform.calls


def test_requires_a_crosstalk_matrix():
    platform = FakePlatform()
    empty = Calibration()
    with pytest.raises(ValueError, match="CrosstalkMatrix"):
        with crosstalk_session(platform, empty):
            pass
    assert platform.calls == []


def test_add_flux_offset_accumulates_on_the_matrix():
    platform = FakePlatform()
    calibration = _calibration()
    with crosstalk_session(platform, calibration) as session:
        session.add_flux_offset("flux_z_q1", 0.01)
        session.add_flux_offset("flux_z_q1", 0.02)
    assert calibration.crosstalk_matrix.flux_offsets["flux_z_q1"] == pytest.approx(0.03)
    assert calibration.crosstalk_matrix.flux_offsets["flux_z_q2"] == pytest.approx(0.0)
