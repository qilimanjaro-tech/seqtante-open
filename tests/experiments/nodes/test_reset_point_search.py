"""Call-assertion test for the ``reset_point_search`` loop, and a unit test of its flux step (QHC-1807).

The loop test drives the full three-phase run with the two-tone measurement and the Lorentzian fit
scripted (a fake :class:`LorentzianPeakModel`, a stubbed two-tone driver, no ``time.sleep``), so the
search loop — measure, move ``flux.z`` through the restore handle, re-measure, then write the point — is
asserted without hardware, a database or a disk write.
"""

import pytest
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.calibration.io import read_calibration_file, write_calibration_file
from seqtante_open.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    ExperimentSection,
    Flux,
    OperatingPoint,
    ReadoutFreqs,
)
from seqtante_open.controllers import calibration_node as node_module
from seqtante_open.controllers.calibration_node import RunContext
from seqtante_open.experiments.nodes import ResetPointSearchNode
from seqtante_open.experiments.nodes import reset_point_search as reset_module
from seqtante_open.experiments.nodes.reset_point_search import next_flux_z_step
from seqtante_open.outputs import output_controller
from seqtante_open.targets import ExperimentTarget

Q1 = ExperimentTarget.from_str("q1")
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1")

DRIVE_LO = 2.9e9
IF_OUTSIDE = -0.08e9  # + LO = 2.82 GHz, above the 2.79-2.81 GHz window
IF_INSIDE = -0.10e9  # + LO = 2.80 GHz, inside the window
FIRST_ID = 100


class FakePlatform:
    def __init__(self):
        self.values: dict = {}
        self.calls: list = []

    def set_crosstalk(self, *, crosstalk):
        self.calls.append("set_crosstalk")

    def set_flux_to_zero(self):
        self.calls.append("set_flux_to_zero")

    def set_bias_to_zero(self):
        self.calls.append("set_bias_to_zero")

    def get_parameter(self, *, alias, parameter):
        return self.values.get((alias, parameter), 0.0)

    def set_parameter(self, *, alias, parameter, value):
        self.values[alias, parameter] = value
        self.calls.append(f"set:{alias}")


class FakeLorentzian:
    """Scripted stand-in for ``LorentzianPeakModel``: no DB load, fit keyed by measurement id."""

    def __init__(self, measurement_id, target=None, path=None, r2_min=0.8):
        self.id = measurement_id
        self.results = {"quality": "fake"}

    def fit(self):
        return self.results

    def plot(self):
        return None

    def save_fitting(self):
        return None

    @property
    def accepted(self):
        return True

    @property
    def fitted_if(self):
        # The first sweep (id FIRST_ID) finds the qubit outside the window; every later one, inside.
        return IF_OUTSIDE if self.id == FIRST_ID else IF_INSIDE

    @property
    def r_squared(self):
        return 0.95


def _crosstalk() -> CrosstalkMatrix:
    return CrosstalkMatrix.from_buses(
        {b: {bb: (1.0 if b == bb else 0.0) for bb in CROSSTALK_BUSES} for b in CROSSTALK_BUSES}
    )


def _file() -> CalibrationFile:
    return CalibrationFile(
        operating_points_version=1,
        operating_points={
            "q1": {
                "reset": OperatingPoint(
                    flux=Flux(z=0.6),
                    drive=DriveFreqs(lo=DRIVE_LO, if_=-0.08e9),
                    readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=50.0),
                )
            }
        },
        experiments={"reset_point_search": {"q1": ExperimentSection(operating_point="reset")}},
    )


def _calibration() -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = _crosstalk()
    write_calibration_file(calibration, _file())
    return calibration


def _make(calibration: Calibration) -> ResetPointSearchNode:
    context = RunContext(
        platform=FakePlatform(),
        platform_path="runcard.yml",
        calibration=calibration,
        calibration_path="calibration.yml",
    )
    return ResetPointSearchNode(name="reset_flux", idx=0, targets=[Q1], parameters={Q1: {}}, context=context)


def _patch(monkeypatch):
    """No hardware, no disk, no sleep: stub the two-tone driver, the fit and the save."""
    counter = {"next": FIRST_ID}

    def fake_two_tone(**kwargs):
        measurement_id = counter["next"]
        counter["next"] += 1
        return measurement_id

    monkeypatch.setattr(reset_module, "two_tone_pulsed_saturation_experiment", fake_two_tone)
    monkeypatch.setattr(reset_module, "LorentzianPeakModel", FakeLorentzian)
    monkeypatch.setattr(reset_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    # db_manager is read (to pass to the driver) but the fake driver ignores it; just make it available.
    monkeypatch.setattr(output_controller, "_initialized", True)
    monkeypatch.setattr(output_controller, "_db_manager", object())


def test_loop_moves_flux_into_the_window_and_writes_the_point(monkeypatch):
    _patch(monkeypatch)
    calibration = _calibration()
    node = _make(calibration)

    failed = node.run({})

    assert failed == []
    # The loop measured move0 (outside) then move1 (inside): two measurements, both ids returned.
    assert node._search[Q1].measurement_id == FIRST_ID + 1
    # One probe move of -flux_z_probe_step (0.0005) from the stored 0.6.
    written = read_calibration_file(calibration).operating_points["q1"]["reset"]
    assert written.flux.z == pytest.approx(0.6 - 0.0005)
    assert written.drive.if_ == pytest.approx(IF_INSIDE)  # drive IF taken from the final fit
    # The flux move went through the platform (via the restore handle).
    assert "set:flux_z_q1" in node.context.platform.calls


def test_giving_up_after_the_move_limit_fails_the_target(monkeypatch):
    _patch(monkeypatch)

    class AlwaysOutside(FakeLorentzian):
        @property
        def fitted_if(self):
            return IF_OUTSIDE  # never reaches the window

    monkeypatch.setattr(reset_module, "LorentzianPeakModel", AlwaysOutside)
    calibration = _calibration()
    node = _make(calibration)

    failed = node.run({})

    assert failed == ["q1"]  # the move limit is reached and the target fails, not the run
    # The point is left untouched when the search gives up.
    assert read_calibration_file(calibration).operating_points["q1"]["reset"].flux.z == pytest.approx(0.6)


# --- next_flux_z_step: probe, secant, clip, and the no-slope guard ---


def test_first_move_is_a_probe_towards_lower_flux():
    step, slope, required = next_flux_z_step([(0.6, 2.82e9)], target_freq=2.80e9, probe_step=5e-4, max_step=1e-3)
    assert step == pytest.approx(-5e-4)
    assert slope is None and required is None


def test_secant_step_steers_towards_the_target():
    history = [(0.6, 2.82e9), (0.5995, 2.81e9)]
    step, slope, required = next_flux_z_step(history, target_freq=2.80e9, probe_step=5e-4, max_step=1e-3)
    assert slope == pytest.approx((2.81e9 - 2.82e9) / (0.5995 - 0.6))
    assert required == pytest.approx(-5e-4)
    assert step == pytest.approx(-5e-4)


def test_secant_step_is_clipped_to_the_maximum():
    history = [(0.6, 2.8001e9), (0.5999, 2.8000e9)]  # tiny slope -> a huge required move
    step, _, required = next_flux_z_step(history, target_freq=2.90e9, probe_step=5e-4, max_step=1e-3)
    assert abs(required) > 1e-3
    assert step == pytest.approx(1e-3)  # clipped to +max_step


def test_no_slope_when_the_frequency_did_not_move():
    with pytest.raises(ValueError, match="cannot take a slope"):
        next_flux_z_step([(0.6, 2.80e9), (0.5995, 2.80e9)], target_freq=2.80e9, probe_step=5e-4, max_step=1e-3)
