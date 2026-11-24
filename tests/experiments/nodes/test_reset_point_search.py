"""Call-assertion test for the ``reset_point_search`` loop, and a unit test of its flux step (QHC-1807).

The loop test drives the full three-phase run with the two-tone measurement and the Lorentzian fit
scripted (a fake :class:`LorentzianPeakModel`, a stubbed two-tone measurement, no ``time.sleep``), so the
search loop — measure, move ``flux.z`` through the restore handle, re-measure, then write the point — is
asserted without hardware, a database or a disk write.
"""

import numpy as np
import pytest
from qililab.qprogram.calibration import Calibration

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
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.nodes import ResetPointSearchNode
from seqtante_open.experiments.nodes import reset_point_search as reset_module
from seqtante_open.experiments.nodes.reset_point_search import _SearchOutcome, next_flux_z_step
from seqtante_open.targets import ExperimentTarget
from tests.experiments.nodes.harness import FakePlatform, identity_crosstalk

Q1 = ExperimentTarget.from_str("q1")

DRIVE_LO = 2.9e9
IF_OUTSIDE = -0.08e9  # + LO = 2.82 GHz, above the 2.79-2.81 GHz window
IF_INSIDE = -0.10e9  # + LO = 2.80 GHz, inside the window
FIRST_ID = 100


class FakeLorentzian:
    """Scripted stand-in for ``LorentzianPeakModel``: no DB load, fit keyed by measurement id.

    ``accept`` scripts whether a fit is accepted, and ``saves`` counts ``save_fitting`` calls, so a test
    can assert a rejected fit is never written to the DB.
    """

    accept = True
    saves = 0

    def __init__(self, measurement_id, target=None, path=None, r2_min=0.8, title=None):
        self.id = measurement_id
        self.results = {"quality": "fake"}

    def fit(self):
        return self.results

    def plot(self):
        return None

    def save_fitting(self):
        type(self).saves += 1

    @property
    def accepted(self):
        return type(self).accept

    @property
    def fitted_if(self):
        # The first sweep (id FIRST_ID) finds the qubit outside the window; every later one, inside.
        return IF_OUTSIDE if self.id == FIRST_ID else IF_INSIDE

    @property
    def r_squared(self):
        return 0.95


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
    calibration.crosstalk_matrix = identity_crosstalk()
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
    """No hardware, no disk, no sleep: stub the shared two-tone measurement, the fit and the save."""
    counter = {"next": FIRST_ID}

    def fake_two_tone(self, target, drive_if_sweep):
        measurement_id = counter["next"]
        counter["next"] += 1
        return measurement_id

    monkeypatch.setattr(ResetPointSearchNode, "_measure_two_tone_saturation", fake_two_tone)
    monkeypatch.setattr(reset_module, "LorentzianPeakModel", FakeLorentzian)
    monkeypatch.setattr(FakeLorentzian, "accept", True)
    monkeypatch.setattr(FakeLorentzian, "saves", 0)
    monkeypatch.setattr(reset_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)


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


def test_a_rejected_final_fit_fails_the_target_and_is_not_saved(monkeypatch):
    _patch(monkeypatch)
    node = _make(_calibration())
    node._search[Q1] = _SearchOutcome(flux_z=0.6, measurement_id=FIRST_ID + 1, plot_path=None)
    monkeypatch.setattr(FakeLorentzian, "accept", False)

    with pytest.raises(AnalysisError, match="final reset-search fit rejected"):
        node.analyze(Q1, [FIRST_ID + 1])

    assert FakeLorentzian.saves == 0  # fit -> check -> save: a rejected fit never reaches the DB


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
    assert slope is None
    assert required is None


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


def test_the_search_starts_on_the_window_around_the_stored_if_and_widens_at_the_same_spacing(monkeypatch):
    """The first sweep is ``drive_if_sweep`` around the stored drive IF; each retry widens it, same spacing."""
    _patch(monkeypatch)
    monkeypatch.setattr(FakeLorentzian, "accept", False)
    sweeps: list = []
    monkeypatch.setattr(
        ResetPointSearchNode,
        "_measure_two_tone_saturation",
        lambda self, target, sweep: sweeps.append(sweep) or FIRST_ID,
    )
    node = _make(_calibration())
    node.parameters[Q1].update({"drive_if_sweep": (-2e6, 2e6, 5), "retry_spans": (8e6,)})

    assert node.run({}) == ["q1"]  # no fit is accepted, so the line is never found

    stored_if = _file().operating_points["q1"]["reset"].drive.if_
    np.testing.assert_allclose(sweeps[0], stored_if + np.linspace(-2e6, 2e6, 5))
    np.testing.assert_allclose(sweeps[1], stored_if + np.arange(-4, 5) * 1e6)
