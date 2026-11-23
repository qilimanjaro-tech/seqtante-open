"""Call-assertion test for the ``two_tone`` sweet-spot node (QHC-1805).

Drives the full three-phase run with the qili two-tone QProgram builder, the stream and the Lorentzian
fit scripted (a fake :class:`LorentzianPeakModel`, a stubbed builder and stream, a fake platform, no
database, no disk), so the node -- measure once at the operating point, fit, then write the drive IF -- is
asserted without any of them. The rejected-fit path checks that the step fails the target and leaves the
point untouched; the execute test checks what the node hands the builder and sets on the hardware.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from qililab import Parameter
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
from seqtante_open.experiments.nodes import TwoToneNode
from seqtante_open.experiments.nodes import two_tone as two_tone_module
from seqtante_open.outputs import FittedParameter, output_controller
from seqtante_open.targets import ExperimentTarget

Q1 = ExperimentTarget.from_str("q1")
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1")

DRIVE_LO = 2.9e9
FITTED_IF = -0.10e9  # + LO = 2.80 GHz
STORED_IF = -0.08e9  # the point's drive IF before the node runs
MEAS_ID = 100

#: The sweep window the step hands the node (absolute Hz).
PARAMS = {"sweep_start": 2.79e9, "sweep_stop": 2.81e9, "sweep_step": 0.4e6}


class FakePlatform:
    def __init__(self):
        self.values: dict = {}
        self.calls: list = []
        self.values_at_execute: dict = {}

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

    def get_element(self, alias):
        """A Rohde & Schwarz drive instrument, so the IQ-modulation/POWER/RF branch runs."""
        return SimpleNamespace(instruments=[SimpleNamespace(name=SimpleNamespace(name="ROHDE_SCHWARZ"))])

    def execute_qprogram(self, qprogram, **kwargs):
        """Snapshot the parameters set so far (before the run restores them) and return a readout result."""
        self.calls.append("execute_qprogram")
        self.values_at_execute = dict(self.values)
        return SimpleNamespace(results={"readout_q1": [SimpleNamespace(array=np.zeros((2, 2)))]})


class FakeStreamArray:
    """Stand-in for ``StreamArray``: a context manager that swallows the write and carries a measurement id."""

    def __init__(self, **kwargs):
        self.measurement = SimpleNamespace(measurement_id=MEAS_ID)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __setitem__(self, key, value):
        return None


class FakeLorentzian:
    """Scripted stand-in for ``LorentzianPeakModel``: no DB load, an accepted fit at ``FITTED_IF``."""

    saved: list = []
    accepted_result = True

    def __init__(self, measurement_id, target=None, path=None, r2_min=0.8, title="Two tone"):
        self.id = measurement_id
        self.results = {"quality": "fake"}

    def fit(self):
        return self.results

    def plot(self):
        return None

    def add_fitted_parameter(self, parameter, value):
        type(self).saved.append((parameter, value))

    def save_fitting(self):
        return None

    @property
    def accepted(self):
        return type(self).accepted_result

    @property
    def fitted_if(self):
        return FITTED_IF if type(self).accepted_result else None

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
                "sweet_spot": OperatingPoint(
                    flux=Flux(z=0.6),
                    drive=DriveFreqs(lo=DRIVE_LO, if_=STORED_IF),
                    readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=50.0),
                )
            }
        },
        experiments={"two_tone": {"q1": ExperimentSection(operating_point="sweet_spot")}},
    )


def _calibration() -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = _crosstalk()
    write_calibration_file(calibration, _file())
    return calibration


def _make(calibration: Calibration, platform: FakePlatform | None = None) -> TwoToneNode:
    context = RunContext(
        platform=platform or FakePlatform(),
        platform_path="runcard.yml",
        calibration=calibration,
        calibration_path="calibration.yml",
    )
    return TwoToneNode(name="two_tone_sweet_spot", idx=0, targets=[Q1], parameters={Q1: dict(PARAMS)}, context=context)


def _patch(monkeypatch):
    """No hardware, no disk: stub the qili QProgram builder, the stream, the fit and the calibration save."""
    calls: dict = {"measure": 0, "built": {}}

    def fake_two_tone_saturation(**kwargs):
        calls["measure"] += 1
        calls["built"] = kwargs
        return SimpleNamespace(name="qprogram")  # sentinel; the fake platform/stream ignore its content

    monkeypatch.setattr(two_tone_module, "two_tone_saturation", fake_two_tone_saturation)
    monkeypatch.setattr(two_tone_module, "StreamArray", FakeStreamArray)
    monkeypatch.setattr(two_tone_module, "LorentzianPeakModel", FakeLorentzian)
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(output_controller, "_initialized", True)
    monkeypatch.setattr(output_controller, "_db_manager", object())
    monkeypatch.setattr(FakeLorentzian, "saved", [])
    monkeypatch.setattr(FakeLorentzian, "accepted_result", True)
    return calls


def test_accepted_fit_writes_the_drive_if_and_keeps_the_lo(monkeypatch):
    calls = _patch(monkeypatch)
    calibration = _calibration()
    node = _make(calibration)

    failed = node.run({})

    assert failed == []
    assert calls["measure"] == 1  # one two-tone sweep
    point = read_calibration_file(calibration).operating_points["q1"]["sweet_spot"]
    assert point.drive.if_ == pytest.approx(FITTED_IF)  # drive IF taken from the fit
    assert point.drive.lo == pytest.approx(DRIVE_LO)  # the LO is left untouched
    # drive_if/fit_r2 come from the shared fit's save_fitting; the node records drive_lo alongside them.
    assert (FittedParameter.DRIVE_LO, DRIVE_LO) in FakeLorentzian.saved


def test_rejected_fit_fails_the_target_and_leaves_the_point(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeLorentzian, "accepted_result", False)
    calibration = _calibration()
    node = _make(calibration)

    failed = node.run({})

    assert failed == ["q1"]  # the fit is rejected: the target fails the step, not the run
    point = read_calibration_file(calibration).operating_points["q1"]["sweet_spot"]
    assert point.drive.if_ == pytest.approx(STORED_IF)  # unchanged


def test_drive_if_sweep_is_centred_on_the_window_minus_the_lo(monkeypatch):
    _patch(monkeypatch)
    node = _make(_calibration())

    sweep = node._drive_if_sweep(Q1, DRIVE_LO)

    assert sweep[0] == pytest.approx(PARAMS["sweep_start"] - DRIVE_LO)
    assert sweep[-1] == pytest.approx(PARAMS["sweep_stop"] - DRIVE_LO)
    assert len(sweep) == round((PARAMS["sweep_stop"] - PARAMS["sweep_start"]) / PARAMS["sweep_step"]) + 1
    assert np.all(np.diff(sweep) > 0)


def test_execute_builds_the_qili_sweep_and_sets_the_hardware(monkeypatch):
    """The real ``execute_node`` builds the qili QProgram at the operating point and sets the HW from it."""
    calls = _patch(monkeypatch)
    calibration = _calibration()
    platform = FakePlatform()
    node = _make(calibration, platform)

    failed = node.run({})

    assert failed == []
    built = calls["built"]
    # The sweep is built at the target's operating point; the readout pulse and IF come from the named point.
    assert built["target"] == "q1"
    assert built["operating_point"] == "sweet_spot"
    assert built["calibration"] is calibration
    np.testing.assert_allclose(built["drive_if_sweep"], node._drive_if_sweep(Q1, DRIVE_LO))
    # Unset tree parameters fall back to the node's defaults, mapped onto qili's argument names.
    assert built["drive_ringup_time"] == 200
    assert built["drive_ringdown_time"] == 200
    assert built["drive_n_sigmas"] == 4
    assert built["readout_ringup_time"] == 600
    # The node set the operating point's hardware itself (snapshotted before the run restored it):
    # drive/readout LOs, readout attenuation, drive power, and the R&S IQ modulation and RF output.
    snap = platform.values_at_execute
    assert snap["drive_q1", Parameter.LO_FREQUENCY] == int(DRIVE_LO)
    assert snap["readout_q1", Parameter.LO_FREQUENCY] == int(7e9)
    assert snap["readout_q1", Parameter.OUT0_ATT] == 50
    assert snap["drive_q1", Parameter.POWER] == -40
    assert snap["drive_q1", Parameter.IQ_MODULATION] is True
    assert snap["drive_q1", Parameter.RF_ON] is True
