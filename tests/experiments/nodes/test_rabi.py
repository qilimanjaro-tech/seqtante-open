"""Call-assertion tests for the ``rabi`` node's two analyses (QHC-1806).

One test per analysis drives the full three-phase run with the measurement and the fit scripted (a fake
:class:`RabiAmplitudeModel`, a stubbed ``execute_node``), so each analysis's write is asserted without
hardware, a database or a disk write: ``readout_levels`` writes the |0>/|1> levels onto the operating
point, ``pi_amplitude`` writes the fitted amplitude into the named DRAG pulse in ``Calibration.waveforms``.
"""

import pytest
from qililab import IQDrag
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
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.nodes import RabiNode
from seqtante_open.experiments.nodes import rabi as rabi_module
from seqtante_open.targets import ExperimentTarget

Q1 = ExperimentTarget.from_str("q1")
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1")

# Scripted fit outputs the fake model returns.
FAKE_PI = 0.42
FAKE_L0, FAKE_L0_STD = -1.5, 0.03
FAKE_L1, FAKE_L1_STD = 1.5, 0.04


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


class FakeRabiModel:
    """A scripted stand-in for ``RabiAmplitudeModel``: no DB load, no fit, fixed outputs."""

    r_squared = 0.99

    def __init__(self, measurement_id, target=None, path=None):
        self.id = measurement_id
        self.target = target
        self.path = path
        self.results = {"angular_frequency": 1.0}

    def fit(self):
        return self.results

    def plot(self):
        return None

    def save_fitting(self):
        return None

    @property
    def pi_amplitude(self):
        return FAKE_PI

    @property
    def readout_levels(self):
        return (FAKE_L0, FAKE_L0_STD, FAKE_L1, FAKE_L1_STD)


def _crosstalk() -> CrosstalkMatrix:
    return CrosstalkMatrix.from_buses(
        {b: {bb: (1.0 if b == bb else 0.0) for bb in CROSSTALK_BUSES} for b in CROSSTALK_BUSES}
    )


def _operating_point() -> OperatingPoint:
    return OperatingPoint(
        flux=Flux(z=0.1),
        drive=DriveFreqs(lo=4e9, if_=1e8),
        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=10.0),
    )


def _rabi_file(point_name: str) -> CalibrationFile:
    """A calibration file with one ``rabi`` section for q1 at ``point_name`` (no pulse overrides)."""
    return CalibrationFile(
        operating_points_version=1,
        operating_points={"q1": {point_name: _operating_point()}},
        experiments={"rabi": {"q1": ExperimentSection(operating_point=point_name)}},
    )


def _calibration(file: CalibrationFile) -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = _crosstalk()
    write_calibration_file(calibration, file)
    return calibration


def _make_rabi(*, analysis: str, calibration: Calibration, parameters: dict | None = None) -> RabiNode:
    context = RunContext(
        platform=FakePlatform(),
        platform_path="runcard.yml",
        calibration=calibration,
        calibration_path="calibration.yml",
    )
    node = RabiNode(
        name="step",
        idx=0,
        targets=[Q1],
        parameters={Q1: parameters or {}},
        context=context,
        analysis=analysis,
    )
    node.execute_node = lambda targets: {target: [777] for target in targets}  # no hardware
    return node


def test_readout_levels_analysis_writes_levels_onto_the_point(monkeypatch):
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("reset"))

    node = _make_rabi(analysis="readout_levels", calibration=calibration)
    failed = node.run({})

    assert failed == []
    levels = read_calibration_file(calibration).operating_points["q1"]["reset"].readout.levels
    assert levels is not None
    assert levels.pulse == "readout"  # the default readout pulse name, tagged on the levels
    assert levels.level_0.nominal == pytest.approx(FAKE_L0)
    assert levels.level_0.std == pytest.approx(FAKE_L0_STD)
    assert levels.level_1.nominal == pytest.approx(FAKE_L1)
    assert levels.level_1.std == pytest.approx(FAKE_L1_STD)


def test_pi_amplitude_analysis_writes_into_the_named_pulse(monkeypatch):
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("sweet_spot"))
    # The pi amplitude is written into this existing DRAG pulse, keeping its other shape parameters.
    calibration.add_waveform(
        bus="drive_q1", name="drag", waveform=IQDrag(amplitude=1.0, duration=40, num_sigmas=4, drag_coefficient=0.1)
    )

    node = _make_rabi(analysis="pi_amplitude", calibration=calibration)
    failed = node.run({})

    assert failed == []
    pulse = calibration.get_waveform(bus="drive_q1", name="drag")
    assert pulse.amplitude == pytest.approx(FAKE_PI)  # amplitude updated to the fitted pi amplitude
    assert pulse.duration == 40  # shape preserved
    assert pulse.num_sigmas == 4
    assert pulse.drag_coefficient == pytest.approx(0.1)
    # The operating point's readout levels are untouched by the pi-amplitude analysis.
    assert read_calibration_file(calibration).operating_points["q1"]["sweet_spot"].readout.levels is None


def test_readout_levels_fit_below_min_r2_fails_the_step(monkeypatch):
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)

    class LowR2Model(FakeRabiModel):
        r_squared = 0.5  # below the default 0.9 minimum

    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", LowR2Model)
    calibration = _calibration(_rabi_file("reset"))

    node = _make_rabi(analysis="readout_levels", calibration=calibration)
    failed = node.run({})

    assert failed == ["q1"]  # a poor fit fails the target
    assert read_calibration_file(calibration).operating_points["q1"]["reset"].readout.levels is None


def test_analyze_raises_analysis_error_below_min_r2(monkeypatch):
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("reset"))
    node = _make_rabi(analysis="readout_levels", calibration=calibration, parameters={"min_r2": 0.999})

    with pytest.raises(AnalysisError, match="below the minimum"):
        node.analyze(Q1, [777])
