"""Call-assertion test for the ``t1`` saturation benchmark node (QHC-1803).

Drives the full three-phase run with the T1 saturation driver and the exponential fit scripted (a fake
:class:`T1Fit`, a stubbed driver, no hardware, no database, no disk), and asserts the node measures at
the operating point, reports its value in microseconds, and lets the run score the benchmark. The
low-R² path checks that a bad fit fails the target and leaves no benchmark result.
"""

import numpy as np
import pytest
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.calibration.io import write_calibration_file
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
from seqtante_open.controllers.verdicts import BenchmarkDirection, BenchmarkThresholds, Verdict
from seqtante_open.experiments.nodes import T1Node
from seqtante_open.experiments.nodes import t1 as t1_module
from seqtante_open.outputs import output_controller
from seqtante_open.targets import ExperimentTarget

Q1 = ExperimentTarget.from_str("q1")
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1")

DRIVE_LO = 2.9e9
DRIVE_IF = -0.1e9
READOUT_LO = 7e9
READOUT_IF = 1e8
T1_US = 42.0
T1_SIGMA_US = 1.5
MEAS_ID = 100

#: Idle sweep the step hands the node (np.linspace args) and the measurement knobs the fake driver ignores.
IDLE_SWEEP = (0, 40_000, 41)
PARAMS = {
    "idle_sweep": IDLE_SWEEP,
    "averages": 1000,
    "relax_duration": 50_000,
    "drive_duration": 20_000,
    "drive_amplitude": 0.5,
    "drive_power": -20,
    "readout_duration": 1000,
    "readout_amplitude": 0.4,
}
#: Tree thresholds, in the benchmark's unit (us); T1_US above `passed` scores Passed.
THRESHOLDS = {
    Q1: BenchmarkThresholds(passed=20.0, warning=10.0, unit="us", direction=BenchmarkDirection.higher_is_better)
}


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


class FakeT1Fit:
    """Scripted stand-in for ``T1Fit``: no DB load, an accepted fit at ``T1_US`` ± ``T1_SIGMA_US``."""

    r2 = 0.99

    def __init__(self, target=None, measurement_id=None, path=None):
        self.target = target
        self.id = measurement_id

    def fit(self):
        return None

    def plot(self):
        return None

    def save_fitting(self):
        return None

    @property
    def r_squared(self):
        return type(self).r2

    @property
    def t1_us(self):
        return T1_US

    @property
    def t1_uncertainty_us(self):
        return T1_SIGMA_US


def _crosstalk() -> CrosstalkMatrix:
    return CrosstalkMatrix.from_buses(
        {b: {bb: (1.0 if b == bb else 0.0) for bb in CROSSTALK_BUSES} for b in CROSSTALK_BUSES}
    )


def _calibration() -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = _crosstalk()
    write_calibration_file(
        calibration,
        CalibrationFile(
            operating_points_version=1,
            operating_points={
                "q1": {
                    "sweet_spot": OperatingPoint(
                        flux=Flux(z=0.6),
                        drive=DriveFreqs(lo=DRIVE_LO, if_=DRIVE_IF),
                        readout=ReadoutFreqs(lo=READOUT_LO, if_=READOUT_IF, attenuation=50.0),
                    )
                }
            },
            experiments={"t1": {"q1": ExperimentSection(operating_point="sweet_spot")}},
        ),
    )
    return calibration


def _make(calibration: Calibration, thresholds) -> T1Node:
    context = RunContext(
        platform=FakePlatform(),
        platform_path="runcard.yml",
        calibration=calibration,
        calibration_path="calibration.yml",
    )
    return T1Node(
        name="t1_sweet_spot", idx=0, targets=[Q1], parameters={Q1: dict(PARAMS)}, context=context, thresholds=thresholds
    )


def _patch(monkeypatch):
    """No hardware, no disk: stub the T1 driver, the fit and the calibration save."""
    calls: dict = {"n": 0}

    def fake_driver(**kwargs):
        calls.update(kwargs)
        calls["n"] += 1
        return MEAS_ID

    monkeypatch.setattr(t1_module, "t1_pulsed_saturation_experiment", fake_driver)
    monkeypatch.setattr(t1_module, "T1Fit", FakeT1Fit)
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(output_controller, "_initialized", True)
    monkeypatch.setattr(output_controller, "_db_manager", object())
    monkeypatch.setattr(FakeT1Fit, "r2", 0.99)
    return calls


def test_runs_the_saturation_sweep_at_the_point_and_scores_the_benchmark(monkeypatch):
    calls = _patch(monkeypatch)
    calibration = _calibration()
    node = _make(calibration, THRESHOLDS)

    failed = node.run({})

    assert failed == []
    assert calls["n"] == 1  # one T1 saturation sweep
    # The driver ran at the target's operating point, over the step's idle sweep.
    assert calls["drive_if_freq"] == pytest.approx(DRIVE_IF)
    assert calls["readout_if_freq"] == int(READOUT_IF)
    assert calls["drive_LO"] == int(DRIVE_LO)
    assert calls["readout_LO"] == int(READOUT_LO)
    np.testing.assert_array_equal(calls["durations_idle"], np.linspace(*IDLE_SWEEP).astype(int))
    # analyze reported the value in us; the run scored it (higher-is-better, 42 us >= 20 us passed).
    result = node.benchmark_results[0]
    assert result.target == "q1"
    assert result.value == pytest.approx(T1_US)
    assert result.uncertainty == pytest.approx(T1_SIGMA_US)
    assert result.verdict is Verdict.passed


def test_low_r2_fails_the_target_and_records_no_benchmark(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeT1Fit, "r2", 0.1)  # below the default minimum R²
    calibration = _calibration()
    node = _make(calibration, THRESHOLDS)

    failed = node.run({})

    assert failed == ["q1"]  # the fit is rejected: the target fails the step
    assert node.benchmark_results == ()
