"""Call-assertion test for the ``t1`` saturation benchmark node (QHC-1803).

Drives the full three-phase run with the exponential fit scripted (a fake :class:`T1Fit`), the
``qili_qprograms`` QProgram builder and the stream/execute boundary stubbed (no hardware, no database,
no disk), and asserts the node builds the sweep at the operating point, sets the operating point's
hardware (LOs, readout attenuation, drive power/RF), reports its value in microseconds, and lets the run
score the benchmark. The rejection paths check that a bad fit (low R², non-physical decay, ill-determined
lifetime) fails the target, records no benchmark result, and never reaches the DB.

The platform fake and the output/save stubbing come from :mod:`tests.experiments.nodes.harness`; the
T1-specific execution fakes (QProgram builder, stream array, ``execute_qprogram``) and the fit stub live
here.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from qililab import Parameter
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
from seqtante_open.controllers.verdicts import BenchmarkDirection, BenchmarkThresholds, Verdict
from seqtante_open.experiments.nodes import T1Node
from seqtante_open.experiments.nodes import t1 as t1_module
from seqtante_open.targets import ExperimentTarget
from tests.experiments.nodes.harness import FakePlatform, build_context, stub_output_and_save

Q1 = ExperimentTarget.from_str("q1")
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1")
DRIVE_BUS = "drive_q1"
READOUT_BUS = "readout_q1"

DRIVE_LO = 2.9e9
DRIVE_IF = -0.1e9
READOUT_LO = 7e9
READOUT_IF = 1e8
READOUT_ATT = 50.0
DRIVE_POWER = -20
T1_US = 42.0
T1_SIGMA_US = 1.5
MEAS_ID = 100

#: Idle sweep the step hands the node (np.linspace args), and its resolved length.
IDLE_SWEEP = (0, 40_000, 41)
N_IDLE = len(np.linspace(*IDLE_SWEEP))
#: The readout pulse and the drive/readout IFs are no longer step knobs: qili_qprograms reads them from
#: the operating point in the calibration file, so the node passes only its name (and sets the hardware).
PARAMS = {
    "idle_sweep": IDLE_SWEEP,
    "averages": 1000,
    "relax_duration": 50_000,
    "drive_duration": 20_000,
    "drive_amplitude": 0.5,
    "drive_power": DRIVE_POWER,
}
#: Tree thresholds, in the benchmark's unit (us); T1_US above `passed` scores Passed.
THRESHOLDS = {
    Q1: BenchmarkThresholds(passed=20.0, warning=10.0, unit="us", direction=BenchmarkDirection.higher_is_better)
}


class FakeT1Fit:
    """Scripted stand-in for ``T1Fit``: no DB load, an accepted decay at ``T1_US`` ± ``T1_SIGMA_US``.

    The class-level knobs (``r2``, ``physical``, ``rel_unc``) let a test script an accepted or a rejected
    fit, and ``save_calls`` counts how often ``save_fitting`` ran, so a test can assert a rejected fit is
    never saved.
    """

    r2 = 0.99
    physical = True
    rel_unc = 0.03
    save_calls = 0

    def __init__(self, target=None, measurement_id=None, path=None):
        self.target = target
        self.id = measurement_id
        self.path = path

    def fit(self):
        return None

    def plot(self):
        return None

    def save_fitting(self):
        type(self).save_calls += 1

    @property
    def r_squared(self):
        return type(self).r2

    @property
    def is_physical_decay(self):
        return type(self).physical

    @property
    def rate(self):
        # Negative for a decay; a positive rate is the non-physical case the node must reject.
        return -1.0e-4 if type(self).physical else 1.0e-4

    @property
    def relative_uncertainty(self):
        return type(self).rel_unc

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
                        readout=ReadoutFreqs(lo=READOUT_LO, if_=READOUT_IF, attenuation=READOUT_ATT),
                    )
                }
            },
            experiments={"t1": {"q1": ExperimentSection(operating_point="sweet_spot")}},
        ),
    )
    return calibration


class FakeStreamArray:
    """Stand-in for ``StreamArray``: a context manager that swallows the per-idle writes and carries MEAS_ID."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.measurement = SimpleNamespace(measurement_id=MEAS_ID)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __setitem__(self, key, value):
        return None


class FakeT1Platform(FakePlatform):
    """A :class:`FakePlatform` that also answers the execution calls the inlined node makes.

    ``get_element`` reports a Rohde & Schwarz drive instrument (so the POWER/RF branch runs) and
    ``execute_qprogram`` returns a readout result of the right length. On execute it snapshots the
    parameters set so far into :attr:`values_at_execute`, before the node's run restores them.
    """

    def __init__(self, values: dict | None = None) -> None:
        super().__init__(values)
        self.values_at_execute: dict = {}

    def get_element(self, alias):
        self.calls.append(f"get_element:{alias}")
        return SimpleNamespace(instruments=[SimpleNamespace(name=SimpleNamespace(name="ROHDE_SCHWARZ"))])

    def execute_qprogram(self, qprogram, **kwargs):
        self.calls.append("execute_qprogram")
        self.values_at_execute = dict(self.values)
        points = [SimpleNamespace(array=np.zeros((2, 1))) for _ in range(N_IDLE)]
        return SimpleNamespace(results={READOUT_BUS: points})


def _make(calibration: Calibration, thresholds, platform=None) -> T1Node:
    context = build_context(calibration, platform if platform is not None else FakeT1Platform())
    return T1Node(
        name="t1_sweet_spot", idx=0, targets=[Q1], parameters={Q1: dict(PARAMS)}, context=context, thresholds=thresholds
    )


def _patch(monkeypatch):
    """No hardware, no disk: stub the QProgram builder, the stream array, the fit and the output/save boundaries."""
    calls: dict = {"build": None}

    def fake_build(**kwargs):
        calls["build"] = kwargs
        return SimpleNamespace(name="qprogram")  # sentinel; the fake platform/stream ignore its content

    monkeypatch.setattr(t1_module, "t1_saturation", fake_build)
    monkeypatch.setattr(t1_module, "StreamArray", FakeStreamArray)
    monkeypatch.setattr(t1_module, "T1Fit", FakeT1Fit)
    stub_output_and_save(monkeypatch)
    # Reset the fit knobs to an accepted decay; each test flips the one it exercises.
    monkeypatch.setattr(FakeT1Fit, "r2", 0.99)
    monkeypatch.setattr(FakeT1Fit, "physical", True)
    monkeypatch.setattr(FakeT1Fit, "rel_unc", 0.03)
    monkeypatch.setattr(FakeT1Fit, "save_calls", 0)
    return calls


def test_runs_the_saturation_sweep_at_the_point_and_scores_the_benchmark(monkeypatch):
    calls = _patch(monkeypatch)
    calibration = _calibration()
    platform = FakeT1Platform()
    node = _make(calibration, THRESHOLDS, platform=platform)

    failed = node.run({})

    assert failed == []
    # The sweep was built at the target's operating point, over the step's idle sweep. The IFs and the
    # readout pulse are not passed: qili_qprograms reads them from the named operating point.
    build = calls["build"]
    assert build["target"] == "q1"
    assert build["operating_point"] == "sweet_spot"
    assert build["calibration"] is calibration
    np.testing.assert_array_equal(build["idle_duration_sweep"], np.linspace(*IDLE_SWEEP).astype(int))
    # The node set the operating point's hardware itself (snapshotted before the run restored it):
    # drive/readout LOs, readout attenuation, drive power and the R&S RF output.
    snap = platform.values_at_execute
    assert snap[DRIVE_BUS, Parameter.LO_FREQUENCY] == int(DRIVE_LO)
    assert snap[READOUT_BUS, Parameter.LO_FREQUENCY] == int(READOUT_LO)
    assert snap[READOUT_BUS, Parameter.OUT0_ATT] == int(READOUT_ATT)
    assert snap[DRIVE_BUS, Parameter.POWER] == DRIVE_POWER
    assert snap[DRIVE_BUS, Parameter.RF_ON] is True
    # analyze reported the value in us; the run scored it (higher-is-better, 42 us >= 20 us passed).
    result = node.benchmark_results[0]
    assert result.target == "q1"
    assert result.value == pytest.approx(T1_US)
    assert result.uncertainty == pytest.approx(T1_SIGMA_US)
    assert result.verdict is Verdict.passed
    assert FakeT1Fit.save_calls == 1  # the accepted fit was saved once


def test_low_r2_fails_the_target_and_records_no_benchmark(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeT1Fit, "r2", 0.1)  # below the default minimum R²
    calibration = _calibration()
    node = _make(calibration, THRESHOLDS)

    failed = node.run({})

    assert failed == ["q1"]  # the fit is rejected: the target fails the step
    assert node.benchmark_results == ()
    assert FakeT1Fit.save_calls == 0  # a rejected fit is never written to the DB


def test_non_physical_decay_fails_the_target_and_is_not_saved(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeT1Fit, "physical", False)  # rate >= 0: T1 would be negative/infinite
    calibration = _calibration()
    node = _make(calibration, THRESHOLDS)

    failed = node.run({})

    assert failed == ["q1"]
    assert node.benchmark_results == ()
    assert FakeT1Fit.save_calls == 0


def test_large_relative_uncertainty_fails_the_target_and_is_not_saved(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeT1Fit, "rel_unc", 2.0)  # sigma_T1 twice T1: above the default maximum
    calibration = _calibration()
    node = _make(calibration, THRESHOLDS)

    failed = node.run({})

    assert failed == ["q1"]
    assert node.benchmark_results == ()
    assert FakeT1Fit.save_calls == 0


def test_infinite_uncertainty_fails_the_target(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeT1Fit, "rel_unc", float("inf"))  # curve_fit could not estimate the covariance
    calibration = _calibration()
    node = _make(calibration, THRESHOLDS)

    failed = node.run({})

    assert failed == ["q1"]
    assert FakeT1Fit.save_calls == 0
