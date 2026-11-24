"""Call-assertion test for the ``t1`` saturation benchmark node (QHC-1803).

Drives the full three-phase run with the exponential fit scripted (a fake :class:`T1Fit`), the
``qili_qprograms`` QProgram builder and the stream/execute boundary stubbed (no hardware, no database,
no disk), and asserts the node builds the sweep at the operating point, sets the operating point's
hardware (LOs, readout attenuation, drive power/RF), reports its value in microseconds, and lets the run
score the benchmark. The rejection paths check that a bad fit (low R², non-physical decay, ill-determined
lifetime) fails the target, records no benchmark result, and never reaches the DB.

The platform and stream-array fakes and the output/save stubbing come from
:mod:`tests.experiments.nodes.harness`; the readout shape the T1 QProgram returns, the QProgram builder
stub and the fit stub live here.
"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from qililab import Parameter, Square
from qililab.qprogram.calibration import Calibration

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
from tests.experiments.nodes.harness import (
    ExecutingPlatform,
    FakeStreamArray,
    build_context,
    identity_crosstalk,
    stub_output_and_save,
)

Q1 = ExperimentTarget.from_str("q1")
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
    "idle_duration_sweep": IDLE_SWEEP,
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
    #: What the node did with the fit, in order (fit, save, plot), and the plot path it was built with.
    events: ClassVar[list[str]] = []
    path_seen: str | None = None

    def __init__(self, target=None, measurement_id=None, path=None):
        self.target = target
        self.id = measurement_id
        self.path = path
        type(self).path_seen = path

    def fit(self):
        type(self).events.append("fit")

    def plot(self):
        type(self).events.append("plot")

    def save_fitting(self):
        type(self).save_calls += 1
        type(self).events.append("save")

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


def _calibration() -> Calibration:
    calibration = Calibration()
    calibration.add_waveform(bus=READOUT_BUS, name="readout-sweet_spot", waveform=Square(amplitude=1.0, duration=1000))
    calibration.crosstalk_matrix = identity_crosstalk()
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


class FakeT1Platform(ExecutingPlatform):
    """An :class:`ExecutingPlatform` with a Rohde & Schwarz drive whose execution returns one point per idle."""

    def __init__(self) -> None:
        super().__init__(instruments={DRIVE_BUS: "ROHDE_SCHWARZ"})

    def results(self) -> dict:
        return {READOUT_BUS: [SimpleNamespace(array=np.zeros((2, 1))) for _ in range(N_IDLE)]}


def _make(calibration: Calibration, thresholds, platform=None, **params) -> T1Node:
    context = build_context(calibration, platform if platform is not None else FakeT1Platform())
    return T1Node(
        name="t1_sweet_spot",
        idx=0,
        targets=[Q1],
        parameters={Q1: {**PARAMS, **params}},
        context=context,
        thresholds=thresholds,
    )


def _patch(monkeypatch):
    """No hardware, no disk: stub the QProgram builder, the stream array, the fit and the output/save boundaries."""
    calls: dict = {"build": None}

    def fake_build(**kwargs):
        calls["build"] = kwargs
        return SimpleNamespace(name="qprogram")  # sentinel; the fake platform/stream ignore its content

    monkeypatch.setattr(t1_module, "t1_saturation", fake_build)
    monkeypatch.setattr(FakeStreamArray, "measurement_id", MEAS_ID)
    monkeypatch.setattr(t1_module, "T1Fit", FakeT1Fit)
    stub_output_and_save(monkeypatch)
    # Reset the fit knobs to an accepted decay; each test flips the one it exercises.
    monkeypatch.setattr(FakeT1Fit, "r2", 0.99)
    monkeypatch.setattr(FakeT1Fit, "physical", True)
    monkeypatch.setattr(FakeT1Fit, "rel_unc", 0.03)
    monkeypatch.setattr(FakeT1Fit, "save_calls", 0)
    monkeypatch.setattr(FakeT1Fit, "events", [])
    monkeypatch.setattr(FakeT1Fit, "path_seen", None)
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
    # Unset optional keys fall back to the node defaults, named like the t1_saturation arguments.
    assert build["drive_ringup_time"] == 200
    assert build["drive_ringdown_time"] == 200
    assert build["drive_n_sigmas"] == 4
    assert build["readout_ringup_time"] == 0
    # The node set the operating point's hardware itself (snapshotted before the run restored it):
    # drive/readout LOs, readout attenuation, drive power, and the R&S IQ modulation and RF output.
    snap = platform.values_at_execute
    assert snap[DRIVE_BUS, Parameter.LO_FREQUENCY] == int(DRIVE_LO)
    assert snap[READOUT_BUS, Parameter.LO_FREQUENCY] == int(READOUT_LO)
    assert snap[READOUT_BUS, Parameter.OUT0_ATT] == int(READOUT_ATT)
    assert snap[DRIVE_BUS, Parameter.POWER] == DRIVE_POWER
    assert snap[DRIVE_BUS, Parameter.IQ_MODULATION] is True
    assert snap[DRIVE_BUS, Parameter.RF_ON] is True
    # analyze reported the value in us; the run scored it (higher-is-better, 42 us >= 20 us passed).
    result = node.benchmark_results[0]
    assert result.target == "q1"
    assert result.value == pytest.approx(T1_US)
    assert result.uncertainty == pytest.approx(T1_SIGMA_US)
    assert result.verdict is Verdict.passed
    assert FakeT1Fit.save_calls == 1  # the accepted fit was saved once
    assert FakeT1Fit.events == ["fit", "save", "plot"]  # fit -> checks -> save -> plot


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


def test_idle_durations_are_rounded_not_truncated(monkeypatch):
    calls = _patch(monkeypatch)
    node = _make(_calibration(), THRESHOLDS, idle_duration_sweep=(0, 1000, 7))

    node.run({})

    # 166.67 and 666.67 round to what t1_saturation plays (167, 667), not truncate to 166/666.
    np.testing.assert_array_equal(calls["build"]["idle_duration_sweep"], [0, 167, 333, 500, 667, 833, 1000])


def test_an_undefined_r2_fails_the_target(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeT1Fit, "r2", float("nan"))  # a flat trace: no variance to explain
    node = _make(_calibration(), THRESHOLDS)

    failed = node.run({})

    assert failed == ["q1"]
    assert FakeT1Fit.save_calls == 0


def test_the_fit_plots_into_the_step_data_folder(monkeypatch, tmp_path):
    _patch(monkeypatch)
    node = _make(_calibration(), THRESHOLDS, data_folder=str(tmp_path))  # the graph sets <data_root>/<step>/<target>

    node.run({})

    assert FakeT1Fit.path_seen == str(tmp_path)
