"""Call-assertion test for the ``t2`` (Ramsey) DRAG benchmark node (QHC-1809).

Drives the full three-phase run with the shared ``qili_qprograms`` T2 (Ramsey) QProgram and the
decaying-oscillation fit scripted (a fake :class:`T2Fit`, a stubbed ``t2_ramsey`` and ``StreamArray``,
a fake platform, no hardware, no database, no disk), and asserts the node builds the sweep at the
operating point, sets the operating point's hardware itself, reports its value in microseconds, and lets
the run score the benchmark. The low-R² path checks that a bad fit fails the target and leaves no
benchmark result.
"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from qililab import Parameter
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
from seqtante_open.experiments.nodes import T2Node
from seqtante_open.experiments.nodes import t2 as t2_module
from seqtante_open.targets import ExperimentTarget
from tests.experiments.nodes.harness import (
    FakePlatform,
    FakeStreamArray,
    build_context,
    identity_crosstalk,
    stub_output_and_save,
)

Q1 = ExperimentTarget.from_str("q1")

DRIVE_LO = 2.9e9
DRIVE_IF = -0.1e9
READOUT_LO = 7e9
READOUT_IF = 1e8
READOUT_ATT = 50.0
DRIVE_POWER = -20
T2_US = 18.0
T2_SIGMA_US = 0.8
MEAS_ID = 777

#: Elapsed-time sweep the step hands the node (np.linspace args) and the measurement knobs the qili QProgram reads from the file.
ELAPSED_SWEEP = (0, 60_000, 61)
PARAMS = {
    "elapsed_sweep": ELAPSED_SWEEP,
    "averages": 1000,
    "relax_duration": 50_000,
    "drive_power": DRIVE_POWER,
}
#: Tree thresholds, in the benchmark's unit (us); T2_US above `passed` scores Passed.
THRESHOLDS = {
    Q1: BenchmarkThresholds(passed=10.0, warning=5.0, unit="us", direction=BenchmarkDirection.higher_is_better)
}


class FakeT2Fit:
    """Scripted stand-in for ``T2Fit``: no DB load, an accepted decay at ``T2_US`` ± ``T2_SIGMA_US``.

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
    def decay(self):
        # Positive for a decaying envelope; a negative one is the non-physical case the node must reject.
        return 18_000.0 if type(self).physical else -18_000.0

    @property
    def relative_uncertainty(self):
        return type(self).rel_unc

    @property
    def t2_us(self):
        return T2_US

    @property
    def t2_uncertainty_us(self):
        return T2_SIGMA_US


def _calibration() -> Calibration:
    calibration = Calibration()
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
            experiments={"t2": {"q1": ExperimentSection(operating_point="sweet_spot")}},
        ),
    )
    return calibration


def _make(calibration: Calibration, thresholds, **params) -> tuple[T2Node, FakePlatform]:
    parameters = {**PARAMS, **params}
    # One readout result per unrolled idle point.
    platform = FakePlatform(readout_points=len(np.linspace(*parameters["elapsed_sweep"])))
    context = build_context(calibration, platform)
    node = T2Node(
        name="t2_sweet_spot", idx=0, targets=[Q1], parameters={Q1: parameters}, context=context, thresholds=thresholds
    )
    return node, platform


def _patch(monkeypatch) -> dict:
    """No hardware, no disk: stub the qili QProgram builder, the stream, the fit and the calibration save."""
    built: dict = {}

    def fake_t2_ramsey(**kwargs):
        built.update(kwargs)
        return SimpleNamespace(name="qprogram")  # sentinel; the fake platform/stream ignore its content

    monkeypatch.setattr(t2_module, "t2_ramsey", fake_t2_ramsey)
    monkeypatch.setattr(t2_module, "StreamArray", FakeStreamArray)
    monkeypatch.setattr(FakeStreamArray, "measurement_id", MEAS_ID)
    monkeypatch.setattr(t2_module, "T2Fit", FakeT2Fit)
    stub_output_and_save(monkeypatch)
    # Reset the fit knobs to an accepted decay; each test flips the one it exercises.
    monkeypatch.setattr(FakeT2Fit, "r2", 0.99)
    monkeypatch.setattr(FakeT2Fit, "physical", True)
    monkeypatch.setattr(FakeT2Fit, "rel_unc", 0.03)
    monkeypatch.setattr(FakeT2Fit, "save_calls", 0)
    monkeypatch.setattr(FakeT2Fit, "events", [])
    monkeypatch.setattr(FakeT2Fit, "path_seen", None)
    return built


def test_builds_the_qili_sweep_at_the_point_sets_the_hardware_and_scores_the_benchmark(monkeypatch):
    built = _patch(monkeypatch)
    calibration = _calibration()
    node, platform = _make(calibration, THRESHOLDS)

    failed = node.run({})

    assert failed == []
    # The sweep was built at the target's operating point; the pulses/IFs come from the named point.
    assert built["target"] == "q1"
    assert built["operating_point"] == "sweet_spot"
    assert built["calibration"] is calibration
    assert built["averages"] == 1000
    assert built["relax_duration"] == 50_000
    np.testing.assert_array_equal(built["idle_duration_sweep"], np.rint(np.linspace(*ELAPSED_SWEEP)).astype(int))
    # The node set the operating point's hardware itself (snapshotted before the run restored it):
    # drive/readout LOs, readout attenuation, drive power, and the R&S IQ modulation and RF output.
    snap = platform.values_at_execute
    assert snap["drive_q1", Parameter.LO_FREQUENCY] == int(DRIVE_LO)
    assert snap["readout_q1", Parameter.LO_FREQUENCY] == int(READOUT_LO)
    assert snap["readout_q1", Parameter.OUT0_ATT] == int(READOUT_ATT)
    assert snap["drive_q1", Parameter.POWER] == DRIVE_POWER
    assert snap["drive_q1", Parameter.IQ_MODULATION] is True
    assert snap["drive_q1", Parameter.RF_ON] is True
    # analyze reported the value in us; the run scored it (higher-is-better, 18 us >= 10 us passed).
    result = node.benchmark_results[0]
    assert result.target == "q1"
    assert result.value == pytest.approx(T2_US)
    assert result.uncertainty == pytest.approx(T2_SIGMA_US)
    assert result.verdict is Verdict.passed
    # Fit, then the checks, then save, then plot: only an accepted fit reaches the DB.
    assert FakeT2Fit.events == ["fit", "save", "plot"]


def test_low_r2_fails_the_target_and_records_no_benchmark(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeT2Fit, "r2", 0.1)  # below the default minimum R²
    calibration = _calibration()
    node, _ = _make(calibration, THRESHOLDS)

    failed = node.run({})

    assert failed == ["q1"]  # the fit is rejected: the target fails the step
    assert node.benchmark_results == ()
    assert FakeT2Fit.save_calls == 0  # a rejected fit is never written to the DB


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("physical", False),  # d <= 0: a growing envelope has no T2
        ("r2", float("nan")),  # a flat trace: no variance to explain
        ("rel_unc", 2.0),  # sigma_T2 twice T2: above the default maximum
        ("rel_unc", float("inf")),  # curve_fit could not estimate the covariance
    ],
)
def test_a_rejected_fit_fails_the_target_and_is_not_saved(monkeypatch, knob, value):
    _patch(monkeypatch)
    monkeypatch.setattr(FakeT2Fit, knob, value)
    node, _ = _make(_calibration(), THRESHOLDS)

    failed = node.run({})

    assert failed == ["q1"]
    assert node.benchmark_results == ()
    assert FakeT2Fit.save_calls == 0
    assert FakeT2Fit.events == ["fit"]  # nothing is saved or plotted for a rejected fit


def test_idle_durations_are_rounded_not_truncated(monkeypatch):
    built = _patch(monkeypatch)
    node, _ = _make(_calibration(), THRESHOLDS, elapsed_sweep=(0, 1000, 7))

    node.run({})

    # 166.67 and 666.67 round to what t2_ramsey plays (167, 667), not truncate to 166/666.
    np.testing.assert_array_equal(built["idle_duration_sweep"], [0, 167, 333, 500, 667, 833, 1000])


def test_the_fit_plots_into_the_step_data_folder(monkeypatch, tmp_path):
    _patch(monkeypatch)
    node, _ = _make(_calibration(), THRESHOLDS, data_folder=str(tmp_path))  # the graph sets <data_root>/<step>/<target>

    node.run({})

    assert FakeT2Fit.path_seen == str(tmp_path)
