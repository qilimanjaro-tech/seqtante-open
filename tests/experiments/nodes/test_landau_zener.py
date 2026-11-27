"""Tests for the ``landau_zener`` benchmark node (QHC-1808, QHC-1921, ADR-0012).

The node runs the full three-phase ``run`` with the real ``generalized_linear_ramps`` QProgram from
``qili-qprograms`` (a spy records its arguments), and a platform that executes nothing: ``StreamArray``,
the scoring fit, the DB and the calibration save are stubbed. The tests check that the node reads the
target's simulated-points table, builds one QProgram per (phi_x, ramp duration) at the step's operating
point, compiles them with the AC crosstalk of the fast-flux lines, sets the readout line, hands the fit
the readout levels and each row's Delta and Delta-epsilon, and reports the ``rmse`` S as the value the run
scores (higher is better).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from qili_qprograms.bus_alias import Role
from qili_qprograms.calibration.io import write_calibration_file
from qili_qprograms.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    ExperimentSection,
    Flux,
    Measured,
    OperatingPoint,
    ReadoutFreqs,
    ReadoutLevels,
)
from qili_qprograms.targets import ExperimentTarget
from qililab import IQPair, Parameter, Square
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.controllers.node_registry import BenchmarkDirection
from seqtante_open.controllers.verdicts import BenchmarkThresholds, Verdict
from seqtante_open.experiments.fitting.landau_zener_fit import Score
from seqtante_open.experiments.nodes import LandauZenerNode
from seqtante_open.experiments.nodes import landau_zener as lz_module
from tests.experiments.nodes.harness import ExecutingPlatform, FakeStreamArray, build_context, stub_output_and_save

Q1 = ExperimentTarget.from_str("q1")
DC_FLUX_BUSES = ["flux_z_q1", "flux_x_q1"]
FAST_FLUX_BUSES = ["fast_flux_z_q1", "fast_flux_x_q1"]
READOUT_BUS = "readout_q1"
READOUT_LO = 7e9
READOUT_IF = 1e8
READOUT_ATT = 50.0
IDLE = (0.1, 0.2)
SWEET_SPOT = (0.0, 0.5)
LEVEL_0 = 0.0
LEVEL_1 = 1.0
THRESHOLD = 0.25
READOUT_PULSE = "readout-reset"
AC_CROSSTALK = [[1.0, 0.1], [0.05, 1.0]]

#: The table as it is seeded, out of phi_x order; the node sweeps it sorted. Neither ramp starts or ends
#: at the reset point's flux, so both need the entry and exit ramps.
SIMULATED_POINTS = [
    {"phi_x": 0.25, "z_start": 0.15, "z_end": 0.3, "min_gap": 0.065, "detuning_span": 1.85},
    {"phi_x": 0.2, "z_start": 0.12, "z_end": 0.28, "min_gap": 0.05, "detuning_span": 1.9},
]
RAMP_DURATIONS = [10, 20, 40, 80]
SCORE = Score(value=0.9, uncertainty=0.02, distance=0.04, reference=0.4)

PARAMS = {
    "ramp_durations": RAMP_DURATIONS,
    "relax_duration": 1_000,
    "averages": 1_000,
    "entry_ramp_duration": [40, 60],
    "exit_ramp_duration": 50,
    "ac_crosstalk": AC_CROSSTALK,
}


class LandauZenerPlatform(ExecutingPlatform):
    """The shared :class:`ExecutingPlatform` with no instruments, recording each execution's calibration and parameters.

    It declares no instrument: the node plays no drive, so asking for one is an error.
    """

    def __init__(self) -> None:
        super().__init__(instruments={})
        self.runs: list[dict] = []

    def execute_qprogram(self, qprogram, calibration=None, **kwargs):
        self.runs.append({"calibration": calibration, "values": dict(self.values)})
        return super().execute_qprogram(qprogram, calibration=calibration, **kwargs)

    def results(self) -> dict:
        return {READOUT_BUS: [SimpleNamespace(array=np.array([0.5, -0.5]), threshold=[THRESHOLD])]}


class FakeLandauZenerFit:
    """Scripted stand-in for ``LandauZenerFit``: no DB load, records its arguments, scores :data:`SCORE`."""

    last: ClassVar[dict] = {}
    saves: ClassVar[int] = 0
    score = SCORE

    def __init__(self, measurement_id, target, level_0, level_1, min_gap, detuning_span, path=None):
        type(self).last = {
            "measurement_id": measurement_id,
            "target": target,
            "level_0": level_0,
            "level_1": level_1,
            "min_gap": list(min_gap),
            "detuning_span": list(detuning_span),
        }

    def fit(self):
        return {}

    def save_fitting(self):
        type(self).saves += 1

    def plot(self):
        return None


def _point(flux: tuple[float, float], levels_pulse: str = READOUT_PULSE) -> OperatingPoint:
    return OperatingPoint(
        flux=Flux(z=flux[0], x=flux[1]),
        drive=DriveFreqs(lo=2.9e9, if_=-0.1e9),
        readout=ReadoutFreqs(
            lo=READOUT_LO,
            if_=READOUT_IF,
            attenuation=READOUT_ATT,
            levels=ReadoutLevels(
                pulse=levels_pulse,
                level_0=Measured(nominal=LEVEL_0, std=0.0),
                level_1=Measured(nominal=LEVEL_1, std=0.0),
            ),
        ),
    )


def _calibration(
    readout_levels_pulse: str = READOUT_PULSE, simulated_points: list | None = SIMULATED_POINTS
) -> Calibration:
    calibration = Calibration()
    # The run needs a DC crosstalk matrix to set the slow flux (separate from the AC one on the fast lines).
    calibration.crosstalk_matrix = CrosstalkMatrix.from_array(buses=DC_FLUX_BUSES, matrix_array=np.eye(2))
    calibration.add_waveform(
        READOUT_BUS,
        READOUT_PULSE,
        IQPair(I=Square(amplitude=0.1, duration=1000), Q=Square(amplitude=0.0, duration=1000)),
    )
    write_calibration_file(
        calibration,
        CalibrationFile(
            operating_points_version=1,
            operating_points={
                "q1": {
                    "reset": _point(IDLE, readout_levels_pulse),
                    "sweet_spot": _point(SWEET_SPOT, "readout-sweet_spot"),
                },
            },
            experiments={
                "landau_zener": {
                    "q1": ExperimentSection(
                        operating_point="reset",
                        pulses={Role.readout: READOUT_PULSE},
                        **({} if simulated_points is None else {"simulated_points": simulated_points}),
                    )
                }
            },
        ),
    )
    return calibration


def _make(
    calibration: Calibration, platform: LandauZenerPlatform, thresholds: BenchmarkThresholds | None = None, **params
) -> LandauZenerNode:
    return LandauZenerNode(
        name="lz",
        idx=0,
        targets=[Q1],
        parameters={Q1: {**PARAMS, **params}},
        context=build_context(calibration, platform),
        thresholds={Q1: thresholds} if thresholds else None,
    )


@pytest.fixture
def builds(monkeypatch) -> list[dict]:
    """Spy on the real QProgram builder; stub the stream, the fit and the output/save."""
    recorded: list[dict] = []
    real_build = lz_module.generalized_linear_ramps

    def spy(**kwargs):
        recorded.append(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(lz_module, "generalized_linear_ramps", spy)
    monkeypatch.setattr(lz_module, "LandauZenerFit", FakeLandauZenerFit)
    monkeypatch.setattr(FakeLandauZenerFit, "last", {})
    monkeypatch.setattr(FakeLandauZenerFit, "saves", 0)
    stub_output_and_save(monkeypatch)
    return recorded


def test_sweeps_each_phi_x_row_over_the_ramp_durations_and_reports_the_score(builds):
    calibration = _calibration()
    platform = LandauZenerPlatform()
    node = _make(calibration, platform)

    failed = node.run({})

    assert failed == []
    # One QProgram per (phi_x, T): rows sorted by phi_x, durations inner, at the step's operating point.
    rows = sorted(SIMULATED_POINTS, key=lambda row: row["phi_x"])
    assert len(builds) == len(rows) * len(RAMP_DURATIONS)
    expected = [(row, duration) for row in rows for duration in RAMP_DURATIONS]
    for build, (row, duration) in zip(builds, expected, strict=True):
        assert build["calibration"] is calibration
        assert build["target"] == "q1"
        assert (build["idle_operating_point"], build["reset_point"], build["readout_point"]) == (
            "reset",
            "reset",
            "reset",
        )
        # The z ramp alone, from z_start to z_end at the row's phi_x, played in T (ADR-0012).
        path = build["flux_path"]
        assert (path.start, path.end, path.duration) == (
            (row["z_start"], row["phi_x"]),
            (row["z_end"], row["phi_x"]),
            duration,
        )
        assert (build["averages"], build["relax_duration"], build["readout_ringup_time"]) == (1_000, 1_000, 0)
        # A YAML [z, x] ramp duration reaches the QProgram as its (z, x) pair; the move order is defaulted.
        assert build["entry_ramp_duration"] == (40, 60)
        assert build["exit_ramp_duration"] == 50
        assert (build["entry_ramp_move_first"], build["exit_ramp_move_first"]) == ("z", "both")
    # Each QProgram ran with the step's AC crosstalk on the fast-flux lines, on a copy of the calibration.
    assert platform.executions == len(builds)
    for execution in platform.runs:
        compiled_with = execution["calibration"]
        assert compiled_with is not calibration
        assert set(compiled_with.crosstalk_matrix_ac.matrix) == set(FAST_FLUX_BUSES)
        assert compiled_with.crosstalk_matrix_ac.matrix["fast_flux_z_q1"]["fast_flux_x_q1"] == pytest.approx(0.1)
        # The readout line sits at the operating point's LO and attenuation; no drive is touched.
        assert execution["values"][READOUT_BUS, Parameter.LO_FREQUENCY] == int(READOUT_LO)
        assert execution["values"][READOUT_BUS, Parameter.OUT0_ATT] == int(READOUT_ATT)
        assert all(alias != "drive_q1" for alias, _ in execution["values"])
    assert calibration.crosstalk_matrix_ac is None
    # The stream is the phi_x-by-T surface: IQ and the threshold per point, phi_x the outer loop.
    stream = FakeStreamArray.last
    assert stream is not None
    assert stream.kwargs["shape"] == (2, 4, 3)
    assert list(stream.kwargs["loops"]) == ["phi_x", "ramp_duration"]
    assert stream.kwargs["loops"]["phi_x"]["array"] == [0.2, 0.25]
    assert stream.kwargs["loops"]["ramp_duration"]["array"] == RAMP_DURATIONS
    assert ((1, 3, 2), THRESHOLD) in stream.writes
    # analyze scored with the operating point's readout levels and each row's Delta and Delta-epsilon, in sweep order.
    assert FakeLandauZenerFit.last == {
        "measurement_id": FakeStreamArray.measurement_id,
        "target": "q1",
        "level_0": LEVEL_0,
        "level_1": LEVEL_1,
        "min_gap": [0.05, 0.065],
        "detuning_span": [1.9, 1.85],
    }
    assert FakeLandauZenerFit.saves == 1
    # No thresholds in this step: the value is reported, but left unscored.
    (unscored,) = node.unscored_benchmarks
    assert (unscored.value, unscored.uncertainty, unscored.reason) == (SCORE.value, SCORE.uncertainty, "no thresholds")


@pytest.mark.parametrize(
    ("passed", "warning", "verdict"),
    [(0.8, 0.5, Verdict.passed), (0.95, 0.85, Verdict.warning), (0.99, 0.95, Verdict.failed)],
)
def test_the_score_is_judged_higher_is_better(builds, passed, warning, verdict):
    thresholds = BenchmarkThresholds(
        passed=passed, warning=warning, unit=None, direction=BenchmarkDirection.higher_is_better
    )
    node = _make(_calibration(), LandauZenerPlatform(), thresholds=thresholds)

    assert node.run({}) == []

    (result,) = node.benchmark_results
    assert result.verdict is verdict


def test_the_reset_point_can_be_another_operating_point(builds):
    _make(_calibration(), LandauZenerPlatform(), reset_point="sweet_spot").run({})

    assert {build["reset_point"] for build in builds} == {"sweet_spot"}
    assert {build["readout_point"] for build in builds} == {"reset"}


def test_without_a_step_matrix_the_calibrations_ac_crosstalk_is_used(builds):
    calibration = _calibration()
    calibration.crosstalk_matrix_ac = CrosstalkMatrix.from_array(buses=FAST_FLUX_BUSES, matrix_array=np.eye(2))
    platform = LandauZenerPlatform()

    failed = _make(calibration, platform, ac_crosstalk=None).run({})

    assert failed == []
    assert all(execution["calibration"] is calibration for execution in platform.runs)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"ac_crosstalk": None}, id="no_ac_crosstalk"),
        pytest.param({"entry_ramp_duration": None}, id="ramp_needs_an_entry_ramp"),
        pytest.param({"ramp_durations": [10, 20, 40]}, id="too_few_durations"),
        pytest.param({"ramp_durations": [10, 40, 20, 80]}, id="durations_not_increasing"),
        pytest.param({"ramp_durations": [10, 20.5, 40, 80]}, id="fractional_duration"),
        pytest.param({"ramp_durations": [2, 20, 40, 80]}, id="duration_under_4_ns"),
    ],
)
def test_an_unrunnable_step_fails_the_target_before_measuring(builds, params):
    platform = LandauZenerPlatform()

    failed = _make(_calibration(), platform, **params).run({})

    assert failed == ["q1"]
    assert platform.executions == 0
    assert FakeStreamArray.last is None


@pytest.mark.parametrize(
    "simulated_points",
    [
        pytest.param(None, id="no_table"),
        pytest.param([], id="empty_table"),
        pytest.param([{**SIMULATED_POINTS[0], "min_gap": -0.01}], id="negative_gap"),
        pytest.param([{**SIMULATED_POINTS[0], "z_end": SIMULATED_POINTS[0]["z_start"]}], id="no_ramp"),
        pytest.param([{**SIMULATED_POINTS[0], "flux": 0.1}], id="unknown_key"),
        pytest.param([SIMULATED_POINTS[0], {**SIMULATED_POINTS[1], "phi_x": 0.25}], id="repeated_phi_x"),
    ],
)
def test_a_missing_or_malformed_table_fails_the_target_before_measuring(builds, simulated_points):
    """Without a usable table there is no ramp to sweep and no theory to score against (ADR-0012)."""
    platform = LandauZenerPlatform()

    failed = _make(_calibration(simulated_points=simulated_points), platform).run({})

    assert failed == ["q1"]
    assert platform.executions == 0
    assert FakeStreamArray.last is None


def test_ac_crosstalk_with_flux_offsets_is_refused(builds):
    calibration = _calibration()
    matrix = CrosstalkMatrix.from_array(buses=FAST_FLUX_BUSES, matrix_array=np.eye(2))
    matrix.flux_offsets["fast_flux_z_q1"] = 0.01
    calibration.crosstalk_matrix_ac = matrix
    platform = LandauZenerPlatform()

    failed = _make(calibration, platform, ac_crosstalk=None).run({})

    assert failed == ["q1"]
    assert platform.executions == 0


def test_refuses_readout_levels_taken_with_a_different_pulse_before_measuring(builds):
    # Levels stored under a different pulse than the QProgram plays -> ADR-0001 D8 refusal, before the sweep.
    platform = LandauZenerPlatform()
    node = _make(_calibration(readout_levels_pulse="readout-other"), platform)

    failed = node.run({})

    assert failed == ["q1"]  # the target fails the step; no benchmark is recorded
    assert platform.executions == 0
    assert FakeStreamArray.last is None
    assert FakeLandauZenerFit.last == {}
    assert node.benchmark_results == ()


def test_a_step_naming_another_point_idles_reads_out_and_normalises_there(builds):
    """The section names ``reset``; a step naming ``sweet_spot`` runs there (QHC-1936)."""
    calibration = _calibration()
    calibration.add_waveform(
        READOUT_BUS,
        "readout-sweet_spot",
        IQPair(I=Square(amplitude=0.1, duration=1000), Q=Square(amplitude=0.0, duration=1000)),
    )

    failed = _make(calibration, LandauZenerPlatform(), operating_point="sweet_spot").run({})

    assert failed == []
    assert {(build["idle_operating_point"], build["readout_point"]) for build in builds} == {
        ("sweet_spot", "sweet_spot")
    }
    # Normalised with sweet_spot's levels, tagged readout-sweet_spot; reset's (readout-reset) would be refused.
    assert FakeLandauZenerFit.last["level_1"] == LEVEL_1


def test_a_missing_readout_pulse_fails_the_target_before_measuring(builds):
    platform = LandauZenerPlatform()

    # No readout-sweet_spot pulse is stored.
    failed = _make(_calibration(), platform, operating_point="sweet_spot").run({})

    assert failed == ["q1"]
    assert builds == []
    assert platform.executions == 0
