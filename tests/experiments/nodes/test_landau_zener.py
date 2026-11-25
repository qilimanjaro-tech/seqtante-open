"""Tests for the ``landau_zener`` benchmark node (QHC-1808).

The node runs the full three-phase ``run`` with the real ``generalized_linear_ramps`` QProgram from
``qili-qprograms`` (a spy records its arguments), and a platform that executes nothing: ``StreamArray``,
the normalising fit, the DB and the calibration save are stubbed. The tests check that the node builds one
QProgram per flux path at the step's operating point, compiles them with the AC crosstalk of the fast-flux
lines, sets the readout line, normalises with the operating point's readout levels and -- with no
thresholds set yet -- leaves the benchmark unscored ("Not evaluated").
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

PARAMS = {
    # Two paths, the second walking further out; neither starts or ends at the reset point's flux.
    "flux_paths": [
        [[0.15, 0.25, 100], [0.2, 0.3, 100]],
        [[0.15, 0.25, 100], [0.3, 0.4, 200]],
    ],
    "annealing_times": [200, 300],
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
    """Scripted stand-in for ``LandauZenerFit``: no DB load, records the readout levels it was given."""

    last: ClassVar[dict] = {}

    def __init__(self, measurement_id=None, target=None, level_0=None, level_1=None, path=None):
        type(self).last = {"measurement_id": measurement_id, "target": target, "level_0": level_0, "level_1": level_1}

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


def _calibration(readout_levels_pulse: str = READOUT_PULSE) -> Calibration:
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
                "landau_zener": {"q1": ExperimentSection(operating_point="reset", pulses={Role.readout: READOUT_PULSE})}
            },
        ),
    )
    return calibration


def _make(calibration: Calibration, platform: LandauZenerPlatform, **params) -> LandauZenerNode:
    # No thresholds: the run leaves the benchmark unscored ("Not evaluated") until QHC-1921.
    return LandauZenerNode(
        name="lz",
        idx=0,
        targets=[Q1],
        parameters={Q1: {**PARAMS, **params}},
        context=build_context(calibration, platform),
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
    monkeypatch.setattr(lz_module, "StreamArray", FakeStreamArray)  # the node streams itself, not via _stream
    monkeypatch.setattr(lz_module, "LandauZenerFit", FakeLandauZenerFit)
    monkeypatch.setattr(FakeLandauZenerFit, "last", {})
    stub_output_and_save(monkeypatch)
    return recorded


def test_runs_one_qprogram_per_path_and_leaves_the_benchmark_unscored(builds):
    calibration = _calibration()
    platform = LandauZenerPlatform()
    node = _make(calibration, platform)

    failed = node.run({})

    assert failed == []
    # One QProgram per path, idle and read out at the step's operating point, reset there by default.
    assert len(builds) == 2
    for build, path in zip(builds, PARAMS["flux_paths"], strict=True):
        assert build["calibration"] is calibration
        assert build["target"] == "q1"
        assert (build["idle_operating_point"], build["reset_point"], build["readout_point"]) == (
            "reset",
            "reset",
            "reset",
        )
        assert build["flux_path"] == path
        assert (build["averages"], build["relax_duration"], build["readout_ringup_time"]) == (1_000, 1_000, 0)
        # A YAML [z, x] ramp duration reaches the QProgram as its (z, x) pair; the move order is defaulted.
        assert build["entry_ramp_duration"] == (40, 60)
        assert build["exit_ramp_duration"] == 50
        assert (build["entry_ramp_move_first"], build["exit_ramp_move_first"]) == ("z", "both")
    # Each QProgram ran with the step's AC crosstalk on the fast-flux lines, on a copy of the calibration.
    assert platform.executions == 2
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
    # The stream holds IQ and the threshold per annealing time.
    stream = FakeStreamArray.last
    assert stream is not None
    assert stream.kwargs["shape"] == (2, 3)
    assert stream.kwargs["loops"]["total_annealing_time"]["array"] == PARAMS["annealing_times"]
    assert ((1, 2), THRESHOLD) in stream.writes
    # analyze normalised with the operating point's readout levels, and reported no value.
    assert FakeLandauZenerFit.last == {
        "measurement_id": FakeStreamArray.measurement_id,
        "target": "q1",
        "level_0": LEVEL_0,
        "level_1": LEVEL_1,
    }
    assert node.benchmark_results == ()


def test_the_reset_point_can_be_another_operating_point(builds):
    _make(_calibration(), LandauZenerPlatform(), reset_point="sweet_spot").run({})

    assert [build["reset_point"] for build in builds] == ["sweet_spot", "sweet_spot"]
    assert [build["readout_point"] for build in builds] == ["reset", "reset"]


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
        pytest.param({"flux_paths": []}, id="no_paths"),
        pytest.param({"annealing_times": [200]}, id="paths_and_times_differ"),
        pytest.param({"entry_ramp_duration": None}, id="path_needs_an_entry_ramp"),
    ],
)
def test_an_unrunnable_step_fails_the_target_before_measuring(builds, params):
    platform = LandauZenerPlatform()

    failed = _make(_calibration(), platform, **params).run({})

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


def test_refuses_readout_levels_taken_with_a_different_pulse(builds):
    # Levels stored under a different pulse than the QProgram plays -> ADR-0001 D8 refusal in analyze.
    node = _make(_calibration(readout_levels_pulse="readout-other"), LandauZenerPlatform())

    failed = node.run({})

    assert failed == ["q1"]  # the target fails the step; no benchmark is recorded
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
    assert [(build["idle_operating_point"], build["readout_point"]) for build in builds] == [
        ("sweet_spot", "sweet_spot"),
        ("sweet_spot", "sweet_spot"),
    ]
    # Normalised with sweet_spot's levels, tagged readout-sweet_spot; reset's (readout-reset) would be refused.
    assert FakeLandauZenerFit.last["level_1"] == LEVEL_1


def test_a_missing_readout_pulse_fails_the_target_before_measuring(builds):
    platform = LandauZenerPlatform()

    # No readout-sweet_spot pulse is stored.
    failed = _make(_calibration(), platform, operating_point="sweet_spot").run({})

    assert failed == ["q1"]
    assert builds == []
    assert platform.executions == 0
