"""Fake-node tests for the three-phase run orchestration (ADR-0006).

These drive ``CalibrationNode.run`` with fake nodes whose phases are scripted, so the orchestration --
phase order, per-target failure gating, benchmark collection, and the crosstalk/restore wrapping -- is
tested without any hardware, fit or disk write.
"""

from typing import ClassVar

import pytest
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter

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
from seqtante_open.controllers.calibration_node import AnalysisResult, CalibrationNode, RunContext
from seqtante_open.controllers.node_registry import BenchmarkSpec
from seqtante_open.controllers.verdicts import (
    AnalysisError,
    BenchmarkDirection,
    BenchmarkResult,
    BenchmarkThresholds,
    Verdict,
)
from seqtante_open.targets import ExperimentTarget

CROSSTALK_BUSES = ("flux_z_q1", "flux_z_q2", "flux_x_q1")
THRESHOLDS = BenchmarkThresholds(passed=1.0, warning=0.5, direction=BenchmarkDirection.higher_is_better)


class FakePlatform:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.calls = []  # ordered call names

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


def _crosstalk() -> CrosstalkMatrix:
    return CrosstalkMatrix.from_buses(
        {b: {bb: (1.0 if b == bb else 0.0) for bb in CROSSTALK_BUSES} for b in CROSSTALK_BUSES}
    )


def _calibration(file: CalibrationFile | None = None) -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = _crosstalk()
    if file is not None:
        write_calibration_file(calibration, file)
    return calibration


def _context(platform=None, calibration=None) -> RunContext:
    return RunContext(
        platform=platform or FakePlatform(),
        platform_path="runcard.yml",
        calibration=calibration if calibration is not None else _calibration(),
        calibration_path="calibration.yml",
    )


def _benchmark_result(target: str, verdict: Verdict) -> BenchmarkResult:
    return BenchmarkResult(target=target, value=1.0, uncertainty=None, verdict=verdict, thresholds=THRESHOLDS)


class ScriptedNode(CalibrationNode):
    """A benchmark node whose phases record their calls; it writes nothing, so no file is read."""

    experiment = "scripted"
    analyses = ("a",)
    applies_operating_point = False
    benchmark = BenchmarkSpec(unit="ns", direction="higher")

    # Per-instance config, set by _make.
    fail: ClassVar[set[str]] = set()
    verdicts: ClassVar[dict[str, Verdict]] = {}
    raise_in_execute: ClassVar[bool] = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.executed: list = []
        self.analyzed: list = []
        self.applied: list = []
        self.seen_flux: dict[str, float] = {}

    def execute_node(self, targets):
        self.executed.extend(targets)  # the base runs one target at a time, so this accumulates
        if self.raise_in_execute:
            raise RuntimeError("hardware boom")
        return {target: [100 + index] for index, target in enumerate(targets)}

    def analyze(self, target, measurements):
        self.analyzed.append(target)
        if str(target) in self.fail:
            raise AnalysisError(f"{target} fit did not converge")
        verdict = self.verdicts.get(str(target), Verdict.passed)
        return AnalysisResult(benchmark=_benchmark_result(str(target), verdict))

    def apply(self, target, result):
        self.applied.append(target)


def _make(cls, *, targets, context=None, **config):
    node = cls(
        name="step",
        idx=0,
        targets=list(targets),
        parameters={target: {} for target in targets},
        context=context or _context(),
    )
    for key, value in config.items():
        setattr(node, key, value)
    return node


Q1 = ExperimentTarget.from_str("q1")
Q2 = ExperimentTarget.from_str("q2")


def test_runs_three_phases_for_each_live_target():
    node = _make(ScriptedNode, targets=[Q1, Q2])
    failed = node.run({})
    assert failed == []
    assert node.executed == [Q1, Q2]
    assert node.analyzed == [Q1, Q2]
    assert node.applied == [Q1, Q2]


def test_dead_targets_are_skipped():
    node = _make(ScriptedNode, targets=[Q1, Q2])
    failed = node.run({"q2": "earlier_step"})
    assert failed == []
    assert node.executed == [Q1]
    assert node.applied == [Q1]


def test_all_dead_returns_without_touching_hardware():
    platform = FakePlatform()
    node = _make(ScriptedNode, targets=[Q1], context=_context(platform=platform))
    failed = node.run({"q1": "earlier_step"})
    assert failed == []
    assert node.executed == []
    assert platform.calls == []


def test_analysis_error_fails_that_target_only():
    node = _make(ScriptedNode, targets=[Q1, Q2], fail={"q1"})
    failed = node.run({})
    assert failed == ["q1"]
    assert node.applied == [Q2]  # q1 never applied


def test_failed_verdict_is_not_a_step_failure():
    node = _make(ScriptedNode, targets=[Q1], verdicts={"q1": Verdict.failed})
    failed = node.run({})
    assert failed == []  # a Failed verdict fails nothing (ADR-0002, D17)
    assert node.applied == [Q1]
    assert [result.verdict for result in node.benchmark_results] == [Verdict.failed]


def test_benchmark_results_are_collected():
    node = _make(ScriptedNode, targets=[Q1, Q2], verdicts={"q1": Verdict.passed, "q2": Verdict.warning})
    node.run({})
    assert {result.target: result.verdict for result in node.benchmark_results} == {
        "q1": Verdict.passed,
        "q2": Verdict.warning,
    }


def test_crosstalk_wraps_the_run():
    platform = FakePlatform()
    node = _make(ScriptedNode, targets=[Q1], context=_context(platform=platform))
    node.run({})
    # Crosstalk installed and flux zeroed before any phase; bias zeroed last.
    assert platform.calls[:2] == ["set_crosstalk", "set_flux_to_zero"]
    assert platform.calls[-1] == "set_bias_to_zero"


def test_bias_is_zeroed_even_when_execute_raises():
    platform = FakePlatform()
    node = _make(ScriptedNode, targets=[Q1], context=_context(platform=platform), raise_in_execute=True)
    failed = node.run({})
    # A hardware error fails the target rather than aborting the node, and bias is still zeroed.
    assert failed == ["q1"]
    assert "set_bias_to_zero" in platform.calls


def test_a_non_analysis_error_fails_that_target_only():
    node = _make(ScriptedNode, targets=[Q1, Q2])
    base_analyze = node.analyze

    def analyze(target, measurements):
        if str(target) == "q1":
            raise RuntimeError("unexpected boom")  # not an AnalysisError
        return base_analyze(target, measurements)

    node.analyze = analyze
    failed = node.run({})
    assert failed == ["q1"]  # the whole run is not aborted
    assert node.applied == [Q2]


def test_a_missing_measurement_fails_that_target_only():
    node = _make(ScriptedNode, targets=[Q1, Q2])

    def execute_node(targets):
        node.executed.extend(targets)
        return {} if any(str(t) == "q1" for t in targets) else {t: [1] for t in targets}

    node.execute_node = execute_node
    failed = node.run({})
    assert failed == ["q1"]  # measurements[q1] KeyError fails q1, not the run
    assert node.applied == [Q2]


def test_restore_handle_is_unavailable_outside_a_run():
    node = _make(ScriptedNode, targets=[Q1])
    with pytest.raises(RuntimeError, match="only available while the node is running"):
        _ = node.restore


# --- operating-point placement and the save path need a calibration with content ---


def _operating_point() -> OperatingPoint:
    return OperatingPoint(
        flux=Flux(z=0.1),
        drive=DriveFreqs(lo=4e9, if_=1e8),
        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=10.0),
    )


def _file_for(experiment: str) -> CalibrationFile:
    return CalibrationFile(
        operating_points_version=1,
        operating_points={"q1": {"sweet": _operating_point()}},
        experiments={experiment: {"q1": ExperimentSection(operating_point="sweet")}},
    )


class PlacingNode(ScriptedNode):
    experiment = "placing"
    applies_operating_point = True

    def execute_node(self, targets):
        self.seen_flux = {
            str(target): self.context.platform.get_parameter(alias=f"flux_z_{target}", parameter=Parameter.FLUX)
            for target in targets
        }
        return super().execute_node(targets)


def test_places_operating_point_before_execute_and_restores_after():
    platform = FakePlatform()
    context = _context(platform=platform, calibration=_calibration(_file_for("placing")))
    node = _make(PlacingNode, targets=[Q1], context=context)

    node.run({})

    # execute_node saw the operating-point flux...
    assert node.seen_flux["q1"] == pytest.approx(0.1)
    # ...and the restore put the bus back afterwards.
    assert platform.get_parameter(alias="flux_z_q1", parameter=Parameter.FLUX) == pytest.approx(0.0)


class RecalibrationNode(ScriptedNode):
    experiment = "recalibrate"
    applies_operating_point = False
    benchmark = None  # a recalibration writes, so run saves once


def test_recalibration_saves_the_calibration_once(monkeypatch):
    saves = []
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: saves.append(path))
    context = _context(calibration=_calibration(_file_for("recalibrate")))
    node = _make(RecalibrationNode, targets=[Q1], context=context)

    node.run({})

    assert saves == ["calibration.yml"]


def test_benchmark_does_not_save(monkeypatch):
    saves = []
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: saves.append(path))
    node = _make(ScriptedNode, targets=[Q1])  # a benchmark

    node.run({})

    assert saves == []
