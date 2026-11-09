"""Tests for the CalibrationNode base contract and the run context (ADR-0004)."""

import pytest
from qililab.qprogram.calibration import Calibration

from seqtante_open.controllers.calibration_node import CalibrationNode, RunContext
from seqtante_open.targets import ExperimentTarget


def _context() -> RunContext:
    # The platform is only stored here, never driven, so a sentinel stands in for it.
    return RunContext(
        platform=object(),  # type: ignore[arg-type]
        platform_path="runcard.yml",
        calibration=Calibration(),
        calibration_path="calibration.yml",
    )


def test_run_context_holds_its_fields():
    calibration = Calibration()
    context = RunContext(
        platform=object(),  # type: ignore[arg-type]
        platform_path="runcard.yml",
        calibration=calibration,
        calibration_path="calibration.yml",
    )
    assert context.platform_path == "runcard.yml"
    assert context.calibration is calibration
    assert context.calibration_path == "calibration.yml"


def test_base_run_raises_until_the_lifecycle_pr():
    # run is a concrete stub on the base, not an abstract method (task 5c fills the body); calling it
    # before then raises NotImplementedError rather than the base being non-instantiable.
    node = CalibrationNode(name="x", idx=0, targets=[], parameters={}, context=_context())
    with pytest.raises(NotImplementedError):
        node.run({})


def test_concrete_node_stores_its_inputs():
    class Example(CalibrationNode):
        experiment = "example"
        analyses = ("a",)

        def run(self, dead_targets):
            return []

    q1 = ExperimentTarget.from_str("q1")
    parameters = {q1: {"amplitude": 0.5}}
    context = _context()
    node = Example(name="step1", idx=3, targets=[q1], parameters=parameters, context=context)

    assert node.name == "step1"
    assert node.idx == 3
    assert node.targets == [q1]
    assert node.parameters is parameters
    assert node.context is context
    assert node.run({}) == []
