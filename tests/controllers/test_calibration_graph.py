# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for CalibrationGraph: construction, selection, cycle detection and per-target skip propagation."""

import networkx as nx
import pytest

from seqtante_open.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    Flux,
    OperatingPoint,
    ReadoutFreqs,
)
from seqtante_open.controllers.calibration_graph import CalibrationGraph, GraphBuildError
from seqtante_open.controllers.calibration_node import AnalysisResult, CalibrationNode, RunContext
from seqtante_open.controllers.calibration_tree_parser import Pipeline, Step
from seqtante_open.controllers.legacy_node import LegacyNode
from seqtante_open.targets import QubitTarget


def _operating_point() -> OperatingPoint:
    return OperatingPoint(
        flux=Flux(z=0.1),
        drive=DriveFreqs(lo=4e9, if_=1e8),
        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=10.0),
    )


def _calibration_file(calibrated: list[str]) -> CalibrationFile:
    """A calibration file where each named target has an operating point (so any dependency is satisfied)."""
    return CalibrationFile(
        operating_points_version=1,
        operating_points={target: {"sweet": _operating_point()} for target in calibrated},
        experiments={},
    )


def _targets(*ids: int) -> list[QubitTarget]:
    return [QubitTarget(i) for i in ids]


def _step(name: str, pipeline: str, targets: list[QubitTarget], experiment: str = "single_tone") -> Step:
    return Step(
        name=name,
        pipeline=pipeline,
        experiment=experiment,
        analysis="lorentzian_peak",
        targets=list(targets),
        parameters={target: {"dummy": 1} for target in targets},
    )


def _pipeline(name, steps, *, after=None, enabled=True) -> Pipeline:
    return Pipeline(
        name=name,
        after=after or [],
        enabled=enabled,
        targets=steps[0].targets if steps else [],
        steps=steps,
    )


def _build(pipelines, **kwargs) -> CalibrationGraph:
    graph = CalibrationGraph(platform=object(), platform_path="runcard.yml")
    graph.build(pipelines, **kwargs)
    return graph


def _order(graph: CalibrationGraph) -> list[str]:
    return [graph.graph.nodes[idx]["data"].name for idx in nx.topological_sort(graph.graph)]


# ----------------------------- Construction -----------------------------


def test_intra_pipeline_steps_run_in_order():
    pipeline = _pipeline(
        "p", [_step("a", "p", _targets(1)), _step("b", "p", _targets(1)), _step("c", "p", _targets(1))]
    )
    order = _order(_build([pipeline]))
    assert order.index("a") < order.index("b") < order.index("c")


def test_after_adds_a_cross_pipeline_edge():
    a = _pipeline("A", [_step("a1", "A", _targets(1))])
    b = _pipeline("B", [_step("b1", "B", _targets(1))], after=["a1"])
    order = _order(_build([a, b]))
    assert order.index("a1") < order.index("b1")


def test_disabled_pipeline_has_no_nodes():
    a = _pipeline("A", [_step("a1", "A", _targets(1))], enabled=False)
    b = _pipeline("B", [_step("b1", "B", _targets(1))])
    assert _order(_build([a, b])) == ["b1"]


def test_only_selected_pipelines_run():
    a = _pipeline("A", [_step("a1", "A", _targets(1))])
    b = _pipeline("B", [_step("b1", "B", _targets(1))])
    assert _order(_build([a, b], selected_names=["B"])) == ["b1"]


def test_selecting_an_unknown_pipeline_errors():
    a = _pipeline("A", [_step("a1", "A", _targets(1))])
    with pytest.raises(GraphBuildError, match="unknown pipeline"):
        _build([a], selected_names=["Z"])


def test_selecting_a_disabled_pipeline_errors():
    a = _pipeline("A", [_step("a1", "A", _targets(1))], enabled=False)
    with pytest.raises(GraphBuildError, match="disabled"):
        _build([a], selected_names=["A"])


# ------------------------------ Cycles ------------------------------


def test_after_cycle_is_rejected_at_build_time():
    a = _pipeline("A", [_step("a1", "A", _targets(1))], after=["b1"])
    b = _pipeline("B", [_step("b1", "B", _targets(1))], after=["a1"])
    with pytest.raises(GraphBuildError, match="cycle"):
        _build([a, b])


# ------------------------- Skip propagation -------------------------


def _record_runs(monkeypatch, failing: dict[str, set[str]] | None = None) -> dict[str, dict]:
    """Patches LegacyNode.run to record what each node ran, and fail the configured targets."""
    failing = failing or {}
    calls: dict[str, dict] = {}

    def fake_run(self, dead_targets):
        ran = [target for target in self.targets if str(target) not in dead_targets]
        calls[self.name] = {"ran": {str(t) for t in ran}, "dead": set(dead_targets)}
        return [target for target in ran if str(target) in failing.get(self.name, set())]

    monkeypatch.setattr(LegacyNode, "run", fake_run)
    return calls


def test_failure_skips_rest_of_pipeline_and_dependents_but_not_others(monkeypatch):
    calls = _record_runs(monkeypatch, failing={"a1": {"q1"}})
    a = _pipeline("A", [_step("a1", "A", _targets(1, 2)), _step("a2", "A", _targets(1, 2))])
    b = _pipeline("B", [_step("b1", "B", _targets(1, 2))], after=["a1"])
    independent = _pipeline("C", [_step("c1", "C", _targets(1, 2))])
    graph = _build([a, b, independent])

    graph.run_calibration()

    # q1 failed at a1 -> skipped in the rest of A (a2) and in the dependent pipeline B; q2 untouched.
    assert calls["a2"] == {"ran": {"q2"}, "dead": {"q1"}}
    assert calls["b1"] == {"ran": {"q2"}, "dead": {"q1"}}
    # The independent pipeline runs both targets.
    assert calls["c1"]["ran"] == {"q1", "q2"}
    assert ("q1", "a1", "failed") in graph.report
    assert ("q1", "a2", "skipped (dependency failed upstream)") in graph.report


def test_unselected_dependency_satisfied_by_calibration_file(monkeypatch):
    calls = _record_runs(monkeypatch)
    a = _pipeline("A", [_step("a1", "A", _targets(1, 2))], enabled=False)  # not run this time
    b = _pipeline("B", [_step("b1", "B", _targets(1, 2))], after=["a1"])
    graph = _build([a, b])

    graph.run_calibration(_calibration_file(["q1"]))  # only q1 is in the calibration file

    assert calls["b1"]["ran"] == {"q1"}  # q1 satisfied by the file
    assert calls["b1"]["dead"] == {"q2"}  # q2 absent -> blocked
    assert any(entry[0] == "q2" and "missing dependency" in entry[2] for entry in graph.report)


def test_a_missing_dependency_also_blocks_the_next_step(monkeypatch):
    calls = _record_runs(monkeypatch)
    a = _pipeline("A", [_step("a1", "A", _targets(1, 2))], enabled=False)  # not run this time
    b = _pipeline("B", [_step("b1", "B", _targets(1, 2)), _step("b2", "B", _targets(1, 2))], after=["a1"])
    graph = _build([a, b])

    graph.run_calibration(_calibration_file(["q1"]))  # q2's dependency is missing

    # q2 is blocked at b1 for the missing dependency, and must stay blocked at b2.
    assert calls["b1"] == {"ran": {"q1"}, "dead": {"q2"}}
    assert calls["b2"] == {"ran": {"q1"}, "dead": {"q2"}}


def test_a_target_first_appearing_in_a_later_step_is_checked(monkeypatch):
    calls = _record_runs(monkeypatch)
    a = _pipeline("A", [_step("a1", "A", _targets(1))], enabled=False)  # not run this time
    # q2 first appears at b2, never at the head b1.
    b = _pipeline("B", [_step("b1", "B", _targets(1)), _step("b2", "B", _targets(1, 2))], after=["a1"])
    graph = _build([a, b])

    graph.run_calibration(_calibration_file(["q1"]))  # q2's dependency is missing

    assert calls["b1"]["ran"] == {"q1"}
    assert calls["b2"] == {"ran": {"q1"}, "dead": {"q2"}}  # q2 checked at b2 and blocked
    assert any(entry[0] == "q2" and entry[1] == "b2" and "missing dependency" in entry[2] for entry in graph.report)


# --------------------------- building real nodes ---------------------------


def test_graph_builds_the_registry_node_class_with_the_context():
    class MyNode(CalibrationNode):
        experiment = "mynode"
        analyses = ("a",)

        def execute_node(self, targets):
            return {}

        def analyze(self, target, measurements):
            return AnalysisResult()

        def apply(self, target, result):
            pass

    context = RunContext(platform=object(), platform_path="runcard.yml", calibration=object(), calibration_path="c.yml")
    pipeline = _pipeline("p", [_step("s", "p", _targets(1), experiment="mynode")])

    graph = _build([pipeline], context=context)

    node = graph.graph.nodes[0]["data"]
    assert isinstance(node, MyNode)
    assert node.context is context


def test_graph_falls_back_to_legacy_when_no_node_class():
    # single_tone is seeded but has no node_cls, so even with a context it runs through the adapter.
    context = RunContext(platform=object(), platform_path="runcard.yml", calibration=object(), calibration_path="c.yml")
    pipeline = _pipeline("p", [_step("s", "p", _targets(1), experiment="single_tone")])

    graph = _build([pipeline], context=context)

    assert isinstance(graph.graph.nodes[0]["data"], LegacyNode)


# --------------------------- _validate_node ---------------------------


def test_validate_node_accepts_target_objects():
    node = LegacyNode(
        parameters={QubitTarget(1): {"dummy": 1}},
        platform=object(),
        platform_path="runcard.yml",
        experiment="single_tone",
        name="n",
        idx=0,
        targets=[QubitTarget(1)],
    )
    # A QubitTarget is no longer skipped as an invalid target type.
    assert node._valid_node[QubitTarget(1)] is True


# --------------------------- empty selection ---------------------------


def test_empty_pipeline_selection_errors():
    a = _pipeline("A", [_step("a1", "A", _targets(1))])
    with pytest.raises(GraphBuildError, match="No pipelines to run"):
        _build([a], selected_names=[])


def test_all_pipelines_disabled_errors():
    a = _pipeline("A", [_step("a1", "A", _targets(1))], enabled=False)
    with pytest.raises(GraphBuildError, match="No pipelines to run"):
        _build([a])


# --------------------------- analysis threading ---------------------------


def test_each_step_analysis_reaches_its_node():
    pipeline = _pipeline("p", [_step("a", "p", _targets(1))])  # analysis="lorentzian_peak"
    graph = _build([pipeline])
    assert graph.graph.nodes[0]["data"].analysis == "lorentzian_peak"


# --------------------------- transitive skip ---------------------------


def test_failure_propagates_through_a_node_that_does_not_run_the_target(monkeypatch):
    calls = _record_runs(monkeypatch, failing={"a1": {"q1"}})
    a = _pipeline("A", [_step("a1", "A", _targets(1, 2))])
    b = _pipeline("B", [_step("b1", "B", _targets(2))], after=["a1"])  # B runs only q2
    c = _pipeline("C", [_step("c1", "C", _targets(1, 2))], after=["b1"])  # C runs q1 and q2
    graph = _build([a, b, c])

    graph.run_calibration()

    # q1 failed at a1; b1 never runs q1, but q1 must still be blocked at c1 downstream.
    assert calls["b1"]["ran"] == {"q2"}
    assert calls["c1"]["ran"] == {"q2"}
    assert "q1" not in calls["c1"]["ran"]
