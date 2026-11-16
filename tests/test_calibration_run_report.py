"""Tests for how the run builds and writes the report (ADR-0009), without running a full calibration."""

import importlib.abc
import sys
from datetime import datetime

from loguru import logger

from seqtante_open import calibration_run as cr_module
from seqtante_open.calibration_run import CalibrationRun
from seqtante_open.controllers.verdicts import BenchmarkDirection, BenchmarkResult, BenchmarkThresholds, Verdict
from seqtante_open.run_results import RunResults


class _FakeNode:
    """A graph node stub carrying only what ``_collect_benchmarks`` reads."""

    def __init__(self, name, experiment, benchmark_results=(), analysis_plots=None):
        self.name = name
        self.experiment = experiment
        self.benchmark_results = benchmark_results
        self.analysis_plots = analysis_plots or {}


class _FakeNxGraph:
    def __init__(self, nodes):  # nodes: dict[idx, _FakeNode]
        self.nodes = {idx: {"data": node} for idx, node in nodes.items()}


class _FakeGraph:
    def __init__(self, report, nodes=None):
        self.report = report
        if nodes is not None:
            self.graph = _FakeNxGraph(nodes)


class _FakeOutputController:
    def __init__(self, storage_path, run_id=7):
        self.storage_path = storage_path
        self.run_id = run_id


def _run(
    monkeypatch, *, storage_path=".", run_id=7, graph_report=None, graph_nodes=None, report=True
) -> CalibrationRun:
    monkeypatch.setattr(cr_module, "output_controller", _FakeOutputController(storage_path, run_id))
    run = CalibrationRun.__new__(CalibrationRun)  # skip __init__: no platform, no real run
    run.report = report
    run.started_at = datetime(2026, 11, 10, 12, 0, 0)
    run.graph = _FakeGraph(graph_report or [], nodes=graph_nodes)
    return run


def _capture_warnings():
    messages: list[str] = []
    handle = logger.add(messages.append, level="WARNING")
    return messages, handle


def test_collect_results_maps_graph_outcomes(monkeypatch):
    run = _run(monkeypatch, run_id=7, graph_report=[("q1", "two_tone", "failed"), ("q2", "t1", "skipped (dep)")])
    results = run._collect_results()

    assert isinstance(results, RunResults)
    assert results.run_id == 7
    assert isinstance(results.seqtante_open_version, str)
    assert [(o.target, o.step, o.reason) for o in results.outcomes] == [
        ("q1", "two_tone", "failed"),
        ("q2", "t1", "skipped (dep)"),
    ]


def test_run_id_minus_one_becomes_none(monkeypatch):
    run = _run(monkeypatch, run_id=-1)
    assert run._collect_results().run_id is None


def test_collect_results_builds_benchmark_rows_from_the_nodes(monkeypatch):
    thresholds = BenchmarkThresholds(
        passed=1e-5, warning=5e-6, direction=BenchmarkDirection.higher_is_better, unit="ns"
    )
    benchmark = BenchmarkResult(
        target="q1", value=1.2e-5, uncertainty=1e-7, verdict=Verdict.passed, thresholds=thresholds
    )
    node = _FakeNode("t1 step", "t1", benchmark_results=(benchmark,), analysis_plots={"q1": ("/tmp/fit.png",)})
    run = _run(monkeypatch, graph_nodes={0: node})

    results = run._collect_results()

    assert len(results.benchmarks) == 1
    row = results.benchmarks[0]
    assert (row.target, row.experiment, row.value, row.uncertainty) == ("q1", "t1", 1.2e-5, 1e-7)
    assert row.verdict is Verdict.passed
    assert (row.passed, row.warning) == (1e-5, 5e-6)
    assert row.unit == "ns"  # carried on the scored thresholds (QHC-1848)
    assert row.direction is BenchmarkDirection.higher_is_better
    assert row.plot == "/tmp/fit.png"


def test_collect_results_leaves_benchmarks_empty_without_a_graph(monkeypatch):
    # The old fake graph carries no `.graph`; collection must skip it rather than raise.
    run = _run(monkeypatch, graph_report=[("q1", "t1", "failed")])
    assert run._collect_results().benchmarks == []


def test_collect_results_carries_the_aborted_flag(monkeypatch):
    run = _run(monkeypatch)
    assert run._collect_results().aborted is False  # default when the attribute is absent
    run.aborted = True
    assert run._collect_results().aborted is True


def test_write_report_writes_a_pdf(monkeypatch, tmp_path):
    run = _run(monkeypatch, storage_path=str(tmp_path), graph_report=[("q1", "two_tone", "failed")])
    run._write_report()
    pdfs = list(tmp_path.glob("report_*.pdf"))
    assert len(pdfs) == 1
    assert pdfs[0].read_bytes().startswith(b"%PDF-")


def _force_reimport_report(monkeypatch):
    """Drop the cached report package so the next import re-executes and re-hits its reportlab imports."""
    for name in list(sys.modules):
        if name == "seqtante_open.report" or name.startswith("seqtante_open.report."):
            monkeypatch.delitem(sys.modules, name, raising=False)


class _BlockReportlab(importlib.abc.MetaPathFinder):
    """A finder that makes ``import reportlab`` fail as if the extra were not installed."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "reportlab" or fullname.startswith("reportlab."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name="reportlab")
        return


def test_a_missing_extra_warns_and_does_not_raise(monkeypatch, tmp_path):
    run = _run(monkeypatch, storage_path=str(tmp_path))
    # Simulate reportlab not being installed: the report import then fails with name "reportlab",
    # exactly as an absent top-level package would (a missing extra, not a bug in the report module).
    _force_reimport_report(monkeypatch)
    for name in list(sys.modules):
        if name == "reportlab" or name.startswith("reportlab."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockReportlab(), *sys.meta_path])

    messages, handle = _capture_warnings()
    try:
        run._write_report()  # must not raise
    finally:
        logger.remove(handle)

    assert any("seqtante_open[report]" in str(m) for m in messages)
    assert list(tmp_path.glob("report_*.pdf")) == []


def test_a_non_extra_import_error_logs_the_traceback_not_the_install_hint(monkeypatch, tmp_path):
    run = _run(monkeypatch, storage_path=str(tmp_path))
    # A platypus symbol that has "moved": reportlab is installed, but a submodule import fails.
    _force_reimport_report(monkeypatch)
    monkeypatch.setitem(sys.modules, "reportlab.platypus", None)

    messages, handle = _capture_warnings()
    try:
        run._write_report()  # must not raise
    finally:
        logger.remove(handle)

    assert not any("seqtante_open[report]" in str(m) for m in messages)  # do not wrongly suggest the extra
    assert any("not a missing extra" in str(m) for m in messages)
    assert list(tmp_path.glob("report_*.pdf")) == []


def test_a_report_failure_is_swallowed(monkeypatch, tmp_path):
    run = _run(monkeypatch, storage_path=str(tmp_path))

    def boom(results, path):
        raise RuntimeError("pdf engine on fire")

    monkeypatch.setattr("seqtante_open.report.write_report", boom)
    messages, handle = _capture_warnings()
    try:
        run._write_report()  # a report failure must never propagate out of the run
    finally:
        logger.remove(handle)

    assert any("report failed" in str(m).lower() for m in messages)
