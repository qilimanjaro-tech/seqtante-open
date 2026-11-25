"""Tests for how the run builds and writes the report (ADR-0009), without running a full calibration."""

import importlib.abc
import sys
import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from loguru import logger

from seqtante_open import calibration_run as cr_module
from seqtante_open.calibration_run import CalibrationRun
from seqtante_open.controllers.verdicts import (
    BenchmarkDirection,
    BenchmarkResult,
    BenchmarkThresholds,
    UnscoredBenchmark,
    Verdict,
)
from seqtante_open.run_results import RunResults


class _FakeNode:
    """A graph node stub carrying only what ``_collect_benchmarks`` reads."""

    def __init__(
        self, name, experiment, benchmark_results=(), analysis_plots=None, unscored_benchmarks=(), is_benchmark=True
    ):
        self.name = name
        self.experiment = experiment
        self.benchmark_results = benchmark_results
        self.unscored_benchmarks = unscored_benchmarks
        self.analysis_plots = analysis_plots or {}
        self.is_benchmark = is_benchmark


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
        self.end_run_calls: list[dict] = []

    def end_run(self, **kwargs):
        self.end_run_calls.append(kwargs)


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


def test_collect_results_lists_an_unscored_benchmark_as_not_evaluated(monkeypatch):
    # T1 thresholds set for q1 but not q2: q2 is measured and fitted but cannot be scored. It must
    # still reach the report, as a "Not evaluated" row, rather than vanish from both tables.
    unscored = UnscoredBenchmark(target="q2", value=9e-6, uncertainty=None, reason="no thresholds")
    node = _FakeNode("t1 step", "t1", unscored_benchmarks=(unscored,), analysis_plots={"q2": ("/tmp/q2.png",)})
    run = _run(monkeypatch, graph_nodes={0: node})

    results = run._collect_results()

    assert len(results.benchmarks) == 1
    row = results.benchmarks[0]
    assert (row.target, row.experiment, row.value, row.verdict) == ("q2", "t1", 9e-6, None)
    assert (row.passed, row.warning, row.direction) == (None, None, None)
    assert row.note == "no thresholds"
    assert row.plot == "/tmp/q2.png"


def test_collect_results_puts_recalibration_fit_plots_on_their_targets(monkeypatch):
    # A recalibration node's fit plots go to the per-target section, in graph order; a benchmark's
    # plot is already shown with its benchmark row, so it is not repeated there.
    two_tone = _FakeNode("two_tone", "two_tone", analysis_plots={"q1": ("/tmp/tt.png",)}, is_benchmark=False)
    rabi = _FakeNode(
        "rabi", "rabi", analysis_plots={"q1": ("/tmp/rabi.png",), "q2": ("/tmp/r2.png",)}, is_benchmark=False
    )
    t1 = _FakeNode("t1", "t1", analysis_plots={"q1": ("/tmp/t1.png",)})
    run = _run(monkeypatch, graph_nodes={0: two_tone, 1: rabi, 2: t1})

    results = run._collect_results()

    assert results.targets["q1"].fit_plots == ["/tmp/tt.png", "/tmp/rabi.png"]
    assert results.targets["q2"].fit_plots == ["/tmp/r2.png"]


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


def test_a_non_import_error_while_importing_the_report_does_not_escape(monkeypatch, tmp_path):
    # pdf.py builds its styles at import time; a later reportlab could raise e.g. a TypeError there.
    # _write_report runs in the run's ``finally``, so such an error must be logged, never raised: it
    # would end a clean run with a traceback, or replace an aborted run's real error.
    import reportlab.lib.styles

    run = _run(monkeypatch, storage_path=str(tmp_path))
    _force_reimport_report(monkeypatch)

    def incompatible_style(*args, **kwargs):
        raise TypeError("ParagraphStyle() got an unexpected keyword argument 'leading'")

    monkeypatch.setattr(reportlab.lib.styles, "ParagraphStyle", incompatible_style)

    messages, handle = _capture_warnings()
    try:
        run._write_report()  # must not raise
    finally:
        logger.remove(handle)

    assert not any("seqtante_open[report]" in str(m) for m in messages)
    assert any("not a missing extra" in str(m) for m in messages)
    assert list(tmp_path.glob("report_*.pdf")) == []


@pytest.mark.parametrize(("abort_traceback"), [None, "Traceback: flux conflict"])
def test_end_calibration_passes_the_abort_traceback_to_the_run_entry(monkeypatch, abort_traceback):
    # Without the traceback qililab saves an aborted run as completed, so qompass shows it as clean.
    run = _run(monkeypatch, report=False)
    run.platform = MagicMock()
    run.start = time.time()
    run.abort_traceback = abort_traceback

    run._end_calibration()

    assert cr_module.output_controller.end_run_calls == [{"traceback": abort_traceback}]
