"""Tests for how the run builds and writes the report (ADR-0009), without running a full calibration."""

import sys
from datetime import datetime

from loguru import logger

from seqtante_open import calibration_run as cr_module
from seqtante_open.calibration_run import CalibrationRun
from seqtante_open.run_results import RunResults


class _FakeGraph:
    def __init__(self, report):
        self.report = report


class _FakeOutputController:
    def __init__(self, storage_path, run_id=7):
        self.storage_path = storage_path
        self.run_id = run_id


def _run(monkeypatch, *, storage_path=".", run_id=7, graph_report=None, report=True) -> CalibrationRun:
    monkeypatch.setattr(cr_module, "output_controller", _FakeOutputController(storage_path, run_id))
    run = CalibrationRun.__new__(CalibrationRun)  # skip __init__: no platform, no real run
    run.report = report
    run.started_at = datetime(2026, 11, 10, 12, 0, 0)
    run.graph = _FakeGraph(graph_report or [])
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


def test_write_report_writes_a_pdf(monkeypatch, tmp_path):
    run = _run(monkeypatch, storage_path=str(tmp_path), graph_report=[("q1", "two_tone", "failed")])
    run._write_report()
    pdfs = list(tmp_path.glob("report_*.pdf"))
    assert len(pdfs) == 1
    assert pdfs[0].read_bytes().startswith(b"%PDF-")


def test_a_missing_extra_warns_and_does_not_raise(monkeypatch, tmp_path):
    run = _run(monkeypatch, storage_path=str(tmp_path))
    monkeypatch.setitem(sys.modules, "seqtante_open.report", None)  # makes the import raise ImportError
    messages, handle = _capture_warnings()
    try:
        run._write_report()  # must not raise
    finally:
        logger.remove(handle)

    assert any("seqtante_open[report]" in str(m) for m in messages)
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
