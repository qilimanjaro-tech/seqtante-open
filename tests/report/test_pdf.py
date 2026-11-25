"""Tests for the PDF report rendering (ADR-0009)."""

import re
from datetime import datetime

from PIL import Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as PlatypusImage
from reportlab.platypus import Paragraph

from seqtante_open.calibration.model import DriveFreqs, Flux, OperatingPoint, ReadoutFreqs
from seqtante_open.controllers.verdicts import BenchmarkDirection, Verdict
from seqtante_open.report import write_report
from seqtante_open.report.pdf import _benchmark_row_cells, _plot
from seqtante_open.run_results import BenchmarkRow, RunResults, StepOutcome


def _operating_point() -> OperatingPoint:
    return OperatingPoint(
        flux=Flux(z=0.1),
        drive=DriveFreqs(lo=4e9, if_=1e8),
        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=10.0),
    )


def _results() -> RunResults:
    results = RunResults(
        timestamp=datetime(2026, 11, 10, 12, 0, 0),
        seqtante_open_version="1.2.3",
        qililab_version="0.35.2",
        run_id=42,
    )
    results.benchmarks.append(
        BenchmarkRow(
            target="q1",
            experiment="t1",
            value=1.2e-5,
            uncertainty=1e-7,
            verdict=Verdict.passed,
            passed=1e-5,
            warning=5e-6,
            unit="ns",
            direction=BenchmarkDirection.higher_is_better,
        )
    )
    results.outcomes.append(StepOutcome(target="q2", step="two_tone", reason="failed"))
    target = results.target("q1")
    target.operating_point_before = _operating_point()
    target.operating_point_after = _operating_point()
    target.reset_point_moves = ["iter 1: flux 0.10 -> 0.12", "iter 2: flux 0.12 -> 0.11"]
    return results


def test_write_report_creates_a_valid_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    write_report(_results(), str(path))
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF-")
    assert path.stat().st_size > 0


def test_write_report_embeds_a_fit_plot(tmp_path):
    plot = tmp_path / "fit.png"
    Image.new("RGB", (8, 8), "white").save(plot)

    results = _results()
    results.target("q1").fit_plots = [str(plot), str(tmp_path / "missing.png")]  # missing one is skipped
    path = tmp_path / "report.pdf"
    write_report(results, str(path))

    assert path.read_bytes().startswith(b"%PDF-")


def test_write_report_renders_a_benchmark_plot(tmp_path):
    plot = tmp_path / "bench.png"
    Image.new("RGB", (8, 8), "white").save(plot)

    results = _results()
    results.benchmarks.append(
        BenchmarkRow(
            target="q2",
            experiment="landau_zener",
            value=0.01,
            uncertainty=None,
            verdict=Verdict.failed,
            passed=1e-3,
            warning=5e-3,
            unit=None,
            direction=BenchmarkDirection.lower_is_better,
            plot=str(plot),
        )
    )
    path = tmp_path / "report.pdf"
    write_report(results, str(path))

    assert path.read_bytes().startswith(b"%PDF-")


def test_write_report_marks_an_aborted_run(tmp_path):
    results = _results()
    results.aborted = True
    path = tmp_path / "report.pdf"
    write_report(results, str(path))  # the aborted banner must render, not crash
    assert path.read_bytes().startswith(b"%PDF-")


def test_write_report_escapes_markup_special_characters(tmp_path):
    # Angle brackets and ampersands reach reportlab's paragraph markup; unescaped, they make the
    # paraparser raise and the whole report fail. They must be escaped everywhere they flow in.
    results = RunResults(
        timestamp=datetime(2026, 11, 10, 12, 0, 0),
        seqtante_open_version="1.0 <dev> & co",
        qililab_version="0.1",
        run_id=None,
    )
    results.benchmarks.append(
        BenchmarkRow(
            target="q1 <a>",
            experiment="t1 & t2",
            value=1.0,
            uncertainty=None,
            verdict=Verdict.passed,
            passed=1.0,
            warning=0.5,
            unit="a<b",
            direction=BenchmarkDirection.higher_is_better,
        )
    )
    target = results.target("q<1>")
    target.reset_point_moves = ["r2 0.75 < 0.8, widening span to 60 MHz & more"]
    path = tmp_path / "report.pdf"
    write_report(results, str(path))  # must not raise despite < > & in the text
    assert path.read_bytes().startswith(b"%PDF-")


def test_write_report_handles_an_empty_run(tmp_path):
    empty = RunResults(
        timestamp=datetime(2026, 11, 10, 12, 0, 0),
        seqtante_open_version="1.2.3",
        qililab_version="0.35.2",
        run_id=None,
    )
    path = tmp_path / "report.pdf"
    write_report(empty, str(path))  # no benchmarks, outcomes or targets -> sections say so, no crash
    assert path.read_bytes().startswith(b"%PDF-")


def _page_count(path) -> int:
    return len(re.findall(rb"/Type /Page\b(?!s)", path.read_bytes()))


def test_an_empty_run_has_no_near_empty_second_page(tmp_path):
    empty = RunResults(timestamp=datetime(2026, 11, 10), seqtante_open_version="1", qililab_version="1")
    path = tmp_path / "report.pdf"
    write_report(empty, str(path))
    assert _page_count(path) == 1  # "No per-target details" stays on page 1


def test_a_run_with_targets_starts_them_on_a_new_page(tmp_path):
    path = tmp_path / "report.pdf"
    write_report(_results(), str(path))
    assert _page_count(path) == 2


def test_an_unreadable_plot_becomes_a_placeholder_not_a_lost_report(tmp_path):
    # kaleido crashing mid-write leaves a truncated PNG; reportlab only reads it in doc.build, where it
    # would raise and lose the whole report. It must cost only that one plot.
    good = tmp_path / "good.png"
    Image.new("RGB", (40, 30), "white").save(good)
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a png")

    styles = {"Meta": getSampleStyleSheet()["Normal"]}
    assert isinstance(_plot(str(good), styles), PlatypusImage)
    for bad in (truncated, junk):
        placeholder = _plot(str(bad), styles)
        assert isinstance(placeholder, Paragraph)
        assert "could not be read" in placeholder.getPlainText()

    results = _results()
    results.target("q1").fit_plots = [str(truncated), str(good)]
    results.benchmarks.append(
        BenchmarkRow(
            target="q2",
            experiment="t1",
            value=1.0,
            uncertainty=None,
            verdict=Verdict.passed,
            passed=1.0,
            warning=0.5,
            unit=None,
            direction=BenchmarkDirection.higher_is_better,
            plot=str(junk),
        )
    )
    path = tmp_path / "report.pdf"
    write_report(results, str(path))
    assert path.read_bytes().startswith(b"%PDF-")


def test_a_not_evaluated_benchmark_renders_with_its_reason(tmp_path):
    row = BenchmarkRow(
        target="q2",
        experiment="t1",
        value=9e-6,
        uncertainty=None,
        verdict=None,
        passed=None,
        warning=None,
        unit="s",
        direction=None,
        note="no thresholds",
    )
    cells = [cell.getPlainText() for cell in _benchmark_row_cells(row)]
    assert cells == ["q2", "t1", "9e-06 s", "Not evaluated", "no thresholds"]

    results = _results()
    results.benchmarks.append(row)
    results.benchmarks.append(
        BenchmarkRow(
            target="q3",
            experiment="t1",
            value=None,
            uncertainty=None,
            verdict=None,
            passed=None,
            warning=None,
            unit="s",
            direction=None,
            note="no measured value",
        )
    )
    path = tmp_path / "report.pdf"
    write_report(results, str(path))
    assert path.read_bytes().startswith(b"%PDF-")


def test_parked_bus_names_are_escaped(tmp_path):
    # model_construct skips the target-name validation, standing in for any parked name with markup.
    base = _operating_point()
    point = OperatingPoint.model_construct(
        flux=base.flux, drive=base.drive, readout=base.readout, parked_flux={"q2 <&> bus": Flux(z=0.3)}
    )
    results = _results()
    results.target("q1").operating_point_after = point
    path = tmp_path / "report.pdf"
    write_report(results, str(path))  # an unescaped '<' or '&' would make the paragraph parser raise
    assert path.read_bytes().startswith(b"%PDF-")
