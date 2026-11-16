"""Tests for the PDF report rendering (ADR-0009)."""

from datetime import datetime

from PIL import Image

from seqtante_open.calibration.model import DriveFreqs, Flux, OperatingPoint, ReadoutFreqs
from seqtante_open.controllers.verdicts import BenchmarkDirection, Verdict
from seqtante_open.report import write_report
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
