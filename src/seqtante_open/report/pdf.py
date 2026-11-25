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

"""Render a run's :class:`~seqtante_open.run_results.RunResults` to a PDF (ADR-0009).

Built with reportlab (the ``seqtante_open[report]`` extra). The layout follows ADR-0002's contents:
a header with the timestamp and versions, the benchmark table, the steps that failed or were skipped,
and a per-target section with operating points before and after, fit plots and the reset-point move
log. A section with nothing to show says so rather than being omitted, so a sparse run still reads.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ``PIL`` is imported eagerly (it is part of the report extra, and reportlab needs it to draw PNGs) so
# that a missing pillow fails the report import with ``name == "PIL"`` and the run prints the install
# hint, instead of surfacing only when the first plot is drawn as a generic report failure.

if TYPE_CHECKING:
    from qili_qprograms.calibration.model import OperatingPoint

    from seqtante_open.run_results import BenchmarkRow, RunResults

_VERDICT_COLOR = {"passed": colors.green, "warning": colors.orange, "failed": colors.red}
_NOT_EVALUATED = "Not evaluated"
_PLOT_WIDTH = 15 * cm

#: Shared header/grid styling for every table, so the benchmark and outcome tables cannot drift apart.
#: The benchmark section layers its verdict colours on by styling the verdict cell itself (see below).
_TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2A44")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
)

#: Fixed column widths for the benchmark table, summing under the ~451 pt A4 frame so a long row (a
#: value with its unit, a thresholds string) wraps inside its cell instead of overflowing the frame.
_BENCHMARK_COL_WIDTHS = [55, 95, 95, 55, 150]

#: Column widths for the outcomes table, summing under the same frame; the Reason column is widest
#: because it holds the step's failure message, which can be a long exception string.
_OUTCOME_COL_WIDTHS = [70, 120, 260]

#: Body cells are ``Paragraph``s so their text wraps within ``_BENCHMARK_COL_WIDTHS``. A ``TableStyle``
#: TEXTCOLOR does not reach a ``Paragraph``, so the verdict colour lives in a per-verdict cell style.
_CELL_STYLE = ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=9, leading=11)
_VERDICT_CELL_STYLE = {
    verdict: ParagraphStyle(name=f"Cell-{verdict}", parent=_CELL_STYLE, textColor=color)
    for verdict, color in _VERDICT_COLOR.items()
}
_NOT_EVALUATED_CELL_STYLE = ParagraphStyle(name="Cell-not-evaluated", parent=_CELL_STYLE, textColor=colors.grey)


def write_report(results: RunResults, path: str) -> None:
    """Write the PDF report for ``results`` to ``path`` (ADR-0009)."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Meta", parent=styles["Normal"], textColor=colors.grey, fontSize=9))
    styles.add(ParagraphStyle(name="Abort", parent=styles["Normal"], textColor=colors.red))
    doc = SimpleDocTemplate(path, pagesize=A4, title="Seqtante-Open calibration report")

    # Versions and the run id are escaped: they reach reportlab's paragraph markup, where an unescaped
    # '<', '>' or '&' makes the parser raise and the whole report fail (the reset-point moves and the
    # per-target names below are escaped for the same reason).
    run_id = "—" if results.run_id is None else escape(str(results.run_id))
    story: list = [
        Paragraph("Seqtante-Open calibration report", styles["Title"]),
        Paragraph(
            f"{results.timestamp:%Y-%m-%d %H:%M:%S} &nbsp;·&nbsp; "
            f"Seqtante-Open {escape(results.seqtante_open_version)} &nbsp;·&nbsp; "
            f"qililab {escape(results.qililab_version)} &nbsp;·&nbsp; run {run_id}",
            styles["Meta"],
        ),
        Spacer(1, 0.6 * cm),
    ]
    if results.aborted:
        story.extend(
            [
                Paragraph(
                    "<b>This run was stopped before it finished</b> (an error or a manual interrupt), so the "
                    "results below are partial: any step after the stop never ran and is not listed.",
                    styles["Abort"],
                ),
                Spacer(1, 0.4 * cm),
            ]
        )

    _benchmarks_section(story, results, styles)
    _outcomes_section(story, results, styles)
    _targets_section(story, results, styles)

    doc.build(story)


def _benchmarks_section(story: list, results: RunResults, styles) -> None:
    story.append(Paragraph("Benchmarks", styles["Heading2"]))
    if not results.benchmarks:
        story.extend([Paragraph("No benchmarks in this run.", styles["Normal"]), Spacer(1, 0.4 * cm)])
        return
    header = ["Target", "Experiment", "Value", "Verdict", "Thresholds"]
    rows = [header] + [_benchmark_row_cells(row) for row in results.benchmarks]
    table = Table(rows, hAlign="LEFT", colWidths=_BENCHMARK_COL_WIDTHS)
    table.setStyle(_TABLE_STYLE)
    story.extend([table, Spacer(1, 0.5 * cm)])
    for row in results.benchmarks:
        if row.plot and os.path.exists(row.plot):
            story.extend(
                [
                    Paragraph(f"<b>{escape(row.target)} · {escape(row.experiment)}</b>", styles["Normal"]),
                    Spacer(1, 0.1 * cm),
                    _plot(row.plot, styles),
                    Spacer(1, 0.4 * cm),
                ]
            )


def _plot(path: str, styles) -> Image | Paragraph:
    """The plot at ``path`` as an image, or a placeholder line when the file cannot be read.

    reportlab reads an ``Image`` lazily, only when the document is built, so a truncated or corrupt PNG
    (kaleido crashing mid-write) would raise inside ``doc.build`` and cost the whole report. Decoding it
    with PIL first moves that failure here, where it costs only this one plot.
    """
    try:
        with PILImage.open(path) as image:
            image.load()
    except Exception:  # ruff: ignore[blind-except] -- one unreadable plot must not cost the whole report
        return Paragraph(f"<i>Plot could not be read: {escape(os.path.basename(path))}</i>", styles["Meta"])
    return Image(path, width=_PLOT_WIDTH, height=_PLOT_WIDTH * 0.5, kind="proportional")


def _cell(text: object) -> Paragraph:
    """A wrapping, markup-escaped table cell."""
    return Paragraph(escape(str(text)), _CELL_STYLE)


def _verdict_cell(verdict: object) -> Paragraph:
    """The verdict cell, coloured by verdict (the colour cannot come from the shared ``TableStyle``).

    ``None`` is a benchmark that was measured but could not be scored, shown greyed as "Not evaluated".
    """
    if verdict is None:
        return Paragraph(_NOT_EVALUATED, _NOT_EVALUATED_CELL_STYLE)
    style = _VERDICT_CELL_STYLE.get(str(verdict), _CELL_STYLE)
    return Paragraph(escape(str(verdict).capitalize()), style)


def _benchmark_row_cells(row: BenchmarkRow) -> list[Paragraph]:
    unit = f" {row.unit}" if row.unit else ""
    if row.value is None:
        value = "—"
    else:
        value = f"{row.value:.4g}"
        if row.uncertainty is not None:
            value += f" ± {row.uncertainty:.2g}"
        value += unit
    if row.passed is None or row.warning is None:
        # Not evaluated: the thresholds column says why instead (no thresholds set, no measured value).
        thresholds = row.note or "—"
    else:
        better = "higher is better" if str(row.direction) == "higher_is_better" else "lower is better"
        thresholds = f"passed {row.passed:g}{unit}, warning {row.warning:g}{unit} ({better})"
    return [_cell(row.target), _cell(row.experiment), _cell(value), _verdict_cell(row.verdict), _cell(thresholds)]


def _outcomes_section(story: list, results: RunResults, styles) -> None:
    story.append(Paragraph("Failed and skipped steps", styles["Heading2"]))
    if not results.outcomes:
        message = (
            "No step failed or was skipped before the run was stopped."
            if results.aborted
            else "Every target completed every step it ran."
        )
        story.extend([Paragraph(message, styles["Normal"]), Spacer(1, 0.4 * cm)])
        return
    header = ["Target", "Step", "Reason"]
    # Body cells are wrapping, escaped Paragraphs (like the benchmark table) so a long reason -- an
    # exception message -- wraps inside its column instead of overflowing the A4 frame.
    rows = [header] + [[_cell(o.target), _cell(o.step), _cell(o.reason)] for o in results.outcomes]
    table = Table(rows, hAlign="LEFT", colWidths=_OUTCOME_COL_WIDTHS)
    table.setStyle(_TABLE_STYLE)
    story.extend([table, Spacer(1, 0.5 * cm)])


def _targets_section(story: list, results: RunResults, styles) -> None:
    if not results.targets:
        # No page break here: a second page that only says "nothing to show" is noise.
        story.extend(
            [
                Paragraph("Per target", styles["Heading2"]),
                Paragraph("No per-target details for this run.", styles["Normal"]),
                Spacer(1, 0.4 * cm),
            ]
        )
        return
    story.extend([PageBreak(), Paragraph("Per target", styles["Heading2"])])
    for name in sorted(results.targets):
        target = results.targets[name]
        story.extend(
            [
                Paragraph(escape(name), styles["Heading3"]),
                # Escaped as a whole: bus names in the parked-flux list are free text.
                Paragraph(
                    f"<b>Operating point before:</b> {escape(_format_operating_point(target.operating_point_before))}",
                    styles["Normal"],
                ),
                Paragraph(
                    f"<b>Operating point after:</b> {escape(_format_operating_point(target.operating_point_after))}",
                    styles["Normal"],
                ),
            ]
        )
        if target.reset_point_moves:
            story.append(Paragraph("<b>Reset-point move log:</b>", styles["Normal"]))
            story.extend(Paragraph(f"• {escape(move)}", styles["Normal"]) for move in target.reset_point_moves)
        for plot in target.fit_plots:
            if os.path.exists(plot):
                story.extend([Spacer(1, 0.2 * cm), _plot(plot, styles)])
        story.append(Spacer(1, 0.5 * cm))


def _format_operating_point(point: OperatingPoint | None) -> str:
    if point is None:
        return "—"
    parts = [f"flux z={point.flux.z:g}"]
    if point.flux.x is not None:
        parts.append(f"x={point.flux.x:g}")
    parts.append(f"drive LO={point.drive.lo:g}+IF={point.drive.if_:g}")
    parts.append(f"readout LO={point.readout.lo:g}+IF={point.readout.if_:g}")
    if point.parked_flux:
        parked = ", ".join(f"{bus}={flux.z:g}" for bus, flux in point.parked_flux.items())
        parts.append(f"parks [{parked}]")
    return "; ".join(parts)
