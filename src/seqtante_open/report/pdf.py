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

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

if TYPE_CHECKING:
    from seqtante_open.calibration.model import OperatingPoint
    from seqtante_open.run_results import BenchmarkRow, RunResults

_VERDICT_COLOR = {"passed": colors.green, "warning": colors.orange, "failed": colors.red}
_PLOT_WIDTH = 15 * cm


def write_report(results: RunResults, path: str) -> None:
    """Write the PDF report for ``results`` to ``path`` (ADR-0009)."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Meta", parent=styles["Normal"], textColor=colors.grey, fontSize=9))
    doc = SimpleDocTemplate(path, pagesize=A4, title="Seqtante-Open calibration report")

    run_id = "—" if results.run_id is None else str(results.run_id)
    story: list = [
        Paragraph("Seqtante-Open calibration report", styles["Title"]),
        Paragraph(
            f"{results.timestamp:%Y-%m-%d %H:%M:%S} &nbsp;·&nbsp; "
            f"Seqtante-Open {results.seqtante_open_version} &nbsp;·&nbsp; qililab {results.qililab_version} "
            f"&nbsp;·&nbsp; run {run_id}",
            styles["Meta"],
        ),
        Spacer(1, 0.6 * cm),
    ]

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
    table = Table(rows, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2A44")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for index, row in enumerate(results.benchmarks, start=1):
        style.append(("TEXTCOLOR", (3, index), (3, index), _VERDICT_COLOR.get(str(row.verdict), colors.black)))
    table.setStyle(TableStyle(style))
    story.extend([table, Spacer(1, 0.5 * cm)])


def _benchmark_row_cells(row: BenchmarkRow) -> list[str]:
    value = f"{row.value:.4g}"
    if row.uncertainty is not None:
        value += f" ± {row.uncertainty:.2g}"
    if row.unit:
        value += f" {row.unit}"
    unit = f" {row.unit}" if row.unit else ""
    better = "higher is better" if str(row.direction) == "higher_is_better" else "lower is better"
    thresholds = f"passed {row.passed:g}{unit}, warning {row.warning:g}{unit} ({better})"
    return [row.target, row.experiment, value, str(row.verdict).capitalize(), thresholds]


def _outcomes_section(story: list, results: RunResults, styles) -> None:
    story.append(Paragraph("Failed and skipped steps", styles["Heading2"]))
    if not results.outcomes:
        story.extend([Paragraph("Every target completed every step it ran.", styles["Normal"]), Spacer(1, 0.4 * cm)])
        return
    rows = [["Target", "Step", "Reason"]] + [[o.target, o.step, o.reason] for o in results.outcomes]
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2A44")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ]
        )
    )
    story.extend([table, Spacer(1, 0.5 * cm)])


def _targets_section(story: list, results: RunResults, styles) -> None:
    if not results.targets:
        return
    story.extend([PageBreak(), Paragraph("Per target", styles["Heading2"])])
    for name in sorted(results.targets):
        target = results.targets[name]
        story.extend(
            [
                Paragraph(name, styles["Heading3"]),
                Paragraph(
                    f"<b>Operating point before:</b> {_format_operating_point(target.operating_point_before)}",
                    styles["Normal"],
                ),
                Paragraph(
                    f"<b>Operating point after:</b> {_format_operating_point(target.operating_point_after)}",
                    styles["Normal"],
                ),
            ]
        )
        if target.reset_point_moves:
            story.append(Paragraph("<b>Reset-point move log:</b>", styles["Normal"]))
            story.extend(Paragraph(f"• {move}", styles["Normal"]) for move in target.reset_point_moves)
        for plot in target.fit_plots:
            if os.path.exists(plot):
                story.extend(
                    [
                        Spacer(1, 0.2 * cm),
                        Image(plot, width=_PLOT_WIDTH, height=_PLOT_WIDTH * 0.5, kind="proportional"),
                    ]
                )
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
