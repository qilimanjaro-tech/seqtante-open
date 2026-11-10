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

"""The in-memory record of a run, from which the PDF report is built (ADR-0009).

A run collects everything the report shows into a :class:`RunResults` while it is alive: the
timestamp and versions, each target's operating points before and after and its fit plots, each
benchmark's value/uncertainty/verdict/threshold, the steps that failed or were skipped with the
reason, and the reset-point move log. The report (:mod:`seqtante_open.report`, an optional extra)
reads this record and writes a PDF; the record itself carries **no** PDF dependency, so the run
assembles it whether or not the report extra is installed. It reads nothing back from the database
(ADR-0009); the run id is recorded only so the PDF can cross-reference it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from seqtante_open.calibration.model import OperatingPoint
    from seqtante_open.controllers.verdicts import BenchmarkDirection, Verdict


@dataclass(frozen=True)
class BenchmarkRow:
    """One benchmark's row in the report table (ADR-0002).

    Args:
        target (str): The target the benchmark ran on.
        experiment (str): The experiment that produced it.
        value (float): The measured value, in ``unit``.
        uncertainty (float | None): The standard uncertainty on ``value``, or ``None`` when the
            analysis gives none.
        verdict (Verdict): Passed / Warning / Failed against the tree's thresholds.
        passed (float): The Passed threshold, in ``unit``.
        warning (float): The Warning threshold, in ``unit``.
        unit (str | None): The unit of the value and thresholds, ``None`` when dimensionless.
        direction (BenchmarkDirection): Whether a higher or a lower value is better.
        plot (str | None): Path to the benchmark's plot, when one was produced.
    """

    target: str
    experiment: str
    value: float
    uncertainty: float | None
    verdict: Verdict
    passed: float
    warning: float
    unit: str | None
    direction: BenchmarkDirection
    plot: str | None = None


@dataclass(frozen=True)
class StepOutcome:
    """A target that did not complete a step, and why (ADR-0006).

    Args:
        target (str): The target.
        step (str): The step (node) name it stopped at.
        reason (str): Why it stopped -- a failure, or a skip with its cause.
    """

    target: str
    step: str
    reason: str


@dataclass
class TargetResults:
    """One target's per-target results for the report (ADR-0001, ADR-0006).

    Args:
        target (str): The target.
        operating_point_before (OperatingPoint | None): Its operating point at the start of the run.
        operating_point_after (OperatingPoint | None): Its operating point after the run.
        fit_plots (list[str]): Paths to the fit plots produced for it, in order.
        reset_point_moves (list[str]): The reset-point search's move log for it, when it ran one.
    """

    target: str
    operating_point_before: OperatingPoint | None = None
    operating_point_after: OperatingPoint | None = None
    fit_plots: list[str] = field(default_factory=list)
    reset_point_moves: list[str] = field(default_factory=list)


@dataclass
class RunResults:
    """Everything the PDF report shows, collected by the run while it is alive (ADR-0009).

    Args:
        timestamp (datetime): When the run started.
        seqtante_open_version (str): The Seqtante-Open version that produced the run.
        qililab_version (str): The qililab version it ran against.
        run_id (int | None): The database run id the ``outputs`` service opened, for cross-reference.
        targets (dict[str, TargetResults]): Per-target results, keyed by target name.
        benchmarks (list[BenchmarkRow]): The benchmark table, one row per benchmark result.
        outcomes (list[StepOutcome]): Every target that failed or was skipped at a step, with the reason.
    """

    timestamp: datetime
    seqtante_open_version: str
    qililab_version: str
    run_id: int | None = None
    targets: dict[str, TargetResults] = field(default_factory=dict)
    benchmarks: list[BenchmarkRow] = field(default_factory=list)
    outcomes: list[StepOutcome] = field(default_factory=list)

    def target(self, name: str) -> TargetResults:
        """The results for ``name``, creating an empty record on first access."""
        if name not in self.targets:
            self.targets[name] = TargetResults(target=name)
        return self.targets[name]
