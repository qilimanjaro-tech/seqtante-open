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

"""What an analysis produces: an analysis error, a benchmark verdict, and a benchmark result (ADR-0002).

A step's ``analyze`` is read-only (ADR-0006). It raises :class:`AnalysisError` when it cannot produce
a result -- the fit does not converge, or its quality is below the threshold set in the tree -- and
that target fails the step (D17). A benchmark additionally scores its value against the tree's
Passed/Warning/Failed thresholds and reports a :class:`BenchmarkResult`; a ``Failed`` verdict is a
result, not a step failure, so it skips nothing that comes after (ADR-0002, D17).

The tree holds the thresholds and the parser converts their units; the node declares the direction
(:class:`BenchmarkDirection`). The parser also rejects thresholds ordered against that direction, so
:func:`verdict_for` assumes ``passed`` and ``warning`` make a reachable band.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    """A benchmark's score against the tree's thresholds (ADR-0002)."""

    passed = "passed"
    warning = "warning"
    failed = "failed"


class BenchmarkDirection(StrEnum):
    """Which way is better for a benchmark's value, declared by the node (D20, D21)."""

    higher_is_better = "higher_is_better"
    lower_is_better = "lower_is_better"


class AnalysisError(Exception):
    """An analysis could not produce a result, so the target fails the step (ADR-0006, D17).

    Raised by a step's ``analyze`` when the fit does not converge or its quality is below the tree's
    threshold. Distinct from a ``Failed`` verdict, which is a benchmark result and fails nothing.
    """


@dataclass(frozen=True)
class BenchmarkThresholds:
    """The tree's Passed/Warning thresholds for a benchmark, in the benchmark's unit (ADR-0002).

    The one threshold type the parser produces and :func:`verdict_for` scores against (QHC-1848).
    ``passed`` is the stricter bound and ``warning`` the looser one; which is numerically larger
    depends on ``direction``. The parser converts the tree's units to the node's and rejects a pair
    ordered against the direction (an unreachable band), so a value only ever scores one verdict.

    Args:
        passed (float): A value at least this good is Passed, in ``unit``.
        warning (float): A value at least this good, but not Passed, is Warning; anything worse is Failed.
        unit (str | None): Unit of ``passed`` and ``warning``, ``None`` when the value is dimensionless.
        direction (BenchmarkDirection): Whether a higher or a lower value is better.
    """

    passed: float
    warning: float
    unit: str | None
    direction: BenchmarkDirection


@dataclass(frozen=True)
class BenchmarkResult:
    """A benchmark's value, uncertainty, verdict and the thresholds it was scored against (ADR-0002).

    Carries what the run's results and the PDF report show per benchmark. ``uncertainty`` is the
    standard uncertainty on ``value`` where the analysis gives one, else ``None``. A recalibration
    produces no benchmark result; its ``apply`` writes to the calibration file instead.
    """

    target: str
    value: float
    uncertainty: float | None
    verdict: Verdict
    thresholds: BenchmarkThresholds


def verdict_for(value: float, thresholds: BenchmarkThresholds) -> Verdict:
    """Score ``value`` against ``thresholds`` into Passed, Warning or Failed (ADR-0002).

    For ``higher_is_better`` a value at or above ``passed`` passes and at or above ``warning`` warns;
    for ``lower_is_better`` the comparisons flip. Assumes the parser has checked the thresholds are
    ordered with the direction, so the bands do not overlap.
    """
    if thresholds.direction is BenchmarkDirection.higher_is_better:
        if value >= thresholds.passed:
            return Verdict.passed
        if value >= thresholds.warning:
            return Verdict.warning
        return Verdict.failed
    if thresholds.direction is BenchmarkDirection.lower_is_better:
        if value <= thresholds.passed:
            return Verdict.passed
        if value <= thresholds.warning:
            return Verdict.warning
        return Verdict.failed
    raise ValueError(f"unknown benchmark direction: {thresholds.direction!r}")
