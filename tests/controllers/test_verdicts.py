"""Tests for benchmark verdicts and the verdict scoring (ADR-0002, D17, D20, D21)."""

import pytest

from seqtante_open.controllers.verdicts import (
    AnalysisError,
    BenchmarkDirection,
    BenchmarkResult,
    BenchmarkThresholds,
    Verdict,
    verdict_for,
)

HIGHER = BenchmarkDirection.higher_is_better
LOWER = BenchmarkDirection.lower_is_better


def _thresholds(passed, warning, direction, unit=None):
    return BenchmarkThresholds(passed=passed, warning=warning, unit=unit, direction=direction)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (60e-6, Verdict.passed),  # at or above passed
        (50e-6, Verdict.passed),  # exactly passed
        (35e-6, Verdict.warning),  # between warning and passed
        (20e-6, Verdict.warning),  # exactly warning
        (10e-6, Verdict.failed),  # below warning
    ],
)
def test_higher_is_better_bands(value, expected):
    # T1-like: passed 50 us, warning 20 us, higher is better.
    assert verdict_for(value, _thresholds(50e-6, 20e-6, HIGHER)) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, Verdict.passed),  # at or below passed
        (1.0, Verdict.passed),  # exactly passed
        (3.0, Verdict.warning),  # between passed and warning
        (5.0, Verdict.warning),  # exactly warning
        (8.0, Verdict.failed),  # above warning
    ],
)
def test_lower_is_better_bands(value, expected):
    # error-rate-like: passed 1.0, warning 5.0, lower is better.
    assert verdict_for(value, _thresholds(1.0, 5.0, LOWER)) is expected


def test_benchmark_result_holds_its_fields():
    thresholds = _thresholds(50e-6, 20e-6, HIGHER)
    result = BenchmarkResult(
        target="q1",
        value=42e-6,
        uncertainty=1e-6,
        verdict=verdict_for(42e-6, thresholds),
        thresholds=thresholds,
    )
    assert result.target == "q1"
    assert result.verdict is Verdict.warning
    assert result.uncertainty == pytest.approx(1e-6)
    assert result.thresholds is thresholds


def test_benchmark_result_allows_no_uncertainty():
    thresholds = _thresholds(1.0, 5.0, LOWER)
    result = BenchmarkResult(target="q1", value=0.5, uncertainty=None, verdict=Verdict.passed, thresholds=thresholds)
    assert result.uncertainty is None


def test_analysis_error_is_an_exception():
    with pytest.raises(AnalysisError, match="did not converge"):
        raise AnalysisError("fit did not converge")
