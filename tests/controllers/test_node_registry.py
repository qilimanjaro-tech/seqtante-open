"""Tests for the experiment registry and node self-registration (ADR-0004, ADR-0007)."""

import pytest

from seqtante_open.controllers.calibration_node import CalibrationNode
from seqtante_open.controllers.node_registry import (
    BenchmarkSpec,
    Parallel,
    experiment_spec,
    known_experiments,
    register_node,
)
from seqtante_open.controllers.verdicts import BenchmarkDirection

HIGHER = BenchmarkDirection.higher_is_better
LOWER = BenchmarkDirection.lower_is_better


def test_subclass_registers_itself_with_its_class():
    class Rabi(CalibrationNode):
        experiment = "rabi"
        analyses = ("pi_amplitude", "readout_levels")

        def run(self, dead_targets):  # pragma: no cover - not exercised here
            return []

    spec = experiment_spec("rabi")
    assert spec is not None
    assert spec.node_cls is Rabi
    assert spec.analyses == ("pi_amplitude", "readout_levels")
    assert spec.default_analysis == "pi_amplitude"


def test_default_analysis_is_the_first():
    class T1(CalibrationNode):
        experiment = "t1"
        analyses = ("exponential_decay",)

        def run(self, dead_targets):  # pragma: no cover
            return []

    assert experiment_spec("t1").default_analysis == "exponential_decay"


def test_a_spine_without_experiment_does_not_register():
    before = set(known_experiments())

    class FixedFluxNode(CalibrationNode):
        """A mid-level spine: no experiment, so not a registrable leaf."""

        def run(self, dead_targets):  # pragma: no cover
            return []

    assert set(known_experiments()) == before


def test_a_leaf_without_analyses_is_rejected():
    with pytest.raises(ValueError, match="declares no analyses"):

        class Broken(CalibrationNode):
            experiment = "broken"

            def run(self, dead_targets):  # pragma: no cover
                return []


def test_two_classes_cannot_claim_the_same_experiment():
    class First(CalibrationNode):
        experiment = "dup"
        analyses = ("a",)

        def run(self, dead_targets):  # pragma: no cover
            return []

    with pytest.raises(ValueError, match="already implemented by First"):

        class Second(CalibrationNode):
            experiment = "dup"
            analyses = ("b",)

            def run(self, dead_targets):  # pragma: no cover
                return []


def test_legacy_experiments_are_seeded_without_a_node_class():
    # During the port the four legacy experiments are known to the parser but carry no node class;
    # the graph builder (QHC-1800) falls back to their legacy function.
    for experiment in ("single_tone", "two_tone", "offset_calibration", "two_tone_vs_flux"):
        spec = experiment_spec(experiment)
        assert spec is not None
        assert spec.node_cls is None


def test_a_node_class_fills_a_seeded_metadata_entry():
    # "t2" is seeded as metadata only; a node class claiming it fills node_cls and keeps a parser entry.
    assert experiment_spec("t2").node_cls is None

    class T2(CalibrationNode):
        experiment = "t2"
        analyses = ("decaying_oscillation",)

        def run(self, dead_targets):  # pragma: no cover
            return []

    assert experiment_spec("t2").node_cls is T2
    assert "t2" in known_experiments()


def test_unknown_experiment_has_no_spec():
    assert experiment_spec("does_not_exist") is None


def test_a_bare_string_analyses_is_one_analysis_not_its_characters():
    # A subclass that writes `analyses = "flux_offset"` means one analysis, not ("f", "l", "u", ...).
    class SingleAnalysis(CalibrationNode):
        experiment = "single_analysis"
        analyses = "flux_offset"  # type: ignore[assignment]

        def run(self, dead_targets):  # pragma: no cover
            return []

    spec = experiment_spec("single_analysis")
    assert spec.analyses == ("flux_offset",)
    assert spec.default_analysis == "flux_offset"


def test_reregistering_the_same_class_is_allowed():
    # A module reload (importlib.reload / IPython %autoreload) re-runs registration for the same
    # class; that is not a clash and must not raise.
    class Reloaded(CalibrationNode):
        experiment = "reloaded"
        analyses = ("a",)

        def run(self, dead_targets):  # pragma: no cover
            return []

    register_node(Reloaded)  # second registration of the same class
    assert experiment_spec("reloaded").node_cls is Reloaded


def test_a_node_class_keeps_the_seeded_benchmark():
    # "t1" is seeded with a benchmark; a node class that does not re-declare one keeps it.
    assert experiment_spec("t1").benchmark == BenchmarkSpec(unit="ns", direction=HIGHER)

    class T1(CalibrationNode):
        experiment = "t1"
        analyses = ("exponential_decay",)

        def run(self, dead_targets):  # pragma: no cover
            return []

    assert experiment_spec("t1").benchmark == BenchmarkSpec(unit="ns", direction=HIGHER)


def test_a_node_class_declares_its_own_benchmark():
    class Benchmarked(CalibrationNode):
        experiment = "benchmarked"
        analyses = ("a",)
        benchmark = BenchmarkSpec(unit="us", direction=LOWER)

        def run(self, dead_targets):  # pragma: no cover
            return []

    assert experiment_spec("benchmarked").benchmark == BenchmarkSpec(unit="us", direction=LOWER)


def test_default_parallel_is_forbidden():
    class Lonely(CalibrationNode):
        experiment = "lonely"
        analyses = ("a",)

        def run(self, dead_targets):  # pragma: no cover
            return []

    assert Lonely.parallel is Parallel.forbidden
    assert Lonely.applies_operating_point is True
