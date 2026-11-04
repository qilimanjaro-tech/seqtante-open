"""Tests for the experiment registry and node self-registration (ADR-0004, ADR-0007)."""

import pytest

from seqtante_open.controllers import experiment_specs
from seqtante_open.controllers.calibration_node import CalibrationNode
from seqtante_open.controllers.node_registry import Parallel, experiment_spec, known_experiments


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


def test_experiment_specs_module_reexports_the_registry_accessors():
    # The parser imports these from the old module path; keep them pointing at the registry.
    assert experiment_specs.experiment_spec is experiment_spec
    assert experiment_specs.known_experiments is known_experiments


def test_default_parallel_is_forbidden():
    class Lonely(CalibrationNode):
        experiment = "lonely"
        analyses = ("a",)

        def run(self, dead_targets):  # pragma: no cover
            return []

    assert Lonely.parallel is Parallel.forbidden
    assert Lonely.applies_operating_point is True
