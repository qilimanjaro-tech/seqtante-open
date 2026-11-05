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

"""Tests for the calibration-tree parser: structure, targets and per-target parameter resolution."""

from io import StringIO
from types import SimpleNamespace
from typing import Any

import pytest
from ruamel.yaml import YAML

from seqtante_open.controllers import CalibrationParser, Pipeline, Step, Thresholds, TreeParseError
from seqtante_open.targets import CouplerTarget, QubitTarget


@pytest.fixture
def platform() -> SimpleNamespace:
    """A platform holding only analog settings, so any other access by the parser fails."""
    return SimpleNamespace(analog_compilation_settings=SimpleNamespace(qubits=[1, 2], topology=[[1, 2]]))


ADR_TREE = """
sample: ConscienceWF02-3.7
cooldown: CD79-Mimas
targets: [q1, q2]

defaults:
  two_tone: {freq_sweep: [-5e6, 5e6, 101], averages: 10000}
  rabi:     {amplitude_sweep: [0.02, 0.8, 40], averages: 10000}

pipelines:
  sweet_spot:
    steps:
      sweet_spot_flux:
        experiment: two_tone_vs_flux
        analysis: parabola_vertex
        operating_point: sweet_spot
        loop: z
        flux_sweep: [-0.02, 0.02, 11]
      two_tone_sweet_spot:
        experiment: two_tone
        analysis: lorentzian_peak
        operating_point: sweet_spot

  t1:
    after: two_tone_sweet_spot
    steps:
      t1_sweet_spot:
        experiment: t1
        operating_point: sweet_spot
        thresholds: {passed: 50 us, warning: 20 us}

  t2:
    after: two_tone_sweet_spot
    steps:
      rabi_sweet_spot:
        experiment: rabi
        analysis: pi_amplitude
        operating_point: sweet_spot
        pulses: {drive: pi-sweet_spot}
      t2_sweet_spot:
        experiment: t2
        operating_point: sweet_spot

  lz:
    after: sweet_spot_flux
    targets: [q2]
    steps:
      reset_flux:
        experiment: reset_point_search
        operating_point: reset
        target_frequency: [2.79e9, 2.81e9]
      rabi_reset:
        experiment: rabi
        analysis: readout_levels
        operating_point: reset
        overrides:
          - targets: [q2]
            amplitude_sweep: [0.02, 0.5, 25]
      lz:
        experiment: landau_zener
        operating_point: reset
"""


def make_tree(**top: Any) -> dict[str, Any]:
    """A minimal valid tree with one pipeline ``p`` and one ``two_tone`` step ``s``; ``top`` replaces top-level keys."""
    tree: dict[str, Any] = {
        "targets": ["q1", "q2"],
        "pipelines": {"p": {"steps": {"s": {"experiment": "two_tone"}}}},
    }
    tree.update(top)
    return tree


def make_step(**keys: Any) -> dict[str, Any]:
    """A ``two_tone`` step; ``keys`` add to or replace its keys."""
    return {"experiment": "two_tone", **keys}


def single_step_tree(step: dict[str, Any], **top: Any) -> dict[str, Any]:
    return make_tree(pipelines={"p": {"steps": {"s": step}}}, **top)


def parse(tree: Any, platform: Any) -> list[Pipeline]:
    return CalibrationParser(calibration_config=tree, platform=platform).parse()


def only_step(tree: Any, platform: Any) -> Step:
    (pipeline,) = parse(tree, platform)
    (step,) = pipeline.steps
    return step


def by_name(pipelines: list[Pipeline]) -> dict[str, Step]:
    return {step.name: step for pipeline in pipelines for step in pipeline.steps}


# ------------------------- The ADR-0003 example, loaded from YAML -------------------------


class TestAdrExample:
    @pytest.fixture
    def pipelines(self, platform) -> list[Pipeline]:
        return parse(YAML().load(StringIO(ADR_TREE)), platform)

    def test_pipelines_in_file_order(self, pipelines):
        assert [p.name for p in pipelines] == ["sweet_spot", "t1", "t2", "lz"]

    def test_steps_in_file_order(self, pipelines):
        assert [[s.name for s in p.steps] for p in pipelines] == [
            ["sweet_spot_flux", "two_tone_sweet_spot"],
            ["t1_sweet_spot"],
            ["rabi_sweet_spot", "t2_sweet_spot"],
            ["reset_flux", "rabi_reset", "lz"],
        ]

    def test_after_is_normalised_to_a_list(self, pipelines):
        assert [p.after for p in pipelines] == [
            [],
            ["two_tone_sweet_spot"],
            ["two_tone_sweet_spot"],
            ["sweet_spot_flux"],
        ]

    def test_every_pipeline_is_enabled_by_default(self, pipelines):
        assert all(p.enabled for p in pipelines)

    def test_steps_know_their_pipeline(self, pipelines):
        assert all(step.pipeline == p.name for p in pipelines for step in p.steps)

    def test_structural_fields(self, pipelines):
        steps = by_name(pipelines)
        assert steps["sweet_spot_flux"].experiment == "two_tone_vs_flux"
        assert steps["sweet_spot_flux"].analysis == "parabola_vertex"
        assert steps["rabi_reset"].analysis == "readout_levels"

    def test_pipeline_targets_narrow_the_steps(self, pipelines):
        steps = by_name(pipelines)
        assert steps["t1_sweet_spot"].targets == ["q1", "q2"]
        assert steps["reset_flux"].targets == ["q2"]
        assert list(steps["reset_flux"].parameters) == ["q2"]

    def test_structural_keys_are_not_parameters(self, pipelines):
        for step in by_name(pipelines).values():
            for parameters in step.parameters.values():
                assert not {"experiment", "analysis", "targets", "overrides"} & set(parameters)

    def test_defaults_and_step_values(self, pipelines):
        steps = by_name(pipelines)
        assert steps["two_tone_sweet_spot"].parameters["q1"] == {
            "freq_sweep": [-5e6, 5e6, 101],
            "averages": 10000,
            "operating_point": "sweet_spot",
        }
        assert steps["sweet_spot_flux"].parameters["q1"] == {
            "operating_point": "sweet_spot",
            "loop": "z",
            "flux_sweep": [-0.02, 0.02, 11],
        }

    def test_overrides(self, pipelines):
        assert by_name(pipelines)["rabi_reset"].parameters["q2"] == {
            "amplitude_sweep": [0.02, 0.5, 25],
            "averages": 10000,
            "operating_point": "reset",
        }

    def test_pulses_and_thresholds_are_per_target_parameters(self, pipelines):
        steps = by_name(pipelines)
        assert steps["rabi_sweet_spot"].parameters["q1"]["pulses"] == {"drive": "pi-sweet_spot"}
        assert steps["t1_sweet_spot"].parameters["q2"]["thresholds"] == Thresholds(
            passed=50_000.0, warning=20_000.0, unit="ns", direction="higher"
        )

    def test_a_benchmark_without_thresholds_has_none(self, pipelines):
        assert "thresholds" not in by_name(pipelines)["t2_sweet_spot"].parameters["q1"]


# ------------------------- Pipelines -------------------------


class TestPipelines:
    @pytest.mark.parametrize("pipelines", [None, {}, [], "p"])
    def test_pipelines_must_be_a_non_empty_mapping(self, platform, pipelines):
        tree = make_tree(pipelines=pipelines)
        with pytest.raises(TreeParseError, match=r"^pipelines:"):
            parse(tree, platform)

    def test_pipelines_missing(self, platform):
        tree = make_tree()
        del tree["pipelines"]
        with pytest.raises(TreeParseError, match=r"^pipelines:"):
            parse(tree, platform)

    def test_pipeline_must_be_a_mapping(self, platform):
        tree = make_tree(pipelines={"p": ["s"]})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p: must be a mapping"):
            parse(tree, platform)

    @pytest.mark.parametrize("steps", [None, {}, [make_step()]])
    def test_steps_must_be_a_non_empty_mapping(self, platform, steps):
        tree = make_tree(pipelines={"p": {"steps": steps}})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.steps:"):
            parse(tree, platform)

    def test_enabled_false(self, platform):
        (pipeline,) = parse(make_tree(pipelines={"p": {"enabled": False, "steps": {"s": make_step()}}}), platform)
        assert pipeline.enabled is False

    def test_disabled_pipelines_are_still_validated(self, platform):
        tree = make_tree(pipelines={"p": {"enabled": False, "steps": {"s": make_step(experiment="nope")}}})
        with pytest.raises(TreeParseError, match="unknown experiment"):
            parse(tree, platform)

    @pytest.mark.parametrize("enabled", ["false", 0, None])
    def test_enabled_must_be_a_bool(self, platform, enabled):
        tree = make_tree(pipelines={"p": {"enabled": enabled, "steps": {"s": make_step()}}})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.enabled:"):
            parse(tree, platform)

    @pytest.mark.parametrize(
        ("pipelines", "expected"),
        [
            pytest.param(
                {
                    "a": {"steps": {"a1": make_step(), "a2": make_step()}},
                    "b": {"steps": {"b1": make_step()}},
                    "c": {"after": ["a2", "b1"], "steps": {"c1": make_step()}},
                },
                ["a2", "b1"],
                id="a_list_of_steps",
            ),
            pytest.param(
                {
                    "a": {"enabled": False, "steps": {"a1": make_step()}},
                    "c": {"after": "a1", "steps": {"c1": make_step()}},
                },
                ["a1"],
                id="a_step_of_a_disabled_pipeline",
            ),
        ],
    )
    def test_after(self, platform, pipelines, expected):
        assert parse(make_tree(pipelines=pipelines), platform)[-1].after == expected

    def test_after_must_name_an_existing_step(self, platform):
        tree = make_tree(pipelines={"p": {"after": "nope", "steps": {"s": make_step()}}})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.after: unknown step 'nope'"):
            parse(tree, platform)

    @pytest.mark.parametrize("after", ["s2", ["s1"], ["other", "s2"]])
    def test_after_cannot_name_a_step_of_the_same_pipeline(self, platform, after):
        tree = make_tree(
            pipelines={
                "o": {"steps": {"other": make_step()}},
                "p": {"after": after, "steps": {"s1": make_step(), "s2": make_step()}},
            }
        )
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.after: 's\d' is a step of this pipeline"):
            parse(tree, platform)

    def test_after_must_name_a_step_not_a_pipeline(self, platform):
        tree = make_tree(
            pipelines={"a": {"steps": {"a1": make_step()}}, "b": {"after": "a", "steps": {"b1": make_step()}}}
        )
        with pytest.raises(TreeParseError, match="unknown step 'a'"):
            parse(tree, platform)

    @pytest.mark.parametrize("after", [1, ["s", 2], {"s": None}])
    def test_after_must_be_names(self, platform, after):
        tree = make_tree(pipelines={"p": {"after": after, "steps": {"s": make_step()}}})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.after: must be"):
            parse(tree, platform)

    def test_step_names_are_unique_across_pipelines(self, platform):
        tree = make_tree(pipelines={"a": {"steps": {"s": make_step()}}, "b": {"steps": {"s": make_step()}}})
        with pytest.raises(TreeParseError, match=r"^pipelines\.b\.steps\.s: step name already used in pipeline 'a'"):
            parse(tree, platform)

    def test_a_step_may_share_its_pipeline_name(self, platform):
        (pipeline,) = parse(make_tree(pipelines={"lz": {"steps": {"lz": make_step()}}}), platform)
        assert pipeline.steps[0].name == "lz"


# ------------------------- Steps -------------------------


class TestSteps:
    def test_step_must_be_a_mapping(self, platform):
        tree = single_step_tree(["two_tone"])  # type: ignore[arg-type]
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.steps\.s: must be a mapping"):
            parse(tree, platform)

    def test_experiment_is_required(self, platform):
        tree = single_step_tree({"operating_point": "sweet_spot"})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.steps\.s\.experiment: missing"):
            parse(tree, platform)

    def test_unknown_experiment(self, platform):
        tree = single_step_tree(make_step(experiment="two_tones"))
        with pytest.raises(TreeParseError, match=r"steps\.s\.experiment: unknown experiment 'two_tones'"):
            parse(tree, platform)

    def test_the_same_experiment_in_several_steps(self, platform):
        tree = make_tree(pipelines={"p": {"steps": {"a": make_step(), "b": make_step()}}})
        assert [s.experiment for s in parse(tree, platform)[0].steps] == ["two_tone", "two_tone"]

    @pytest.mark.parametrize(
        ("step", "expected"),
        [
            pytest.param(make_step(), "lorentzian_peak", id="default"),
            pytest.param(make_step(experiment="rabi"), "pi_amplitude", id="default_of_several"),
            pytest.param(make_step(experiment="rabi", analysis="readout_levels"), "readout_levels", id="named"),
        ],
    )
    def test_analysis(self, platform, step, expected):
        assert only_step(single_step_tree(step), platform).analysis == expected

    def test_analysis_of_another_experiment(self, platform):
        tree = single_step_tree(make_step(analysis="readout_levels"))
        with pytest.raises(
            TreeParseError, match=r"steps\.s\.analysis: unknown analysis 'readout_levels' for 'two_tone'"
        ):
            parse(tree, platform)

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(make_step(experiment="two_tone_vs_flux"), id="flux_sweep_without_operating_point_or_loop"),
            pytest.param(make_step(operating_point="", loop="y"), id="values_the_node_will_reject"),
        ],
    )
    def test_operating_point_and_loop_are_not_checked_here(self, platform, step):
        expected = {key: value for key, value in step.items() if key != "experiment"}
        assert only_step(single_step_tree(step), platform).parameters == {"q1": expected, "q2": expected}

    def test_operating_point_and_loop_resolve_per_target(self, platform):
        step = make_step(
            operating_point="sweet_spot",
            overrides=[{"targets": ["q2"], "operating_point": "reset", "loop": "x"}],
        )
        tree = single_step_tree(step, defaults={"two_tone": {"loop": "z"}})
        assert only_step(tree, platform).parameters == {
            "q1": {"loop": "z", "operating_point": "sweet_spot"},
            "q2": {"loop": "x", "operating_point": "reset"},
        }


# ------------------------- Targets -------------------------


class TestTargets:
    def test_top_level_targets_are_required(self, platform):
        tree = make_tree()
        del tree["targets"]
        with pytest.raises(TreeParseError, match=r"^targets: missing"):
            parse(tree, platform)

    @pytest.mark.parametrize("targets", [[], "q1", None])
    def test_targets_must_be_a_non_empty_list(self, platform, targets):
        tree = make_tree(targets=targets)
        with pytest.raises(TreeParseError, match=r"^targets: must be a non-empty list"):
            parse(tree, platform)

    def test_top_level_targets_reach_every_step(self, platform):
        step = only_step(make_tree(), platform)
        assert step.targets == ["q1", "q2"]
        assert list(step.parameters) == ["q1", "q2"]

    def test_pipeline_narrows(self, platform):
        (pipeline,) = parse(make_tree(pipelines={"p": {"targets": ["q2"], "steps": {"s": make_step()}}}), platform)
        assert pipeline.targets == ["q2"]
        assert pipeline.steps[0].targets == ["q2"]

    def test_step_narrows(self, platform):
        tree = make_tree(pipelines={"p": {"steps": {"a": make_step(targets=["q1"]), "b": make_step()}}})
        a, b = parse(tree, platform)[0].steps
        assert a.targets == ["q1"]
        assert list(a.parameters) == ["q1"]
        assert b.targets == ["q1", "q2"]

    def test_step_narrows_the_pipeline_not_the_top(self, platform):
        tree = make_tree(pipelines={"p": {"targets": ["q2"], "steps": {"s": make_step(targets=["q1"])}}})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.steps\.s\.targets: \['q1'\] are not in"):
            parse(tree, platform)

    def test_pipeline_cannot_widen(self, platform):
        tree = make_tree(targets=["q1"], pipelines={"p": {"targets": ["q1", "q2"], "steps": {"s": make_step()}}})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.targets: \['q2'\] are not in"):
            parse(tree, platform)

    @pytest.mark.parametrize("targets", [[], "q1"])
    def test_narrowed_targets_must_be_a_non_empty_list(self, platform, targets):
        tree = make_tree(pipelines={"p": {"targets": targets, "steps": {"s": make_step()}}})
        with pytest.raises(TreeParseError, match=r"^pipelines\.p\.targets: must be a non-empty list"):
            parse(tree, platform)

    @pytest.mark.parametrize(
        ("targets", "expected"),
        [
            pytest.param(["all_q"], ["q1", "q2"], id="all_q"),
            pytest.param(["all_c"], ["c1_2"], id="all_c"),
            pytest.param(["c1_2", "all_q"], ["c1_2", "q1", "q2"], id="expansion_keeps_its_place"),
            pytest.param(["all_q", "all_c"], ["q1", "q2", "c1_2"], id="all_q_and_all_c"),
            pytest.param(["all_q", "q1"], ["q1", "q2"], id="dedup_after_all_q"),
            pytest.param(["q2", "all_q"], ["q2", "q1"], id="dedup_keeps_the_first"),
            pytest.param(["c1_2", "all_c"], ["c1_2"], id="dedup_after_all_c"),
            pytest.param(["q1", "q1"], ["q1"], id="dedup_a_repeated_name"),
        ],
    )
    def test_expansion(self, platform, targets, expected):
        step = only_step(make_tree(targets=targets), platform)
        assert step.targets == expected
        assert list(step.parameters) == expected

    @pytest.mark.parametrize(
        ("step_targets", "expected"),
        [
            pytest.param(["all_q"], ["q1", "q2"], id="all_q"),
            pytest.param(["q2", "all_q", "q2"], ["q2", "q1"], id="dedup"),
        ],
    )
    def test_expansion_in_a_narrowing(self, platform, step_targets, expected):
        tree = make_tree(targets=["all_q", "all_c"], pipelines={"p": {"steps": {"s": make_step(targets=step_targets)}}})
        assert only_step(tree, platform).targets == expected

    def test_dedup_in_overrides(self, platform):
        step = make_step(overrides=[{"targets": ["q1", "all_q"], "averages": 100}])
        assert only_step(single_step_tree(step), platform).parameters == {
            "q1": {"averages": 100},
            "q2": {"averages": 100},
        }

    def test_all_c_orders_the_topology(self):
        platform = SimpleNamespace(analog_compilation_settings=SimpleNamespace(qubits=[1, 2], topology=[[2, 1]]))
        assert only_step(make_tree(targets=["all_c"]), platform).targets == [CouplerTarget(1, 2)]

    @pytest.mark.parametrize(
        ("settings", "target", "match"),
        [
            pytest.param(None, "all_q", r"^targets: all_q needs the qubits", id="all_q_without_analog_settings"),
            pytest.param(None, "all_c", r"^targets: all_c needs the topology", id="all_c_without_analog_settings"),
            pytest.param(
                SimpleNamespace(qubits=[], topology=[[1, 2]]),
                "all_q",
                r"^targets: all_q needs the qubits",
                id="all_q_without_qubits",
            ),
            pytest.param(
                SimpleNamespace(qubits=[1, 2], topology=[]),
                "all_c",
                r"^targets: all_c needs the topology",
                id="all_c_without_topology",
            ),
            pytest.param(
                SimpleNamespace(qubits=[-1], topology=[]),
                "all_q",
                r"^targets: all_q: Invalid qubit_id",
                id="all_q_with_an_invalid_qubit",
            ),
            pytest.param(
                SimpleNamespace(qubits=[1], topology=[[1, 1]]),
                "all_c",
                r"^targets: all_c: Invalid coupler",
                id="all_c_with_a_self_coupling",
            ),
        ],
    )
    def test_expansion_needs_valid_analog_settings(self, settings, target, match):
        platform = SimpleNamespace(analog_compilation_settings=settings)
        tree = make_tree(targets=[target])
        with pytest.raises(TreeParseError, match=match):
            parse(tree, platform)


class TestTargetObjects:
    def test_targets_and_parameter_keys_are_target_objects(self, platform):
        (pipeline,) = parse(make_tree(targets=["q1", "c1_2"]), platform)
        step = pipeline.steps[0]
        assert [type(target) for target in pipeline.targets] == [QubitTarget, CouplerTarget]
        assert [type(target) for target in step.targets] == [QubitTarget, CouplerTarget]
        assert [type(target) for target in step.parameters] == [QubitTarget, CouplerTarget]

    def test_expanded_targets_are_target_objects(self, platform):
        step = only_step(make_tree(targets=["all_q", "all_c"]), platform)
        assert step.targets == [QubitTarget(1), QubitTarget(2), CouplerTarget(1, 2)]
        assert [type(target) for target in step.targets] == [QubitTarget, QubitTarget, CouplerTarget]

    @pytest.mark.parametrize("name", ["q01", "c2_1", "c1_1", "Q1", "qubit1", 1, None, ["q1"]])
    def test_invalid_top_level_target(self, platform, name):
        tree = make_tree(targets=["q1", name])
        with pytest.raises(TreeParseError, match=r"^targets: Invalid target"):
            parse(tree, platform)

    @pytest.mark.parametrize(
        ("tree", "match"),
        [
            pytest.param(
                make_tree(pipelines={"p": {"targets": ["c2_1"], "steps": {"s": make_step()}}}),
                r"^pipelines\.p\.targets: Invalid target 'c2_1'",
                id="pipeline",
            ),
            pytest.param(
                single_step_tree(make_step(targets=["q01"])),
                r"^pipelines\.p\.steps\.s\.targets: Invalid target 'q01'",
                id="step",
            ),
            pytest.param(
                single_step_tree(make_step(overrides=[{"targets": ["q1", "q-2"], "averages": 1}])),
                r"^pipelines\.p\.steps\.s\.overrides\[0\]\.targets: Invalid target 'q-2'",
                id="override",
            ),
        ],
    )
    def test_invalid_target_below_the_top(self, platform, tree, match):
        with pytest.raises(TreeParseError, match=match):
            parse(tree, platform)

    def test_messages_name_targets(self, platform):
        tree = make_tree(targets=["q1"], pipelines={"p": {"targets": ["q1", "c1_2"], "steps": {"s": make_step()}}})
        with pytest.raises(
            TreeParseError, match=r"^pipelines\.p\.targets: \['c1_2'\] are not in the enclosing targets \['q1'\]\.$"
        ):
            parse(tree, platform)


# ------------------------- Parameter resolution -------------------------


class TestResolution:
    @pytest.mark.parametrize(
        ("tree", "expected"),
        [
            pytest.param(make_tree(), {"q1": {}, "q2": {}}, id="no_parameters"),
            pytest.param(
                make_tree(defaults={"two_tone": {"averages": 1000}}),
                {"q1": {"averages": 1000}, "q2": {"averages": 1000}},
                id="defaults_only",
            ),
            pytest.param(
                make_tree(defaults={"rabi": {"averages": 1000}}),
                {"q1": {}, "q2": {}},
                id="defaults_of_another_experiment_do_not_apply",
            ),
            pytest.param(
                make_tree(defaults={"not_yet_an_experiment": {"a": 1}}),
                {"q1": {}, "q2": {}},
                id="defaults_for_an_experiment_no_step_uses_are_accepted",
            ),
            pytest.param(
                single_step_tree(make_step(averages=500), defaults={"two_tone": {"averages": 1000, "span": 5e6}}),
                {"q1": {"averages": 500, "span": 5e6}, "q2": {"averages": 500, "span": 5e6}},
                id="step_overrides_defaults",
            ),
            pytest.param(
                single_step_tree(make_step(averages=500, overrides=[{"targets": ["q2"], "averages": 100}])),
                {"q1": {"averages": 500}, "q2": {"averages": 100}},
                id="override_overrides_step",
            ),
            pytest.param(
                single_step_tree(
                    make_step(overrides=[{"targets": ["q2"], "averages": 100}]),
                    defaults={"two_tone": {"averages": 1000}},
                ),
                {"q1": {"averages": 1000}, "q2": {"averages": 100}},
                id="override_overrides_defaults",
            ),
            pytest.param(
                single_step_tree(
                    make_step(
                        overrides=[
                            {"targets": ["q1", "q2"], "averages": 100, "span": 1},
                            {"targets": ["q2"], "averages": 200},
                            {"targets": ["q2"], "span": 3},
                        ]
                    )
                ),
                {"q1": {"averages": 100, "span": 1}, "q2": {"averages": 200, "span": 3}},
                id="overrides_apply_in_file_order",
            ),
            pytest.param(
                single_step_tree(make_step(overrides=[{"targets": ["all_q"], "averages": 100}])),
                {"q1": {"averages": 100}, "q2": {"averages": 100}},
                id="override_with_all_q",
            ),
            pytest.param(
                single_step_tree(
                    make_step(
                        pulses={"drive": "pi-reset", "readout": "readout-reset"},
                        overrides=[{"targets": ["q2"], "pulses": {"drive": "pi-sweet_spot"}}],
                    )
                ),
                {
                    "q1": {"pulses": {"drive": "pi-reset", "readout": "readout-reset"}},
                    "q2": {"pulses": {"drive": "pi-sweet_spot"}},
                },
                id="override_replaces_a_whole_value",
            ),
            pytest.param(
                single_step_tree(make_step(overrides=[{"targets": ["q1"]}])),
                {"q1": {}, "q2": {}},
                id="override_with_only_targets",
            ),
        ],
    )
    def test_resolution(self, platform, tree, expected):
        assert only_step(tree, platform).parameters == expected

    def test_override_target_outside_the_step_is_warned_and_ignored(self, platform, warnings_log):
        step = make_step(targets=["q1"], overrides=[{"targets": ["q1", "q2"], "averages": 100}])
        assert only_step(single_step_tree(step), platform).parameters == {"q1": {"averages": 100}}
        assert warnings_log == [
            "pipelines.p.steps.s.overrides[0].targets: ['q2'] are not targets of the step. (Ignored)"
        ]

    def test_targets_do_not_share_values(self, platform):
        tree = single_step_tree(make_step(sweep=[0, 1, 2]), defaults={"two_tone": {"nested": {"a": 1}}})
        parameters = only_step(tree, platform).parameters
        parameters["q1"]["sweep"].append(3)
        parameters["q1"]["nested"]["a"] = 2
        assert parameters["q2"] == {"sweep": [0, 1, 2], "nested": {"a": 1}}

    def test_parsing_does_not_change_the_tree(self, platform):
        tree = single_step_tree(make_step(sweep=[0, 1], overrides=[{"targets": ["q2"], "sweep": [2, 3]}]))
        parameters = only_step(tree, platform).parameters
        parameters["q1"]["sweep"].append(9)
        parameters["q2"]["sweep"].append(9)
        assert tree["pipelines"]["p"]["steps"]["s"]["sweep"] == [0, 1]
        assert tree["pipelines"]["p"]["steps"]["s"]["overrides"] == [{"targets": ["q2"], "sweep": [2, 3]}]

    @pytest.mark.parametrize("overrides", [{"targets": ["q1"]}, {"q1": {"averages": 1}}, "q1"])
    def test_overrides_must_be_a_list(self, platform, overrides):
        tree = single_step_tree(make_step(overrides=overrides))
        with pytest.raises(TreeParseError, match=r"steps\.s\.overrides: must be a list"):
            parse(tree, platform)

    def test_override_must_be_a_mapping(self, platform):
        tree = single_step_tree(make_step(overrides=[["q1"]]))
        with pytest.raises(TreeParseError, match=r"steps\.s\.overrides\[0\]: must be a mapping"):
            parse(tree, platform)

    def test_override_needs_targets(self, platform):
        tree = single_step_tree(make_step(overrides=[{"targets": ["q1"]}, {"averages": 1}]))
        with pytest.raises(TreeParseError, match=r"steps\.s\.overrides\[1\]\.targets: missing"):
            parse(tree, platform)

    @pytest.mark.parametrize("targets", [[], "q1"])
    def test_override_targets_must_be_a_non_empty_list(self, platform, targets):
        tree = single_step_tree(make_step(overrides=[{"targets": targets, "averages": 1}]))
        with pytest.raises(TreeParseError, match=r"steps\.s\.overrides\[0\]\.targets: must be a non-empty list"):
            parse(tree, platform)

    @pytest.mark.parametrize("key", ["experiment", "analysis", "overrides"])
    def test_structural_keys_cannot_be_overridden(self, platform, key):
        step = make_step(overrides=[{"targets": ["q1"], key: "x"}])
        tree = single_step_tree(step)
        with pytest.raises(TreeParseError, match=rf"steps\.s\.overrides\[0\]: \['{key}'\] can only be set on the step"):
            parse(tree, platform)

    @pytest.mark.parametrize("key", ["experiment", "analysis", "targets", "overrides"])
    def test_structural_keys_cannot_be_defaults(self, platform, key):
        tree = make_tree(defaults={"two_tone": {key: "x"}})
        with pytest.raises(TreeParseError, match=rf"^defaults\.two_tone: \['{key}'\] can only be set on the step"):
            parse(tree, platform)

    @pytest.mark.parametrize("defaults", [["two_tone"], "two_tone"])
    def test_defaults_must_be_a_mapping(self, platform, defaults):
        tree = make_tree(defaults=defaults)
        with pytest.raises(TreeParseError, match=r"^defaults: must be a mapping"):
            parse(tree, platform)

    def test_defaults_of_an_experiment_must_be_a_mapping(self, platform):
        tree = make_tree(defaults={"two_tone": [1, 2]})
        with pytest.raises(TreeParseError, match=r"^defaults\.two_tone: must be a mapping"):
            parse(tree, platform)


class TestDataFolder:
    @pytest.mark.parametrize(
        ("tree", "match"),
        [
            pytest.param(
                single_step_tree(make_step(data_folder="x")),
                r"^pipelines\.p\.steps\.s: \['data_folder'\] cannot be set",
                id="step",
            ),
            pytest.param(
                make_tree(defaults={"two_tone": {"data_folder": "x"}}),
                r"^defaults\.two_tone: \['data_folder'\] cannot be set",
                id="defaults",
            ),
            pytest.param(
                single_step_tree(make_step(overrides=[{"targets": ["q1"], "data_folder": "x"}])),
                r"^pipelines\.p\.steps\.s\.overrides\[0\]: \['data_folder'\] cannot be set",
                id="override",
            ),
        ],
    )
    def test_cannot_be_set(self, platform, tree, match):
        with pytest.raises(TreeParseError, match=match):
            parse(tree, platform)


# ------------------------- Benchmark thresholds -------------------------


def t1_step(**keys: Any) -> dict[str, Any]:
    """A ``t1`` step; ``keys`` add to or replace its keys."""
    return make_step(experiment="t1", **keys)


def lz_step(**keys: Any) -> dict[str, Any]:
    """A ``landau_zener`` step, a dimensionless benchmark where lower is better; ``keys`` add to or replace its keys."""
    return make_step(experiment="landau_zener", operating_point="reset", **keys)


def t1_thresholds(passed: object = "50 us", warning: object = "20 us") -> dict[str, object]:
    return {"passed": passed, "warning": warning}


class TestThresholds:
    @pytest.mark.parametrize(
        "passed",
        [
            pytest.param("50 us", id="us"),
            pytest.param("50us", id="no_space"),
            pytest.param("  50 us ", id="surrounding_spaces"),
            pytest.param("50 \u00b5s", id="micro_sign"),
            pytest.param("50 \u03bcs", id="greek_mu"),
            pytest.param("5e1 us", id="exponent"),
            pytest.param("50.0 us", id="decimal"),
            pytest.param("50000 ns", id="ns"),
            pytest.param("0.05 ms", id="ms"),
            pytest.param(".05 ms", id="leading_dot"),
        ],
    )
    def test_time_is_converted_to_the_benchmark_unit(self, platform, passed):
        step = only_step(single_step_tree(t1_step(thresholds=t1_thresholds(passed=passed))), platform)
        assert step.parameters["q1"]["thresholds"] == Thresholds(
            passed=50_000.0, warning=20_000.0, unit="ns", direction="higher"
        )

    @pytest.mark.parametrize("passed", [1, 1.5])
    def test_dimensionless_takes_a_plain_number(self, platform, passed):
        step = only_step(single_step_tree(lz_step(thresholds={"passed": passed, "warning": 3})), platform)
        assert step.parameters["q1"]["thresholds"] == Thresholds(
            passed=float(passed), warning=3.0, unit=None, direction="lower"
        )

    def test_zero_is_accepted(self, platform):
        step = only_step(single_step_tree(t1_step(thresholds=t1_thresholds(warning="0 us"))), platform)
        assert step.parameters["q1"]["thresholds"] == Thresholds(50_000.0, 0.0, "ns", "higher")

    def test_a_benchmark_without_thresholds_has_none(self, platform):
        assert only_step(single_step_tree(t1_step()), platform).parameters == {"q1": {}, "q2": {}}

    def test_resolve_per_target(self, platform):
        step = t1_step(
            thresholds=t1_thresholds("40 us", "10 us"),
            overrides=[{"targets": ["q2"], "thresholds": t1_thresholds("60 us", "30 us")}],
        )
        tree = single_step_tree(step, defaults={"t1": {"thresholds": t1_thresholds()}})
        parameters = only_step(tree, platform).parameters
        assert parameters["q1"]["thresholds"] == Thresholds(40_000.0, 10_000.0, "ns", "higher")
        assert parameters["q2"]["thresholds"] == Thresholds(60_000.0, 30_000.0, "ns", "higher")

    def test_from_defaults(self, platform):
        tree = single_step_tree(t1_step(), defaults={"t1": {"thresholds": t1_thresholds()}})
        assert only_step(tree, platform).parameters["q2"]["thresholds"] == Thresholds(
            50_000.0, 20_000.0, "ns", "higher"
        )

    def test_defaults_of_an_unknown_experiment_are_not_checked(self, platform):
        tree = make_tree(defaults={"not_yet_an_experiment": {"thresholds": "anything"}})
        assert only_step(tree, platform).parameters == {"q1": {}, "q2": {}}

    def test_parsing_does_not_change_the_tree(self, platform):
        tree = single_step_tree(t1_step(thresholds=t1_thresholds()))
        only_step(tree, platform)
        assert tree["pipelines"]["p"]["steps"]["s"]["thresholds"] == t1_thresholds()

    def test_yaml_with_micro_sign(self, platform):
        tree = YAML().load(
            StringIO(
                "targets: [q1]\n"
                "pipelines:\n"
                "  p:\n"
                "    steps:\n"
                "      s: {experiment: t1, operating_point: sweet_spot, thresholds: {passed: 50 \u00b5s, warning: 20 us}}\n"
            )
        )
        assert only_step(tree, platform).parameters["q1"]["thresholds"] == Thresholds(
            50_000.0, 20_000.0, "ns", "higher"
        )

    @pytest.mark.parametrize(
        ("tree", "match"),
        [
            pytest.param(
                single_step_tree(make_step(thresholds=t1_thresholds())),
                r"^pipelines\.p\.steps\.s\.thresholds: 'two_tone' is not a benchmark",
                id="not_a_benchmark",
            ),
            pytest.param(
                make_tree(defaults={"two_tone": {"thresholds": t1_thresholds()}}),
                r"^defaults\.two_tone\.thresholds: 'two_tone' is not a benchmark",
                id="not_a_benchmark_in_defaults",
            ),
            pytest.param(
                single_step_tree(make_step(overrides=[{"targets": ["q1"], "thresholds": t1_thresholds()}])),
                r"^pipelines\.p\.steps\.s\.overrides\[0\]\.thresholds: 'two_tone' is not a benchmark",
                id="not_a_benchmark_in_override",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds="50 us")),
                r"^pipelines\.p\.steps\.s\.thresholds: must be a mapping with exactly 'passed' and 'warning'",
                id="not_a_mapping",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds={"passed": "50 us"})),
                r"^pipelines\.p\.steps\.s\.thresholds: must be a mapping with exactly 'passed' and 'warning'",
                id="missing_warning",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds={**t1_thresholds(), "failed": "1 us"})),
                r"^pipelines\.p\.steps\.s\.thresholds: must be a mapping with exactly 'passed' and 'warning'",
                id="extra_key",
            ),
            pytest.param(
                single_step_tree(
                    t1_step(
                        thresholds=t1_thresholds(),
                        overrides=[{"targets": ["q2"], "thresholds": {"passed": "60 us"}}],
                    )
                ),
                r"^pipelines\.p\.steps\.s\.overrides\[0\]\.thresholds: must be a mapping with exactly",
                id="override_replaces_the_whole_thresholds",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds(passed=50))),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: 50 needs a unit",
                id="time_without_unit",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds(warning=True))),
                r"^pipelines\.p\.steps\.s\.thresholds\.warning: True is not a number",
                id="bool",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds(passed="fast"))),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: 'fast' must be a number with a unit",
                id="not_a_quantity",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds(passed="-50 us"))),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: '-50 us' must be a number with a unit",
                id="negative",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds(passed=["50", "us"]))),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: \['50', 'us'\] must be a number with a unit",
                id="list",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds(passed="50 GHz"))),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: unknown unit 'GHz'",
                id="wrong_kind_of_unit",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds(passed="1 s"))),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: unknown unit 's'",
                id="seconds_not_accepted",
            ),
            pytest.param(
                single_step_tree(lz_step(thresholds={"passed": "1 us", "warning": 3})),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: '1 us': the benchmark is dimensionless",
                id="dimensionless_with_unit",
            ),
            pytest.param(
                single_step_tree(lz_step(thresholds={"passed": -1, "warning": 3})),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: -1 is negative",
                id="dimensionless_negative",
            ),
            pytest.param(
                single_step_tree(lz_step(thresholds={"passed": float("nan"), "warning": 3})),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: nan is not a finite number",
                id="nan",
            ),
            pytest.param(
                single_step_tree(lz_step(thresholds={"passed": 1.5, "warning": float("inf")})),
                r"^pipelines\.p\.steps\.s\.thresholds\.warning: inf is not a finite number",
                id="inf",
            ),
            pytest.param(
                single_step_tree(lz_step(thresholds={"passed": float("-inf"), "warning": 3})),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: -inf is not a finite number",
                id="minus_inf",
            ),
            pytest.param(
                single_step_tree(lz_step(thresholds={"passed": 1.5, "warning": 10**400})),
                r"^pipelines\.p\.steps\.s\.thresholds\.warning: 1\d+ is not a finite number",
                id="int_too_large_for_a_float",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds(passed="1e999 us"))),
                r"^pipelines\.p\.steps\.s\.thresholds\.passed: '1e999 us' is not a finite number",
                id="time_overflow",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds("50 us", "50000 ns"))),
                r"^pipelines\.p\.steps\.s\.thresholds: passed and warning are equal",
                id="empty_band_after_conversion",
            ),
            pytest.param(
                single_step_tree(t1_step(thresholds=t1_thresholds("20 us", "50 us"))),
                r"^pipelines\.p\.steps\.s\.thresholds: passed \(20 us\) must be above warning \(50 us\), "
                r"since higher is better for 't1'",
                id="higher_is_better_reversed",
            ),
            pytest.param(
                single_step_tree(lz_step(thresholds={"passed": 3, "warning": 1})),
                r"^pipelines\.p\.steps\.s\.thresholds: passed \(3\) must be below warning \(1\), "
                r"since lower is better for 'landau_zener'",
                id="lower_is_better_reversed",
            ),
        ],
    )
    def test_invalid(self, platform, tree, match):
        with pytest.raises(TreeParseError, match=match):
            parse(tree, platform)


# --- Pipeline- and step-level key validation (QHC-1800) ---


def test_unknown_pipeline_key_is_rejected(platform):
    tree = make_tree(pipelines={"p": {"enable": False, "steps": {"s": make_step()}}})
    with pytest.raises(TreeParseError, match="unknown pipeline key"):
        parse(tree, platform)


def test_pipeline_level_defaults_are_rejected(platform):
    tree = make_tree(pipelines={"p": {"defaults": {}, "steps": {"s": make_step()}}})
    with pytest.raises(TreeParseError, match="unknown pipeline key"):
        parse(tree, platform)


def test_enabled_on_a_step_is_rejected(platform):
    with pytest.raises(TreeParseError, match="can only be set on the pipeline"):
        parse(single_step_tree(make_step(enabled=False)), platform)


def test_after_on_a_step_is_rejected(platform):
    with pytest.raises(TreeParseError, match="can only be set on the pipeline"):
        parse(single_step_tree(make_step(after="x")), platform)
