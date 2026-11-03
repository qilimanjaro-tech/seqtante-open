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

"""Parses a calibration tree (ADR-0003) into pipelines of steps with parameters resolved per target (ADR-0005)."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from loguru import logger
from qililab import Platform

from seqtante_open.controllers.experiment_specs import experiment_spec, known_experiments
from seqtante_open.targets import CouplerTarget, ExperimentTarget, QubitTarget, Target

Loop = Literal["z", "x"]

_LOOPS: tuple[Loop, ...] = ("z", "x")
_STEP_KEYS = frozenset({"experiment", "analysis", "operating_point", "loop", "targets", "overrides"})
_RESERVED_PARAMETERS = frozenset({"data_folder"})


class TreeParseError(ValueError):
    """The calibration tree is malformed or names something that does not exist."""


@dataclass(frozen=True)
class Step:
    """One named step of a pipeline.

    Args:
        name (str): Step name, unique across the tree.
        pipeline (str): Name of the pipeline the step belongs to.
        experiment (str): Experiment the step runs.
        analysis (str): Analysis the step runs, the experiment's default when the tree names none.
        operating_point (str): Operating point the step runs at and writes into.
        loop (Loop | None): Flux loop the step sweeps, only for flux-sweep experiments.
        targets (list[Target]): Targets the step runs on.
        parameters (dict[Target, dict[str, Any]]): Resolved parameters per target.
    """

    name: str
    pipeline: str
    experiment: str
    analysis: str
    operating_point: str
    loop: Loop | None
    targets: list[Target]
    parameters: dict[Target, dict[str, Any]]


@dataclass(frozen=True)
class Pipeline:
    """A named, ordered group of steps.

    Args:
        name (str): Pipeline name.
        after (list[str]): Steps that must have succeeded for a target before the first step runs.
        enabled (bool): Whether the tree switches the pipeline on.
        targets (list[Target]): Targets the pipeline runs on.
        steps (list[Step]): Steps, in run order.
    """

    name: str
    after: list[str]
    enabled: bool
    targets: list[Target]
    steps: list[Step]


class CalibrationParser:
    """Parses a calibration tree into pipelines of steps.

    The platform is only read for the qubits and topology that ``all_q`` and ``all_c`` expand to.

    Args:
        calibration_config (Mapping[str, Any]): The calibration tree, as loaded from YAML.
        platform (Platform): Platform the run uses.
    """

    def __init__(self, calibration_config: Mapping[str, Any], platform: Platform):
        self.calibration_config = calibration_config
        self.platform = platform

    def parse(self) -> list[Pipeline]:
        """Parses every pipeline, enabled or not.

        Raises:
            TreeParseError: If the tree is malformed or names something that does not exist.
        """
        config = self.calibration_config

        # Top-level targets and defaults
        if "targets" not in config:
            raise TreeParseError("targets: missing.")
        targets = self._targets(config["targets"], parent=None, path="targets")
        defaults = self._defaults(config.get("defaults", {}))

        # Pipelines and their steps
        raw_pipelines = config.get("pipelines")
        if not isinstance(raw_pipelines, Mapping) or not raw_pipelines:
            raise TreeParseError("pipelines: must be a non-empty mapping of pipeline names to pipelines.")
        pipelines = [self._pipeline(name, raw, targets, defaults) for name, raw in raw_pipelines.items()]

        # Step names unique across the tree
        step_pipeline: dict[str, str] = {}
        for pipeline in pipelines:
            for step in pipeline.steps:
                if step.name in step_pipeline:
                    raise TreeParseError(
                        f"pipelines.{pipeline.name}.steps.{step.name}: step name already used in pipeline "
                        f"'{step_pipeline[step.name]}'. Step names must be unique across the tree."
                    )
                step_pipeline[step.name] = pipeline.name

        # Pipeline after names existing steps of other pipelines
        for pipeline in pipelines:
            for name in pipeline.after:
                if name not in step_pipeline:
                    raise TreeParseError(f"pipelines.{pipeline.name}.after: unknown step '{name}'.")
                if step_pipeline[name] == pipeline.name:
                    raise TreeParseError(
                        f"pipelines.{pipeline.name}.after: '{name}' is a step of this pipeline. "
                        "Steps in a pipeline already run in order."
                    )
        return pipelines

    def _pipeline(
        self, name: str, raw: Any, parent_targets: list[Target], defaults: dict[str, dict[str, Any]]
    ) -> Pipeline:
        """Parses one pipeline, narrowing ``parent_targets`` with its own ``targets``."""
        path = f"pipelines.{name}"
        if not isinstance(raw, Mapping):
            raise TreeParseError(f"{path}: must be a mapping.")
        targets = self._narrowed_targets(raw, parent_targets, path)

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TreeParseError(f"{path}.enabled: must be true or false.")

        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, Mapping) or not raw_steps:
            raise TreeParseError(f"{path}.steps: must be a non-empty mapping of step names to steps.")
        steps = [self._step(step_name, name, raw_step, targets, defaults) for step_name, raw_step in raw_steps.items()]
        return Pipeline(name=name, after=_after(raw.get("after"), path), enabled=enabled, targets=targets, steps=steps)

    def _step(
        self, name: str, pipeline: str, raw: Any, parent_targets: list[Target], defaults: dict[str, dict[str, Any]]
    ) -> Step:
        """Parses one step and resolves its parameters per target: defaults, then the step, then overrides."""
        path = f"pipelines.{pipeline}.steps.{name}"
        if not isinstance(raw, Mapping):
            raise TreeParseError(f"{path}: must be a mapping.")

        experiment = raw.get("experiment")
        if not isinstance(experiment, str):
            raise TreeParseError(f"{path}.experiment: missing.")
        spec = experiment_spec(experiment)
        if spec is None:
            raise TreeParseError(
                f"{path}.experiment: unknown experiment '{experiment}'. Known: {', '.join(sorted(known_experiments()))}."
            )

        analysis = raw.get("analysis", spec.default_analysis)
        if analysis not in spec.analyses:
            raise TreeParseError(
                f"{path}.analysis: unknown analysis '{analysis}' for '{experiment}'. Known: {', '.join(spec.analyses)}."
            )

        operating_point = raw.get("operating_point")
        if not isinstance(operating_point, str) or not operating_point:
            raise TreeParseError(f"{path}.operating_point: missing.")

        loop = raw.get("loop")
        if spec.flux_sweep and loop not in _LOOPS:
            raise TreeParseError(f"{path}.loop: '{experiment}' sweeps flux, so loop must be one of {_LOOPS}.")
        if not spec.flux_sweep and loop is not None:
            raise TreeParseError(f"{path}.loop: '{experiment}' does not sweep flux, so it takes no loop.")

        targets = self._narrowed_targets(raw, parent_targets, path)

        own = {key: value for key, value in raw.items() if key not in _STEP_KEYS}
        _check_reserved(own, path)
        overrides = self._overrides(raw.get("overrides", []), targets, path)

        base = {**defaults.get(experiment, {}), **own}
        parameters: dict[Target, dict[str, Any]] = {}
        for target in targets:
            resolved = base
            for override_targets, values in overrides:
                if target in override_targets:
                    resolved = resolved | values
            parameters[target] = copy.deepcopy(resolved)

        return Step(
            name=name,
            pipeline=pipeline,
            experiment=experiment,
            analysis=analysis,
            operating_point=operating_point,
            loop=cast("Loop | None", loop),
            targets=targets,
            parameters=parameters,
        )

    def _overrides(
        self, raw: Any, step_targets: list[Target], step_path: str
    ) -> list[tuple[list[Target], dict[str, Any]]]:
        """Parses a step's overrides into ``(targets, values)`` pairs, dropping targets the step does not run on."""
        path = f"{step_path}.overrides"
        if not isinstance(raw, list):
            raise TreeParseError(f"{path}: must be a list of entries, each with its own targets.")
        overrides = []
        for idx, entry in enumerate(raw):
            entry_path = f"{path}[{idx}]"
            if not isinstance(entry, Mapping):
                raise TreeParseError(f"{entry_path}: must be a mapping.")
            if "targets" not in entry:
                raise TreeParseError(f"{entry_path}.targets: missing.")
            kept: list[Target] = []
            ignored: list[Target] = []
            for target in self._targets(entry["targets"], parent=None, path=f"{entry_path}.targets"):
                (kept if target in step_targets else ignored).append(target)
            if ignored:
                logger.warning(
                    "{path}.targets: {ignored} are not targets of the step. (Ignored)",
                    path=entry_path,
                    ignored=_names(ignored),
                )
            values = {key: value for key, value in entry.items() if key != "targets"}
            _check_structural(values, entry_path)
            _check_reserved(values, entry_path)
            overrides.append((kept, values))
        return overrides

    @staticmethod
    def _defaults(raw: Any) -> dict[str, dict[str, Any]]:
        """Parses the ``defaults`` block into parameters per experiment."""
        if not isinstance(raw, Mapping):
            raise TreeParseError("defaults: must be a mapping of experiment names to parameters.")
        defaults = {}
        for experiment, values in raw.items():
            path = f"defaults.{experiment}"
            if not isinstance(values, Mapping):
                raise TreeParseError(f"{path}: must be a mapping of parameters.")
            _check_structural(values, path)
            _check_reserved(values, path)
            defaults[experiment] = dict(values)
        return defaults

    def _narrowed_targets(self, raw: Mapping[str, Any], parent: list[Target], path: str) -> list[Target]:
        """Returns the ``targets`` of a pipeline or step, or ``parent`` when it sets none."""
        if "targets" not in raw:
            return list(parent)
        return self._targets(raw["targets"], parent=parent, path=f"{path}.targets")

    def _targets(self, raw: Any, parent: list[Target] | None, path: str) -> list[Target]:
        """Parses target names, expands ``all_q`` and ``all_c`` in place, drops repeats keeping the first, and checks
        the result is within ``parent``, when given."""
        if not isinstance(raw, list) or not raw:
            raise TreeParseError(f"{path}: must be a non-empty list of targets.")
        targets: list[Target] = []
        for name in raw:
            if name == "all_q":
                targets += self._all_qubits(path)
            elif name == "all_c":
                targets += self._all_couplers(path)
            else:
                targets.append(_target(name, path))
        # Drop repeats keeping the first; a dict keeps file order, a set would not.
        targets = list(dict.fromkeys(targets))
        if parent is not None:
            outside = [target for target in targets if target not in parent]
            if outside:
                raise TreeParseError(f"{path}: {_names(outside)} are not in the enclosing targets {_names(parent)}.")
        return targets

    def _all_qubits(self, path: str) -> list[Target]:
        """Returns every qubit in the runcard's analog settings."""
        settings = self.platform.analog_compilation_settings
        if settings is None or not settings.qubits:
            raise TreeParseError(f"{path}: all_q needs the qubits in the runcard's analog settings.")
        try:
            return [QubitTarget(qubit) for qubit in settings.qubits]
        except ValueError as error:
            raise TreeParseError(f"{path}: all_q: {error}") from error

    def _all_couplers(self, path: str) -> list[Target]:
        """Returns every coupler in the runcard's analog topology, with its qubits in either order."""
        settings = self.platform.analog_compilation_settings
        if settings is None or not settings.topology:
            raise TreeParseError(f"{path}: all_c needs the topology in the runcard's analog settings.")
        try:
            return [CouplerTarget.between(edge[0], edge[1]) for edge in settings.topology]
        except ValueError as error:
            raise TreeParseError(f"{path}: all_c: {error}") from error


def _target(name: Any, path: str) -> Target:
    """Parses one target name written in the tree."""
    try:
        return ExperimentTarget.from_str(name)
    except ValueError as error:
        raise TreeParseError(f"{path}: {error}") from error


def _names(targets: list[Target]) -> list[str]:
    """Target names, for messages."""
    return [str(target) for target in targets]


def _after(raw: Any, path: str) -> list[str]:
    """Normalises a pipeline's ``after`` to a list of step names."""
    if raw is None:
        return []
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise TreeParseError(f"{path}.after: must be a step name or a list of step names.")
    return list(names)


def _check_structural(values: Mapping[str, Any], path: str) -> None:
    """Rejects keys that can only be set on the step itself."""
    structural = sorted(_STEP_KEYS.intersection(values))
    if structural:
        raise TreeParseError(f"{path}: {structural} can only be set on the step itself.")


def _check_reserved(values: Mapping[str, Any], path: str) -> None:
    """Rejects parameters the tree cannot set."""
    reserved = sorted(_RESERVED_PARAMETERS.intersection(values))
    if reserved:
        raise TreeParseError(f"{path}: {reserved} cannot be set from the calibration tree.")
