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
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from loguru import logger
from qililab import Platform

from seqtante_open.controllers.experiment_specs import (
    BenchmarkSpec,
    Direction,
    ExperimentSpec,
    experiment_spec,
    known_experiments,
)
from seqtante_open.targets import CouplerTarget, ExperimentTarget, QubitTarget, Target

_STEP_KEYS = frozenset({"experiment", "analysis", "targets", "overrides"})
_RESERVED_PARAMETERS = frozenset({"data_folder"})
# Each shorthand: the analog-settings field it expands from, and how one item of that field becomes a target.
_EXPANSIONS: dict[str, tuple[str, Callable[[Any], Target]]] = {
    "all_q": ("qubits", QubitTarget),
    "all_c": ("topology", lambda edge: CouplerTarget.between(edge[0], edge[1])),
}
_THRESHOLD_KEYS = frozenset({"passed", "warning"})
_TIME_UNITS = {
    "ns": 1,
    "us": 1_000,
    # Micro sign, U+00B5.
    "\u00b5s": 1_000,
    # Greek mu, U+03BC.
    "\u03bcs": 1_000,
    "ms": 1_000_000,
}
_QUANTITY = re.compile(r"(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>\S+)", re.ASCII)


class TreeParseError(ValueError):
    """The calibration tree is malformed or names something that does not exist."""


@dataclass(frozen=True)
class Thresholds:
    """Benchmark thresholds, converted to the unit the benchmark reports.

    Args:
        passed (float): A value at least this good is Passed.
        warning (float): A value at least this good, but not Passed, is Warning. Anything worse is Failed.
        unit (str | None): Unit of ``passed`` and ``warning``, ``None`` when dimensionless.
        direction (Direction): Whether a higher or a lower value is better.
    """

    passed: float
    warning: float
    unit: str | None
    direction: Direction


@dataclass(frozen=True)
class Step:
    """One named step of a pipeline.

    Args:
        name (str): Step name, unique across the tree.
        pipeline (str): Name of the pipeline the step belongs to.
        experiment (str): Experiment the step runs.
        analysis (str): Analysis the step runs, the experiment's default when the tree names none.
        targets (list[Target]): Targets the step runs on.
        parameters (dict[Target, dict[str, Any]]): Resolved parameters per target.
    """

    name: str
    pipeline: str
    experiment: str
    analysis: str
    targets: list[Target]
    parameters: dict[Target, dict[str, Any]]


class _Override(NamedTuple):
    """One entry of a step's ``overrides``.

    Args:
        targets (list[Target]): Targets of the entry that the step runs on.
        values (dict[str, Any]): Parameters the entry sets.
    """

    targets: list[Target]
    values: dict[str, Any]


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
        self, name: str, raw: object, parent_targets: list[Target], defaults: dict[str, dict[str, Any]]
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
        self, name: str, pipeline: str, raw: object, parent_targets: list[Target], defaults: dict[str, dict[str, Any]]
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

        targets = self._narrowed_targets(raw, parent_targets, path)

        own = {key: value for key, value in raw.items() if key not in _STEP_KEYS}
        _check_settable(own, path)
        own = _with_thresholds(own, experiment, spec, path)
        overrides = self._overrides(raw.get("overrides", []), targets, experiment, spec, path)

        base = {**defaults.get(experiment, {}), **own}
        parameters: dict[Target, dict[str, Any]] = {}
        for target in targets:
            resolved = base
            for override in overrides:
                if target in override.targets:
                    resolved = resolved | override.values
            parameters[target] = copy.deepcopy(resolved)

        return Step(
            name=name,
            pipeline=pipeline,
            experiment=experiment,
            analysis=analysis,
            targets=targets,
            parameters=parameters,
        )

    def _overrides(
        self, raw: object, step_targets: list[Target], experiment: str, spec: ExperimentSpec, step_path: str
    ) -> list[_Override]:
        """Parses a step's overrides, dropping targets the step does not run on."""
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
            targets = self._targets(entry["targets"], parent=None, path=f"{entry_path}.targets")
            kept = [target for target in targets if target in step_targets]
            ignored = [target for target in targets if target not in step_targets]
            if ignored:
                logger.warning(
                    "{path}.targets: {ignored} are not targets of the step. (Ignored)",
                    path=entry_path,
                    ignored=_names(ignored),
                )
            values = {key: value for key, value in entry.items() if key != "targets"}
            _check_settable(values, entry_path)
            overrides.append(_Override(targets=kept, values=_with_thresholds(values, experiment, spec, entry_path)))
        return overrides

    @staticmethod
    def _defaults(raw: object) -> dict[str, dict[str, Any]]:
        """Parses the ``defaults`` block into parameters per experiment."""
        if not isinstance(raw, Mapping):
            raise TreeParseError("defaults: must be a mapping of experiment names to parameters.")
        defaults = {}
        for experiment, values in raw.items():
            path = f"defaults.{experiment}"
            if not isinstance(values, Mapping):
                raise TreeParseError(f"{path}: must be a mapping of parameters.")
            _check_settable(values, path)
            spec = experiment_spec(experiment)
            defaults[experiment] = dict(values) if spec is None else _with_thresholds(values, experiment, spec, path)
        return defaults

    def _narrowed_targets(self, raw: Mapping[str, object], parent: list[Target], path: str) -> list[Target]:
        """Returns the ``targets`` of a pipeline or step, or ``parent`` when it sets none."""
        if "targets" not in raw:
            return list(parent)
        return self._targets(raw["targets"], parent=parent, path=f"{path}.targets")

    def _targets(self, raw: object, parent: list[Target] | None, path: str) -> list[Target]:
        """Parses target names, expands ``all_q`` and ``all_c`` in place, drops repeats keeping the first, and checks
        the result is within ``parent``, when given."""
        if not isinstance(raw, list) or not raw:
            raise TreeParseError(f"{path}: must be a non-empty list of targets.")
        targets: list[Target] = []
        for name in raw:
            if isinstance(name, str) and name in _EXPANSIONS:
                targets += self._expand(name, path)
            else:
                targets.append(_target(name, path))
        # Drop repeats keeping the first; a dict keeps file order, a set would not.
        targets = list(dict.fromkeys(targets))
        if parent is not None:
            outside = [target for target in targets if target not in parent]
            if outside:
                raise TreeParseError(f"{path}: {_names(outside)} are not in the enclosing targets {_names(parent)}.")
        return targets

    def _expand(self, name: str, path: str) -> list[Target]:
        """Expands ``all_q`` to every qubit in the runcard's analog settings, or ``all_c`` to every coupler in its
        topology."""
        field, to_target = _EXPANSIONS[name]
        items = getattr(self.platform.analog_compilation_settings, field, None)
        if not items:
            raise TreeParseError(f"{path}: {name} needs the {field} in the runcard's analog settings.")
        try:
            return [to_target(item) for item in items]
        except ValueError as error:
            raise TreeParseError(f"{path}: {name}: {error}") from error


def _target(name: object, path: str) -> Target:
    """Parses one target name written in the tree."""
    if not isinstance(name, str):
        raise TreeParseError(f"{path}: Invalid target {name!r}: must be a string such as 'q1' or 'c1_2'.")
    try:
        return ExperimentTarget.from_str(name)
    except ValueError as error:
        raise TreeParseError(f"{path}: {error}") from error


def _names(targets: list[Target]) -> list[str]:
    """Target names, for messages."""
    return [str(target) for target in targets]


def _after(raw: object, path: str) -> list[str]:
    """Normalises a pipeline's ``after`` to a list of step names."""
    if raw is None:
        return []
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise TreeParseError(f"{path}.after: must be a step name or a list of step names.")
    return list(names)


def _check_settable(values: Mapping[str, Any], path: str) -> None:
    """Rejects keys that only the step itself can set, then parameters the tree cannot set at all."""
    for keys, reason in (
        (_STEP_KEYS, "can only be set on the step itself"),
        (_RESERVED_PARAMETERS, "cannot be set from the calibration tree"),
    ):
        found = sorted(keys.intersection(values))
        if found:
            raise TreeParseError(f"{path}: {found} {reason}.")


def _with_thresholds(values: Mapping[str, Any], experiment: str, spec: ExperimentSpec, path: str) -> dict[str, Any]:
    """Returns a copy of ``values`` with its ``thresholds``, if any, parsed for ``experiment``."""
    parsed = dict(values)
    if "thresholds" in parsed:
        parsed["thresholds"] = _thresholds(parsed["thresholds"], experiment, spec.benchmark, f"{path}.thresholds")
    return parsed


def _thresholds(raw: object, experiment: str, benchmark: BenchmarkSpec | None, path: str) -> Thresholds:
    """Parses ``passed`` and ``warning`` into the benchmark's unit and checks they leave a Warning band."""
    if benchmark is None:
        raise TreeParseError(f"{path}: '{experiment}' is not a benchmark, so it takes no thresholds.")
    if not isinstance(raw, Mapping) or set(raw) != _THRESHOLD_KEYS:
        raise TreeParseError(f"{path}: must be a mapping with exactly 'passed' and 'warning'.")
    passed = _threshold_value(raw["passed"], benchmark, f"{path}.passed")
    warning = _threshold_value(raw["warning"], benchmark, f"{path}.warning")
    if passed == warning:
        raise TreeParseError(f"{path}: passed and warning are equal, so no value can be Warning.")
    higher = benchmark.direction == "higher"
    if (passed > warning) != higher:
        raise TreeParseError(
            f"{path}: passed ({raw['passed']}) must be {'above' if higher else 'below'} warning ({raw['warning']}), "
            f"since {benchmark.direction} is better for '{experiment}'."
        )
    return Thresholds(passed=passed, warning=warning, unit=benchmark.unit, direction=benchmark.direction)


def _threshold_value(raw: object, benchmark: BenchmarkSpec, path: str) -> float:
    """Parses one threshold, a plain number when dimensionless or a number with a time unit, into the benchmark's
    unit."""
    if isinstance(raw, bool):
        raise TreeParseError(f"{path}: {raw!r} is not a number.")
    if benchmark.unit is None:
        if not isinstance(raw, int | float):
            raise TreeParseError(f"{path}: {raw!r}: the benchmark is dimensionless, so it takes a plain number.")
        return float(raw)
    if isinstance(raw, int | float):
        raise TreeParseError(f"{path}: {raw!r} needs a unit, such as '50 us'.")
    match = _QUANTITY.fullmatch(raw.strip()) if isinstance(raw, str) else None
    if match is None:
        raise TreeParseError(f"{path}: {raw!r} must be a number with a unit, such as '50 us'.")
    unit = match["unit"]
    if unit not in _TIME_UNITS:
        raise TreeParseError(f"{path}: unknown unit '{unit}'. Time thresholds take ns, us or ms.")
    return float(match["number"]) * _TIME_UNITS[unit] / _TIME_UNITS[benchmark.unit]
