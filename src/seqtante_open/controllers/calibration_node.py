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

"""The abstract base every experiment subclasses, and the three-phase lifecycle it runs (ADR-0004, ADR-0006).

A :class:`CalibrationNode` is one step of the calibration tree made runnable (ADR-0003). Each subclass
declares its ``experiment`` name and the analyses it supports, registers itself in
:data:`~seqtante_open.controllers.node_registry.NODE_REGISTRY` through ``__init_subclass__``, and
implements three phases that :meth:`CalibrationNode.run` calls in order (ADR-0006):

- ``execute_node`` -- hardware only; returns the measurement ids per target.
- ``analyze`` -- read-only; fits the measurement and returns an :class:`AnalysisResult`, raising
  :class:`~seqtante_open.controllers.verdicts.AnalysisError` when it cannot (the target fails the step).
- ``apply`` -- the only writer; writes the operating point, a pulse, a section or the crosstalk offset.
  It runs only for targets whose ``analyze`` succeeded, and a benchmark's ``apply`` writes nothing.

``run`` wraps the phases in the crosstalk session and the restore context manager, places the targets
at their operating point when ``applies_operating_point``, collects benchmark results for the report,
and returns the targets that failed so the graph can skip their later steps (ADR-0004).

The two mid-level spines :class:`FixedFluxNode` and :class:`FluxSweepNode` set no ``experiment`` and
are not registered; they hold the code the two experiment shapes share as the nodes are ported.
"""

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
from loguru import logger
from qili_qprograms.bus_alias import AXES, Axis, Role, bus_alias
from qili_qprograms.calibration import pulse_name
from qili_qprograms.calibration.io import read_calibration_file, save_calibration, write_calibration_file
from qili_qprograms.calibration.model import CalibrationFile, ExperimentSection
from qili_qprograms.targets import Target
from qililab import Parameter, Platform
from qililab.qprogram.calibration import Calibration
from qililab.result import StreamArray
from qililab.typings import ParameterValue

from seqtante_open.controllers.crosstalk_session import CrosstalkSession, crosstalk_session
from seqtante_open.controllers.drive_source import drive_source_writes
from seqtante_open.controllers.node_registry import BenchmarkSpec, Parallel, experiment_spec, register_node
from seqtante_open.controllers.operating_point import (
    apply_operating_point,
    operating_point_flux_writes,
    operating_point_for,
)
from seqtante_open.controllers.parameter_restore import ParameterRestore, restoring_parameters
from seqtante_open.controllers.verdicts import (
    AnalysisError,
    BenchmarkResult,
    BenchmarkThresholds,
    UnscoredBenchmark,
    verdict_for,
)
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from qililab.qprogram import QProgram
    from qililab.result import AutocalMeasurement, Measurement

#: How :meth:`CalibrationNode._stream` writes into the stream: ``write(stream_array, averaged)``, once per
#: execution, where ``averaged`` holds the readout bus's acquisition arrays averaged over the executions so
#: far. With ``repetitions == 1`` it is that one execution's arrays.
StreamWrite = Callable[[StreamArray, Sequence[np.ndarray]], None]


def write_per_point(stream_array: StreamArray, averaged: Sequence[np.ndarray]) -> None:
    """One acquisition per sweep point (an unrolled sweep, e.g. T1/T2 idle times): write each at its index."""
    for index, array in enumerate(averaged):
        stream_array[index,] = array.T


def write_whole(stream_array: StreamArray, averaged: Sequence[np.ndarray]) -> None:
    """One acquisition holding the whole sweep (a hardware loop, e.g. rabi, two-tone): write it in one go."""
    stream_array[()] = averaged[0].T


#: The pulse purposes a node asks :meth:`CalibrationNode._pulse` for: the ``<purpose>`` of a
#: ``<purpose>-<operating_point>`` pulse name (ADR-0008).
PulsePurpose = Literal["pi", "pi_half", "readout"]

#: The node-specific fitted payload an ``analyze`` hands its ``apply`` (a frequency, a pi amplitude,
#: readout levels...). The base passes it through untouched; each node fixes its own concrete type.
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RunContext:
    """What a run hands to every node: the platform and the one calibration file the run reads (ADR-0004).

    The run deserialises the ``Calibration`` once from ``calibration_path`` and passes this context to
    every node, so a node reads and writes the same in-memory calibration rather than loading its own.

    Args:
        platform (Platform): The platform the run drives.
        platform_path (str): Path to the runcard the platform was built from. The runcard is never
            saved from a run (ADR-0001).
        calibration (Calibration): The calibration file for the run, deserialised once.
        calibration_path (str): Path the calibration file is read from and saved back to.
    """

    platform: Platform
    platform_path: str
    calibration: Calibration
    calibration_path: str


@dataclass(frozen=True)
class AnalysisResult(Generic[ResultT]):
    """What a node's ``analyze`` produces for one target (ADR-0006). Read-only; ``apply`` does the writing.

    Args:
        fitted (ResultT | None): The values ``apply`` writes into the calibration file -- a fitted
            frequency, a pi amplitude, a reset flux, readout levels. Its shape is the node's own; the
            base passes it from ``analyze`` to ``apply`` untouched. A pure benchmark leaves it ``None``.
        value (float | None): A benchmark's measured value, in the benchmark's unit. The run scores it
            against the tree's thresholds with :func:`~seqtante_open.controllers.verdicts.verdict_for`
            (QHC-1848), so ``analyze`` reports the value and never the verdict. ``None`` for a recalibration.
        uncertainty (float | None): The standard uncertainty on ``value`` where the analysis gives one.
        plots (tuple[str, ...]): Paths to the fit plots the report shows per target (ADR-0002).
    """

    fitted: ResultT | None = None
    value: float | None = None
    uncertainty: float | None = None
    plots: tuple[str, ...] = ()


@runtime_checkable
class GraphNode(Protocol):
    """What :class:`CalibrationGraph` needs of a node, whatever implements it (ADR-0007).

    Both :class:`CalibrationNode` subclasses and the legacy-function adapter (QHC-1800) satisfy this,
    so the graph drives them the same way.
    """

    idx: int
    name: str

    def run(self, dead_targets: dict[str, str]) -> list[str]:
        """Run the node, returning the targets whose calibration failed or was skipped."""
        ...


class CalibrationNode(ABC, Generic[ResultT]):
    """The abstract base every experiment subclasses (ADR-0004, ADR-0006).

    A subclass declares its ``experiment`` name and ``analyses`` as class variables, registers itself
    in :data:`~seqtante_open.controllers.node_registry.NODE_REGISTRY` automatically, and implements
    ``execute_node``, ``analyze`` and ``apply``. The mid-level spines set no ``experiment`` and are not
    registered; only leaf experiments are.

    Class variables:
        experiment (str): The experiment name the step's ``experiment`` field matches. A subclass that
            sets none is treated as an abstract spine and is not registered.
        analyses (tuple[str, ...]): The analyses the experiment supports (ADR-0003). The first is the
            default when a step names none.
        applies_operating_point (bool): Whether ``run`` places the targets at the step's operating
            point before ``execute_node`` (ADR-0004). Default ``True``.
        parallel (Parallel): Whether the experiment may run on several targets at once. Default
            :attr:`~seqtante_open.controllers.node_registry.Parallel.forbidden`.
        benchmark (BenchmarkSpec | None): How a benchmark reports its value (unit and direction,
            ADR-0002), or ``None`` for a recalibration. Registered into the experiment's
            :class:`~seqtante_open.controllers.node_registry.ExperimentSpec` and consumed by the
            threshold parsing and the verdict logic.
        defaults (Mapping[str, Any]): The values :meth:`_param` falls back to for a step parameter the
            tree leaves unset; a key with no default is required. Default: none.
    """

    experiment: ClassVar[str | None] = None
    analyses: ClassVar[tuple[str, ...]] = ()
    applies_operating_point: ClassVar[bool] = True
    parallel: ClassVar[Parallel] = Parallel.forbidden
    benchmark: ClassVar[BenchmarkSpec | None] = None
    defaults: ClassVar[Mapping[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A spine (FixedFluxNode/FluxSweepNode) sets no experiment and is not a registrable experiment;
        # only a leaf experiment registers. A leaf must declare at least one analysis.
        if cls.__dict__.get("experiment") is None:
            return
        if not cls.analyses:
            raise ValueError(f"{cls.__name__} sets experiment '{cls.experiment}' but declares no analyses")
        register_node(cls)

    def __init__(
        self,
        *,
        name: str,
        idx: int,
        targets: list[Target],
        parameters: dict[Target, dict[str, Any]],
        context: RunContext,
        analysis: str | None = None,
        thresholds: dict[Target, BenchmarkThresholds] | None = None,
    ) -> None:
        """Build one runnable step.

        Args:
            name (str): The step's name, unique across the tree (ADR-0003).
            idx (int): The node's position key in the graph.
            targets (list[Target]): The targets the step runs on.
            parameters (dict[Target, dict[str, Any]]): Resolved parameters per target (ADR-0005).
            context (RunContext): The run's platform and calibration file.
            analysis (str | None): The analysis this step runs, chosen from the experiment's ``analyses``
                (ADR-0003). ``None`` falls back to the experiment's default analysis, so a node built
                directly (e.g. in a test) still has one. A node with more than one analysis dispatches
                its ``analyze``/``apply`` on this.
            thresholds (dict[Target, BenchmarkThresholds] | None): The benchmark's Passed/Warning
                thresholds per target, parsed into the node's unit (QHC-1848). ``None`` or empty for a
                recalibration, or a benchmark target the tree sets no thresholds for. Kept apart from
                ``parameters`` so acceptance policy is not mixed into the experiment parameters.
        """
        self.name = name
        self.idx = idx
        self.targets = targets
        self.parameters = parameters
        self.context = context
        self.analysis = analysis if analysis is not None else self._default_analysis()
        self._thresholds: dict[Target, BenchmarkThresholds] = thresholds or {}
        self.benchmark_results: tuple[BenchmarkResult, ...] = ()
        # Benchmark targets measured this run but not scored (no thresholds or no value), so the report
        # can list them as "Not evaluated" instead of dropping them (ADR-0002, QHC-1848).
        self.unscored_benchmarks: tuple[UnscoredBenchmark, ...] = ()
        # Fit plots produced this run, keyed by target, so the report can show a benchmark's plot and
        # (later) the per-target fit plots. Populated in ``_collect`` and reset at the start of ``run``.
        self.analysis_plots: dict[str, tuple[str, ...]] = {}
        self._file: CalibrationFile | None = None
        self._restore: ParameterRestore | None = None
        self._crosstalk: CrosstalkSession | None = None

    def _default_analysis(self) -> str | None:
        """The experiment's default analysis (first declared), or ``None`` for a spine with no experiment."""
        if self.experiment is not None:
            spec = experiment_spec(self.experiment)
            if spec is not None:
                return spec.default_analysis
        return self.analyses[0] if self.analyses else None

    @property
    def is_benchmark(self) -> bool:
        """Whether this node is a benchmark: it writes nothing and reports a verdict (ADR-0002).

        Reads the registry spec, not the class's own ``benchmark``: ``register_node`` keeps a seeded
        ``BenchmarkSpec`` when a node class does not redeclare one, so a node that leaves ``benchmark``
        unset but whose experiment is seeded as a benchmark is still a benchmark (and must not save).
        """
        if self.experiment is None:
            return False
        spec = experiment_spec(self.experiment)
        return spec is not None and spec.benchmark is not None

    @property
    def calibration_file(self) -> CalibrationFile:
        """The run's calibration-file content, parsed from the shared ``Calibration`` once and cached."""
        if self._file is None:
            self._file = read_calibration_file(self.context.calibration)
        return self._file

    @property
    def restore(self) -> ParameterRestore:
        """The restore handle a phase sets platform parameters through; valid only during ``run``."""
        if self._restore is None:
            raise RuntimeError("the restore handle is only available while the node is running")
        return self._restore

    @property
    def crosstalk(self) -> CrosstalkSession:
        """The run's crosstalk session a recalibration's ``apply`` writes offsets through; only during ``run``."""
        if self._crosstalk is None:
            raise RuntimeError("the crosstalk session is only available while the node is running")
        return self._crosstalk

    def _param(self, target: Target, key: str) -> Any:
        """The step's ``key`` for ``target``, else the node's :attr:`defaults` entry for it.

        Raises:
            KeyError: if the step sets no ``key`` for ``target`` and the node has no default for it.
        """
        params = self.parameters[target]
        if key in params:
            return params[key]
        if key in self.defaults:
            return self.defaults[key]
        raise KeyError(f"step '{self.name}': {target} sets no '{key}' and {type(self).__name__} has no default for it")

    def _leaf_experiment(self) -> str:
        """The node's ``experiment``.

        Raises:
            TypeError: on a spine, which sets no experiment; only a leaf node measures (QHC-1932).
        """
        experiment = self.experiment
        if experiment is None:
            raise TypeError(f"{type(self).__name__} sets no experiment; only a leaf node can measure")
        return experiment

    def _section(self, target: Target) -> ExperimentSection:
        """``target``'s section of the node's experiment in the calibration file (ADR-0008)."""
        experiment = self._leaf_experiment()
        return self.calibration_file.experiments[experiment][str(target)]

    def _sweep(self, target: Target, key: str) -> np.ndarray:
        """The step's ``key`` window: ``np.linspace`` arguments ``[start, stop, num]``.

        Raises:
            ValueError: if the window has fewer than 2 points, which no fit can use.
        """
        sweep = np.linspace(*self._param(target, key))
        if len(sweep) < 2:
            raise ValueError(f"step '{self.name}': {target} {key} must have at least 2 points, got {len(sweep)}")
        return sweep

    def _absolute_sweep(self, target: Target, key: str, centre: float) -> np.ndarray:
        """The step's ``key`` window (:meth:`_sweep`, as offsets) shifted onto ``centre``.

        Sweeps are offsets from the operating point's stored value (QHC-1932), so the same tree recentres
        after every recalibration.
        """
        return centre + self._sweep(target, key)

    def _require_r2(self, target: Target, r_squared: float) -> None:
        """Fail ``target`` unless the fit's R² reaches the step's ``min_r2``.

        Raises:
            AnalysisError: if R² is below ``min_r2``, or undefined (NaN, a flat trace). Every comparison
                with NaN is False, so it is rejected explicitly.
        """
        min_r2 = self._param(target, "min_r2")
        if math.isnan(r_squared) or r_squared < min_r2:
            raise AnalysisError(f"{target} {self.experiment} fit R² {r_squared:.3f} is below the minimum {min_r2}")

    def _require_rel_uncertainty(self, target: Target, relative_uncertainty: float) -> None:
        """Fail ``target`` unless the fitted value's relative uncertainty is within the step's ``max_rel_uncertainty``.

        Raises:
            AnalysisError: if it exceeds ``max_rel_uncertainty`` or is not finite (``curve_fit`` could not
                estimate the covariance, or the value is non-physical).
        """
        max_rel_uncertainty = self._param(target, "max_rel_uncertainty")
        if not math.isfinite(relative_uncertainty) or relative_uncertainty > max_rel_uncertainty:
            raise AnalysisError(
                f"{target} {self.experiment} relative uncertainty {relative_uncertainty:.3f} exceeds the maximum "
                f"{max_rel_uncertainty}"
            )

    def _pulse(self, target: Target, role: Role, purpose: PulsePurpose) -> str:
        """The pulse ``target`` plays on ``role``: ``<purpose>-<operating_point>`` (ADR-0008, QHC-1932).

        That is the name the shared ``qili_qprograms`` experiments play (``readout-<op>``, ``pi-<op>``).

        Raises:
            ValueError: if the section names a different pulse for ``role``, which the QProgram would not play.
        """
        section = self._section(target)
        played = pulse_name(purpose, section.operating_point)
        named = section.pulses.get(role, played)
        if named != played:
            raise ValueError(
                f"{target}: the {self.experiment} section names {role.value} pulse '{named}', but its QProgram "
                f"plays '{played}' at operating point '{section.operating_point}'"
            )
        return played

    def _require_pulse(self, target: Target, role: Role, purpose: PulsePurpose) -> str:
        """:meth:`_pulse`, checked to exist on its bus, so a missing pulse fails before any hardware is set.

        Raises:
            ValueError: if the section names another pulse, or the pulse is not in ``Calibration.waveforms``.
        """
        name = self._pulse(target, role, purpose)
        bus = bus_alias(target, role)
        if not self.context.calibration.has_waveform(bus, name):
            raise ValueError(
                f"{target}: {self.experiment} needs the '{name}' pulse on {bus}, which is not in "
                "Calibration.waveforms; it comes with the manual calibration"
            )
        return name

    def _stream(
        self,
        target: Target,
        qprogram: "QProgram",
        *,
        experiment_name: str,
        loops: Mapping[str, Any],
        shape: tuple[int, ...],
        write: StreamWrite = write_whole,
        repetitions: int = 1,
        optional_identifier: str | None = None,
    ) -> int:
        """Run ``qprogram`` ``repetitions`` times, stream the running average with ``write``, and return the measurement id.

        Every node measures through here (QHC-1932): the stream array is built the same way, and only how
        the averaged acquisitions land in it differs (:func:`write_per_point`, :func:`write_whole`, or a
        node's own). After every execution the stream holds the average over the executions so far, so an
        interrupted run keeps what it measured.

        Raises:
            ValueError: if ``repetitions`` is below 1, which would run nothing and save an empty measurement.
        """
        if repetitions < 1:
            raise ValueError(f"{target} {self.experiment}: repetitions must be at least 1, not {repetitions}")
        platform = self.context.platform
        readout_bus = bus_alias(target, Role.readout)
        stream_array = StreamArray(
            shape=shape,
            loops=loops,
            platform=platform,
            experiment_name=experiment_name,
            db_manager=output_controller.db_manager,
            qprogram=qprogram,
            calibration=self.context.calibration,
            optional_identifier=optional_identifier,
            qubit_idx=str(target),
        )
        total: list[np.ndarray] = []
        with stream_array:
            for execution in range(1, repetitions + 1):
                arrays = [
                    np.asarray(acquisition.array)
                    for acquisition in platform.execute_qprogram(qprogram).results[readout_bus]
                ]
                total = (
                    arrays if execution == 1 else [summed + array for summed, array in zip(total, arrays, strict=True)]
                )
                write(stream_array, [summed / execution for summed in total])
        return cast("Measurement | AutocalMeasurement", stream_array.measurement).measurement_id

    def _drive_source_writes(self, target: Target) -> list[tuple[Parameter, ParameterValue]]:
        """The writes that arm ``target``'s drive source, its level read from the step or :attr:`defaults`.

        ``drive_power`` (Rohde & Schwarz, dBm) or ``drive_gain`` (QCM-RF, [-1, 1]), whichever the drive
        bus's instrument takes (:func:`~seqtante_open.controllers.drive_source.drive_source_writes`).

        Raises:
            ValueError: if the drive instrument's level is neither set in the step nor defaulted.
        """
        params = self.parameters[target]
        return drive_source_writes(
            self.context.platform,
            bus_alias(target, Role.drive),
            drive_power=params.get("drive_power", self.defaults.get("drive_power")),
            drive_gain=params.get("drive_gain", self.defaults.get("drive_gain")),
        )

    def _set_drive_source(self, target: Target) -> None:
        """Arm ``target``'s drive source through the restore handle; any node can, with no operating point.

        Raises:
            ValueError: if the drive instrument's level is neither set in the step nor defaulted.
        """
        drive_bus = bus_alias(target, Role.drive)
        for parameter, value in self._drive_source_writes(target):
            self.restore.set(drive_bus, parameter, value)

    def _set_operating_point_hardware(self, target: Target) -> None:
        """Set ``target``'s line at its operating point, then arm its drive, through the restore handle (ADR-0006).

        The QProgram sets the IFs and plays the pulses; this sets what it does not: the drive and readout
        LOs and the readout attenuation from the operating point, then the drive source
        (:meth:`_set_drive_source`). Every write goes through :attr:`restore`, so ``run`` puts the line
        back as it was after each target and nothing leaks into the next node. The drive level is
        checked before the first write, so a missing one leaves the line untouched.

        Raises:
            TypeError: if called on a node that sets no ``experiment`` (a spine, not a leaf).
            ValueError: if the readout attenuation is not a whole number of dB, or the drive instrument's
                level is neither set in the step nor defaulted.
        """
        experiment = self._leaf_experiment()
        point = operating_point_for(self.calibration_file, experiment, target)
        attenuation = point.readout.attenuation
        if not float(attenuation).is_integer():
            raise ValueError(f"{target}: readout attenuation {attenuation} dB is not a whole number of dB")
        drive_writes = self._drive_source_writes(target)
        drive_bus = bus_alias(target, Role.drive)
        readout_bus = bus_alias(target, Role.readout)

        restore = self.restore
        restore.set(drive_bus, Parameter.LO_FREQUENCY, round(point.drive.lo))
        restore.set(readout_bus, Parameter.LO_FREQUENCY, round(point.readout.lo))
        restore.set(readout_bus, Parameter.OUT0_ATT, int(attenuation))
        for parameter, value in drive_writes:
            restore.set(drive_bus, parameter, value)

    def run(self, dead_targets: dict[str, str]) -> list[str]:
        """Run the three phases for every live target and return the ones that failed (ADR-0006).

        Targets already stopped by an earlier node are skipped, and the crosstalk session zeroes flux
        at the start and bias at the end around the whole node. A ``forbidden``-parallel node runs each
        target on its own, placing its operating point right before it runs and undoing it right after,
        so a point that parks a neighbour is never held while that neighbour is measured and two
        targets' fluxes never sit on the fridge at once. An ``allowed``/``required`` node runs its
        targets as one group: their points are placed together and ``execute_node`` is called once with
        the whole group (:attr:`parallel`). Either way a target whose phase raises -- a failed fit
        (``AnalysisError``) or any other error -- fails that target alone and the node moves on (D17);
        benchmark results are collected on :attr:`benchmark_results`, and benchmark targets that could
        not be scored on :attr:`unscored_benchmarks`.

        Args:
            dead_targets (dict[str, str]): Targets already stopped by an earlier node, mapped to the
                node that stopped them.

        Returns:
            list[str]: The targets that failed in this node (a fit error or any other error).
        """
        # Reset before the early return below, so a re-run of this node keeps no stale verdicts or plots.
        self.benchmark_results = ()
        self.unscored_benchmarks = ()
        self.analysis_plots = {}
        live = [target for target in self.targets if str(target) not in dead_targets]
        if not live:
            return []

        failed: list[str] = []
        benchmarks: list[BenchmarkResult | UnscoredBenchmark] = []
        platform = self.context.platform
        with crosstalk_session(platform, self.context.calibration) as crosstalk:
            self._crosstalk = crosstalk
            try:
                if self.parallel is Parallel.forbidden:
                    applied = self._run_one_at_a_time(live, failed, benchmarks)
                else:
                    applied = self._run_as_group(live, failed, benchmarks)
                if applied and not self.is_benchmark:
                    self._save_calibration()
            finally:
                self._crosstalk, self._file = None, None

        self.benchmark_results = tuple(b for b in benchmarks if isinstance(b, BenchmarkResult))
        self.unscored_benchmarks = tuple(b for b in benchmarks if isinstance(b, UnscoredBenchmark))
        return failed

    def _run_one_at_a_time(
        self, live: list[Target], failed: list[str], benchmarks: list[BenchmarkResult | UnscoredBenchmark]
    ) -> bool:
        """Run each target in its own operating-point and restore scope (``forbidden`` parallel)."""
        platform = self.context.platform
        applied = False
        for target in live:
            try:
                with restoring_parameters(platform) as restore:
                    self._restore = restore
                    if self.applies_operating_point:
                        self._place_operating_point(target)
                    measurements = self.execute_node([target])[target]
                    # analyze/apply run inside the restore scope, so anything a phase sets is undone.
                    applied = self._collect(target, measurements, failed, benchmarks) or applied
            except Exception:  # ruff: ignore[blind-except] -- a placement/measurement/restore failure fails this target
                logger.opt(exception=True).warning(
                    "Node <i>{node}</i>: target <r>{target}</r> failed", node=self.name, target=str(target)
                )
                failed.append(str(target))
            finally:
                self._restore = None
        return applied

    def _run_as_group(
        self, live: list[Target], failed: list[str], benchmarks: list[BenchmarkResult | UnscoredBenchmark]
    ) -> bool:
        """Place every target's point and measure the whole group in one call (``allowed``/``required``)."""
        platform = self.context.platform
        applied = False
        # Reject a flux conflict between targets' points up front -- this is meant to fail loudly (no
        # parallel-placement policy yet), so it sits outside the restore-failure guard below.
        if self.applies_operating_point and self.experiment is not None:
            self._reject_flux_conflicts(live, self.experiment)
        try:
            with restoring_parameters(platform) as restore:
                self._restore = restore
                group = self._place_group(live, failed) if self.applies_operating_point else list(live)
                measurements = None
                if group:
                    try:
                        measurements = self.execute_node(group)
                    except Exception:  # ruff: ignore[blind-except] -- a group-wide measurement failure fails the group
                        logger.opt(exception=True).warning(
                            "Node <i>{node}</i>: group measurement failed", node=self.name
                        )
                        failed.extend(str(target) for target in group)
                if measurements is not None:
                    for target in group:
                        try:
                            target_measurements = measurements[target]
                        except KeyError:
                            logger.opt(exception=True).warning(
                                "Node <i>{node}</i>: no measurement for <r>{target}</r>",
                                node=self.name,
                                target=str(target),
                            )
                            failed.append(str(target))
                            continue
                        applied = self._collect(target, target_measurements, failed, benchmarks) or applied
        except Exception:  # ruff: ignore[blind-except] -- a restore-on-exit failure must not abort the graph (parity with forbidden mode)
            logger.opt(exception=True).warning(
                "Node <i>{node}</i>: restoring parameters failed after the group ran", node=self.name
            )
        finally:
            self._restore = None
        return applied

    def _place_group(self, live: list[Target], failed: list[str]) -> list[Target]:
        """Place every target's operating point for a parallel group, before measuring (ADR-0006).

        Places each target, dropping one whose placement raises so a single bad section does not fail
        the whole group. The conflict check that must fail loudly runs in :meth:`_run_as_group` first.
        """
        placed: list[Target] = []
        for target in live:
            try:
                self._place_operating_point(target)
                placed.append(target)
            except Exception:  # ruff: ignore[blind-except] -- a target with no placeable point drops from the group
                logger.opt(exception=True).warning(
                    "Node <i>{node}</i>: placing <r>{target}</r>'s operating point failed",
                    node=self.name,
                    target=str(target),
                )
                failed.append(str(target))
        return placed

    def _reject_flux_conflicts(self, live: list[Target], experiment: str) -> None:
        """Raise if two targets' operating points write the same flux bus to different values."""
        planned: dict[str, tuple[str, float]] = {}
        for target in live:
            try:
                writes = operating_point_flux_writes(self.calibration_file, experiment, target)
            except Exception:  # ruff: ignore[blind-except] -- an uncomputable point drops here; placing for real handles it
                writes = {}
            for bus, value in writes.items():
                prior = planned.get(bus)
                if prior is not None and prior[1] != value:
                    raise RuntimeError(
                        f"parallel placement conflict on bus '{bus}' between {prior[0]} and {target}: "
                        f"{prior[1]} vs {value}. Parallel operating-point placement has no policy yet."
                    )
                planned[bus] = (str(target), value)

    def _collect(
        self,
        target: Target,
        measurements: list[int],
        failed: list[str],
        benchmarks: list[BenchmarkResult | UnscoredBenchmark],
    ) -> bool:
        """Analyze and apply one target, isolating its failure; return whether ``apply`` ran (D17).

        A failed ``apply`` is rolled back: the cached calibration file and the crosstalk offsets are
        restored to their pre-apply state, so a half-written target is never saved (ADR-0002).
        """
        try:
            result = self.analyze(target, measurements)
        except Exception:  # ruff: ignore[blind-except] -- a failed analysis fails only this target
            logger.opt(exception=True).warning(
                "Node <i>{node}</i>: analysing <r>{target}</r> failed", node=self.name, target=str(target)
            )
            failed.append(str(target))
            return False

        file_before = copy.deepcopy(self._file)
        offsets_before = self._crosstalk.snapshot_offsets() if self._crosstalk is not None else None
        try:
            self.apply(target, result)
        except Exception:  # ruff: ignore[blind-except] -- roll the partial write back, then fail only this target
            self._file = file_before
            if offsets_before is not None and self._crosstalk is not None:
                self._crosstalk.restore_offsets(offsets_before)
            logger.opt(exception=True).warning(
                "Node <i>{node}</i>: applying <r>{target}</r> failed; rolled back", node=self.name, target=str(target)
            )
            failed.append(str(target))
            return False

        if result.plots:
            self.analysis_plots[str(target)] = result.plots
        if self.is_benchmark:
            # Scoring runs inside this per-target guard so a non-finite value or a scoring error
            # fails only this target (D17), rather than escaping to the group-level handler.
            try:
                scored = self._score(target, result)
            except Exception:  # ruff: ignore[blind-except] -- a bad benchmark value fails only this target (D17)
                logger.opt(exception=True).warning(
                    "Node <i>{node}</i>: scoring <r>{target}</r> failed", node=self.name, target=str(target)
                )
                failed.append(str(target))
                return False
            benchmarks.append(scored)
        return True

    def _score(self, target: Target, result: AnalysisResult[ResultT]) -> BenchmarkResult | UnscoredBenchmark:
        """Score one benchmark target's measured value against its thresholds (QHC-1848, ADR-0002, D17).

        Returns the :class:`~seqtante_open.controllers.verdicts.BenchmarkResult`, or an
        :class:`~seqtante_open.controllers.verdicts.UnscoredBenchmark` when the benchmark cannot be
        scored -- no measured value or no thresholds -- which is logged and reported as "Not evaluated"
        rather than as a step failure. Raises
        :class:`~seqtante_open.controllers.verdicts.AnalysisError` when the value is non-finite or the
        thresholds do not match the benchmark's unit/direction, so the target fails the step.
        """
        thresholds = self._thresholds_for(target)
        if result.value is None or thresholds is None:
            missing = "measured value" if result.value is None else "thresholds"
            logger.warning(
                "Node <i>{node}</i>: <r>{target}</r> has no {missing}; benchmark left unscored",
                node=self.name,
                target=str(target),
                missing=missing,
            )
            return UnscoredBenchmark(
                target=str(target), value=result.value, uncertainty=result.uncertainty, reason=f"no {missing}"
            )
        if not math.isfinite(result.value):
            raise AnalysisError(f"{target}: benchmark value {result.value} is not finite")
        self._check_scoring_units(thresholds)
        return BenchmarkResult(
            target=str(target),
            value=result.value,
            uncertainty=result.uncertainty,
            verdict=verdict_for(result.value, thresholds),
            thresholds=thresholds,
        )

    def _check_scoring_units(self, thresholds: BenchmarkThresholds) -> None:
        """Guard that ``thresholds`` were parsed in the benchmark's own unit and direction (QHC-1848).

        The parser converts the tree's thresholds with whichever :class:`BenchmarkSpec` the experiment
        declared then; this checks that spec still matches at scoring time, so a value in one unit is
        never scored against thresholds in another (which would score a good qubit Failed silently).
        """
        spec = experiment_spec(self.experiment) if self.experiment is not None else None
        benchmark = spec.benchmark if spec is not None else None
        if benchmark is not None and (thresholds.unit != benchmark.unit or thresholds.direction != benchmark.direction):
            raise AnalysisError(
                f"thresholds unit/direction ({thresholds.unit}, {thresholds.direction}) do not match "
                f"the '{self.experiment}' benchmark spec ({benchmark.unit}, {benchmark.direction})"
            )

    def _thresholds_for(self, target: Target) -> BenchmarkThresholds | None:
        """The parsed Passed/Warning thresholds for ``target`` (ADR-0002).

        The parser converts the tree's thresholds to the benchmark's unit and the node holds them per
        target, apart from ``parameters``; :meth:`_score` scores the measured value against them (QHC-1848).
        """
        return self._thresholds.get(target)

    def _place_operating_point(self, target: Target) -> None:
        experiment = self._leaf_experiment()
        apply_operating_point(self.calibration_file, experiment, target, self.restore)

    def _save_calibration(self) -> None:
        write_calibration_file(self.context.calibration, self.calibration_file)
        save_calibration(self.context.calibration, self.context.calibration_path)

    @abstractmethod
    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Measure every target and return its measurement ids; hardware only, writes nothing (ADR-0006).

        Runs at the operating point ``run`` has placed (unless the node opts out and places it here).
        A node that re-measures in a loop, like the reset-point search, returns the id of every
        measurement so the report shows each one. Temporary platform settings go through
        :attr:`restore`, so ``run`` undoes them.
        """

    @abstractmethod
    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[ResultT]:
        """Fit ``target``'s measurement and return its result; read-only (ADR-0006).

        Raises:
            AnalysisError: when the fit does not converge or its quality is below the tree's threshold.
                The target then fails the step; a ``Failed`` benchmark verdict does not (ADR-0002, D17).
        """

    @abstractmethod
    def apply(self, target: Target, result: AnalysisResult[ResultT]) -> None:
        """Write ``target``'s result into the calibration file or the crosstalk offset (ADR-0006).

        The only writer. Runs only for a target whose ``analyze`` succeeded. Writes into
        :attr:`calibration_file` (``run`` persists it) or through :attr:`crosstalk`; a benchmark's
        ``apply`` writes nothing.
        """


class FixedFluxNode(CalibrationNode[ResultT]):
    """A node that sweeps at the step's operating point (ADR-0004).

    The home for the code the fixed-flux experiments share (``two_tone`` at the sweet spot, ``rabi``,
    ``t1``, ``t2``, ``landau_zener`` and the ``reset_point_search`` loop). It lands as those nodes are
    ported (tasks 6, 8-12); for now the spine fixes :attr:`applies_operating_point` and leaves the
    three phases to the experiment. The step parameters (:meth:`CalibrationNode._param`), the drive
    source (:meth:`CalibrationNode._set_drive_source`) and the operating point's LOs and attenuation
    (:meth:`CalibrationNode._set_operating_point_hardware`) live on the base, since every node uses them.
    """

    applies_operating_point: ClassVar[bool] = True


class FluxSweepNode(CalibrationNode[ResultT]):
    """A node that sweeps a flux loop across the operating point (ADR-0004).

    The home for the code the per-loop flux-sweep experiments share (``two_tone_vs_flux``, QHC-1804,
    and ``offset_calibration`` after December). The node moves flux itself rather than being pinned to
    one operating point, so the spine opts out of the automatic placement; ``two_tone_vs_flux`` places
    the point right before its sweep, so the unswept loop and the parked neighbours hold their biases.
    The coupler-to-readout-qubit mapping lands with the first node that measures a coupler.
    """

    applies_operating_point: ClassVar[bool] = False

    def _loop(self, target: Target) -> Axis:
        """The flux loop the step sweeps on ``target`` (its ``flux_axis``), checked to be ``z`` or ``x``.

        Raises:
            KeyError: if the step sets no ``flux_axis`` and the node has no default for it.
            ValueError: if ``flux_axis`` is not one of :data:`~qili_qprograms.bus_alias.AXES`.
        """
        loop = self._param(target, "flux_axis")
        if loop not in AXES:
            raise ValueError(f"step '{self.name}': {target} flux_axis must be one of {AXES}, not {loop!r}")
        return loop
