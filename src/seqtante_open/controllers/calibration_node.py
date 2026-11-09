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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from qililab import Platform
from qililab.qprogram.calibration import Calibration

from seqtante_open.calibration.io import read_calibration_file, save_calibration, write_calibration_file
from seqtante_open.calibration.model import CalibrationFile
from seqtante_open.controllers.crosstalk_session import CrosstalkSession, crosstalk_session
from seqtante_open.controllers.node_registry import BenchmarkSpec, Parallel, register_node
from seqtante_open.controllers.operating_point import apply_operating_point
from seqtante_open.controllers.parameter_restore import ParameterRestore, restoring_parameters
from seqtante_open.controllers.verdicts import BenchmarkResult
from seqtante_open.targets import Target

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
        benchmark (BenchmarkResult | None): The scored result a benchmark reports to the run and the
            report (ADR-0002); ``None`` for a recalibration.
        plots (tuple[str, ...]): Paths to the fit plots the report shows per target (ADR-0002).
    """

    fitted: ResultT | None = None
    benchmark: BenchmarkResult | None = None
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
    """

    experiment: ClassVar[str | None] = None
    analyses: ClassVar[tuple[str, ...]] = ()
    applies_operating_point: ClassVar[bool] = True
    parallel: ClassVar[Parallel] = Parallel.forbidden
    benchmark: ClassVar[BenchmarkSpec | None] = None

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
    ) -> None:
        """Build one runnable step.

        Args:
            name (str): The step's name, unique across the tree (ADR-0003).
            idx (int): The node's position key in the graph.
            targets (list[Target]): The targets the step runs on.
            parameters (dict[Target, dict[str, Any]]): Resolved parameters per target (ADR-0005).
            context (RunContext): The run's platform and calibration file.
        """
        self.name = name
        self.idx = idx
        self.targets = targets
        self.parameters = parameters
        self.context = context
        self.benchmark_results: tuple[BenchmarkResult, ...] = ()
        self._file: CalibrationFile | None = None
        self._restore: ParameterRestore | None = None
        self._crosstalk: CrosstalkSession | None = None

    @property
    def is_benchmark(self) -> bool:
        """Whether this node is a benchmark: it writes nothing and reports a verdict (ADR-0002)."""
        return self.benchmark is not None

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

    def run(self, dead_targets: dict[str, str]) -> list[str]:
        """Run the three phases for every live target and return the ones that failed (ADR-0006).

        Targets already stopped by an earlier node are skipped. The crosstalk session zeroes flux at
        the start and bias at the end around the whole node; each target is then run on its own, with
        its operating point placed right before it runs and undone right after, so a point that parks
        a neighbour is never held while that neighbour is measured, and two targets' fluxes never sit
        on the fridge at once. A target whose phase raises -- a failed fit (``AnalysisError``) or any
        other error -- fails that target alone and the node moves on (D17); benchmark results are
        collected on :attr:`benchmark_results`.

        Args:
            dead_targets (dict[str, str]): Targets already stopped by an earlier node, mapped to the
                node that stopped them.

        Returns:
            list[str]: The targets that failed in this node (a fit error or any other error).
        """
        live = [target for target in self.targets if str(target) not in dead_targets]
        if not live:
            return []

        failed: list[str] = []
        benchmarks: list[BenchmarkResult] = []
        applied = False
        platform = self.context.platform
        with crosstalk_session(platform, self.context.calibration) as crosstalk:
            self._crosstalk = crosstalk
            try:
                for target in live:
                    try:
                        with restoring_parameters(platform) as restore:
                            self._restore = restore
                            if self.applies_operating_point:
                                self._place_operating_point(target)
                            measurements = self.execute_node([target])
                            result = self.analyze(target, measurements[target])
                            self.apply(target, result)
                        applied = True
                        if result.benchmark is not None:
                            benchmarks.append(result.benchmark)
                    except Exception:  # ruff: ignore[blind-except] -- one target's failure must not abort the node
                        failed.append(str(target))
                    finally:
                        self._restore = None
                if applied and not self.is_benchmark:
                    self._save_calibration()
            finally:
                self._crosstalk, self._file = None, None

        self.benchmark_results = tuple(benchmarks)
        return failed

    def _place_operating_point(self, target: Target) -> None:
        experiment = self.experiment
        # Spines set no experiment; placing an operating point on one is a programming error.
        if experiment is None:
            raise RuntimeError(f"{type(self).__name__} applies an operating point but declares no experiment")
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
    three phases to the experiment.
    """

    applies_operating_point: ClassVar[bool] = True


class FluxSweepNode(CalibrationNode[ResultT]):
    """A node that sweeps a flux loop across the operating point (ADR-0004).

    The home for the code the per-loop flux-sweep experiments share (``two_tone_vs_flux``, and
    ``offset_calibration`` after December), including the coupler-to-readout-qubit mapping. It lands
    with task 7; the node moves flux itself rather than being pinned to one operating point, so the
    spine opts out of the automatic placement.
    """

    applies_operating_point: ClassVar[bool] = False
