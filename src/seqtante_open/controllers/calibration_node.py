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

"""The abstract base every experiment subclasses, and the context a run hands each node (ADR-0004).

A :class:`CalibrationNode` is one step of the calibration tree made runnable: the parser builds one
per step (ADR-0003). Each subclass declares its ``experiment`` name and the analyses it supports, and
registers itself in :data:`~seqtante_open.controllers.node_registry.NODE_REGISTRY` through
``__init_subclass__``. The three-phase lifecycle (``execute_node`` / ``analyze`` / ``apply``, ADR-0006)
lands on this base in a later PR; here the base carries the declarations, the run context and the graph
contract (``idx``, ``name``, ``run``) that :class:`~seqtante_open.controllers.calibration_graph.CalibrationGraph`
drives.

During the port the graph also runs the experiments that are still legacy functions, through a
legacy-function adapter that satisfies the same :class:`GraphNode` contract without subclassing this
base; that adapter and the graph build live with the graph task (QHC-1800, ADR-0007).
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from qililab import Platform
from qililab.qprogram.calibration import Calibration

from seqtante_open.controllers.node_registry import BenchmarkSpec, Parallel, register_node
from seqtante_open.targets import Target


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


class CalibrationNode(ABC):
    """The abstract base every experiment subclasses (ADR-0004).

    A subclass declares its ``experiment`` name and ``analyses`` as class variables, and registers
    itself in :data:`~seqtante_open.controllers.node_registry.NODE_REGISTRY` automatically. The
    mid-level spines (``FixedFluxNode``, ``FluxSweepNode``) set no ``experiment`` and are not
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

    def run(self, dead_targets: dict[str, str]) -> list[str]:
        """Run the three phases for every live target (ADR-0006).

        The three-phase body lands on this base with the lifecycle PR (task 5c); until then the base
        raises. It is a concrete method, not an abstract one: 5c fills this body rather than every
        subclass implementing its own, so a leaf subclass is not forced to override it (ADR-0006).

        Args:
            dead_targets (dict[str, str]): Targets already stopped by an earlier node, mapped to the
                node that stopped them.

        Returns:
            list[str]: The targets whose calibration failed or was skipped in this node.
        """
        raise NotImplementedError("the three-phase run lands with the lifecycle PR (task 5c)")
