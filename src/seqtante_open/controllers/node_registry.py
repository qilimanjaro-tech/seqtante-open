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

"""The one registry of experiments: name -> declarations (and the node class that implements it).

``NODE_REGISTRY`` replaces ``experiment_functions_dict`` (ADR-0004). Each :class:`CalibrationNode`
subclass registers itself here through ``__init_subclass__`` from its ``experiment`` name, declaring
the analyses it supports and, for a benchmark, how it reports its value (ADR-0003, ADR-0006). The
parser reads this registry to validate a step's ``experiment`` and ``analysis`` and to convert its
thresholds to the unit the benchmark declares (ADR-0002) — nothing else needs to know which
experiments exist.

During the port (ADR-0007) not every experiment is a node class yet: the four experiments that still
run as legacy functions live in :data:`seqtante_open.experiments.experiment_functions`, and the graph
builder picks the legacy function when a registry entry carries no node class. The experiments that
have neither a class nor a legacy function yet (``rabi``, ``t1``, ``t2`` ...) are seeded here as
declarations only, so the parser accepts a tree that names them before their node lands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from seqtante_open.controllers.calibration_node import CalibrationNode


#: Whether a higher or a lower benchmark value is the better calibration (ADR-0002).
Direction = Literal["higher", "lower"]


class Parallel(StrEnum):
    """Whether an experiment may run on several targets at once (ADR-0004).

    Every current experiment is ``forbidden``: parallel execution is not supported on fluxonium yet,
    so the tree has no ``simultaneous`` key. The value is declared so that supporting it later does
    not touch every node.
    """

    forbidden = "forbidden"
    allowed = "allowed"
    required = "required"


@dataclass(frozen=True)
class BenchmarkSpec:
    """How a benchmark reports its value (ADR-0002).

    Args:
        unit (str | None): Unit of the value, ``None`` when it is dimensionless.
        direction (Direction): Whether a higher or a lower value is better.
    """

    unit: str | None
    direction: Direction


@dataclass(frozen=True)
class ExperimentSpec:
    """What the parser and the graph builder need to know about one experiment.

    Args:
        analyses (tuple[str, ...]): Analyses the experiment supports (ADR-0003). The first is the default.
        benchmark (BenchmarkSpec | None): How the experiment reports its value when it is a benchmark,
            or ``None`` for a recalibration (ADR-0002).
        node_cls (type[CalibrationNode] | None): The node class that implements the experiment, or
            ``None`` while the experiment is still a legacy function or has no implementation yet (ADR-0007).
    """

    analyses: tuple[str, ...]
    benchmark: BenchmarkSpec | None = None
    node_cls: type[CalibrationNode] | None = None

    @property
    def default_analysis(self) -> str:
        """The analysis a step runs when it names none: the first declared (ADR-0003)."""
        return self.analyses[0]


#: Every experiment the parser accepts, keyed by name. Seeded with declarations for the experiments
#: that have no node class yet; :func:`register_node` fills in ``node_cls`` as each class lands.
NODE_REGISTRY: dict[str, ExperimentSpec] = {
    "offset_calibration": ExperimentSpec(analyses=("flux_offset",)),
    "single_tone": ExperimentSpec(analyses=("lorentzian_peak",)),
    "two_tone": ExperimentSpec(analyses=("lorentzian_peak",)),
    "two_tone_vs_flux": ExperimentSpec(analyses=("parabola_vertex",)),
    "reset_point_search": ExperimentSpec(analyses=("lorentzian_peak",)),
    "rabi": ExperimentSpec(analyses=("pi_amplitude", "readout_levels")),
    "t1": ExperimentSpec(analyses=("exponential_decay",), benchmark=BenchmarkSpec(unit="ns", direction="higher")),
    "t2": ExperimentSpec(analyses=("decaying_oscillation",), benchmark=BenchmarkSpec(unit="ns", direction="higher")),
    "landau_zener": ExperimentSpec(
        analyses=("theory_agreement",), benchmark=BenchmarkSpec(unit=None, direction="lower")
    ),
}


def _as_analyses(analyses: str | Iterable[str]) -> tuple[str, ...]:
    """Normalise a declared ``analyses`` to a tuple, accepting a lone name as a one-element tuple.

    A subclass that writes ``analyses = "flux_offset"`` means the single analysis ``"flux_offset"``,
    not the characters ``("f", "l", "u", ...)`` that ``tuple("flux_offset")`` would give.
    """
    if isinstance(analyses, str):
        return (analyses,)
    return tuple(analyses)


def register_node(cls: type[CalibrationNode]) -> None:
    """Record ``cls`` as the implementation of its ``experiment``, called from ``__init_subclass__``.

    The class's ``analyses`` and ``benchmark`` replace a seeded declaration if one exists, so a node
    class is the single source of truth for its experiment once it lands (ADR-0004). A class that
    declares no ``benchmark`` keeps the seeded one, so a benchmark declared here is not lost when the
    node lands without re-declaring it.

    Re-importing a node module (``importlib.reload``, IPython ``%autoreload``) re-runs
    ``__init_subclass__`` for the same class; that is not a clash and is allowed to re-register.

    Raises:
        ValueError: if the class sets no ``experiment`` name, or a *different* node class already
            claims it.
    """
    experiment = cls.experiment
    if experiment is None:
        raise ValueError(f"{cls.__name__} has no experiment name to register under")
    existing = NODE_REGISTRY.get(experiment)
    if existing is not None and existing.node_cls is not None and not _same_class(existing.node_cls, cls):
        raise ValueError(
            f"experiment '{experiment}' is already implemented by {existing.node_cls.__name__}; "
            f"{cls.__name__} cannot claim the same name"
        )
    analyses = _as_analyses(cls.analyses)
    seeded_benchmark = existing.benchmark if existing is not None else None
    benchmark = cls.benchmark if cls.benchmark is not None else seeded_benchmark
    if existing is not None:
        NODE_REGISTRY[experiment] = replace(existing, analyses=analyses, benchmark=benchmark, node_cls=cls)
    else:
        NODE_REGISTRY[experiment] = ExperimentSpec(analyses=analyses, benchmark=benchmark, node_cls=cls)


def _same_class(a: type, b: type) -> bool:
    """Whether two class objects are the same class, including across a module reload.

    A reload builds a new class object with the same module and qualified name; identity (``is``) does
    not hold, so compare by ``__module__`` and ``__qualname__``.
    """
    return (a.__module__, a.__qualname__) == (b.__module__, b.__qualname__)


def experiment_spec(experiment: str) -> ExperimentSpec | None:
    """The declarations of ``experiment``, or ``None`` if it does not exist."""
    return NODE_REGISTRY.get(experiment)


def known_experiments() -> set[str]:
    """The names of every experiment the parser accepts."""
    return set(NODE_REGISTRY)
