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
the analyses it supports (ADR-0003, ADR-0006). The parser reads this registry to validate a step's
``experiment`` and ``analysis`` — nothing else needs to know which experiments exist.

During the port (ADR-0007) not every experiment is a node class yet: the four experiments that still
run as legacy functions live in :data:`seqtante_open.experiments.experiment_functions`, and the graph
builder picks the legacy function when a registry entry carries no node class. The experiments that
have neither a class nor a legacy function yet (``rabi``, ``t1``, ``t2`` ...) are seeded here as
declarations only, so the parser accepts a tree that names them before their node lands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seqtante_open.controllers.calibration_node import CalibrationNode


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
class ExperimentSpec:
    """What the parser and the graph builder need to know about one experiment.

    Args:
        analyses (tuple[str, ...]): Analyses the experiment supports (ADR-0003). The first is the default.
        node_cls (type[CalibrationNode] | None): The node class that implements the experiment, or
            ``None`` while the experiment is still a legacy function or has no implementation yet (ADR-0007).
    """

    analyses: tuple[str, ...]
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
    "t1": ExperimentSpec(analyses=("exponential_decay",)),
    "t2": ExperimentSpec(analyses=("decaying_oscillation",)),
    "landau_zener": ExperimentSpec(analyses=("theory_agreement",)),
}


def register_node(cls: type[CalibrationNode]) -> None:
    """Record ``cls`` as the implementation of its ``experiment``, called from ``__init_subclass__``.

    The class's ``analyses`` replace a seeded declaration if one exists, so a node class is the single
    source of truth for its experiment once it lands (ADR-0004).

    Raises:
        ValueError: if the class sets no ``experiment`` name, or another node class already claims it.
    """
    experiment = cls.experiment
    if experiment is None:
        raise ValueError(f"{cls.__name__} has no experiment name to register under")
    existing = NODE_REGISTRY.get(experiment)
    if existing is not None and existing.node_cls is not None:
        raise ValueError(
            f"experiment '{experiment}' is already implemented by {existing.node_cls.__name__}; "
            f"{cls.__name__} cannot claim the same name"
        )
    analyses = tuple(cls.analyses)
    if existing is not None:
        NODE_REGISTRY[experiment] = replace(existing, analyses=analyses, node_cls=cls)
    else:
        NODE_REGISTRY[experiment] = ExperimentSpec(analyses=analyses, node_cls=cls)


def experiment_spec(experiment: str) -> ExperimentSpec | None:
    """The declarations of ``experiment``, or ``None`` if it does not exist."""
    return NODE_REGISTRY.get(experiment)


def known_experiments() -> set[str]:
    """The names of every experiment the parser accepts."""
    return set(NODE_REGISTRY)
