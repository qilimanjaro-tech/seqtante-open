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

"""What the calibration-tree parser needs to know about each experiment.

Stand-in until node classes declare it themselves through ``NODE_REGISTRY`` (ADR-0004).
"""

from dataclasses import dataclass
from typing import Literal

Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class BenchmarkSpec:
    """How a benchmark reports its value.

    Args:
        unit (str | None): Unit of the value, ``None`` when it is dimensionless.
        direction (Direction): Whether a higher or a lower value is better.
    """

    unit: str | None
    direction: Direction


@dataclass(frozen=True)
class ExperimentSpec:
    """Declarations of one experiment.

    Args:
        analyses (tuple[str, ...]): Analyses the experiment supports. The first one is the default.
        benchmark (BenchmarkSpec | None): How the experiment reports its value, when it is a benchmark.
    """

    analyses: tuple[str, ...]
    benchmark: BenchmarkSpec | None = None

    @property
    def default_analysis(self) -> str:
        return self.analyses[0]


_EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
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


def experiment_spec(experiment: str) -> ExperimentSpec | None:
    """Returns the declarations of ``experiment``, or ``None`` if it does not exist."""
    return _EXPERIMENT_SPECS.get(experiment)


def known_experiments() -> set[str]:
    """Returns the names of every experiment."""
    return set(_EXPERIMENT_SPECS)
