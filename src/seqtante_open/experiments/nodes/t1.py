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

"""The ``t1`` node: a T1 saturation benchmark at the sweet spot (ADR-0002, ADR-0006, ADR-0007, QHC-1803).

The first vertical slice and the first benchmark node. One T1 saturation sweep runs at the step's
operating point (the base places it), an exponential-decay fit gives the relaxation time with its
uncertainty, and ``analyze`` reports that value in microseconds -- never a verdict: the run scores it
against the tree's thresholds (QHC-1848). T1 is higher-is-better. ``apply`` writes nothing, as a
benchmark does (ADR-0002). The fitted ``t1`` and ``fit_r2`` go to the DB (QHC-1497, QHC-1791).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.experiment_classes import t1_pulsed_saturation_experiment
from seqtante_open.experiments.fitting import T1Fit
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from seqtante_open.targets import Target

EXPONENTIAL_DECAY = "exponential_decay"

#: Default minimum R² a T1 fit must reach to pass the step (overridable per step).
DEFAULT_MIN_R2 = 0.8


class T1Node(FixedFluxNode[None]):
    """One T1 saturation sweep at the sweet spot, fit to an exponential decay and scored as a benchmark (QHC-1803).

    The benchmark unit (``us``) and direction (higher-is-better) come from the ``t1`` entry in
    ``NODE_REGISTRY``; the run reads them to score the reported value against the tree's thresholds.
    """

    experiment: ClassVar[str | None] = "t1"
    analyses: ClassVar[tuple[str, ...]] = (EXPONENTIAL_DECAY,)

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the T1 saturation sweep on each target at its operating point; return its measurement id."""
        db_manager = output_controller.db_manager
        experiment = self.experiment
        # A registered leaf node always sets experiment (enforced in __init_subclass__); narrow for the type checker.
        if experiment is None:
            raise RuntimeError(f"{type(self).__name__} declares no experiment")
        measurements: dict[Target, list[int]] = {}
        for target in targets:
            params = self.parameters[target]
            point = operating_point_for(self.calibration_file, experiment, target)
            durations_idle = np.linspace(*params["idle_sweep"]).astype(int)
            measurement_id = t1_pulsed_saturation_experiment(
                platform=self.context.platform,
                db_manager=db_manager,
                readout_bus=bus_alias(target, Role.readout),
                drive_bus=bus_alias(target, Role.drive),
                durations_idle=durations_idle,
                drive_if_freq=point.drive.if_,
                drive_duration=params["drive_duration"],
                drive_amplitude=params["drive_amplitude"],
                drive_gain=params["drive_power"],
                readout_if_freq=int(point.readout.if_),
                readout_duration=params["readout_duration"],
                readout_amplitude=params["readout_amplitude"],
                overlap_time=params.get("overlap_time", 0),
                relax_duration=params["relax_duration"],
                averages=params["averages"],
                ringup_time=params.get("ringup_time", 0),
                duration_ringup_drive=params.get("duration_ringup_drive", 200),
                drive_sigmas=params.get("drive_sigmas", 4),
                drive_LO=int(point.drive.lo),
                readout_LO=int(point.readout.lo),
                target=str(target),
                calibration=self.context.calibration,
            )
            measurements[target] = [measurement_id]
        return measurements

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[None]:
        """Fit the decay, save ``t1``/``fit_r2`` to the DB, and report T1 and its uncertainty in us; read-only.

        Raises:
            AnalysisError: when the fit's R² is below the step's minimum (the target fails the step).
        """
        params = self.parameters[target]
        model = T1Fit(target=str(target), measurement_id=measurements[-1], path=params.get("data_folder"))
        model.fit()
        plot = model.plot()
        model.save_fitting()

        min_r2 = params.get("min_r2", DEFAULT_MIN_R2)
        if model.r_squared < min_r2:
            raise AnalysisError(f"{target} T1 fit R² {model.r_squared:.3f} is below the minimum {min_r2}")

        return AnalysisResult(
            value=model.t1_us, uncertainty=model.t1_uncertainty_us, plots=(plot,) if plot else ()
        )

    def apply(self, target: Target, result: AnalysisResult[None]) -> None:
        """A benchmark writes nothing to the calibration file; the run records the scored verdict (ADR-0002)."""
