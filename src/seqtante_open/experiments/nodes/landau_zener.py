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

"""The ``landau_zener`` node: a compartmentalized Landau-Zener benchmark at the reset point (QHC-1808).

MVP port of the compartmentalized experiment (QHC-1822). ``run`` biases the slow flux lines to the
operating point before ``execute_node`` (the base places it); the experiment then plays its reset,
Landau-Zener path and readout ramps relative to that idle point. ``analyze`` normalises the measured
signal to a transition probability with the reset-point readout levels and plots it; ``apply`` writes
nothing, as a benchmark does (ADR-0002).

The benchmark is left **"Not evaluated"**: the agreement metric and the theory curve that give it a
verdict are a follow-up (QHC-1921), so ``analyze`` reports no value and the run does not score it (the
tree sets no thresholds for ``landau_zener`` yet, so :meth:`CalibrationNode._score` leaves it unscored).

Inputs are split between the calibration and the tree (QHC-1808):

- From the calibration's ``landau_zener`` operating point: the idle DC flux (``idle_point_dc``), the
  readout LO/IF, and the |0>/|1> readout levels.
- From the step's tree parameters: the annealing paths and times, the ramp durations, the reset and
  readout flux points, the reset/readout durations, the averages, the readout amplitude/ring-up, and
  the AC crosstalk matrix (a raw array wrapped with ``CrosstalkMatrix.from_array``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.calibration.io import readout_levels_for
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.experiment_classes import lz_transition_not_idling_reset__custom_linear_ramp
from seqtante_open.experiments.fitting.landau_zener_fit import LandauZenerFit
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from seqtante_open.targets import Target

EXPERIMENT = "landau_zener"
THEORY_AGREEMENT = "theory_agreement"


class LandauZenerNode(FixedFluxNode[None]):
    """A compartmentalized Landau-Zener transition at the reset point, scored as a benchmark (QHC-1808).

    The benchmark unit (dimensionless) and direction (lower-is-better) come from the ``landau_zener``
    entry in ``NODE_REGISTRY``. Until the agreement metric and theory curve land (QHC-1921) the node
    reports no value, so the run leaves the benchmark "Not evaluated".
    """

    experiment: ClassVar[str | None] = EXPERIMENT
    analyses: ClassVar[tuple[str, ...]] = (THEORY_AGREEMENT,)

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the compartmentalized Landau-Zener sweep on each target; return its measurement id.

        The base has biased the slow flux lines to the operating point (the idle point); the
        experiment plays the reset hold, ramps and path relative to it, with the AC crosstalk from the
        tree baked into the waveforms.
        """
        db_manager = output_controller.db_manager
        measurements: dict[Target, list[int]] = {}
        for target in targets:
            params = self.parameters[target]
            point = operating_point_for(self.calibration_file, EXPERIMENT, target)
            if point.flux.x is None:
                raise AnalysisError(f"{target}: the '{EXPERIMENT}' operating point has no x flux for the idle point")

            flux_buses = [bus_alias(target, Role.flux, "z"), bus_alias(target, Role.flux, "x")]
            crosstalk = CrosstalkMatrix.from_array(
                buses=flux_buses, matrix_array=np.asarray(params["ac_crosstalk"], dtype=float)
            )

            measurement_id, _ = lz_transition_not_idling_reset__custom_linear_ramp(
                platform=self.context.platform,
                db_manager=db_manager,
                readout_bus=bus_alias(target, Role.readout),
                readout_if_freq=int(point.readout.if_),
                flux_buses=flux_buses,
                flux_readout=list(params["flux_readout"]),
                flux_reset=list(params["flux_reset"]),
                idle_point_dc=[point.flux.z, point.flux.x],
                reset_dur=params["reset_duration"],
                annealing_paths_list=params["annealing_paths"],
                landau_zener_annealing_time=params["annealing_times"],
                readout_dur=params["readout_duration"],
                crosstalk=crosstalk,
                averages=params["averages"],
                ramp_reset_duration=list(params["ramp_reset_duration"]),
                ramp_readout_duration=list(params["ramp_readout_duration"]),
                readout_amp=params.get("readout_amplitude", 0.1),
                ringup_time=params.get("ringup_time", 0),
                readout_LO=int(point.readout.lo),
            )
            measurements[target] = [measurement_id]
        return measurements

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[None]:
        """Normalise the signal with the reset-point readout levels and plot it; report no value yet.

        Reads the |0>/|1> readout levels from the operating point, refusing levels taken with a
        different readout pulse than this step plays (ADR-0001, D8). The agreement metric that would
        give a value (and so a verdict) is a follow-up (QHC-1921), so ``value`` is ``None`` and the run
        leaves the benchmark unscored.

        Raises:
            AnalysisError: when the operating point stores no readout levels, or the section declares
                no readout pulse to check them against.
        """
        params = self.parameters[target]
        readout_pulse = self._readout_pulse(target)
        levels = readout_levels_for(self.calibration_file, EXPERIMENT, str(target), readout_pulse)
        if levels is None:
            raise AnalysisError(
                f"{target}: the '{EXPERIMENT}' operating point stores no readout levels to normalise with"
            )

        model = LandauZenerFit(
            measurement_id=measurements[-1],
            target=str(target),
            level_0=levels.level_0.nominal,
            level_1=levels.level_1.nominal,
            path=params.get("data_folder"),
        )
        plot = model.plot()
        # MVP: no agreement metric yet (QHC-1921), so report no value; the run leaves this unscored.
        return AnalysisResult(value=None, plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[None]) -> None:
        """A benchmark writes nothing to the calibration file; the run records the verdict (ADR-0002)."""

    def _readout_pulse(self, target: Target) -> str:
        """The readout pulse this step plays, for the ADR-0001 D8 readout-levels check.

        The compartmentalized experiment plays an inline ``Square`` readout, so the pulse the stored
        levels must match is the one the ``landau_zener`` section declares for the readout role.

        Raises:
            AnalysisError: if the section names no readout pulse, so the levels cannot be checked.
        """
        section = self.calibration_file.experiments[EXPERIMENT][str(target)]
        pulse = section.pulses.get(Role.readout)
        if pulse is None:
            raise AnalysisError(
                f"{target}: the '{EXPERIMENT}' section declares no readout pulse, so the readout levels "
                "cannot be checked against the pulse this step plays (ADR-0001, D8)"
            )
        return pulse
