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

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from qili_qprograms.bus_alias import Role, bus_alias
from qili_qprograms.experiments import t1_saturation
from qililab.result import StreamArray

from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import T1Fit
from seqtante_open.experiments.results import measurement_id
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qili_qprograms.targets import Target

EXPONENTIAL_DECAY = "exponential_decay"

#: Stream/experiment name the T1 measurement is stored under.
_EXPERIMENT_NAME = "t1_pulsed_saturation"

#: Defaults for the optional step parameters (the node's ``defaults``); every one is overridable in the
#: tree. The measurement keys are named like the ``t1_saturation`` arguments. ``idle_sweep``, ``averages``,
#: ``relax_duration``, ``drive_duration`` and ``drive_amplitude`` have no default, and the drive instrument
#: decides which of ``drive_power`` (Rohde & Schwarz, dBm) or ``drive_gain`` (QCM-RF, [-1, 1]) the step
#: must set.
T1_DEFAULTS: Mapping[str, Any] = {
    "drive_ringup_time": 200,
    "drive_ringdown_time": 200,
    "drive_n_sigmas": 4,
    "overlap_time": 0,
    "readout_ringup_time": 0,
    # Minimum R² a T1 fit must reach to pass the step.
    "min_r2": 0.8,
    # Maximum relative uncertainty (sigma_T1 / T1): catches a fit that clears R² but whose lifetime is
    # ill-determined, and the ``inf`` uncertainty ``curve_fit`` returns when it cannot estimate the covariance.
    "max_rel_uncertainty": 0.5,
}


class T1Node(FixedFluxNode[None]):
    """One T1 saturation sweep at the sweet spot, fit to an exponential decay and scored as a benchmark (QHC-1803).

    The benchmark unit (``us``) and direction (higher-is-better) come from the ``t1`` entry in
    ``NODE_REGISTRY``; the run reads them to score the reported value against the tree's thresholds.
    """

    experiment = "t1"
    analyses = (EXPONENTIAL_DECAY,)
    defaults = T1_DEFAULTS

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the T1 saturation sweep on each target at its operating point; return its measurement id."""
        return {target: [self._run_saturation_sweep(target)] for target in targets}

    def _run_saturation_sweep(self, target: Target) -> int:
        """Build the T1 saturation QProgram, set the operating point's hardware, run the sweep and stream it.

        The QProgram is the shared ``qili_qprograms`` experiment: it reads the readout pulse, its weights
        and the drive/readout IFs for the operating point from the calibration file, and plays the
        saturation drive on the target's buses. The LOs, readout attenuation and drive source are set by
        the shared :meth:`CalibrationNode._set_operating_point_hardware`, through the restore handle.
        """
        section = self.calibration_file.experiments[self.experiment][str(target)]
        readout_bus = bus_alias(target, Role.readout)
        # Round, not truncate, so the stored time axis is the one t1_saturation plays.
        durations_idle = np.rint(np.linspace(*self._param(target, "idle_sweep"))).astype(int)

        qprogram = t1_saturation(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=section.operating_point,
            idle_duration_sweep=durations_idle,
            averages=self._param(target, "averages"),
            relax_duration=self._param(target, "relax_duration"),
            drive_duration=self._param(target, "drive_duration"),
            drive_amplitude=self._param(target, "drive_amplitude"),
            drive_ringup_time=self._param(target, "drive_ringup_time"),
            drive_ringdown_time=self._param(target, "drive_ringdown_time"),
            drive_n_sigmas=self._param(target, "drive_n_sigmas"),
            overlap_time=self._param(target, "overlap_time"),
            readout_ringup_time=self._param(target, "readout_ringup_time"),
        )

        self._set_operating_point_hardware(target)

        platform = self.context.platform
        stream_array = StreamArray(
            shape=(len(durations_idle), 2),
            loops={
                "time": {
                    "array": durations_idle,
                    "units": "ns",
                    "bus": bus_alias(target, Role.drive),
                    "parameter": "time",
                }
            },
            platform=platform,
            experiment_name=_EXPERIMENT_NAME,
            db_manager=output_controller.db_manager,
            qprogram=qprogram,
            calibration=self.context.calibration,
            qubit_idx=str(target),
        )
        with stream_array:
            results = platform.execute_qprogram(qprogram).results
            for idx in range(len(durations_idle)):
                stream_array[idx,] = results[readout_bus][idx].array.T
        return measurement_id(stream_array)

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[None]:
        """Fit the decay, check it, save ``t1``/``fit_r2``, plot, and report T1 and its uncertainty in us; read-only.

        Fit, then every acceptance check, then save, then plot: a rejected fit never reaches the DB (a
        reader would otherwise see a plausible ``t1`` for a step the run marked failed, QHC-1803).

        Raises:
            AnalysisError: when the fit is not a physical decay (``rate >= 0``), its R² is below the
                step's minimum or undefined (a flat trace), or its relative uncertainty exceeds the step's
                maximum. The target then fails the step.
        """
        params = self.parameters[target]
        # The graph sets data_folder to <data_root>/<step>/<target>; None shows the plot instead of saving it.
        model = T1Fit(target=str(target), measurement_id=measurements[-1], path=params.get("data_folder"))
        model.fit()

        if not model.is_physical_decay:
            raise AnalysisError(f"{target} T1 fit is not a decay (rate {model.rate:.3e} >= 0); T1 is not physical")

        min_r2 = self._param(target, "min_r2")
        # A flat trace leaves R² NaN; reject it explicitly, since every comparison with NaN is False.
        if math.isnan(model.r_squared) or model.r_squared < min_r2:
            raise AnalysisError(f"{target} T1 fit R² {model.r_squared:.3f} is below the minimum {min_r2}")

        max_rel_uncertainty = self._param(target, "max_rel_uncertainty")
        relative_uncertainty = model.relative_uncertainty
        if not math.isfinite(relative_uncertainty) or relative_uncertainty > max_rel_uncertainty:
            raise AnalysisError(
                f"{target} T1 relative uncertainty {relative_uncertainty:.3f} exceeds the maximum {max_rel_uncertainty}"
            )

        model.save_fitting()
        plot = model.plot()
        return AnalysisResult(value=model.t1_us, uncertainty=model.t1_uncertainty_us, plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[None]) -> None:
        """A benchmark writes nothing to the calibration file; the run records the scored verdict (ADR-0002)."""
