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

"""The ``t2`` node: a T2 (Ramsey) DRAG benchmark at the sweet spot (ADR-0002, ADR-0006, ADR-0007, QHC-1809).

One T2 (Ramsey) DRAG sweep runs at the step's operating point (the base places it), a decaying-oscillation
fit gives the dephasing time with its uncertainty, and ``analyze`` reports that value in microseconds --
never a verdict: the run scores it against the tree's thresholds (QHC-1848). T2 is higher-is-better.
``apply`` writes nothing, as a benchmark does (ADR-0002). The fitted ``t2`` and ``fit_r2`` go to the DB
(QHC-1497, QHC-1791). Operationally the T2 pipeline runs after a calibrated pi pulse (rabi), but this
node's code does not depend on the rabi node: the QProgram reads the pi-half pulse from the calibration
file (or the pi pulse at half amplitude).

The T2 (Ramsey) QProgram is the shared ``qili_qprograms`` experiment (QHC-1928): the node builds it and
runs it directly -- no ``experiment_class`` driver layer -- setting the operating point's hardware through
the restore handle (ADR-0006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from qili_qprograms.bus_alias import Role, bus_alias
from qili_qprograms.calibration import pulse_name
from qili_qprograms.experiments import t2_ramsey

from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode, write_per_point
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import T2Fit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qili_qprograms.targets import Target

DECAYING_OSCILLATION = "decaying_oscillation"

#: Stream/experiment name the T2 measurement is stored under.
_EXPERIMENT_NAME = "t2_drag"

#: The node's ``defaults``; every one is overridable in the tree. ``idle_duration_sweep``, ``averages`` and
#: ``relax_duration`` have no default, and the drive instrument decides which of ``drive_power``
#: (Rohde & Schwarz, dBm) or ``drive_gain`` (QCM-RF, [-1, 1]) the step must set.
T2_DEFAULTS: Mapping[str, Any] = {
    "readout_ringup_time": 0,
    # Minimum R² a T2 fit must reach to pass the step.
    "min_r2": 0.8,
    # Maximum relative uncertainty (sigma_T2 / T2): catches a fit that clears R² but whose dephasing time
    # is ill-determined, and the ``inf`` uncertainty ``curve_fit`` returns when it cannot estimate the covariance.
    "max_rel_uncertainty": 0.5,
}


class T2Node(FixedFluxNode[None]):
    """One T2 (Ramsey) DRAG sweep at the sweet spot, fit to a decaying oscillation and scored as a benchmark (QHC-1809).

    The benchmark unit (``us``) and direction (higher-is-better) come from the ``t2`` entry in
    ``NODE_REGISTRY``; the run reads them to score the reported value against the tree's thresholds.
    """

    experiment = "t2"
    analyses = (DECAYING_OSCILLATION,)
    defaults = T2_DEFAULTS

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the T2 (Ramsey) DRAG sweep on each target at its operating point; return its measurement id."""
        return {target: [self._run_ramsey_sweep(target)] for target in targets}

    def _run_ramsey_sweep(self, target: Target) -> int:
        """Build the T2 (Ramsey) QProgram, set the operating point's hardware, run the sweep and stream it.

        The QProgram is the shared ``qili_qprograms`` experiment: it plays the calibrated ``pi_half-<op>``
        pulse (or ``pi-<op>`` at half amplitude) twice around an idle wait and reads the ``readout-<op>``
        pulse, its weights and the drive/readout IFs from the calibration file. What it does not set -- the
        drive/readout LOs, the readout attenuation and the drive source -- the shared
        :meth:`~seqtante_open.controllers.calibration_node.CalibrationNode._set_operating_point_hardware`
        sets from the operating point, through the restore handle so the run undoes it (ADR-0006).
        """
        self._require_pulse(target, Role.readout, "readout")
        operating_point = self._operating_point_name(target)
        self._warn_if_pi_half_is_stored(target, operating_point)
        # Round, not truncate, so the stored time axis is the one t2_ramsey plays.
        durations_idle = np.rint(self._sweep(target, "idle_duration_sweep")).astype(int)

        qprogram = t2_ramsey(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=operating_point,
            idle_duration_sweep=durations_idle,
            averages=self._param(target, "averages"),
            relax_duration=self._param(target, "relax_duration"),
            readout_ringup_time=self._param(target, "readout_ringup_time"),
        )

        self._set_operating_point_hardware(target)
        # t2_ramsey unrolls the idle sweep: one acquisition per idle time.
        return self._stream(
            target,
            qprogram,
            experiment_name=_EXPERIMENT_NAME,
            loops={
                "t2_time": {
                    "array": durations_idle,
                    "units": "ns",
                    "bus": bus_alias(target, Role.drive),
                    "parameter": "t2_elapsed_time",
                }
            },
            shape=(len(durations_idle), 2),
            write=write_per_point,
        )

    def _warn_if_pi_half_is_stored(self, target: Target, operating_point: str) -> None:
        """Warn when ``pi_half-<operating_point>`` is stored, since ``t2_ramsey`` then plays it instead of half the pi pulse.

        The calibration tree recalibrates only ``pi-<operating_point>`` (rabi), so a stored pi-half pulse is not
        kept in step with it and can rotate by other than pi/2 once rabi has moved the pi amplitude. A wrong
        rotation lowers the Ramsey contrast rather than the decay time, and a badly wrong one fails the R²
        check, so the step still runs.
        """
        pi_half = pulse_name("pi_half", operating_point)
        if self.context.calibration.has_waveform(bus_alias(target, Role.drive), pi_half):
            logger.warning(
                "Node <i>{node}</i>: <r>{target}</r> plays the stored '{pi_half}' pulse, which the calibration tree "
                "does not recalibrate, instead of '{pi}' at half amplitude",
                node=self.name,
                target=str(target),
                pi_half=pi_half,
                pi=pulse_name("pi", operating_point),
            )

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[None]:
        """Fit the oscillation, check it, save ``t2``/``fit_r2``, plot, and report T2 and its uncertainty in us; read-only.

        Fit, then every acceptance check, then save, then plot: a rejected fit never reaches the DB (a
        reader would otherwise see a plausible ``t2`` for a step the run marked failed), as for T1.

        Raises:
            AnalysisError: when the fit is not a physical decay (``d <= 0``), its R² is below the step's
                minimum or undefined (a flat trace), or its relative uncertainty exceeds the step's
                maximum. The target then fails the step.
        """
        # The graph sets data_folder to <data_root>/<step>/<target>; None shows the plot instead of saving it.
        model = T2Fit(
            target=str(target), measurement_id=measurements[-1], path=self.parameters[target].get("data_folder")
        )
        model.fit()

        if not model.is_physical_decay:
            raise AnalysisError(f"{target} T2 fit is not a decay (d {model.decay:.3e} <= 0); T2 is not physical")

        self._require_r2(target, model.r_squared)
        self._require_rel_uncertainty(target, model.relative_uncertainty)

        model.save_fitting()
        plot = model.plot()
        return AnalysisResult(value=model.t2_us, uncertainty=model.t2_uncertainty_us, plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[None]) -> None:
        """A benchmark writes nothing to the calibration file; the run records the scored verdict (ADR-0002)."""
