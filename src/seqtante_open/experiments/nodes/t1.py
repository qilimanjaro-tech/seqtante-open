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
import os
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from qili_qprograms.experiments import t1_saturation
from qililab import Parameter
from qililab.result import StreamArray

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.parameter_restore import restoring_platform
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import T1Fit
from seqtante_open.experiments.results import measurement_id
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from qililab.platform import Platform

    from seqtante_open.targets import Target

EXPONENTIAL_DECAY = "exponential_decay"

#: Stream/experiment name the T1 measurement is stored under.
_EXPERIMENT_NAME = "t1_pulsed_saturation"

#: Warned when the drive bus has no instrument whose power this node knows how to set.
_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."

#: Default minimum R² a T1 fit must reach to pass the step (overridable per step).
DEFAULT_MIN_R2 = 0.8

#: Default maximum relative uncertainty (sigma_T1 / T1) a T1 fit may have to pass the step (overridable
#: per step). Catches a fit that clears R² but whose lifetime is ill-determined, and the ``inf``
#: uncertainty ``curve_fit`` returns when it cannot estimate the covariance.
DEFAULT_MAX_REL_UNCERTAINTY = 0.5


class T1Node(FixedFluxNode[None]):
    """One T1 saturation sweep at the sweet spot, fit to an exponential decay and scored as a benchmark (QHC-1803).

    The benchmark unit (``us``) and direction (higher-is-better) come from the ``t1`` entry in
    ``NODE_REGISTRY``; the run reads them to score the reported value against the tree's thresholds.
    """

    experiment = "t1"
    analyses = (EXPONENTIAL_DECAY,)

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the T1 saturation sweep on each target at its operating point; return its measurement id."""
        # Every hardware write goes through the restore handle, so run() puts the drive/readout line back
        # as it was and nothing leaks into the next node -- including the drive RF output (ADR-0006).
        platform = restoring_platform(self.context.platform, self.restore)
        return {target: [self._run_saturation_sweep(platform, target)] for target in targets}

    def _run_saturation_sweep(self, platform: Platform, target: Target) -> int:
        """Build the T1 saturation QProgram, set the operating point's hardware, run the sweep and stream it.

        The QProgram is the shared ``qili_qprograms`` experiment: it reads the readout pulse, its weights
        and the drive/readout IFs for the operating point from the calibration file, and plays the
        saturation drive on the target's buses. This node sets what the QProgram does not -- the
        drive/readout LOs, the readout attenuation and the drive power/RF -- from the operating point.
        """
        params = self.parameters[target]
        section = self.calibration_file.experiments[self.experiment][str(target)]
        operating_point_name = section.operating_point
        operating_point = self.calibration_file.operating_points[str(target)][operating_point_name]
        drive_bus = bus_alias(target, Role.drive)
        readout_bus = bus_alias(target, Role.readout)
        durations_idle = np.linspace(*params["idle_sweep"]).astype(int)

        qprogram = t1_saturation(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=operating_point_name,
            idle_duration_sweep=durations_idle,
            averages=params["averages"],
            relax_duration=params["relax_duration"],
            drive_duration=params["drive_duration"],
            drive_amplitude=params["drive_amplitude"],
            drive_ringup_time=params.get("duration_ringup_drive", 200),
            drive_n_sigmas=params.get("drive_sigmas", 4),
            overlap_time=params.get("overlap_time", 0),
            readout_ringup_time=params.get("ringup_time", 0),
        )

        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=int(operating_point.drive.lo))
        platform.set_parameter(
            alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=int(operating_point.readout.lo)
        )
        platform.set_parameter(
            alias=readout_bus, parameter=Parameter.OUT0_ATT, value=int(operating_point.readout.attenuation)
        )
        self._set_drive_source(platform, drive_bus, params["drive_power"])

        stream_array = StreamArray(
            shape=(len(durations_idle), 2),
            loops={"time": {"array": durations_idle.astype(int), "units": "ns", "bus": drive_bus, "parameter": "time"}},
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

    def _set_drive_source(self, platform: Platform, drive_bus: str, drive_power: float) -> None:
        """Set the drive power, and arm a Rohde & Schwarz source to play the IQ drive.

        The power goes through the parameter the bus's drive instrument uses (R&S ``POWER``, QCMRF
        ``GAIN``). A Rohde & Schwarz source also needs IQ modulation switched on -- otherwise it emits a
        bare carrier instead of the saturation drive's IQ waveform -- and its RF output switched on. Both,
        like the power, go through the restore handle, so the node's run turns them back off at the end;
        no explicit teardown is needed here.
        """
        drive_instruments = [instrument.name.name for instrument in platform.get_element(drive_bus).instruments]
        if "ROHDE_SCHWARZ" in drive_instruments:
            platform.set_parameter(alias=drive_bus, parameter=Parameter.IQ_MODULATION, value=True)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_power)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif "QCMRF" in drive_instruments:
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_power)
        else:
            warn(_NO_DRIVE_INSTRUMENT_MSG, stacklevel=2)

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[None]:
        """Fit the decay, check it, save ``t1``/``fit_r2`` only once it passes, and report T1 and its uncertainty in us; read-only.

        Every acceptance check runs before ``save_fitting`` so a rejected fit never reaches the DB: a
        reader would otherwise see a plausible ``t1`` for a step the run marked failed (QHC-1803).

        Raises:
            AnalysisError: when the fit is not a physical decay (``rate >= 0``), its R² is below the
                step's minimum, or its relative uncertainty exceeds the step's maximum. The target
                then fails the step.
        """
        params = self.parameters[target]
        model = T1Fit(target=str(target), measurement_id=measurements[-1], path=self._plot_dir(params, target))
        model.fit()
        plot = model.plot()

        if not model.is_physical_decay:
            raise AnalysisError(f"{target} T1 fit is not a decay (rate {model.rate:.3e} >= 0); T1 is not physical")

        min_r2 = params.get("min_r2", DEFAULT_MIN_R2)
        if model.r_squared < min_r2:
            raise AnalysisError(f"{target} T1 fit R² {model.r_squared:.3f} is below the minimum {min_r2}")

        max_rel_uncertainty = params.get("max_rel_uncertainty", DEFAULT_MAX_REL_UNCERTAINTY)
        relative_uncertainty = model.relative_uncertainty
        if not math.isfinite(relative_uncertainty) or relative_uncertainty > max_rel_uncertainty:
            raise AnalysisError(
                f"{target} T1 relative uncertainty {relative_uncertainty:.3f} exceeds the maximum {max_rel_uncertainty}"
            )

        # Every check passed: only now does the accepted fit reach the DB.
        model.save_fitting()

        return AnalysisResult(value=model.t1_us, uncertainty=model.t1_uncertainty_us, plots=(plot,) if plot else ())

    def _plot_dir(self, params: dict, target: Target) -> str | None:
        """The directory the fit writes its plot into: a per-experiment, per-target subfolder of ``data_folder``.

        ``data_folder`` is shared across the whole tree, so each experiment and target gets its own
        subfolder; otherwise plots from different steps or qubits would collide. ``None`` (no folder
        configured) is passed straight through, and the fit shows the plot instead of saving it.
        """
        data_folder = params.get("data_folder")
        if not data_folder:
            return None
        return os.path.join(data_folder, str(self.experiment), str(target))

    def apply(self, target: Target, result: AnalysisResult[None]) -> None:
        """A benchmark writes nothing to the calibration file; the run records the scored verdict (ADR-0002)."""
