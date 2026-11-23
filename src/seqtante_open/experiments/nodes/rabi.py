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

"""The ``rabi`` node: one amplitude (DRAG drive) measurement with two analyses (ADR-0003, QHC-1806).

A single Rabi amplitude sweep feeds two analyses, chosen per step:

- ``readout_levels`` (step ``rabi_reset``, at the ``reset`` point) writes the |0>/|1> readout levels
  onto the operating point, tagged with the readout pulse they were taken with. A fit below the step's
  minimum R² fails the step.
- ``pi_amplitude`` (step ``rabi_sweet_spot``, at the ``sweet_spot`` point) writes the fitted pi
  amplitude into the named drive (DRAG) pulse in ``Calibration.waveforms``, not onto the point.

This is the first experiment ported to a real :class:`CalibrationNode` subclass (ADR-0007); the fit
lives in :class:`~seqtante_open.experiments.fitting.RabiAmplitudeModel` and the measurement driver in
:func:`~seqtante_open.experiments.experiment_classes.rabi_amplitude`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from qili_qprograms.experiments import rabi_amplitude
from qililab import IQDrag
from qililab.result import StreamArray

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.calibration.model import Measured, ReadoutLevels
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import RabiAmplitudeModel
from seqtante_open.experiments.results import measurement_id
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from collections.abc import Mapping

    from seqtante_open.targets import Target

#: Stream/experiment name the measurement is stored under.
_EXPERIMENT_NAME = "rabi_amplitude_drag"

#: Analysis names (ADR-0003). The first is the default when a step names none.
READOUT_LEVELS = "readout_levels"
PI_AMPLITUDE = "pi_amplitude"

#: Default minimum R² a ``readout_levels`` fit must reach to pass its step (overridable per step).
DEFAULT_MIN_R2 = 0.9

#: The node's ``defaults``; every one is overridable in the tree. ``drive_amplitude_sweep``, ``averages``
#: and ``relax_duration`` have no default, and the drive instrument decides which of ``drive_power``
#: (Rohde & Schwarz, dBm) or ``drive_gain`` (QCM-RF, [-1, 1]) the step must set.
RABI_DEFAULTS: Mapping[str, Any] = {
    "readout_ringup_time": 0,
    "min_r2": DEFAULT_MIN_R2,
    # Pulse names when a section does not override the role (ADR-0008).
    "drive_pulse": "drag",
    "readout_pulse": "readout",
    # DRAG shape the pi pulse is built with when no drive waveform exists yet to inherit from.
    "drive_duration": 40,
    "drive_n_sigmas": 4.0,
    "drive_drag": 0.0,
}


@dataclass(frozen=True)
class RabiFitted:
    """What ``analyze`` hands ``apply``: the pi amplitude and the two readout levels (with uncertainty)."""

    pi_amplitude: float
    level_0: float
    level_0_std: float
    level_1: float
    level_1_std: float


class RabiNode(FixedFluxNode[RabiFitted]):
    """One Rabi amplitude sweep, analysed as ``pi_amplitude`` or ``readout_levels`` per step (QHC-1806)."""

    experiment: ClassVar[str] = "rabi"
    analyses: ClassVar[tuple[str, ...]] = (PI_AMPLITUDE, READOUT_LEVELS)
    defaults = RABI_DEFAULTS

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the amplitude sweep on each target at its operating point and return its measurement id."""
        return {target: [self._run_amplitude_sweep(target)] for target in targets}

    def _run_amplitude_sweep(self, target: Target) -> int:
        """Build the Rabi amplitude QProgram, set the operating point's hardware, run the sweep and stream it.

        The QProgram is the shared ``qili_qprograms`` experiment: it plays the calibrated ``pi-<op>``
        pulse (normalised to unit amplitude) through a gain sweep and reads the ``readout-<op>`` pulse,
        all from the calibration file. What it does not set -- the drive/readout LOs, the readout
        attenuation and the drive source -- the shared
        :meth:`~seqtante_open.controllers.calibration_node.CalibrationNode._set_operating_point_hardware`
        sets from the operating point, through the restore handle so the run undoes it (ADR-0006).
        """
        section = self.calibration_file.experiments[self.experiment][str(target)]
        operating_point_name = section.operating_point
        drive_bus = bus_alias(target, Role.drive)
        readout_bus = bus_alias(target, Role.readout)
        sweep = np.linspace(*self._param(target, "drive_amplitude_sweep"))
        if len(sweep) < 2:
            raise ValueError(f"{target}: drive_amplitude_sweep must have at least 2 points, got {len(sweep)}")

        qprogram = rabi_amplitude(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=operating_point_name,
            drive_amplitude_sweep=sweep,
            averages=self._param(target, "averages"),
            relax_duration=self._param(target, "relax_duration"),
            readout_ringup_time=self._param(target, "readout_ringup_time"),
        )

        self._set_operating_point_hardware(target)

        platform = self.context.platform
        stream_array = StreamArray(
            shape=(len(sweep), 2),
            loops={
                "drive_amplitude": {"array": sweep, "units": "amp", "bus": drive_bus, "parameter": "drive_amplitude"}
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
            stream_array[()] = results[readout_bus][0].array.T
        return measurement_id(stream_array)

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[RabiFitted]:
        """Fit the sweep, check it, save the fit to the DB, plot, and return the pi amplitude and readout levels.

        Read-only. Fit, then every acceptance check, then save, then plot: a rejected fit never reaches the
        DB, as for T1 (QHC-1803). Both analyses fail (``AnalysisError``) when the fit's R² is below the
        step's minimum or undefined (a flat trace); a ``pi_amplitude`` step also fails when the fit shows
        no oscillation (no pi pulse exists).
        """
        model = RabiAmplitudeModel(
            measurements[-1], target=str(target), path=self.parameters[target].get("data_folder")
        )
        model.fit()

        min_r2 = self._param(target, "min_r2")
        # A flat trace leaves R² NaN; reject it explicitly, since every comparison with NaN is False.
        if math.isnan(model.r_squared) or model.r_squared < min_r2:
            raise AnalysisError(f"{target} rabi fit R² {model.r_squared:.3f} is below the minimum {min_r2}")
        if self.analysis == PI_AMPLITUDE and model.results["angular_frequency"] <= 0:
            raise AnalysisError(f"{target} rabi fit shows no oscillation; there is no pi amplitude to apply")

        model.save_fitting()
        plot = model.plot()
        level_0, level_0_std, level_1, level_1_std = model.readout_levels
        fitted = RabiFitted(
            pi_amplitude=model.pi_amplitude,
            level_0=level_0,
            level_0_std=level_0_std,
            level_1=level_1,
            level_1_std=level_1_std,
        )
        return AnalysisResult(fitted=fitted, plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[RabiFitted]) -> None:
        """Write the chosen analysis's result (``readout_levels`` onto the point, ``pi_amplitude`` to the pulse)."""
        # rabi always fits a result; the guard is defensive against a benchmark path rabi never takes.
        if result.fitted is not None:
            if self.analysis == READOUT_LEVELS:
                self._apply_readout_levels(target, result.fitted)
            else:
                self._apply_pi_amplitude(target, result.fitted)

    def _apply_readout_levels(self, target: Target, fitted: RabiFitted) -> None:
        """Write the |0>/|1> levels onto the target's operating point, tagged with the readout pulse."""
        point = operating_point_for(self.calibration_file, self.experiment, target)
        point.readout.levels = ReadoutLevels(
            pulse=self._readout_pulse(target),
            level_0=Measured(nominal=fitted.level_0, std=fitted.level_0_std),
            level_1=Measured(nominal=fitted.level_1, std=fitted.level_1_std),
        )

    def _apply_pi_amplitude(self, target: Target, fitted: RabiFitted) -> None:
        """Write the pi amplitude into the named drive (DRAG) pulse, keeping its other shape parameters."""
        drive_bus = bus_alias(target, Role.drive)
        drive_pulse = self._drive_pulse(target)
        if self.context.calibration.has_waveform(drive_bus, drive_pulse):
            existing = self.context.calibration.get_waveform(drive_bus, drive_pulse)
            if not isinstance(existing, IQDrag):
                raise AnalysisError(
                    f"{target}: drive pulse '{drive_pulse}' on {drive_bus} is a {type(existing).__name__}, "
                    "not an IQDrag; cannot update its pi amplitude"
                )
            pulse = IQDrag(
                amplitude=fitted.pi_amplitude,
                duration=existing.duration,
                num_sigmas=existing.num_sigmas,
                drag_coefficient=existing.drag_coefficient,
            )
        else:
            pulse = IQDrag(
                amplitude=fitted.pi_amplitude,
                duration=self._param(target, "drive_duration"),
                num_sigmas=self._param(target, "drive_n_sigmas"),
                drag_coefficient=self._param(target, "drive_drag"),
            )
        self.context.calibration.add_waveform(bus=drive_bus, name=drive_pulse, waveform=pulse)

    def _readout_pulse(self, target: Target) -> str:
        section = self.calibration_file.experiments[self.experiment][str(target)]
        return section.pulses.get(Role.readout, self._param(target, "readout_pulse"))

    def _drive_pulse(self, target: Target) -> str:
        section = self.calibration_file.experiments[self.experiment][str(target)]
        return section.pulses.get(Role.drive, self._param(target, "drive_pulse"))
