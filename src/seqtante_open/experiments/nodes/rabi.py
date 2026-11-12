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

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from qililab import IQDrag

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.calibration.model import Measured, ReadoutLevels
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.experiment_classes import rabi_amplitude
from seqtante_open.experiments.fitting import RabiAmplitudeModel
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from seqtante_open.targets import Target

#: Analysis names (ADR-0003). The first is the default when a step names none.
READOUT_LEVELS = "readout_levels"
PI_AMPLITUDE = "pi_amplitude"

#: Default minimum R² a ``readout_levels`` fit must reach to pass its step (overridable per step).
DEFAULT_MIN_R2 = 0.9

#: Default pulse names when a section does not override the role (ADR-0008).
_DEFAULT_DRIVE_PULSE = "drag"
_DEFAULT_READOUT_PULSE = "readout"

#: DRAG-shape defaults the pi pulse is built with when no drive waveform exists yet to inherit from.
_DRIVE_SHAPE_DEFAULTS = {"drive_duration": 40, "drive_sigmas": 4.0, "drive_drag": 0.0}


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

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the amplitude sweep on each target and return its measurement id (hardware only)."""
        db_manager = output_controller.db_manager
        measurements: dict[Target, list[int]] = {}
        for target in targets:
            params = self.parameters[target]
            point = operating_point_for(self.calibration_file, self.experiment, target)
            sweep = np.linspace(*params["drive_amplitude_sweep"])
            measurement_id = rabi_amplitude(
                platform=self.context.platform,
                db_manager=db_manager,
                readout_bus=bus_alias(target, Role.readout),
                readout_if_freq=point.readout.if_,
                readout_duration=params["readout_duration"],
                readout_amplitude=params["readout_amplitude"],
                drive_bus=bus_alias(target, Role.drive),
                drive_power=params["drive_power"],
                drive_if_freq=point.drive.if_,
                drive_amplitude_sweep=sweep,
                drive_duration=params["drive_duration"],
                averages=params["averages"],
                relax_duration=params["relax_duration"],
                ringup_time=params.get("ringup_time", 0),
                drive_sigmas=params.get("drive_sigmas", _DRIVE_SHAPE_DEFAULTS["drive_sigmas"]),
                drive_drag=params.get("drive_drag", _DRIVE_SHAPE_DEFAULTS["drive_drag"]),
                drive_LO=point.drive.lo,
                readout_LO=point.readout.lo,
                target=str(target),
                calibration=self.context.calibration,
            )
            measurements[target] = [measurement_id]
        return measurements

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[RabiFitted]:
        """Fit the sweep, save the fit to the DB, and return the pi amplitude and readout levels.

        Read-only. A ``readout_levels`` step fails (``AnalysisError``) when the fit's R² is below its
        minimum; a ``pi_amplitude`` step fails when the fit shows no oscillation (no pi pulse exists).
        """
        params = self.parameters[target]
        model = RabiAmplitudeModel(measurements[-1], target=str(target), path=params.get("data_folder"))
        model.fit()
        plot = model.plot()
        model.save_fitting()

        level_0, level_0_std, level_1, level_1_std = model.readout_levels
        if self.analysis == READOUT_LEVELS and model.r_squared < params.get("min_r2", DEFAULT_MIN_R2):
            raise AnalysisError(
                f"{target} rabi fit R² {model.r_squared:.3f} is below the minimum "
                f"{params.get('min_r2', DEFAULT_MIN_R2)} for readout levels"
            )
        if self.analysis == PI_AMPLITUDE and model.results["angular_frequency"] <= 0:
            raise AnalysisError(f"{target} rabi fit shows no oscillation; there is no pi amplitude to apply")

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
        if result.fitted is None:  # pragma: no cover - a benchmark path rabi never takes
            return
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
            pulse = IQDrag(
                amplitude=fitted.pi_amplitude,
                duration=existing.duration,
                num_sigmas=existing.num_sigmas,
                drag_coefficient=existing.drag_coefficient,
            )
        else:
            params = self.parameters[target]
            pulse = IQDrag(
                amplitude=fitted.pi_amplitude,
                duration=params.get("drive_duration", _DRIVE_SHAPE_DEFAULTS["drive_duration"]),
                num_sigmas=params.get("drive_sigmas", _DRIVE_SHAPE_DEFAULTS["drive_sigmas"]),
                drag_coefficient=params.get("drive_drag", _DRIVE_SHAPE_DEFAULTS["drive_drag"]),
            )
        self.context.calibration.add_waveform(bus=drive_bus, name=drive_pulse, waveform=pulse)

    def _readout_pulse(self, target: Target) -> str:
        section = self.calibration_file.experiments[self.experiment][str(target)]
        return section.pulses.get(Role.readout, self.parameters[target].get("readout_pulse", _DEFAULT_READOUT_PULSE))

    def _drive_pulse(self, target: Target) -> str:
        section = self.calibration_file.experiments[self.experiment][str(target)]
        return section.pulses.get(Role.drive, self.parameters[target].get("drive_pulse", _DEFAULT_DRIVE_PULSE))
