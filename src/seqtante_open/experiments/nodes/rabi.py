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
  onto the operating point, tagged with the ``readout-<operating_point>`` pulse they were taken with. A
  fit below the step's minimum R² fails the step.
- ``pi_amplitude`` (step ``rabi_sweet_spot``, at the ``sweet_spot`` point) writes the fitted pi
  amplitude into the ``pi-<operating_point>`` DRAG pulse in ``Calibration.waveforms``, not onto the point.

The pulses are the ones ``qili_qprograms``' ``rabi_amplitude`` plays, named by
:func:`~qili_qprograms.calibration.pulse_name` (ADR-0008). Both must already exist: the node seeds
none, they come with the manual calibration.

This is the first experiment ported to a real :class:`CalibrationNode` subclass (ADR-0007); the fit
lives in :class:`~seqtante_open.experiments.fitting.RabiAmplitudeModel` and the measurement driver in
:func:`~seqtante_open.experiments.experiment_classes.rabi_amplitude`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qili_qprograms.experiments import rabi_amplitude
from qililab import IQDrag

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.calibration.model import Measured, ReadoutLevels
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode, write_whole
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import RabiAmplitudeModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from seqtante_open.targets import Target

#: Stream/experiment name the measurement is stored under.
_EXPERIMENT_NAME = "rabi_amplitude_drag"

#: Analysis names (ADR-0003). The first is the default when a step names none.
READOUT_LEVELS = "readout_levels"
PI_AMPLITUDE = "pi_amplitude"

#: The node's ``defaults``; every one is overridable in the tree. ``drive_amplitude_sweep``, ``averages``
#: and ``relax_duration`` have no default, and the drive instrument decides which of ``drive_power``
#: (Rohde & Schwarz, dBm) or ``drive_gain`` (QCM-RF, [-1, 1]) the step must set.
RABI_DEFAULTS: Mapping[str, Any] = {
    "readout_ringup_time": 0,
    # Minimum R² the cosine fit must reach to pass the step.
    "min_r2": 0.9,
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

    experiment = "rabi"
    analyses = (PI_AMPLITUDE, READOUT_LEVELS)
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
        self._require_pulse(target, Role.drive, "pi")
        self._require_pulse(target, Role.readout, "readout")
        sweep = self._sweep(target, "drive_amplitude_sweep")

        qprogram = rabi_amplitude(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=self._section(target).operating_point,
            drive_amplitude_sweep=sweep,
            averages=self._param(target, "averages"),
            relax_duration=self._param(target, "relax_duration"),
            readout_ringup_time=self._param(target, "readout_ringup_time"),
        )

        self._set_operating_point_hardware(target)
        # The gain sweep runs in hardware: one acquisition holds the whole sweep.
        return self._stream(
            target,
            qprogram,
            experiment_name=_EXPERIMENT_NAME,
            loops={
                "drive_amplitude": {
                    "array": sweep,
                    "units": "amp",
                    "bus": bus_alias(target, Role.drive),
                    "parameter": "drive_amplitude",
                }
            },
            shape=(len(sweep), 2),
            write=write_whole,
        )

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

        self._require_r2(target, model.r_squared)
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
        operating_point = operating_point_for(self.calibration_file, self.experiment, target)
        operating_point.readout.levels = ReadoutLevels(
            pulse=self._pulse(target, Role.readout, "readout"),
            level_0=Measured(nominal=fitted.level_0, std=fitted.level_0_std),
            level_1=Measured(nominal=fitted.level_1, std=fitted.level_1_std),
        )

    def _apply_pi_amplitude(self, target: Target, fitted: RabiFitted) -> None:
        """Write the pi amplitude into the ``pi-<operating_point>`` DRAG pulse, keeping its other shape parameters."""
        drive_bus = bus_alias(target, Role.drive)
        drive_pulse = self._require_pulse(target, Role.drive, "pi")
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
        self.context.calibration.add_waveform(bus=drive_bus, name=drive_pulse, waveform=pulse)
