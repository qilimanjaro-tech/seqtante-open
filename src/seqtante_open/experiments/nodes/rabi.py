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
from warnings import warn

import numpy as np
from qili_qprograms.experiments import rabi_amplitude
from qililab import IQDrag, Parameter
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
    from seqtante_open.controllers.parameter_restore import ParameterRestore
    from seqtante_open.targets import Target

#: Stream/experiment name the measurement is stored under.
_EXPERIMENT_NAME = "rabi_amplitude_drag"

#: Warned when the drive bus has no instrument whose power this node knows how to set.
_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."

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
        """Run the amplitude sweep on each target at its operating point and return its measurement id."""
        return {target: [self._run_amplitude_sweep(target)] for target in targets}

    def _run_amplitude_sweep(self, target: Target) -> int:
        """Build the Rabi amplitude QProgram, set the operating point's hardware, run the sweep and stream it.

        The QProgram is the shared ``qili_qprograms`` experiment: it plays the calibrated ``pi-<op>``
        pulse (normalised to unit amplitude) through a gain sweep and reads the ``readout-<op>`` pulse,
        all from the calibration file. This node sets what the QProgram does not -- the drive/readout
        LOs, the readout attenuation and the drive power/RF -- from the operating point, through the
        restore handle so the run undoes them (ADR-0006).
        """
        params = self.parameters[target]
        section = self.calibration_file.experiments[self.experiment][str(target)]
        operating_point_name = section.operating_point
        point = self.calibration_file.operating_points[str(target)][operating_point_name]
        drive_bus = bus_alias(target, Role.drive)
        readout_bus = bus_alias(target, Role.readout)
        sweep = np.linspace(*params["drive_amplitude_sweep"])
        if len(sweep) < 2:
            raise ValueError(f"{target}: drive_amplitude_sweep must have at least 2 points, got {len(sweep)}")

        qprogram = rabi_amplitude(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=operating_point_name,
            drive_amplitude_sweep=sweep,
            averages=params["averages"],
            relax_duration=params["relax_duration"],
            readout_ringup_time=params.get("ringup_time", 0),
        )

        restore = self.restore
        restore.set(drive_bus, Parameter.LO_FREQUENCY, int(point.drive.lo))
        restore.set(readout_bus, Parameter.LO_FREQUENCY, int(point.readout.lo))
        restore.set(readout_bus, Parameter.OUT0_ATT, int(point.readout.attenuation))
        self._set_drive_source(restore, drive_bus, params["drive_power"])

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

    def _set_drive_source(self, restore: ParameterRestore, drive_bus: str, drive_power: float) -> None:
        """Set the drive power, and arm a Rohde & Schwarz source to play the IQ drive.

        The power goes through the parameter the bus's drive instrument uses (R&S ``POWER``, QCMRF
        ``GAIN``). A Rohde & Schwarz source also needs IQ modulation switched on -- otherwise it emits a
        bare carrier instead of the DRAG pulse's IQ waveform -- and its RF output switched on. Both, like
        the power, go through the restore handle, so the node's run turns them back off at the end.
        """
        drive_instruments = [
            instrument.name.name for instrument in self.context.platform.get_element(drive_bus).instruments
        ]
        if "ROHDE_SCHWARZ" in drive_instruments:
            restore.set(drive_bus, Parameter.IQ_MODULATION, True)
            restore.set(drive_bus, Parameter.POWER, drive_power)
            restore.set(drive_bus, Parameter.RF_ON, True)
        elif "QCMRF" in drive_instruments:
            restore.set(drive_bus, Parameter.GAIN, drive_power)
        else:
            warn(_NO_DRIVE_INSTRUMENT_MSG, stacklevel=2)

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[RabiFitted]:
        """Fit the sweep, save the fit to the DB, and return the pi amplitude and readout levels.

        Read-only. Both analyses fail (``AnalysisError``) when the fit's R² is below the step's minimum;
        a ``pi_amplitude`` step also fails when the fit shows no oscillation (no pi pulse exists).
        """
        params = self.parameters[target]
        model = RabiAmplitudeModel(measurements[-1], target=str(target), path=params.get("data_folder"))
        model.fit()
        plot = model.plot()
        model.save_fitting()

        level_0, level_0_std, level_1, level_1_std = model.readout_levels
        min_r2 = params.get("min_r2", DEFAULT_MIN_R2)
        if model.r_squared < min_r2:
            raise AnalysisError(f"{target} rabi fit R² {model.r_squared:.3f} is below the minimum {min_r2}")
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
