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

"""The ``rabi`` node: one Rabi amplitude measurement with two analyses (ADR-0003, QHC-1806).

A single Rabi amplitude sweep feeds two analyses, chosen per step:

- ``readout_levels`` (step ``rabi_reset``, at the ``reset`` point) writes the |0>/|1> readout levels
  onto the operating point, tagged with the ``readout-<operating_point>`` pulse they were taken with. A
  fit below the step's minimum R² fails the step.
- ``pi_amplitude`` (step ``rabi_sweet_spot``, at the ``sweet_spot`` point) writes the fitted pi
  amplitude into the ``pi-<operating_point>`` pulse in ``Calibration.waveforms``, not onto the point.

The pulses are the ones ``qili_qprograms``' ``rabi_amplitude`` plays, named by
:func:`~qili_qprograms.calibration.pulse_name` (ADR-0008). Both must already exist: the node seeds
none, they come with the manual calibration.

The node is a :class:`FixedFluxNode` (ADR-0007): it builds qili's ``rabi_amplitude`` QProgram and runs
it itself, and the fit lives in :class:`~seqtante_open.experiments.fitting.RabiAmplitudeModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qili_qprograms.bus_alias import Role, bus_alias
from qili_qprograms.calibration.model import Measured, ReadoutLevels
from qili_qprograms.experiments import rabi_amplitude
from qili_qprograms.waveforms import scaled, unit_amplitude
from qililab import Parameter

from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode, write_whole
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import RabiAmplitudeModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qili_qprograms.targets import Target
    from qililab.typings import ParameterValue

#: Stream/experiment name the measurement is stored under.
_EXPERIMENT_NAME = "rabi_amplitude_drag"

#: Analysis names (ADR-0003). ``RabiNode.analyses`` lists ``pi_amplitude`` first, so it is the default when a step names none.
READOUT_LEVELS = "readout_levels"
PI_AMPLITUDE = "pi_amplitude"

#: The node's ``defaults``; every one is overridable in the tree. ``drive_amplitude_sweep``, ``averages``
#: and ``relax_duration`` have no default. A Rohde & Schwarz drive needs the step's ``drive_power`` (dBm);
#: a QCM-RF drive runs at ``drive_gain`` 1, the only value the node accepts (see :meth:`RabiNode._drive_source_writes`).
RABI_DEFAULTS: Mapping[str, Any] = {
    "readout_ringup_time": 0,
    "drive_gain": 1.0,
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
            operating_point=self._operating_point_name(target),
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

    def _drive_source_writes(self, target: Target) -> list[tuple[Parameter, ParameterValue]]:
        """The base's drive-source writes, refusing a QCM-RF ``drive_gain`` other than 1.

        rabi sweeps the drive gain itself (the QProgram's ``set_awg_gain``), and writes the fitted pi
        amplitude as the pulse's amplitude for a consumer that plays it at gain 1. A static sequencer gain
        other than 1 has no place in that: Qblox documents the path gain as multiplying ``set_awg_gain``,
        which would scale the sweep, and if the sweep replaces it instead, the step's ``drive_gain`` would
        silently do nothing. Either way the tree would not run what it says, so the step fails.
        :meth:`_set_operating_point_hardware` takes these writes before its first hardware write, so the
        refusal leaves the line untouched, and the platform is asked for the drive instrument only once.

        Raises:
            ValueError: if the drive is a QCM-RF and the step sets ``drive_gain`` to anything but 1, or the
                drive instrument's level is neither set in the step nor defaulted.
        """
        writes = super()._drive_source_writes(target)
        for parameter, value in writes:
            if parameter == Parameter.GAIN and value != 1:
                raise ValueError(
                    f"step '{self.name}': {target} drive_gain {value} is not 1; rabi sweeps the drive gain "
                    "itself, so it runs a QCM-RF drive at drive_gain 1"
                )
        return writes

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[RabiFitted]:
        """Fit the sweep, check it, save the fit to the DB, plot, and return the pi amplitude and readout levels.

        Read-only. Fit, then every acceptance check, then save, then plot: a rejected fit never reaches the
        DB, as for T1 (QHC-1803). Both analyses fail (``AnalysisError``) when the fit's R² is below the
        step's minimum or undefined (a flat trace), when the pi amplitude is undefined (no oscillation) or
        outside the swept window (the sweep stopped short of the excited level, so neither the pi amplitude
        nor |1> was measured), or when the value the step writes has no uncertainty estimate.
        """
        model = RabiAmplitudeModel(
            measurements[-1], target=str(target), path=self.parameters[target].get("data_folder")
        )
        model.fit()

        self._require_r2(target, model.r_squared)
        self._require_pi_in_window(target, model.pi_amplitude, model.amplitudes)
        level_0, level_0_std, level_1, level_1_std = model.readout_levels
        stds = (level_0_std, level_1_std) if self.analysis == READOUT_LEVELS else (model.results["pi_amplitude_std"],)
        if not np.all(np.isfinite(stds)):
            raise AnalysisError(f"{target} rabi fit has no uncertainty estimate (singular covariance); not storing it")

        model.save_fitting()
        plot = model.plot()
        fitted = RabiFitted(
            pi_amplitude=model.pi_amplitude,
            level_0=level_0,
            level_0_std=level_0_std,
            level_1=level_1,
            level_1_std=level_1_std,
        )
        return AnalysisResult(fitted=fitted, plots=(plot,) if plot else ())

    @staticmethod
    def _require_pi_in_window(target: Target, pi_amplitude: float, amplitudes: np.ndarray) -> None:
        """Fail the target unless the fitted pi amplitude is defined and inside the swept window."""
        low, high = float(np.min(amplitudes)), float(np.max(amplitudes))
        # NaN (no oscillation) fails too.
        if not low <= pi_amplitude <= high:
            raise AnalysisError(
                f"{target} rabi fit puts the pi amplitude at {pi_amplitude:.4g}, outside the swept window "
                f"[{low:.4g}, {high:.4g}]; widen drive_amplitude_sweep"
            )

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
        operating_point = self._operating_point(target)
        operating_point.readout.levels = ReadoutLevels(
            pulse=self._pulse(target, Role.readout, "readout"),
            level_0=Measured(nominal=fitted.level_0, std=fitted.level_0_std),
            level_1=Measured(nominal=fitted.level_1, std=fitted.level_1_std),
        )

    def _apply_pi_amplitude(self, target: Target, fitted: RabiFitted) -> None:
        """Write the pulse ``rabi_amplitude`` played at the fitted gain into ``pi-<operating_point>``.

        ``rabi_amplitude`` plays the stored pulse scaled to unit amplitude and sweeps a gain, so the pi
        pulse is that unit pulse scaled by the fitted pi amplitude. Its shape and sign are kept.
        """
        drive_bus = bus_alias(target, Role.drive)
        drive_pulse = self._require_pulse(target, Role.drive, "pi")
        existing = self.context.calibration.get_waveform(drive_bus, drive_pulse)
        pulse = scaled(unit_amplitude(existing), fitted.pi_amplitude)
        self.context.calibration.add_waveform(bus=drive_bus, name=drive_pulse, waveform=pulse)
