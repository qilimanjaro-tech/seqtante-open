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

"""The ``two_tone`` node: one pulsed-saturation sweep at the sweet spot, fit to a Lorentzian (QHC-1805).

Ports the legacy ``two_tone`` function to a real :class:`CalibrationNode` (ADR-0004, ADR-0007). The node
runs at the step's operating point (the base places it before ``execute_node``), sweeps the drive IF with
one two-tone pulsed-saturation measurement, and fits the qubit line with
:class:`~seqtante_open.experiments.fitting.LorentzianPeakModel`.

The QProgram is the shared ``qili_qprograms`` experiment ``two_tone_saturation`` (QHC-1916): it reads the
``readout-<op>`` pulse, its integration weights and the readout IF from the calibration file and sweeps the
drive IF with a smooth ring-up / square / ring-down saturation drive. The node keeps the execution half --
drive/readout LOs, readout attenuation and drive power (plus IQ modulation and RF for a Rohde & Schwarz
source), all through the restore handle -- then runs the sweep and streams it.

``apply`` -- the only writer -- stores the fitted drive IF on the operating point (the ``two_tone_sweet_spot``
point). The drive LO is left as it was, and the legacy ``save_platform`` call is gone: the run persists the
calibration file (ADR-0001). The Lorentzian fit is shared with ``reset_point_search`` (QHC-1807); the fitted
``drive_if``, ``drive_lo`` and ``fit_r2`` are written to the measurement entry (QHC-1497, QHC-1791).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar
from warnings import warn

import numpy as np
from qili_qprograms.experiments import two_tone_saturation
from qililab import Parameter
from qililab.result import StreamArray

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import LorentzianPeakModel
from seqtante_open.experiments.results import measurement_id
from seqtante_open.outputs import FittedParameter, output_controller

if TYPE_CHECKING:
    from seqtante_open.controllers.parameter_restore import ParameterRestore
    from seqtante_open.targets import Target

LORENTZIAN_PEAK = "lorentzian_peak"

#: Stream/experiment name the measurement is stored under.
_EXPERIMENT_NAME = "two_tone_saturation"

#: Warned when the drive bus has no instrument whose power this node knows how to set.
_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."

#: Default minimum R² a Lorentzian fit must reach to pass the step (overridable per step).
DEFAULT_MIN_R2 = 0.8

#: Two-tone pulsed-saturation measurement defaults; every one is overridable as a tree parameter. The
#: readout pulse (amplitude, duration, weights) comes from the calibration file, not from parameters.
_DEFAULTS = {
    "averages": 5000,
    "relax_duration": 50_000,
    "ringup_time": 600,
    "drive_duration": 30_000,
    "drive_amplitude": 0.8,
    "drive_power": -40,
    "overlap_time": 0,
    "drive_sigmas": 4,
    "duration_ringup_drive": 200,
    "drive_ringdown_time": 200,
}


@dataclass(frozen=True)
class TwoToneSweetSpot:
    """What ``analyze`` hands ``apply``: the fitted drive IF to store on the operating point, in Hz."""

    drive_if: float


class TwoToneNode(FixedFluxNode[TwoToneSweetSpot]):
    """One two-tone pulsed-saturation sweep at the sweet spot, fit to a Lorentzian peak (QHC-1805)."""

    experiment: ClassVar[str] = "two_tone"
    analyses: ClassVar[tuple[str, ...]] = (LORENTZIAN_PEAK,)

    def _param(self, target: Target, key: str) -> Any:
        return self.parameters[target].get(key, _DEFAULTS[key])

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run one two-tone sweep per target and return its measurement id (hardware only)."""
        return {target: [self._measure(target)] for target in targets}

    def _measure(self, target: Target) -> int:
        """Build the qili two-tone QProgram, set the operating point's hardware, run the sweep; return its id.

        The QProgram plays the ``readout-<op>`` pulse with its weights and sets the readout IF, all from the
        calibration file, and sweeps the drive IF itself. This node sets what the QProgram does not -- the
        drive/readout LOs, the readout attenuation and the drive power/RF -- from the operating point,
        through the restore handle so the run undoes them (ADR-0006).
        """
        section = self.calibration_file.experiments[self.experiment][str(target)]
        point = operating_point_for(self.calibration_file, self.experiment, target)
        drive_bus = bus_alias(target, Role.drive)
        readout_bus = bus_alias(target, Role.readout)
        drive_if_sweep = self._drive_if_sweep(target, point.drive.lo)
        if len(drive_if_sweep) < 2:
            raise ValueError(f"{target}: the drive-IF sweep must have at least 2 points, got {len(drive_if_sweep)}")

        qprogram = two_tone_saturation(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=section.operating_point,
            drive_if_sweep=drive_if_sweep,
            averages=self._param(target, "averages"),
            relax_duration=self._param(target, "relax_duration"),
            drive_duration=self._param(target, "drive_duration"),
            drive_amplitude=self._param(target, "drive_amplitude"),
            drive_ringup_time=self._param(target, "duration_ringup_drive"),
            drive_ringdown_time=self._param(target, "drive_ringdown_time"),
            drive_n_sigmas=self._param(target, "drive_sigmas"),
            overlap_time=self._param(target, "overlap_time"),
            readout_ringup_time=self._param(target, "ringup_time"),
        )

        restore = self.restore
        restore.set(drive_bus, Parameter.LO_FREQUENCY, int(point.drive.lo))
        restore.set(readout_bus, Parameter.LO_FREQUENCY, int(point.readout.lo))
        restore.set(readout_bus, Parameter.OUT0_ATT, int(point.readout.attenuation))
        self._set_drive_source(restore, drive_bus, self._param(target, "drive_power"))

        platform = self.context.platform
        stream_array = StreamArray(
            shape=(len(drive_if_sweep), 2),
            loops={
                "IF_frequency": {"array": drive_if_sweep, "units": "Hz", "bus": drive_bus, "parameter": "IF_frequency"}
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
        bare carrier instead of the saturation drive's IQ waveform -- and its RF output switched on. Both,
        like the power, go through the restore handle, so the node's run turns them back off at the end.
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

    def _drive_if_sweep(self, target: Target, drive_lo: float) -> np.ndarray:
        """Build the drive-IF sweep (Hz) from the step's absolute-frequency window and the placed LO."""
        start = self.parameters[target]["sweep_start"]
        stop = self.parameters[target]["sweep_stop"]
        step = self.parameters[target]["sweep_step"]
        npoints = round((stop - start) / step) + 1
        return np.linspace(start, stop, num=npoints, endpoint=True) - drive_lo

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[TwoToneSweetSpot]:
        """Fit the sweep, save ``drive_if``/``drive_lo``/``fit_r2`` to the DB, return the drive IF; read-only.

        Raises:
            AnalysisError: when the Lorentzian fit is rejected (the step fails for this target).
        """
        point = operating_point_for(self.calibration_file, self.experiment, target)
        model = LorentzianPeakModel(
            measurements[-1],
            target=str(target),
            path=self.parameters[target].get("data_folder"),
            r2_min=self.parameters[target].get("min_r2", DEFAULT_MIN_R2),
            title="Two tone (sweet spot)",
        )
        model.fit()
        plot = model.plot()
        # The shared fit writes drive_if and fit_r2; the placed LO is recorded alongside them (QHC-1497).
        model.add_fitted_parameter(FittedParameter.DRIVE_LO, point.drive.lo)
        model.save_fitting()
        if not model.accepted or model.fitted_if is None:
            raise AnalysisError(f"{target}: two-tone fit rejected: {model.results['quality']}")
        return AnalysisResult(fitted=TwoToneSweetSpot(drive_if=model.fitted_if), plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[TwoToneSweetSpot]) -> None:
        """Write the fitted drive IF onto the sweet-spot operating point; the only writer (the LO is untouched)."""
        # analyze returns a fitted point or raises, so apply always has one; the guard is just a type narrow.
        if result.fitted is not None:
            point = operating_point_for(self.calibration_file, self.experiment, target)
            point.drive.if_ = result.fitted.drive_if
