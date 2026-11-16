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

``apply`` -- the only writer -- stores the fitted drive IF on the operating point (the ``two_tone_sweet_spot``
point). The drive LO, placed outside the sweep by the base helper, is left as it was, and the legacy
``save_platform`` call is gone: the run persists the calibration file (ADR-0001). The measurement and the
Lorentzian fit are shared with ``reset_point_search`` (QHC-1807); the fitted ``drive_if``, ``drive_lo`` and
``fit_r2`` are written to the measurement entry (QHC-1497, QHC-1791).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.experiment_classes import two_tone_pulsed_saturation_experiment
from seqtante_open.experiments.fitting import LorentzianPeakModel
from seqtante_open.outputs import FittedParameter, output_controller

if TYPE_CHECKING:
    from seqtante_open.targets import Target

LORENTZIAN_PEAK = "lorentzian_peak"

#: Default minimum R² a Lorentzian fit must reach to pass the step (overridable per step).
DEFAULT_MIN_R2 = 0.8

#: Two-tone pulsed-saturation measurement defaults; every one is overridable as a tree parameter.
_DEFAULTS = {
    "averages": 5000,
    "relax_duration": 50_000,
    "readout_amplitude": 0.7,
    "readout_duration": 1000,
    "ringup_time": 600,
    "drive_duration": 30_000,
    "drive_amplitude": 0.8,
    "drive_power": -40,
    "overlap_time": 0,
    "drive_sigmas": 4,
    "duration_ringup_drive": 200,
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
        """Run one two-tone pulsed-saturation sweep at the target's operating point; return its id."""
        point = operating_point_for(self.calibration_file, self.experiment, target)
        drive_lo = point.drive.lo
        return two_tone_pulsed_saturation_experiment(
            platform=self.context.platform,
            db_manager=output_controller.db_manager,
            readout_bus=bus_alias(target, Role.readout),
            drive_bus=bus_alias(target, Role.drive),
            readout_if_freq=int(point.readout.if_),
            averages=self._param(target, "averages"),
            readout_duration=self._param(target, "readout_duration"),
            readout_amplitude=self._param(target, "readout_amplitude"),
            drive_duration=self._param(target, "drive_duration"),
            drive_amplitude=self._param(target, "drive_amplitude"),
            relax_duration=self._param(target, "relax_duration"),
            overlap_time=self._param(target, "overlap_time"),
            ringup_time=self._param(target, "ringup_time"),
            drive_if_sweep=self._drive_if_sweep(target, drive_lo),
            drive_gain=self._param(target, "drive_power"),
            drive_sigmas=self._param(target, "drive_sigmas"),
            duration_ringup_drive=self._param(target, "duration_ringup_drive"),
            drive_LO=drive_lo,
            readout_LO=point.readout.lo,
            target=str(target),
            calibration=self.context.calibration,
        )

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
        if result.fitted is None:  # pragma: no cover - two_tone always fits a point
            return
        point = operating_point_for(self.calibration_file, self.experiment, target)
        point.drive.if_ = result.fitted.drive_if
