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

The measurement is the shared qili ``two_tone_saturation`` QProgram (QHC-1916), run by the
:class:`~seqtante_open.experiments.nodes.two_tone_saturation.TwoToneSaturationNode` spine it shares with
``reset_point_search`` (QHC-1807).

``apply`` -- the only writer -- stores the fitted drive IF on the operating point (the ``two_tone_sweet_spot``
point). The drive LO is left as it was, and the legacy ``save_platform`` call is gone: the run persists the
calibration file (ADR-0001). The Lorentzian fit is shared with ``reset_point_search`` (QHC-1807); the fitted
``drive_if``, ``drive_lo`` and ``fit_r2`` are written to the measurement entry (QHC-1497, QHC-1791).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from seqtante_open.controllers.calibration_node import AnalysisResult
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import LorentzianPeakModel
from seqtante_open.experiments.nodes.two_tone_saturation import TWO_TONE_SATURATION_DEFAULTS, TwoToneSaturationNode
from seqtante_open.outputs import FittedParameter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from seqtante_open.targets import Target

LORENTZIAN_PEAK = "lorentzian_peak"

#: Default minimum R² a Lorentzian fit must reach to pass the step (overridable per step).
DEFAULT_MIN_R2 = 0.8

#: The node's ``defaults``: the shared two-tone measurement's, plus the fit acceptance. The sweep window
#: (``sweep_start``, ``sweep_stop``, ``sweep_step``, absolute Hz) has no default: the step must set it.
TWO_TONE_DEFAULTS: Mapping[str, Any] = {**TWO_TONE_SATURATION_DEFAULTS, "min_r2": DEFAULT_MIN_R2}


@dataclass(frozen=True)
class TwoToneSweetSpot:
    """What ``analyze`` hands ``apply``: the fitted drive IF to store on the operating point, in Hz."""

    drive_if: float


class TwoToneNode(TwoToneSaturationNode[TwoToneSweetSpot]):
    """One two-tone pulsed-saturation sweep at the sweet spot, fit to a Lorentzian peak (QHC-1805)."""

    experiment = "two_tone"
    analyses = (LORENTZIAN_PEAK,)
    defaults = TWO_TONE_DEFAULTS

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run one two-tone sweep per target and return its measurement id (hardware only)."""
        return {target: [self._measure(target)] for target in targets}

    def _measure(self, target: Target) -> int:
        """Run one two-tone pulsed-saturation sweep at the target's operating point; return its id."""
        operating_point = operating_point_for(self.calibration_file, self.experiment, target)
        return self._measure_two_tone_saturation(target, self._drive_if_sweep(target, operating_point.drive.lo))

    def _drive_if_sweep(self, target: Target, drive_lo: float) -> np.ndarray:
        """Build the drive-IF sweep (Hz) from the step's absolute-frequency window and the placed LO."""
        start = self._param(target, "sweep_start")
        stop = self._param(target, "sweep_stop")
        step = self._param(target, "sweep_step")
        npoints = round((stop - start) / step) + 1
        return np.linspace(start, stop, num=npoints, endpoint=True) - drive_lo

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[TwoToneSweetSpot]:
        """Fit the sweep, check it, save ``drive_if``/``drive_lo``/``fit_r2``, plot, and return the drive IF; read-only.

        Fit, then the acceptance check, then save, then plot: a rejected fit never reaches the DB (a reader
        would otherwise see a plausible ``drive_if`` for a step the run marked failed), as for T1 (QHC-1803).

        Raises:
            AnalysisError: when the Lorentzian fit is rejected (the step fails for this target).
        """
        operating_point = operating_point_for(self.calibration_file, self.experiment, target)
        model = LorentzianPeakModel(
            measurements[-1],
            target=str(target),
            path=self.parameters[target].get("data_folder"),
            r2_min=self._param(target, "min_r2"),
            title="Two tone (sweet spot)",
        )
        model.fit()
        if not model.accepted or model.fitted_if is None:
            raise AnalysisError(f"{target}: two-tone fit rejected: {model.results['quality']}")
        # The shared fit writes drive_if and fit_r2; the placed LO is recorded alongside them (QHC-1497).
        model.add_fitted_parameter(FittedParameter.DRIVE_LO, operating_point.drive.lo)
        model.save_fitting()
        plot = model.plot()
        return AnalysisResult(fitted=TwoToneSweetSpot(drive_if=model.fitted_if), plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[TwoToneSweetSpot]) -> None:
        """Write the fitted drive IF onto the sweet-spot operating point; the only writer (the LO is untouched)."""
        # analyze returns a fitted point or raises, so apply always has one; the guard is just a type narrow.
        if result.fitted is not None:
            operating_point = operating_point_for(self.calibration_file, self.experiment, target)
            operating_point.drive.if_ = result.fitted.drive_if
