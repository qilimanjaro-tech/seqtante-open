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

"""The ``reset_point_search`` node: a closed-loop search for the reset flux (ADR-0006, QHC-1807).

Ports the experimentalists' calibration script (``qilimanjaro-tech/calibration_script``) as its own
experiment, so Landau-Zener through QiliSDK matches the script's quality instead of a fixed wide sweep.
A read-only ``analyze`` that runs once cannot re-measure until the qubit is in the window, so the search
lives in ``execute_node``:

1. **Find the line.** Two-tone pulsed-saturation sweep at the current reset flux; if the Lorentzian fit
   is below its minimum R², widen the span and retry (the script's 30 MHz, then 60 MHz).
2. **Move into the window.** While the qubit is outside the target frequency window, move ``flux.z``
   along the measured slope — a small probe first, then secant steps clipped to a maximum step — waiting
   for the flux to settle between moves. Give up past a maximum total shift or number of moves.

Flux moves go through the restore handle (ADR-0004), so the run returns the fridge to where it was;
``execute_node`` returns every measurement id so the report shows each move. ``analyze`` checks the
final fit, and ``apply`` — the only writer — writes the reset point's absolute ``flux.z`` and drive IF
(ADR-0011: the measured flux is stored, the crosstalk matrix is not touched).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from qililab.typings.enums import Parameter

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.experiment_classes import two_tone_pulsed_saturation_experiment
from seqtante_open.experiments.fitting import LorentzianPeakModel
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from seqtante_open.targets import Target

LORENTZIAN_PEAK = "lorentzian_peak"

#: Defaults are the experimentalists' calibration-script values; every one is a tree parameter.
_DEFAULTS = {
    # Fit acceptance and span widening.
    "min_r2": 0.8,
    "retry_spans": (30e6, 60e6),
    # Target frequency window (absolute Hz) and the sweep that first looks for the line (absolute Hz).
    "target_freq_min": 2.79e9,
    "target_freq_max": 2.81e9,
    "sweep_start": 2.79e9,
    "sweep_stop": 2.81e9,
    "sweep_step": 0.4e6,
    # flux_z tuning.
    "flux_z_probe_step": 0.0005,
    "flux_z_max_step": 0.001,
    "flux_z_max_shift": 0.003,
    "flux_z_max_moves": 6,
    "flux_settle_s": 5.0,
    # Two-tone measurement.
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

#: A tolerance so a move that lands exactly on the shift limit is not rejected by floating-point noise.
_SHIFT_EPS = 1e-12


def next_flux_z_step(
    history: list[tuple[float, float]], target_freq: float, probe_step: float, max_step: float
) -> tuple[float, float | None, float | None]:
    """Choose the next ``flux_z`` move that should bring the qubit frequency to ``target_freq``.

    With one measurement the local slope is unknown, so the first move is a probe of ``probe_step``
    towards lower ``flux_z``. After that it is a secant step through the last two measurements, clipped
    to ``max_step`` either way. Ported verbatim from the calibration script.

    Args:
        history: ``(flux_z, qubit_frequency_hz)`` per measurement, oldest first; at least one entry.
        target_freq: Frequency to steer to, in Hz.
        probe_step: Size of the first, slope-finding move, in Φ0.
        max_step: Largest move allowed in one step, in Φ0.

    Returns:
        The clipped ``flux_z`` move in Φ0; the local slope in Hz/Φ0 it came from; and the unclipped move
        the secant predicts would reach ``target_freq``. The last two are ``None`` for the probe.

    Raises:
        ValueError: if the last two measurements share a ``flux_z`` or a qubit frequency, so no slope
            can be taken.
    """
    if len(history) < 2:
        return -probe_step, None, None
    (flux_1, freq_1), (flux_2, freq_2) = history[-2:]
    if flux_2 == flux_1 or freq_2 == freq_1:
        raise ValueError(
            f"cannot take a slope from flux_z {flux_1:.5f} -> {flux_2:.5f}, "
            f"f_q {freq_1 * 1e-9:.6f} -> {freq_2 * 1e-9:.6f} GHz"
        )
    slope = (freq_2 - freq_1) / (flux_2 - flux_1)
    required = (target_freq - freq_2) / slope
    return float(np.clip(required, -max_step, max_step)), slope, float(required)


@dataclass(frozen=True)
class ResetPoint:
    """What ``analyze`` hands ``apply``: the reset point's absolute flux and drive IF to write."""

    flux_z: float
    drive_if: float


@dataclass
class _SearchOutcome:
    """Per-target loop state ``execute_node`` leaves for ``analyze``/``apply`` (hardware state, not in the data)."""

    flux_z: float
    measurement_id: int
    plot_path: str | None


class ResetPointSearchNode(FixedFluxNode[ResetPoint]):
    """Closed-loop reset-flux search ported from the experimentalists' calibration script (QHC-1807)."""

    experiment: ClassVar[str] = "reset_point_search"
    analyses: ClassVar[tuple[str, ...]] = (LORENTZIAN_PEAK,)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._search: dict[Target, _SearchOutcome] = {}

    def _param(self, target: Target, key: str):
        return self.parameters[target].get(key, _DEFAULTS[key])

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the find-the-line / move-into-the-window loop per target; return every measurement id."""
        measurements: dict[Target, list[int]] = {}
        for target in targets:
            measurements[target] = self._search_one(target)
        return measurements

    def _search_one(self, target: Target) -> list[int]:
        point = operating_point_for(self.calibration_file, self.experiment, target)
        drive_lo = point.drive.lo
        flux_z_bus = bus_alias(target, Role.flux, "z")

        target_freq_min = self._param(target, "target_freq_min")
        target_freq_max = self._param(target, "target_freq_max")
        target_freq = (target_freq_min + target_freq_max) / 2
        max_shift = self._param(target, "flux_z_max_shift")
        max_moves = self._param(target, "flux_z_max_moves")

        flux_z_initial = point.flux.z
        flux_z = flux_z_initial
        ids: list[int] = []

        sweep_start = self._param(target, "sweep_start")
        sweep_stop = self._param(target, "sweep_stop")
        initial_center = (sweep_start + sweep_stop) / 2
        qubit_if, plot_path = self._find_qubit(target, point, initial_center, ids)
        if qubit_if is None:
            raise AnalysisError(f"{target}: no qubit line found at flux_z = {flux_z:.5f} in any sweep width")
        qubit_freq_abs = qubit_if + drive_lo
        history: list[tuple[float, float]] = [(flux_z, qubit_freq_abs)]
        moves = 0

        while not target_freq_min <= qubit_freq_abs <= target_freq_max:
            if moves >= max_moves:
                raise AnalysisError(f"{target}: qubit still outside the target window after {moves} flux_z moves")
            try:
                step, slope, required = next_flux_z_step(
                    history,
                    target_freq,
                    self._param(target, "flux_z_probe_step"),
                    self._param(target, "flux_z_max_step"),
                )
            except ValueError as error:
                raise AnalysisError(f"{target}: the qubit frequency is not responding to flux_z: {error}") from error
            next_flux_z = flux_z + step
            if (required is not None and abs(flux_z + required - flux_z_initial) > max_shift + _SHIFT_EPS) or abs(
                next_flux_z - flux_z_initial
            ) > max_shift + _SHIFT_EPS:
                raise AnalysisError(
                    f"{target}: reaching {target_freq * 1e-9:.6f} GHz needs a flux_z shift beyond the "
                    f"±{max_shift} limit from {flux_z_initial:.5f}"
                )
            expected_freq = qubit_freq_abs if slope is None else qubit_freq_abs + slope * step

            # The flux move goes through the restore handle, so the run returns the bus to where it was.
            self.restore.set(flux_z_bus, Parameter.FLUX, next_flux_z)
            time.sleep(self._param(target, "flux_settle_s"))
            flux_z = next_flux_z
            moves += 1

            qubit_if, plot_path = self._find_qubit(target, point, expected_freq, ids)
            if qubit_if is None:
                raise AnalysisError(f"{target}: lost the qubit line after moving flux_z to {flux_z:.5f}")
            qubit_freq_abs = qubit_if + drive_lo
            history.append((flux_z, qubit_freq_abs))

        self._search[target] = _SearchOutcome(flux_z=flux_z, measurement_id=ids[-1], plot_path=plot_path)
        return ids

    def _find_qubit(self, target: Target, point, center: float, ids: list[int]) -> tuple[float | None, str | None]:
        """Look for the qubit line around ``center`` (absolute Hz), widening the sweep until a fit is accepted.

        Every attempt is plotted (accepted or not), so a failed search still leaves its sweeps in the
        report. Returns the fitted IF and the accepted fit's plot path, so :meth:`analyze` can reuse that
        plot instead of re-plotting the same measurement.
        """
        sweep_span = self._param(target, "sweep_stop") - self._param(target, "sweep_start")
        spans = [sweep_span, *self._param(target, "retry_spans")]
        for span in spans:
            measurement_id = self._measure_two_tone(target, point, center - span / 2, center + span / 2)
            ids.append(measurement_id)
            model = LorentzianPeakModel(
                measurement_id,
                target=str(target),
                path=self.parameters[target].get("data_folder"),
                r2_min=self._param(target, "min_r2"),
            )
            model.fit()
            plot = model.plot()
            if model.accepted:
                return model.fitted_if, plot
        return None, None

    def _measure_two_tone(self, target: Target, point, freq_start: float, freq_stop: float) -> int:
        """Run one two-tone pulsed-saturation sweep over ``[freq_start, freq_stop]`` (absolute Hz); return its id."""
        drive_lo = point.drive.lo
        step = self._param(target, "sweep_step")
        npoints = round((freq_stop - freq_start) / step) + 1
        drive_if_sweep = np.linspace(freq_start, freq_stop, num=npoints, endpoint=True) - drive_lo
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
            drive_if_sweep=drive_if_sweep,
            drive_gain=self._param(target, "drive_power"),
            drive_sigmas=self._param(target, "drive_sigmas"),
            duration_ringup_drive=self._param(target, "duration_ringup_drive"),
            drive_LO=drive_lo,
            readout_LO=point.readout.lo,
            target=str(target),
            calibration=self.context.calibration,
        )

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[ResetPoint]:
        """Check the final fit and return the reset point's flux_z and drive IF; read-only.

        Raises:
            AnalysisError: when the final two-tone fit is rejected (the step fails for this target).
        """
        outcome = self._search[target]
        model = LorentzianPeakModel(
            outcome.measurement_id,
            target=str(target),
            path=self.parameters[target].get("data_folder"),
            r2_min=self._param(target, "min_r2"),
        )
        model.fit()
        model.save_fitting()
        if not model.accepted or model.fitted_if is None:
            raise AnalysisError(f"{target}: final reset-search fit rejected: {model.results['quality']}")
        # execute_node already plotted this accepted sweep during the search; reuse that plot.
        fitted = ResetPoint(flux_z=outcome.flux_z, drive_if=model.fitted_if)
        return AnalysisResult(fitted=fitted, plots=(outcome.plot_path,) if outcome.plot_path else ())

    def apply(self, target: Target, result: AnalysisResult[ResetPoint]) -> None:
        """Write the reset point's absolute ``flux.z`` and drive IF (ADR-0011); the only writer."""
        if result.fitted is None:  # pragma: no cover - reset_point_search always fits a point
            return
        point = operating_point_for(self.calibration_file, self.experiment, target)
        point.flux.z = result.fitted.flux_z
        point.drive.if_ = result.fitted.drive_if
