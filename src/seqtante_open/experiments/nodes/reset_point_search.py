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

1. **Find the line.** Two-tone pulsed-saturation sweep (the qili ``two_tone_saturation`` QProgram, run by
   :class:`~seqtante_open.experiments.nodes.two_tone_saturation.TwoToneSaturationNode`) at the current reset flux; if the Lorentzian fit
   is below its minimum R², widen the span and retry (the script's 30 MHz, then 60 MHz).
2. **Move into the window.** While the qubit is outside the target frequency window, move ``flux.z``
   along the measured slope — a small probe first, then secant steps clipped to a maximum step — waiting
   for the flux to settle after the placement jump and after every move, as the script does. Give up
   past a maximum total shift or number of moves.

Flux moves go through the restore handle (ADR-0004), so the run returns the fridge to where it was;
``execute_node`` returns every measurement id so the report shows each move. ``analyze`` checks the
final fit, and ``apply`` — the only writer — writes the reset point's absolute ``flux.z`` and drive IF
(ADR-0011: the measured flux is stored, the crosstalk matrix is not touched).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qili_qprograms.bus_alias import Role, bus_alias
from qililab.typings.enums import Parameter

from seqtante_open.controllers.calibration_node import AnalysisResult
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import LorentzianPeakModel
from seqtante_open.experiments.nodes.two_tone_saturation import TWO_TONE_SATURATION_DEFAULTS, TwoToneSaturationNode

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qili_qprograms.targets import Target

LORENTZIAN_PEAK = "lorentzian_peak"

#: Defaults are the experimentalists' calibration-script values (the node's ``defaults``); every one is a
#: tree parameter. The two-tone measurement's own defaults come from the shared spine. The target
#: frequency window, ``target_freq_min`` / ``target_freq_max`` (absolute Hz), has no default: it is the
#: device's, so a tree that leaves it out fails the step instead of steering to another device's window.
RESET_POINT_SEARCH_DEFAULTS: Mapping[str, Any] = {
    # Fit acceptance and span widening.
    "min_r2": 0.8,
    "retry_spans": (30e6, 60e6),
    # The drive-IF window that first looks for the line: ``np.linspace`` args as offsets from the stored
    # drive IF (QHC-1932). Its spacing is kept when the search re-centres the sweep or widens it to
    # ``retry_spans``; a retry span no wider than this window is skipped.
    "drive_if_sweep": (-10e6, 10e6, 51),
    # flux_z tuning.
    "flux_z_probe_step": 0.0005,
    "flux_z_max_step": 0.001,
    "flux_z_max_shift": 0.003,
    "flux_z_max_moves": 6,
    # Wait for the flux to settle after each move, in ns (5 s).
    "flux_settle_duration": 5_000_000_000,
    # Two-tone measurement.
    **TWO_TONE_SATURATION_DEFAULTS,
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


class ResetPointSearchNode(TwoToneSaturationNode[ResetPoint]):
    """Closed-loop reset-flux search ported from the experimentalists' calibration script (QHC-1807)."""

    experiment = "reset_point_search"
    analyses = (LORENTZIAN_PEAK,)
    defaults = RESET_POINT_SEARCH_DEFAULTS

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._search: dict[Target, _SearchOutcome] = {}

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the find-the-line / move-into-the-window loop per target; return every measurement id.

        Every target's target window and sweeps are checked before anything is measured. ``run`` has just placed the
        operating point, a jump from the zeroed flux much larger than any search move, so the first
        sweep waits ``flux_settle_duration`` as every later move does (the script sleeps after its
        initial flux set too): a first point read while the flux still drifts seeds a wrong secant.

        Raises:
            KeyError: if a target sets no ``target_freq_min`` or ``target_freq_max``.
            ValueError: if a target's window is empty (:meth:`_target_window`), or its ``drive_if_sweep``
                gives a sweep no fit can use (:meth:`_sweep_offsets`).
        """
        offsets = {}
        for target in targets:
            self._target_window(target)
            offsets[target] = self._sweep_offsets(target)
        time.sleep(max(self._param(target, "flux_settle_duration") for target in targets) * 1e-9)
        return {target: self._search_one(target, offsets[target]) for target in targets}

    def _target_window(self, target: Target) -> tuple[float, float]:
        """The ``(target_freq_min, target_freq_max)`` window the search steers the qubit into, in Hz.

        Raises:
            KeyError: if the step sets no ``target_freq_min`` or ``target_freq_max`` (no default).
            ValueError: if ``target_freq_min`` is not below ``target_freq_max``, so no frequency is inside.
        """
        target_freq_min = self._param(target, "target_freq_min")
        target_freq_max = self._param(target, "target_freq_max")
        if target_freq_min >= target_freq_max:
            raise ValueError(
                f"step '{self.name}': {target} target_freq_min {target_freq_min:g} Hz is not below "
                f"target_freq_max {target_freq_max:g} Hz, so no qubit frequency is inside the window"
            )
        return target_freq_min, target_freq_max

    def _sweep_offsets(self, target: Target) -> list[np.ndarray]:
        """The drive-IF offsets of each sweep :meth:`_find_qubit` may take, centred on 0, in Hz.

        The first is the step's ``drive_if_sweep`` window; each retry widens it to the next of
        ``retry_spans`` at the same point spacing. A retry span no wider than the sweep before it
        would re-measure inside a window that just failed, so it is skipped.

        Raises:
            ValueError: if ``drive_if_sweep`` is not a usable window (:meth:`_sweep`).
        """
        window = self._sweep(target, "drive_if_sweep")
        spacing = abs(window[1] - window[0])
        offsets = [window - (window[0] + window[-1]) / 2]
        for span in self._param(target, "retry_spans"):
            half = round(span / spacing / 2)
            # Widths are whole spacings, so half a spacing absorbs floating-point noise.
            if 2 * half * spacing > np.ptp(offsets[-1]) + spacing / 2:
                offsets.append(np.arange(-half, half + 1) * spacing)
        return offsets

    def _search_one(self, target: Target, offsets: list[np.ndarray]) -> list[int]:
        operating_point = self._operating_point(target)
        drive_lo = operating_point.drive.lo
        flux_z_bus = bus_alias(target, Role.flux, "z")

        target_freq_min, target_freq_max = self._target_window(target)
        target_freq = (target_freq_min + target_freq_max) / 2
        max_shift = self._param(target, "flux_z_max_shift")
        max_moves = self._param(target, "flux_z_max_moves")

        flux_z_initial = operating_point.flux.z
        flux_z = flux_z_initial
        ids: list[int] = []

        window = self._sweep(target, "drive_if_sweep")
        window_centre = (window[0] + window[-1]) / 2
        qubit_if, plot_path = self._find_qubit(target, operating_point.drive.if_ + window_centre, offsets, ids)
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
            time.sleep(self._param(target, "flux_settle_duration") * 1e-9)
            flux_z = next_flux_z
            moves += 1

            qubit_if, plot_path = self._find_qubit(target, expected_freq - drive_lo, offsets, ids)
            if qubit_if is None:
                raise AnalysisError(f"{target}: lost the qubit line after moving flux_z to {flux_z:.5f}")
            qubit_freq_abs = qubit_if + drive_lo
            history.append((flux_z, qubit_freq_abs))

        self._search[target] = _SearchOutcome(flux_z=flux_z, measurement_id=ids[-1], plot_path=plot_path)
        return ids

    def _find_qubit(
        self, target: Target, centre_if: float, offsets: list[np.ndarray], ids: list[int]
    ) -> tuple[float | None, str | None]:
        """Look for the qubit line around ``centre_if`` (Hz), widening the sweep until a fit is accepted.

        Each sweep is the next of ``offsets`` (:meth:`_sweep_offsets`) centred on ``centre_if``. Every
        attempt is plotted (accepted or not), so a failed search still leaves its sweeps in the report.
        Returns the fitted IF and the accepted fit's plot path, so :meth:`analyze` can reuse that plot
        instead of re-plotting it.
        """
        for offset in offsets:
            measurement_id = self._measure_two_tone_saturation(target, centre_if + offset)
            ids.append(measurement_id)
            model = LorentzianPeakModel(
                measurement_id,
                target=str(target),
                path=self.parameters[target].get("data_folder"),
                r2_min=self._param(target, "min_r2"),
                title="Two tone (reset search)",
            )
            model.fit()
            plot = model.plot()
            if model.accepted:
                return model.fitted_if, plot
        return None, None

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[ResetPoint]:
        """Fit the final sweep, check it, save it, and return the reset point's flux_z and drive IF; read-only.

        Fit, then the acceptance check, then save: a rejected fit never reaches the DB (a reader would
        otherwise see a plausible ``drive_if`` for a step the run marked failed), as for T1 (QHC-1803).
        The plot was already made during the search and is reused.

        Raises:
            AnalysisError: when the final two-tone fit is rejected (the step fails for this target).
        """
        outcome = self._search[target]
        model = LorentzianPeakModel(
            outcome.measurement_id,
            target=str(target),
            path=self.parameters[target].get("data_folder"),
            r2_min=self._param(target, "min_r2"),
            title="Two tone (reset search)",
        )
        model.fit()
        if not model.accepted or model.fitted_if is None:
            raise AnalysisError(f"{target}: final reset-search fit rejected: {model.results['quality']}")
        model.save_fitting()
        # execute_node already plotted this accepted sweep during the search; reuse that plot.
        fitted = ResetPoint(flux_z=outcome.flux_z, drive_if=model.fitted_if)
        return AnalysisResult(fitted=fitted, plots=(outcome.plot_path,) if outcome.plot_path else ())

    def apply(self, target: Target, result: AnalysisResult[ResetPoint]) -> None:
        """Write the reset point's absolute ``flux.z`` and drive IF (ADR-0011); the only writer."""
        # reset_point_search always fits a point, so result.fitted is set here; the guard is defensive.
        if result.fitted is not None:
            operating_point = self._operating_point(target)
            operating_point.flux.z = result.fitted.flux_z
            operating_point.drive.if_ = result.fitted.drive_if
