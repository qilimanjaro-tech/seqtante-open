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

from typing import TypedDict

import numpy as np
import plotly.graph_objects as go
from lmfit import Model

from seqtante_open.experiments.fitting.fit_base import FittingClass


class _RabiResults(TypedDict):
    offset: float
    amplitude: float
    angular_frequency: float
    phase: float
    r_squared: float
    pi_amplitude: float
    level_0: float
    level_0_std: float
    level_1: float
    level_1_std: float


def _cosine(x: np.ndarray, offset: float, amplitude: float, angular_frequency: float, phase: float) -> np.ndarray:
    """An undamped cosine: ``offset + amplitude * cos(angular_frequency * x + phase)``."""
    return offset + amplitude * np.cos(angular_frequency * x + phase)


class RabiAmplitudeModel(FittingClass):
    """Fit a Rabi amplitude sweep (DRAG drive) and read the pi amplitude and the |0>/|1> readout levels.

    The measurement sweeps the drive amplitude; the rotated readout signal oscillates as a cosine of
    the drive amplitude. An undamped cosine is fit (an amplitude sweep has no time decay), and from it:

    - the two readout **levels** are the cosine's extrema, ``offset ± |amplitude|``. Which one is |0> is
      decided by the first sweep point (the qubit sits in the ground state at the lowest drive). Each
      level carries the uncertainty propagated from the fitted ``offset`` and ``amplitude``. The levels
      fold in thermal population and readout SNR, an accepted caveat until single-shot readout
      calibration exists (QHC-1806).
    - the **pi amplitude** is the drive amplitude of the first excited-state extremum after the ground
      point: half a Rabi period on from the ground extremum nearest the first sweep point.

    Fitted parameters (``rabi_amplitude``, ``fit_r2``; QHC-1497) are written to the measurement entry
    with :meth:`save_fitting`.

    Args:
        measurement_id: Autocalibration database id of the amplitude sweep to load.
        target: Swept target token, e.g. ``"q1"``.
        path: Folder to save the plot into; if ``None`` the plot is shown.
    """

    results: _RabiResults

    def __init__(self, measurement_id: int, target: str | None = None, path: str | None = None):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.target = target
        xarr = self.get_xarray()
        amplitude_coord = xarr[xarr.dims[0]]
        self.amplitudes = np.asarray(amplitude_coord.data, dtype=float)
        self.drive_bus = amplitude_coord.attrs.get("bus", "")
        self.signal = np.real(self.rotate_iq(xarr.to_numpy()))
        # ``results`` is populated by fit(); left unset until then, so it reads as unpopulated.

    def fit(self) -> _RabiResults:
        """Fit the undamped cosine and derive the pi amplitude and the two readout levels."""
        amplitudes, signal = self.amplitudes, self.signal

        offset_guess = float(np.mean(signal))
        amplitude_guess = float((np.max(signal) - np.min(signal)) / 2) or 1.0
        angular_frequency_guess = self._angular_frequency_guess(amplitudes, signal)

        model = Model(_cosine)
        model.set_param_hint("amplitude", value=amplitude_guess, min=0.0)
        model.set_param_hint("angular_frequency", value=angular_frequency_guess, min=0.0)
        model.set_param_hint("offset", value=offset_guess)
        model.set_param_hint("phase", value=0.0, min=-np.pi, max=np.pi)
        params = model.make_params()
        fit = model.fit(data=signal, params=params, x=amplitudes)

        offset = fit.params["offset"].value
        amplitude = fit.params["amplitude"].value
        angular_frequency = fit.params["angular_frequency"].value
        phase = fit.params["phase"].value
        offset_std = fit.params["offset"].stderr or 0.0
        amplitude_std = fit.params["amplitude"].stderr or 0.0

        # Both levels (offset ± amplitude) carry the same std, propagated from offset and amplitude
        # ignoring their covariance -- an accepted approximation for the level uncertainty.
        level_std = float(np.hypot(offset_std, amplitude_std))
        top, bottom = offset + amplitude, offset - amplitude
        # The first sweep point is the ground state; |0> is the extremum its fitted value sits closest to.
        first = _cosine(amplitudes[0], offset, amplitude, angular_frequency, phase)
        ground, excited = (top, bottom) if abs(first - top) <= abs(first - bottom) else (bottom, top)

        self.results = {
            "offset": offset,
            "amplitude": amplitude,
            "angular_frequency": angular_frequency,
            "phase": phase,
            "r_squared": float(fit.rsquared),
            "pi_amplitude": self._pi_amplitude(offset, amplitude, angular_frequency, phase, excited),
            "level_0": float(ground),
            "level_0_std": level_std,
            "level_1": float(excited),
            "level_1_std": level_std,
        }
        return self.results

    @staticmethod
    def _angular_frequency_guess(amplitudes: np.ndarray, signal: np.ndarray) -> float:
        """Estimate the oscillation's angular frequency from the FFT of the (centred) signal."""
        step = float(np.mean(np.diff(amplitudes)))
        if step == 0:
            return 1.0
        spectrum = np.abs(np.fft.rfft(signal - np.mean(signal)))
        frequencies = np.fft.rfftfreq(len(signal), d=abs(step))
        peak = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
        dominant = frequencies[peak] if peak else 1.0 / (abs(step) * len(signal))
        return float(2 * np.pi * dominant)

    def _pi_amplitude(
        self, offset: float, amplitude: float, angular_frequency: float, phase: float, excited: float
    ) -> float:
        """The first drive amplitude at which the fitted cosine reaches the excited level, after the first point.

        Walks a dense grid across the swept range and returns the first amplitude whose fitted value is
        closest to the excited extremum. Falls back to the first sweep point's amplitude if the fit is
        degenerate (no oscillation), which then fails the step's R² gate upstream.
        """
        if angular_frequency <= 0:
            return float(self.amplitudes[0])
        grid = np.linspace(self.amplitudes[0], self.amplitudes[-1], max(1000, 10 * len(self.amplitudes)))
        curve = _cosine(grid, offset, amplitude, angular_frequency, phase)
        # The excited extremum is reached where the fitted curve is closest to the excited level.
        reached = grid[np.argmin(np.abs(curve - excited))]
        return float(reached)

    @property
    def pi_amplitude(self) -> float:
        self._require_fit()
        return self.results["pi_amplitude"]

    @property
    def readout_levels(self) -> tuple[float, float, float, float]:
        """``(level_0, level_0_std, level_1, level_1_std)`` of the |0>/|1> readout response."""
        self._require_fit()
        return (
            self.results["level_0"],
            self.results["level_0_std"],
            self.results["level_1"],
            self.results["level_1_std"],
        )

    @property
    def r_squared(self) -> float:
        self._require_fit()
        return self.results["r_squared"]

    def save_fitting(self):
        """Write the pi amplitude and the fit R² to the measurement entry (QHC-1497, QHC-1791)."""
        from seqtante_open.outputs import FittedParameter

        self._require_fit()
        self.add_fitted_parameter(FittedParameter.RABI_AMPLITUDE, self.results["pi_amplitude"])
        self.add_fitted_parameter(FittedParameter.FIT_R2, self.results["r_squared"])
        self.save_fitting_to_database()

    def _require_fit(self):
        if not getattr(self, "results", None):
            raise RuntimeError("No fit results available, call fit() first.")

    def plot(self):
        """Plot the sweep, its cosine fit, the two readout levels and the pi amplitude."""
        self._require_fit()
        results = self.results
        title = f"Rabi amplitude {self.target}"
        curve = _cosine(
            self.amplitudes, results["offset"], results["amplitude"], results["angular_frequency"], results["phase"]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.amplitudes,
                y=self.signal,
                mode="lines+markers",
                name="Data",
                line={"color": "royalblue", "dash": "dash"},
                marker={"color": "royalblue", "size": 5},
            )
        )
        fig.add_trace(
            go.Scatter(x=self.amplitudes, y=curve, mode="lines", name="Fit", line={"color": "red", "width": 2})
        )
        for level, name, colour in (
            (results["level_0"], "|0>", "green"),
            (results["level_1"], "|1>", "orange"),
        ):
            fig.add_hline(y=level, line={"color": colour, "dash": "dot"}, annotation_text=name)
        fig.add_vline(
            x=results["pi_amplitude"],
            line={"color": "purple", "dash": "dot"},
            annotation_text=f"pi amp = {results['pi_amplitude']:.4g}",
        )
        fig.update_layout(
            title=f"{title}, ID: {self.id} (r² = {results['r_squared']:.3f})",
            xaxis_title="Drive amplitude (a.u.)",
            yaxis_title="Integrated Voltage (a.u.)",
            width=1000,
            height=600,
        )
        return self.save_plot(fig, title, save_to_db=True)
