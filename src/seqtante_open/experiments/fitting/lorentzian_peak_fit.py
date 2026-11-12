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
from scipy.optimize import curve_fit

from seqtante_open.experiments.fitting.fit_base import FittingClass


class _LorentzianResults(TypedDict):
    fitted_if: float | None
    r_squared: float
    linewidth: float
    accepted: bool
    quality: str
    params: tuple[float, float, float, float] | None


def _lorentzian(x: np.ndarray, x0: float, a: float, gamma: float, c: float) -> np.ndarray:
    """A Lorentzian peak: ``c + (a * gamma / 2) / pi / ((x - x0)^2 + (gamma/2)^2)``."""
    return c + (a * gamma / 2) / np.pi / ((x - x0) ** 2 + (gamma / 2) ** 2)


class LorentzianPeakModel(FittingClass):
    """Fit a Lorentzian peak to a two-tone sweep's magnitude and return the qubit IF (QHC-1807).

    Ported from the experimentalists' calibration script (``two_tones_lorentzian_fit_and_plot``). The
    magnitude ``|I + iQ|`` is fit with a Lorentzian whose starting values come from the data; the fit is
    **accepted** only if it converges, its R² reaches the minimum, its centre lies inside the sweep, and
    its linewidth is wider than one sweep step and narrower than the whole sweep. A rejected fit leaves
    :attr:`results` ``accepted`` false and ``fitted_if`` ``None`` — the reset-point search reads that to
    widen the sweep or fail the step.

    The fitted drive IF and R² (``drive_if``, ``fit_r2``; QHC-1497) are written with :meth:`save_fitting`.

    Args:
        measurement_id: Autocalibration database id of the two-tone sweep to load.
        target: Swept target token, e.g. ``"q14"``.
        path: Folder to save the plot into; if ``None`` the plot is shown.
        r2_min: Smallest R² accepted as a fit. Defaults to 0.8 (the script's value).
    """

    results: _LorentzianResults

    def __init__(self, measurement_id: int, target: str | None = None, path: str | None = None, r2_min: float = 0.8):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.target = target
        self.r2_min = r2_min
        xarr = self.get_xarray()
        frequency_coord = xarr[xarr.dims[0]]
        self.frequencies = np.asarray(frequency_coord.data, dtype=float)
        self.magnitude = np.abs(xarr.to_numpy())
        # ``results`` is populated by fit(); left unset until then, so it reads as unpopulated.

    def fit(self) -> _LorentzianResults:
        """Fit the Lorentzian, apply the acceptance checks and record the outcome."""
        frequency, magnitude = self.frequencies, self.magnitude
        guess = self._initial_guess(frequency, magnitude)

        fitted_if: float | None = None
        r_squared = 0.0
        linewidth = 0.0
        params: tuple[float, float, float, float] | None = None
        try:
            best, _ = curve_fit(_lorentzian, frequency, magnitude, p0=guess)
        except (RuntimeError, ValueError) as error:
            quality = f"fit did not converge ({error})"
        else:
            params = (float(best[0]), float(best[1]), float(best[2]), float(best[3]))
            model = _lorentzian(frequency, *best)
            r_squared = float(1 - np.sum((magnitude - model) ** 2) / np.sum((magnitude - np.mean(magnitude)) ** 2))
            step = abs(frequency[1] - frequency[0])
            span = abs(frequency[-1] - frequency[0])
            linewidth = abs(best[2])
            quality = f"R2 = {r_squared:.2f}, f_q IF = {best[0] * 1e-6:.3f} MHz, linewidth = {linewidth * 1e-3:.0f} kHz"
            if r_squared < self.r2_min:
                quality += f": rejected, R2 below {self.r2_min}"
            elif not min(frequency) <= best[0] <= max(frequency):
                quality += ": rejected, centre outside the sweep"
            elif not step < linewidth < span:
                quality += ": rejected, linewidth not resolved by the sweep"
            else:
                fitted_if = float(best[0])

        self.results = {
            "fitted_if": fitted_if,
            "r_squared": r_squared,
            "linewidth": linewidth,
            "accepted": fitted_if is not None,
            "quality": quality,
            "params": params,
        }
        return self.results

    @staticmethod
    def _initial_guess(frequency: np.ndarray, magnitude: np.ndarray) -> tuple[float, float, float, float]:
        """Starting ``(x0, amplitude, linewidth, offset)`` for the Lorentzian, as the script picks them."""
        edge_vs_mean = np.mean(magnitude[0:5]) - np.mean(magnitude)
        if edge_vs_mean < 0:
            extreme_index = int(np.argmax(magnitude))
            sign = 1.0
        else:
            extreme_index = int(np.argmin(magnitude))
            sign = -1.0
        x0 = float(frequency[extreme_index])
        amplitude = sign * 30e5
        linewidth = abs((frequency[0] + frequency[-1]) / 10)
        offset = float(magnitude[0])
        return x0, amplitude, linewidth, offset

    @property
    def fitted_if(self) -> float | None:
        """The fitted qubit IF in Hz, or ``None`` when the fit was rejected."""
        self._require_fit()
        return self.results["fitted_if"]

    @property
    def r_squared(self) -> float:
        self._require_fit()
        return self.results["r_squared"]

    @property
    def accepted(self) -> bool:
        """Whether the fit passed every acceptance check."""
        self._require_fit()
        return self.results["accepted"]

    def save_fitting(self):
        """Write the fitted drive IF and the fit R² to the measurement entry (QHC-1497, QHC-1791)."""
        from seqtante_open.outputs import FittedParameter

        self._require_fit()
        if self.results["fitted_if"] is not None:
            self.add_fitted_parameter(FittedParameter.DRIVE_IF, self.results["fitted_if"])
        self.add_fitted_parameter(FittedParameter.FIT_R2, self.results["r_squared"])
        self.save_fitting_to_database()

    def _require_fit(self):
        if not getattr(self, "results", None):
            raise RuntimeError("No fit results available, call fit() first.")

    def plot(self):
        """Plot the sweep and, when accepted, its Lorentzian fit; save it next to the measurement."""
        self._require_fit()
        results = self.results
        title = f"Two tone (reset search) {self.target}"
        frequencies_mhz = self.frequencies * 1e-6

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=frequencies_mhz,
                y=self.magnitude,
                mode="markers",
                name="Data",
                marker={"color": "royalblue", "size": 5},
            )
        )
        if results["accepted"] and results["params"] is not None:
            dense = np.linspace(self.frequencies[0], self.frequencies[-1], 501)
            fig.add_trace(
                go.Scatter(
                    x=dense * 1e-6,
                    y=_lorentzian(dense, *results["params"]),
                    mode="lines",
                    name="Fit",
                    line={"color": "red", "width": 2},
                )
            )
        fig.update_layout(
            title=f"{title}, ID: {self.id} ({results['quality']})",
            xaxis_title="IF frequency (MHz)",
            yaxis_title="Amplitude (a.u.)",
            width=1000,
            height=600,
        )
        return self.save_plot(fig, title, save_to_db=True)
