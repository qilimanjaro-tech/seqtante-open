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

import math
from typing import TypedDict

import numpy as np
import plotly.graph_objects as go

from seqtante_open.experiments.fitting.fit_base import FittingClass


class _LorentzianResults(TypedDict):
    fitted_if: float | None
    r_squared: float
    linewidth: float
    accepted: bool
    quality: str
    #: ``(amplitude, center, width, offset)`` of :meth:`FittingClass.lorentzian`, NaN when the fit failed.
    params: tuple[float, float, float, float]


class LorentzianPeakModel(FittingClass[_LorentzianResults]):
    """Fit a Lorentzian peak to a two-tone sweep's magnitude and return the qubit IF (QHC-1807).

    The acceptance rules are the experimentalists' calibration script's (``two_tones_lorentzian_fit_and_plot``).
    The rotated real quadrature ``Re(rotate_iq(I + iQ))`` is fit with the shared
    :meth:`~seqtante_open.experiments.fitting.fit_base.FittingClass.fit_lorentzian`, the one Lorentzian
    fit (QHC-1932 item 5). The fit is **accepted** only if its R² reaches the minimum, its centre lies
    inside the sweep, and its linewidth is wider than one sweep step and narrower than the whole sweep. A rejected fit leaves
    :attr:`results` ``accepted`` false and ``fitted_if`` ``None`` — the reset-point search reads that to
    widen the sweep or fail the step.

    The fitted drive IF and R² (``drive_if``, ``fit_r2``; QHC-1497) are written with :meth:`save_fitting`.

    Args:
        measurement_id: Autocalibration database id of the two-tone sweep to load.
        target: Swept target token, e.g. ``"q14"``.
        path: Folder to save the plot into; if ``None`` the plot is shown.
        r2_min: Smallest R² accepted as a fit. Defaults to 0.8 (the script's value).
        title: Plot-title prefix, so each caller labels its own sweep (the target is appended). Defaults
            to ``"Two tone"``.
    """

    def __init__(
        self,
        measurement_id: int,
        target: str | None = None,
        path: str | None = None,
        r2_min: float = 0.8,
        title: str = "Two tone",
    ):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.target = target
        self.r2_min = r2_min
        self._plot_title = title
        xarr = self.get_xarray()
        frequency_coord = xarr[xarr.dims[0]]
        self.frequencies = np.asarray(frequency_coord.data, dtype=float)
        self.signal = np.real(self.rotate_iq(xarr.to_numpy()))

    def fit(self) -> _LorentzianResults:
        """Fit the Lorentzian, apply the acceptance checks and record the outcome."""
        # A refit that finds nothing must not leave the previous result readable.
        self._results = None
        frequency = self.frequencies
        fit = self.fit_lorentzian(frequency, self.signal)
        r_squared, centre, linewidth = fit["r_squared"], fit["center"], abs(fit["width"])
        params = (fit["amplitude"], centre, fit["width"], fit["offset"])
        step = abs(frequency[1] - frequency[0])
        span = abs(frequency[-1] - frequency[0])
        fitted_if: float | None = None
        quality = f"R2 = {r_squared:.2f}, f_q IF = {centre * 1e-6:.3f} MHz, linewidth = {linewidth * 1e-3:.0f} kHz"
        # Every comparison with NaN is False, so an undefined R² (a flat trace or a failed fit) is rejected explicitly.
        if math.isnan(r_squared) or r_squared < self.r2_min:
            quality += f": rejected, R2 below {self.r2_min}"
        elif not min(frequency) <= centre <= max(frequency):
            quality += ": rejected, centre outside the sweep"
        elif not step < linewidth < span:
            quality += ": rejected, linewidth not resolved by the sweep"
        else:
            fitted_if = centre

        # Stored even for a rejected fit: ``accepted`` and ``quality`` are its result, read by the node.
        self.results = _LorentzianResults(
            fitted_if=fitted_if,
            r_squared=r_squared,
            linewidth=linewidth,
            accepted=fitted_if is not None,
            quality=quality,
            params=params,
        )
        return self.results

    @property
    def fitted_if(self) -> float | None:
        """The fitted qubit IF in Hz, or ``None`` when the fit was rejected."""
        return self.results["fitted_if"]

    @property
    def r_squared(self) -> float:
        return self.results["r_squared"]

    @property
    def accepted(self) -> bool:
        """Whether the fit passed every acceptance check."""
        return self.results["accepted"]

    def save_fitting(self):
        """Write the fitted drive IF and the fit R² to the measurement entry (QHC-1497, QHC-1791)."""
        from seqtante_open.outputs import FittedParameter

        if self.results["fitted_if"] is not None:
            self.add_fitted_parameter(FittedParameter.DRIVE_IF, self.results["fitted_if"])
        self.add_fitted_parameter(FittedParameter.FIT_R2, self.results["r_squared"])
        self.save_fitting_to_database()

    def plot(self):
        """Plot the sweep and, when accepted, its Lorentzian fit; save it next to the measurement."""
        results = self.results
        title = f"{self._plot_title} {self.target}"
        frequencies_mhz = self.frequencies * 1e-6

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=frequencies_mhz,
                y=self.signal,
                mode="markers",
                name="Data",
                marker={"color": "royalblue", "size": 5},
            )
        )
        if results["accepted"]:
            dense = np.linspace(self.frequencies[0], self.frequencies[-1], 501)
            fig.add_trace(
                go.Scatter(
                    x=dense * 1e-6,
                    y=self.lorentzian(dense, *results["params"]),
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
        return self.save_plot(fig, title)
