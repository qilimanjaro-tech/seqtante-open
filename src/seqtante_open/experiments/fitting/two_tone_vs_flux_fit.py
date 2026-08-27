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

import os
from typing import cast

import numpy as np
import plotly.graph_objects as go
from qililab.data_management import build_platform
from qililab.typings.enums import Parameter

from .fit_base import FittingClass

# Fit logic constants
_MIN_TRACE_R_SQUARED = 0.9
_MAX_TRACE_ATTEMPTS = 20
_MIN_TRACES_FOR_PARABOLA = 3


class FluxoniumTwoToneFluxModel(FittingClass):
    """Fit and plot a two-tone (qubit spectroscopy) frequency-vs-flux map.

    Rotates the IQ plane of every flux row onto its signal quadrature, fits a Lorentzian to
    each to get the qubit IF at that flux, then fits a parabola through those IFs. The
    parabola's vertex is the flux sweet spot (:attr:`center`); :attr:`offset` is ``-center``,
    the bias correction that moves the sweet spot to zero flux. :meth:`plot` renders the
    rotated map with the per-flux fits, the parabola and the sweet spot.

    Rows whose Lorentzian never reaches ``r² >= 0.9`` are dropped, and the parabola is only
    fitted when at least three rows survive and its vertex falls inside the swept range;
    otherwise :attr:`fitted` stays ``False`` and callers must not use :attr:`offset`.

    Args:
        measurement_id: Autocalibration database id of the sweep to load.
        target: Swept target token, e.g. ``"q1"`` or ``"c1_2"``.
        path: Folder to save the plot into; if ``None`` the plot is shown.
        lo: Drive LO in Hz, used to report absolute qubit frequencies; if ``None`` it is read
            from the runcard stored with the measurement.
        flux_bus: Bus the flux loop swept, used to label the plot. Defaults to the bus the
            measurement recorded for that loop.
    """

    center: float
    offset: float
    r_squared: float
    coefficients: np.ndarray

    def __init__(
        self,
        measurement_id: int,
        target: str | None = None,
        path: str | None = None,
        lo: float | None = None,
        flux_bus: str | None = None,
    ):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.target = target
        self.lo = lo
        self.frequencies = np.asarray(self.loops["IF_frequency"]["array"])
        self.fluxes = np.asarray(self.loops["flux"]["array"])
        self.drive_bus = self.loops["IF_frequency"]["bus"]
        self.flux_bus = flux_bus if flux_bus is not None else self.loops["flux"]["bus"]
        self.rotated = np.zeros((len(self.fluxes), len(self.frequencies)))
        self.fitted_if = np.full(len(self.fluxes), np.nan)
        self.mask = np.zeros(len(self.fluxes), dtype=bool)
        self.fitted = False

    def _drive_lo(self) -> float:
        """Drive-bus LO frequency in Hz, taken from the runcard stored with the measurement."""
        platform = build_platform(cast("dict", self.measurement.platform_before))
        return platform.get_parameter(alias=self.drive_bus, parameter=Parameter.LO_FREQUENCY)

    def fit(self):
        """Fit the qubit IF at every flux, then the parabola whose vertex is the sweet spot."""
        for ii in range(len(self.fluxes)):
            self.rotated[ii] = np.real(self.rotate_iq(self.array[ii, :, 0] + 1j * self.array[ii, :, 1]))
            r_squared, attempts, fitted_if = 0.0, 0, np.nan
            while r_squared < _MIN_TRACE_R_SQUARED and attempts < _MAX_TRACE_ATTEMPTS:
                fitted_if, _, r_squared = self.lorentzian_fit(self.rotated[ii], self.frequencies)
                attempts += 1
            if r_squared >= _MIN_TRACE_R_SQUARED:
                self.mask[ii] = True
                self.fitted_if[ii] = fitted_if

        if self.mask.sum() < _MIN_TRACES_FOR_PARABOLA:
            return

        coefficients = np.polyfit(self.fluxes[self.mask], self.fitted_if[self.mask], 2)
        if coefficients[0] == 0:
            return

        center = -coefficients[1] / (2 * coefficients[0])
        if not self.fluxes.min() <= center <= self.fluxes.max():
            return

        residuals = self.fitted_if[self.mask] - np.polyval(coefficients, self.fluxes[self.mask])
        spread = self.fitted_if[self.mask] - self.fitted_if[self.mask].mean()
        self.coefficients = coefficients
        self.r_squared = float(1 - residuals @ residuals / (spread @ spread)) if spread.any() else 0.0
        self.center = float(center)
        self.offset = -self.center
        self.fitted = True

    def plot(self):
        """Plot the rotated map with the per-flux fits, the parabola and the sweet spot."""
        title = f"Two Tone vs Flux \n {self.target} {self.flux_bus}"
        lo = self.lo if self.lo is not None else self._drive_lo()
        frequencies_ghz = (self.frequencies + lo) * 1e-9

        fig = go.Figure(
            go.Heatmap(
                x=frequencies_ghz,
                y=self.fluxes,
                z=self.rotated,
                colorscale="Viridis",
                colorbar={"title": {"text": "Integrated Voltage (a.u.)", "side": "right"}},
            )
        )
        fig.add_scatter(
            x=(self.fitted_if[self.mask] + lo) * 1e-9,
            y=self.fluxes[self.mask],
            mode="markers",
            name="Fitted IF",
            marker={"color": "red", "size": 12},
        )
        if self.fitted:
            fluxes = np.linspace(self.fluxes.min(), self.fluxes.max(), 101)
            fig.add_scatter(
                x=(np.polyval(self.coefficients, fluxes) + lo) * 1e-9,
                y=fluxes,
                mode="lines",
                name=f"Parabola (r² = {self.r_squared:.3f})",
                line={"color": "red", "width": 2},
            )
            fig.add_hline(
                y=self.center,
                line={"color": "darkorange", "dash": "dot", "width": 3},
                annotation_text=f"Sweet spot = {self.center:.4f}, offset = {self.offset:.4f}",
            )

        fig.update_xaxes(title_text="Qubit Frequency (GHz)")
        fig.update_yaxes(title_text="Flux (phi_0)")
        fig.update_layout(
            title={
                "text": f"{title}, ID: {self.id}".replace("\n", "<br>"),
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 22},
            },
            width=1000,
            height=700,
            legend={"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(0,0,0,0.2)"},
            showlegend=True,
        )

        self.save_plot(fig, title)
