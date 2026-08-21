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

from typing import cast

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qililab.data_management import build_platform
from qililab.typings.enums import Parameter

from seqtante_open.experiments.fitting.fit_base import FittingClass

_QUADRATURE_LABELS = {
    "signal": "Signal quadrature (rotated)",
    "noise": "Noise quadrature (orthogonal)",
}


class FluxoniumSingleToneModel(FittingClass):
    """Fit and plot a single-tone (resonator spectroscopy) frequency sweep.

    Loads the measurement's 1D ``S21`` trace and rotates the IQ plane so the
    response collapses onto one quadrature (``signal``) while the orthogonal one
    holds only noise (``noise``), then fits a Lorentzian to each. :meth:`plot`
    renders both quadratures side by side, each with its data and its fit, and
    marks the fitted IF. All frequencies are handled in Hz and only converted
    for display.

    Args:
        measurement_id: Autocalibration database id of the sweep to load.
        target: Swept target token, e.g. ``"q1"``.
        path: Folder to save the plot into; if ``None`` the plot is shown.
        lo: Readout LO in Hz, used to report the absolute readout frequency; if
            ``None`` it is read from the runcard stored with the measurement.
    """

    results: dict[str, dict[str, float | np.ndarray]]

    def __init__(
        self, measurement_id: int, target: str | None = None, path: str | None = None, lo: float | None = None
    ):

        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.target = target
        self.lo = lo
        self.results = {}
        xarr = self.get_xarray()
        self.array = xarr
        freq_coord = xarr[xarr.dims[0]]
        self.frequencies = freq_coord.data
        self.readout_bus = freq_coord.attrs["bus"]
        rota = self.rotate_iq(xarr.to_numpy())
        self.signal = np.real(rota)
        self.noise = np.imag(rota)

    @property
    def quadratures(self) -> dict[str, np.ndarray]:
        """Rotated quadratures to fit, keyed as :attr:`results` and :data:`_QUADRATURE_LABELS`."""
        return {"signal": self.signal, "noise": self.noise}

    def _readout_lo(self) -> float:
        """Readout-bus LO frequency in Hz, taken from the runcard stored with the measurement."""
        platform = build_platform(cast("dict", self.measurement.platform_before))
        return platform.get_parameter(alias=self.readout_bus, parameter=Parameter.LO_FREQUENCY)

    def fit(self):
        """Fit a Lorentzian to each rotated quadrature."""
        for quadrature, values in self.quadratures.items():
            fitted_if, fit_values, r_squared = self.lorentzian_fit(values, self.frequencies)
            self.results[quadrature] = {
                "fitted_if": fitted_if,
                "fit_values": fit_values,
                "r_squared": r_squared,
            }

    def plot(self):
        """Plot both quadratures with their fits, side by side."""
        if not self.results:
            raise RuntimeError("No fit results available, call fit() before plot().")

        title = f"Single Tone {self.target}"
        lo = self.lo if self.lo is not None else self._readout_lo()
        frequencies_mhz = self.frequencies * 1e-6

        subplot_titles = [
            f"{label}<br>IF = {self.results[quadrature]['fitted_if'] * 1e-6:.3f} MHz, "
            f"f_readout = {(self.results[quadrature]['fitted_if'] + lo) * 1e-9:.4f} GHz, "
            f"r² = {self.results[quadrature]['r_squared']:.3f}"
            for quadrature, label in _QUADRATURE_LABELS.items()
        ]

        fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles, horizontal_spacing=0.08)

        for col, (quadrature, values) in enumerate(self.quadratures.items(), start=1):
            res = self.results[quadrature]
            fit_values = cast("np.ndarray", res["fit_values"])
            fig.add_trace(
                go.Scatter(
                    x=frequencies_mhz,
                    y=values,
                    mode="lines+markers",
                    name="Data",
                    legendgroup="data",
                    showlegend=col == 1,
                    line={"color": "royalblue", "dash": "dash"},
                    marker={"color": "royalblue", "size": 5},
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=frequencies_mhz,
                    y=fit_values,
                    mode="lines",
                    name="Fit",
                    legendgroup="fit",
                    showlegend=col == 1,
                    line={"color": "red", "dash": "dash", "width": 2},
                ),
                row=1,
                col=col,
            )
            fitted_if_mhz = cast("float", res["fitted_if"]) * 1e-6
            fig.add_trace(
                go.Scatter(
                    x=[fitted_if_mhz, fitted_if_mhz],
                    y=[min(values.min(), fit_values.min()), max(values.max(), fit_values.max())],
                    mode="lines",
                    name="Fitted IF",
                    legendgroup="fitted_if",
                    showlegend=col == 1,
                    line={"color": "green", "dash": "dot", "width": 2},
                ),
                row=1,
                col=col,
            )
            fig.update_xaxes(title_text="IF (MHz)", row=1, col=col)

        fig.update_yaxes(title_text="Integrated Voltage (a.u.)", row=1, col=1)
        fig.update_layout(
            title=f"{title}, ID: {self.id}",
            width=1400,
            height=600,
            margin={"t": 120, "b": 110},
            legend={"orientation": "h", "x": 0.5, "y": -0.15, "xanchor": "center", "yanchor": "top"},
            showlegend=True,
        )

        self.save_plot(fig, title)
