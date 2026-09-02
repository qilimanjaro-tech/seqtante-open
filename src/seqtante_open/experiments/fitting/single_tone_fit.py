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
from qililab.data_management import build_platform
from qililab.typings.enums import Parameter

from seqtante_open.experiments.fitting.fit_base import FittingClass


class FluxoniumSingleToneModel(FittingClass):
    """Locate and plot the resonator dip of a single-tone frequency sweep.

    Loads the measurement's 1D ``S21`` trace, reduces it to a magnitude in dB and
    takes the deepest point of the sweep as the resonance. No curve is fitted, so
    the answer is always one of the swept frequencies and its resolution is the
    sweep step. :meth:`plot` renders the magnitude trace against the absolute
    readout frequency and marks the dip. Frequencies are handled in Hz and only
    converted for display.

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
        self.s21 = xarr.to_numpy()

    def _readout_lo(self) -> float:
        """Readout-bus LO frequency in Hz, taken from the runcard stored with the measurement."""
        platform = build_platform(cast("str", self.measurement.platform_before))
        return platform.get_parameter(alias=self.readout_bus, parameter=Parameter.LO_FREQUENCY)

    def fit(self):
        """Take the deepest point of the magnitude trace as the resonance.

        Returns:
            float: The fitted IF in Hz, also stored under ``results["signal"]``.
        """
        magnitude = self.decibels(self.s21)
        fitted_if = self.frequencies[np.argmin(magnitude)]
        self.results = {"signal": {"fitted_if": fitted_if, "magnitude": magnitude}}

        return fitted_if

    def plot(self):
        """Plot the magnitude trace and mark the dip."""
        if not self.results:
            raise RuntimeError("No fit results available, call fit() before plot().")

        title = f"Single Tone {self.target}"
        lo = self.lo if self.lo is not None else self._readout_lo()
        fitted_if = cast("float", self.results["signal"]["fitted_if"])
        magnitude = cast("np.ndarray", self.results["signal"]["magnitude"])
        resonance_ghz = (fitted_if + lo) * 1e-9

        fig = go.Figure(
            go.Scatter(
                x=(self.frequencies + lo) * 1e-9,
                y=magnitude,
                mode="lines+markers",
                name="Data",
                line={"color": "royalblue", "dash": "dash"},
                marker={"color": "royalblue", "size": 5},
            )
        )
        fig.add_vline(
            x=resonance_ghz,
            line={"color": "red", "dash": "dot", "width": 2},
            annotation_text=f"Resonator Freq = {resonance_ghz:.4f} GHz",
        )
        fig.update_layout(
            title=f"{title}, ID: {self.id}<br>Fitted Intermediate Frequency = {fitted_if * 1e-6:.4f} MHz",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Integrated Voltage (dB)",
            width=1000,
            height=600,
            margin={"t": 120},
            showlegend=True,
        )

        self.save_plot(fig, title)
