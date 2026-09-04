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

import numpy as np
import plotly.graph_objects as go
import scipy.optimize as sp

from seqtante_open.experiments.fitting.fit_base import FittingClass


class T1Fit(FittingClass):
    """Handle the correction of the data and fitting
    of T1 data.
    Args:
        target (str): Measured target token, e.g. ``"q1"``.
        measurement_id (int): ID of the measurement to fit.
        path (str | None, optional): Directory where the plot is saved. If None, the plot is shown. Defaults to None.
    """
    T1: int
    results: np.ndarray | None

    def __init__(self, target: str, measurement_id: int, path: str | None = None):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.results = None
        xarr = self.get_xarray()
        self.wait_values = xarr[xarr.dims[0]].data
        self.arr = np.real(self.rotate_iq(self.array[:, 0] + 1j * self.array[:, 1]))

    def fit(self):
        self.results, _ = sp.curve_fit(
            self.exponential,
            self.wait_values,
            self.arr,
            p0=self.exponential_initial_guess(self.wait_values, self.arr),
        )
        self.T1 = -1 / self.results[1]

    def plot(self):
        """Plot the rotated decay and the fitted exponential."""
        if self.results is None:
            raise RuntimeError("No fit results available, call fit() before plot().")

        title = f"{self.target}_T1"

        fig = go.Figure(
            go.Scatter(
                x=self.wait_values,
                y=self.arr,
                mode="markers",
                name="Data",
                marker={"color": "royalblue", "size": 5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.wait_values,
                y=self.exponential(self.wait_values, *self.results),
                mode="lines",
                name="Fitted exponential",
                line={"color": "red"},
            )
        )
        fig.update_layout(
            title=f"{self.target}, ID: {self.id}<br>Rotated T1 = {self.T1 / 1000:.3f} us",
            xaxis_title="Wait Time (ns)",
            yaxis_title="Integrated Voltage (a.u.)",
            width=1000,
            height=600,
            margin={"t": 120},
            showlegend=True,
        )

        self.save_plot(fig, title)
