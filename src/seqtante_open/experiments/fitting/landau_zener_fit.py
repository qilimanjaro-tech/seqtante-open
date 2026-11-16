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

"""Normalise and plot the compartmentalized Landau-Zener transition (QHC-1808, ADR-0002).

MVP: the averaged readout quadratures are rotated to the real axis and mapped to a transition
probability with the reset-point readout levels (|0>, |1>), then plotted against the total annealing
time. No model is fit and no agreement metric is computed yet: the theory curve and the metric that
give the benchmark its verdict are a follow-up (QHC-1921), so the node reports no value and the run
leaves the benchmark "Not evaluated".
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from seqtante_open.experiments.fitting.fit_base import FittingClass


class LandauZenerFit(FittingClass):
    """Normalise the Landau-Zener transition signal with the readout levels and plot it (QHC-1808).

    The averaged readout quadratures are rotated to the real axis (:meth:`FittingClass.rotate_iq`) and
    mapped to a transition probability ``P = (signal - level_0) / (level_1 - level_0)`` with the
    |0>/|1> readout levels taken at the reset point (ADR-0001). The levels are passed in by the node,
    which reads them from the calibration and enforces the readout-pulse match (ADR-0001, D8).

    No model is fit: the agreement metric and theory curve are a follow-up (QHC-1921).

    Args:
        measurement_id (int): Id of the measurement to normalise and plot.
        target (str): Measured target token, e.g. ``"q1"``.
        level_0 (float): The |0> readout level, in the rotated-signal units.
        level_1 (float): The |1> readout level, in the rotated-signal units.
        path (str | None, optional): Directory the plot is saved into; if None the plot is shown.
    """

    annealing_times: np.ndarray
    transition_probability: np.ndarray

    def __init__(self, measurement_id: int, target: str, level_0: float, level_1: float, path: str | None = None):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        if level_1 == level_0:
            raise ValueError(f"the |0> and |1> readout levels are equal ({level_0}); cannot normalise the signal")
        self.level_0 = level_0
        self.level_1 = level_1

        xarr = self.get_xarray()
        self.annealing_times = np.asarray(xarr[xarr.dims[0]].data)
        signal = np.real(self.rotate_iq(self.array[:, 0] + 1j * self.array[:, 1]))
        self.transition_probability = (signal - level_0) / (level_1 - level_0)

    def plot(self) -> str | None:
        """Plot the normalised transition probability against the total annealing time.

        Returns:
            str | None: Path the plot was written to, or None when it was shown instead.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.annealing_times,
                y=self.transition_probability,
                mode="lines+markers",
                name="P(transition)",
            )
        )
        fig.update_layout(
            title=f"Landau-Zener transition {self.target}",
            xaxis_title="Total annealing time (ns)",
            yaxis_title="Transition probability",
        )
        return self.save_plot(fig, title=f"landau_zener_{self.target}")
