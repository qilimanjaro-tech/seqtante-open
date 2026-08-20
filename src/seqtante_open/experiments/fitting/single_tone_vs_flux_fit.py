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
from qililab.data_management import build_platform
from qililab.typings.enums import Parameter

from seqtante_open.experiments.fitting.fit_base import FittingClass
from seqtante_open.experiments.plotting import plot_measurement_2d_heatmap_updated
from seqtante_open.outputs import output_controller


class FluxoniumSingleToneFluxModel(FittingClass):
    """Fit and plot a single-tone resonator-vs-flux sweep.

    Loads the measurement's 2D ``|S21|`` map (flux x frequency), finds the
    flux-symmetry point of each frequency trace and stores the median as
    ``center`` (the flux sweet spot); ``offset`` is ``-center``, the bias
    needed to sit at zero net flux. :meth:`plot` renders the heatmap with the
    readout ``LO + IF`` axis and overlays the per-trace centers.

    Args:
        measurement_id: Autocalibration database id of the sweep to load.
        target: Swept target token, e.g. ``"q1"`` or ``"c1_2"``.
        path: Folder to save the plot into; if ``None`` the plot is shown.
        lo: Readout LO in Hz for the ``LO + IF`` axis; if ``None`` it is read
            from the runcard stored with the measurement.
    """

    center: float
    offset: float
    per_trace: np.ndarray
    result: dict[str, dict[str, np.ndarray]]

    def __init__(
        self, measurement_id: int, target: str | None = None, path: str | None = None, lo: float | None = None
    ):

        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.lo = lo
        xarr = self.get_xarray()
        xarr = self.convert_plot_units(xarr)
        self.array = xarr
        self.frequencies = xarr[xarr.dims[1]].data
        self.fluxes = xarr[xarr.dims[0]].data
        self.readout_bus = xarr[xarr.dims[1]].attrs["bus"]
        self.flux_bus = xarr[xarr.dims[0]].attrs["bus"]

    def _readout_lo(self) -> float:
        """Readout-bus LO frequency, taken from the runcard stored with the measurement."""
        platform = build_platform(cast("dict", self.measurement.platform_before))
        return platform.get_parameter(alias=self.readout_bus, parameter=Parameter.LO_FREQUENCY)

    @staticmethod
    def _auto_convolve_trace(z_col, axis, mode="abs", p=0.5, half=8, unwrap=True):
        """Locate the symmetry point of one 1D feature via auto-convolution.

        Reduces a complex column to a real signal (amplitude or phase), mean-centers
        it, and convolves it with itself. The lobe is symmetric about 2*centroid, so
        its smoothed argmax maps back to the feature's position on ``axis``.

        Args:
            z_col (np.ndarray): Complex trace to locate the symmetry point of.
            axis (np.ndarray): Values the returned center is expressed on.
            mode (str, optional): ``"abs"`` to reduce by amplitude, ``"angle"`` by phase.
                Defaults to ``"abs"``.
            p (float, optional): Geometric weight of the smoothing kernel. Defaults to 0.5.
            half (int, optional): Half-width of the smoothing kernel in samples. Defaults to 8.
            unwrap (bool, optional): Unwrap the phase before centering, used only when
                ``mode`` is ``"angle"``. Defaults to True.

        Returns:
            tuple[float, np.ndarray, np.ndarray]: ``center``, the symmetry point on
                ``axis``; ``axis_s``, the trimmed convolution axis; and ``smooth``, the
                smoothed auto-convolution. The last two are returned for plotting.

        Raises:
            ValueError: If ``mode`` is neither ``"abs"`` nor ``"angle"``.
        """
        z_col = np.asarray(z_col)

        if mode == "abs":
            sig = np.abs(z_col - np.mean(z_col))
        elif mode == "angle":
            phase = np.angle(z_col)
            if unwrap:
                phase = np.unwrap(phase)
            sig = np.abs(phase - np.mean(phase))
        else:
            raise ValueError(f"Mode must be 'abs' or 'angle', got {mode!r}")

        sig = sig.astype(float)
        sig = sig - np.mean(sig)

        conv = np.convolve(sig, sig, mode="full")  # peaks at 2*centroid
        conv_axis = np.linspace(np.min(axis), np.max(axis), len(conv))

        weights = np.array([p ** abs(k) for k in range(-half, half + 1)])
        weights /= weights.sum()
        w = len(weights)

        axis_s = conv_axis[w // 2 : -1 - w // 2]
        smooth = np.array([np.dot(weights, conv[ii : ii + w]) for ii in range(len(axis_s))])

        center = axis_s[np.argmax(smooth)]
        return center, axis_s, smooth

    @staticmethod
    def _find_median_symmetry_point_of_image(z_arr, loop1, modes=("abs", "angle"), **kwargs):
        n = z_arr.shape[0]
        result = {m: {"per_trace": np.empty(n)} for m in modes}

        for i in range(n):
            for m in modes:
                c, _, _ = FluxoniumSingleToneFluxModel._auto_convolve_trace(z_arr[i, :], loop1, mode=m, **kwargs)
                result[m]["per_trace"][i] = c
        for m in modes:
            result[m]["center"] = np.median(result[m]["per_trace"])

        return result

    def fit(self, mode="abs", p=0.5, half=8, unwrap=True):
        res = self._find_median_symmetry_point_of_image(
            self.array.data.T, self.fluxes, modes=(mode,), p=p, half=half, unwrap=unwrap
        )
        self.result = res
        self.center = res[mode]["center"]
        self.per_trace = res[mode]["per_trace"]
        self.offset = -self.center

    def plot(self):
        title = f"Single Tone vs Flux \n {self.target} {self.flux_bus}"
        lo = self.lo if self.lo is not None else self._readout_lo()
        fig = plot_measurement_2d_heatmap_updated(xarr=self.array.transpose(..., self.array.dims[0]), title=title + f" ID: {self.id}", fixed_LO_freq=lo, dataprocessing=self.decibels)
        fig.add_scatter(
            x=self.per_trace,
            y=self.frequencies,
            mode="markers",
            name="Center per Trace",
            line={"color": "white", "width": 1.5},
            marker={"size": 4},
        )
        fig.add_vline(
            x=self.center, line={"color": "red", "dash": "dot", "width": 4}, annotation_text=f"Median Center = {self.center:.4f}"
        )
        fig.update_layout(
            legend={"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(0,0,0,0.2)"},
            showlegend=True,
        )

        self.save_plot(fig, title)
