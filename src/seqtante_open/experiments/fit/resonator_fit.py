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
"""Module for handling correction and fitting of Resonators data.

This module provides classes and functions to correct
and fit resonator data.

Classes:
    ResonatorSpectroscopyFit: Handles the correction and fitting
    of Resonator Spectroscopy data.

Functions:
    fit: fit the resonator spectroscopy data.
    plot: plot the fitted resonator spectroscopy data.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import qililab as ql
import xarray as xr
from scipy.signal import find_peaks, savgol_filter
from tqdm.auto import tqdm

from seqtante_open.experiments.fit import FittingClass
from seqtante_open.experiments.plotting import (
    auto_plot,
    convert_plot_units,
    get_xarray_from_meas,
    rotated_IQ_divide_by_median_col,
)
from seqtante_open.experiments.utils import lorentzian_fit_custom


class ResonatorSpectroscopyFit(FittingClass):
    """Handle the correction of the data and fitting
    of Resonator Spectroscopy data.
    """

    def __init__(
        self,
        qubit_idx: int,
        measurement_id: int,
        path: str | None = None
    ):
        """Initialize the class

        Args:
            qubit_idx (int): qubit index.
            measurement_id (int): id of the experiment in the database.
            path (str | None): path to save plots.
        """
        super().__init__(measurement_id=measurement_id, target=qubit_idx, path=path)
        calibration = ql.deserialize(self.measurement.calibration)
        parameters: dict = calibration.parameters

        self.data_folder = parameters["data_folder"]
        self.peak_axis: int | str = parameters["peak_axis"]
        self.filter_on: bool = parameters.get("filter_on", False)
        self.filter_window_length: int = parameters.get("filter_window_length", 21)
        self.filter_polyorder: int = parameters.get("filter_polyorder", 3)
        self.fit_lorentzian = parameters.get("fit_lorentzian", False)
        self.dataprocessing = np.abs if parameters["dataprocessing"] == "absolute" else rotated_IQ_divide_by_median_col

        self.x_axis: str = parameters["x_axis"]

        self.readout_if_list = None

    def _find_peaks_from_arrays(self,
                                data: np.ndarray,
                                x_vals: np.ndarray,
                                ):
        """Find peaks in a 2D array (num_traces, trace_length)"""
        peak_indices = []
        fitted_ifs = []

        for y in tqdm(data):
            if self.filter_on and len(y) >= self.filter_window_length:
                y_smooth = savgol_filter(y, window_length=self.filter_window_length, polyorder=self.filter_polyorder)
            else:
                y_smooth = y

            peaks, props = find_peaks(-y_smooth, prominence=0.01)
            if len(peaks) > 0:
                best_peak_index = peaks[np.argmax(props["prominences"])]
            else:
                best_peak_index = np.argmax(y_smooth)

            peak_indices.append(best_peak_index)

            if self.fit_lorentzian:
                fitted_if, _, _ = lorentzian_fit_custom(
                    y_values=y_smooth,
                    x_values=x_vals,
                    peak_pos_guess=best_peak_index,
                )
                fitted_ifs.append(fitted_if)

        if self.fit_lorentzian:
            return np.array(fitted_ifs)
        return x_vals[np.array(peak_indices)]

    def fit(self):
        xarr = get_xarray_from_meas(self.measurement)
        xarr = convert_plot_units(xarr)
        xarr = xr.apply_ufunc(self.dataprocessing, xarr.T)

        # Determine axis name
        if isinstance(self.peak_axis, str):
            axis_name = self.peak_axis
        elif isinstance(self.peak_axis, int):
            axis_name = xarr.dims[self.peak_axis]
        else:
            raise ValueError("peak_axis must be int or str")

        # Prepare data
        data = xarr.transpose(..., axis_name).to_numpy()
        x_vals = xarr[axis_name].to_numpy()

        self.readout_if_list = self._find_peaks_from_arrays(
            data=data,
            x_vals=x_vals,
        )

    def plot(self):
        title = "Resonator vs Flux"
        fig = auto_plot(self.measurement, x=self.x_axis)
        _, loops = self.measurement.load_old_h5()
        x_flux = loops["flux"]["array"]

        fig.add_trace(go.Scatter(
            x=x_flux,
            y=self.readout_if_list,
            mode="markers+lines",
            name="Peak IF freq",
            marker={"color": "red", "size": 6},
            line={"dash": "dot", "color": "red"},
        ))

        if self.path:
            qubit_path = os.path.join(self.path, f"q{self.qubit_idx}")
            os.makedirs(qubit_path, exist_ok=True)
            filepath = os.path.join(qubit_path, title + ".png")
            fig.write_image(filepath)
        else:
            fig.show()
