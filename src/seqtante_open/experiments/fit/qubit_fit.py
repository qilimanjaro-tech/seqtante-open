# Copyright 2024 Qilimanjaro Quantum Tech
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
"""Module for handling correction and fitting of Qubits data.

This module provides classes and functions to correct
and fit qubit data.

Classes:
    QubitSpectroscopyFit: Handles the correction and fitting
    of Qubit Spectroscopy data.

Functions:
    fit: fit the qubit spectroscopy data.
    plot: plot the fitted qubit spectroscopy data.
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
)
from seqtante_open.experiments.analysis import rotate_iq
from seqtante_open.experiments.analysis import lorentzian


class QubitSpectroscopyFit:
    """Handle the correction of the data and fitting
    of Qubit Spectroscopy data.
    """

    def __init__(
        self, qubit_idx: int,
        measurement: np.ndarray,
        loop: dict,
        path: str | None = None
    ):
        """Initialize the class

        Args:
            qubit_idx (int): qubit index
            measurement (): array with the measurement result
            path (str | None): path to save plots
        """
        self.qubit_idx = qubit_idx
        self.measurement = measurement
        self.loop = loop
        self.path = path
        self.qubit_freq = None

    def _find_qubit_freq(self,
                        data: np.ndarray,
                        x_vals: np.ndarray,
                        ):
        """Find qubit frequency in a 2D array"""
        if_sweep = self.loop["frequency"]["array"]
        flux_sweep = self.loop["flux"]["array"]
        fitted_ifs = np.empty((len(flux_sweep)))
        r_squareds = np.empty((len(flux_sweep)))
        i = data[:,:,0]
        q = data[:,:,1]
        rotated_signal = rotate_iq(i + 1j * q)
        
        fit_values = np.empty((len(x_vals), len(if_sweep)))
        fitted_ifs = np.empty((len(x_vals)))
        r_squareds = np.empty((len(x_vals)))
        mask_i = np.empty(len(x_vals), dtype=bool)


        for ii in range(len(x_vals)):
            i_fitted_if, i_fitvals, i_rsquared = lorentzian(np.real(rotated_signal[ii]), if_sweep)

            if i_rsquared < 0.60:
                i_fitted_if = np.nan
                mask_i[ii] = False
            else:
                mask_i[ii] = True
            
            fitted_ifs[ii] = i_fitted_if
            fit_values[ii] = i_fitvals
            r_squareds[ii] = i_rsquared

        i_coeffs = np.polyfit(flux_sweep[mask_i], fitted_ifs[:][mask_i], 2)
        i_sweetspot = -i_coeffs[1]/(2*i_coeffs[0])

        return i_sweetspot

    def fit(self):
        """Empty placeholder for now"""
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

        self.qubit_if = self._find_qubit_freq(
            data=data,
            x_vals=x_vals,
        )

        return None

    def plot(self, x_axis: str):
        title = "Two tone vs Flux"
        fig = auto_plot(self.measurement,
                        x=x_axis,
                        dataprocessing=center_phase_around_median)

        if self.path:
            qubit_path = os.path.join(self.path, f"q{self.qubit_idx}")
            os.makedirs(qubit_path, exist_ok=True)
            filepath = os.path.join(qubit_path, title + ".png")
            fig.write_image(filepath)
        else:
            fig.show()