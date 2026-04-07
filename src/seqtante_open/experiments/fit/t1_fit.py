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

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import qililab as ql
import xarray as xr
from scipy.signal import find_peaks
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
import scipy.optimize as sp
from seqtante_open.experiments.fit import FittingClass
from seqtante_open.experiments.plotting import (
    auto_plot,
    center_phase_around_median,
    convert_plot_units,
    get_xarray_from_meas,
)
from seqtante_open.experiments.analysis import rotate_iq
from seqtante_open.experiments.fit.utils import decaying_exponential


class T1Fit:
    """Handle the correction of the data and fitting
    of T1 data.
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
            loop (dict): dictionary with the loop and units
            path (str | None): path to save plots
        """
        self.qubit_idx = qubit_idx
        self.measurement = measurement
        self.loop = loop
        self.path = path
        self.drive_amp = None

    def _find_t1_coherence(self,
                        data: np.ndarray,
                        x_vals: np.ndarray,
                        ):
        """Find drive amplitude in a 1D array"""
        wait_sweep = self.loop["time"]["array"]
        i = data[:,0]
        q = data[:,1]
        rotated_signal = rotate_iq(i + 1j * q)
        signal = np.real(rotated_signal)
        
        initial_guess = [1, 8_000, 0]  # initial guess for the parameters

        optimized_params_1, _ = sp.curve_fit(
            decaying_exponential, wait_sweep, signal, p0=initial_guess
        )
        t1_fitted = optimized_params_1[1]/1000

        return t1_fitted

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

        self.drive_amp = self._find_t1_coherence(
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