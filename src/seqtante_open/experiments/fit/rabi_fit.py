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

from seqtante_open.experiments.fit import FittingClass
from seqtante_open.experiments.plotting import (
    auto_plot,
    center_phase_around_median,
    convert_plot_units,
    get_xarray_from_meas,
)
from seqtante_open.experiments.analysis import rotate_iq
from seqtante_open.experiments.fit.utils import find_peaks_poly, sinus


class RabiFit:
    """Handle the correction of the data and fitting
    of Rabi data.
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

    def _find_drive_amp(self,
                        data: np.ndarray,
                        x_vals: np.ndarray,
                        ):
        """Find drive amplitude in a 1D array"""
        amp_sweep = self.loop["amplitude"]["array"]
        i = data[:,0]
        q = data[:,1]
        rotated_signal = rotate_iq(i + 1j * q)
        signal = np.real(rotated_signal)
        
        # Poly fit
        z = np.polyfit(amp_sweep, signal, 8)
        p = np.poly1d(z)
        fit_poly = p(amp_sweep)

        # Sinus fit
        mod = lmfit.Model(sinus)

        tt = np.array(amp_sweep)
        yy = np.array(signal)
        ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0])) * 2 * np.pi  # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.0**0.5
        guess_offset = np.mean(yy)

        # Set initial parameter values
        mod.set_param_hint("a", value=guess_amp)
        mod.set_param_hint("b", value=guess_freq)
        mod.set_param_hint("c", value=0)
        mod.set_param_hint("d", value=guess_offset)

        params = mod.make_params()
        fit = mod.fit(data=signal, params=params, x=amp_sweep, method="basinhopping")
        
        a_value = fit.params["a"].value
        b_value = fit.params["b"].value
        c_value = fit.params["c"].value
        d_value = fit.params["d"].value

        popt = [a_value, b_value, c_value, d_value]

        # T/8 with T as the sweep interval
        peak_threshold = amp_sweep[0] + np.abs((amp_sweep[-1] - amp_sweep[0]) / 12)

        fit_sinus = sinus(amp_sweep, *popt)
        peaks_max_sinus, _ = find_peaks(fit_sinus)
        peaks_min_sinus, _ = find_peaks(-fit_sinus)

        all_peaks_indices_sinus = np.concatenate([peaks_max_sinus, peaks_min_sinus])
        all_peaks_indices_sinus.sort()

        all_peaks_indices_poly = find_peaks_poly(fit_poly)

        peak_sinus = amp_sweep[all_peaks_indices_sinus[0]] if len(all_peaks_indices_sinus) != 0 else amp_sweep[-1]
        peak_2_sinus = min((peak_sinus + np.abs(1 / (popt[1]))), amp_sweep[-1])

        peak_poly = amp_sweep[all_peaks_indices_poly[0]] if len(all_peaks_indices_poly) != 0 else amp_sweep[-1]
        peak_2_poly = amp_sweep[all_peaks_indices_poly[1]] if len(all_peaks_indices_poly) > 1 else amp_sweep[-1]

        fitted_pi_pulse_amplitude_sinus = peak_sinus if peak_sinus >= peak_threshold else peak_2_sinus
        fitted_pi_pulse_amplitude_poly = peak_poly if peak_poly >= peak_threshold else peak_2_poly

        r2_poly = r2_score(signal, fit_poly)
        r2_sinus = r2_score(signal, fit_sinus)

        if r2_poly > r2_sinus:
            chosen_fit = "poly"
        else:
            chosen_fit = "sinus"

        fits = {
            "poly": [fitted_pi_pulse_amplitude_poly, fit_poly, r2_poly],
            "sinus": [fitted_pi_pulse_amplitude_sinus, fit_sinus, r2_sinus],
        }

        pi_amp = fits[chosen_fit][0]

        return pi_amp

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

        self.drive_amp = self._find_drive_amp(
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