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
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from scipy.special import erf as _erf
from seqtante_open.outputs import output_controller


class FittingClass(ABC):
    """Base class for all fittings."""

    def __init__(self,
                 measurement_id: int,
                 target: int | list[int] | tuple[int],
                 path: str | None = None,
                 ):
        """Base class for all fittings. It includes many functions for the fittings.

        Args:
            measurement_id (int): ID of the experiment to fit in the autocalibration database.
            target (int | list[int]): _description_
            path (str | None, optional): Directory of the folder where the plot/s are saved, if None it shows the plot. Defaults to None.
        """
        self.id = measurement_id
        self.measurement = output_controller.db_manager.load_calibration_by_id(measurement_id)
        self.array, self.loops = self.measurement.load_h5()

        self.path = path
        if isinstance(target, int):
            self.qubit_idx = target
        if isinstance(target, (tuple, list)):
            self.control_qubit_idx = target[0]
            self.target_qubit_idx = target[1]

    def fit(self):
        """Fits the experimental data to the corresponding function."""

    def plot(self):
        """Plots the fitted data. if a folder is given, it saves ther plot, else it shows it."""

    def fit_and_plot(self):
        """Sequentially does the fit and the plot functions."""
        self.fit()
        self.plot()

    def save_plot(self, title):
        if self.path:
            filepath = os.path.join(self.path, title + ".png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # ------------------- Utils Block -------------------

    @staticmethod
    def rotate_iq(arr: np.ndarray):
        """
        Function to rotate the I vs Q matrix to synthesized the data obtained
        into a single dimension (Real component)
        leaving the other only with noise (Imaginary component)

        Args:
            arr (np.array(float)): array of complex numbers corresponding to IQ signal

        Returns:
            np.array(float): rotated values
        """
        # Compute the covariance matrix
        cov = np.cov(arr.real, arr.imag)
        # Get the eigenvalues and eigenvectors of the covariance matrix
        w, v = np.linalg.eig(cov)
        # Find the index of the max eigenvalue
        max_idx = np.argmax(w)
        # Compute the angle of rotation
        angle = np.arctan2(v[max_idx, 1], v[max_idx, 0])
        # Rotate the array
        rotated = arr * np.exp(1j * angle)
        return rotated

    @staticmethod
    def wrap_pi(angle: float) -> float:
        """Wrap angle to (-π, π]."""
        angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
        if angle <= -np.pi:
            angle = np.pi
        return angle

    @staticmethod
    def find_peaks_poly(poly_fit):
        sorted_peaks_indices = []
        aux = poly_fit[:-1] - poly_fit[1:]
        aux = aux > 0
        bool_comulative = aux[0]
        for i, bool_curr in enumerate(aux[1:]):
            if bool_curr != bool_comulative:
                bool_comulative = bool_curr
                sorted_peaks_indices.append(i + 1)

        return sorted_peaks_indices

    @staticmethod
    def fit_drag(loop_values, signal_0, signal_1):
        difference = signal_0 - signal_1

        z = np.polyfit(loop_values, difference, 2)
        p = np.poly1d(z)
        fit = p(loop_values)
        roots = p.roots
        fitted_drag_coeff = roots[0] if loop_values[0] < roots[0] < loop_values[1] else roots[1]

        return fit, fitted_drag_coeff

    @staticmethod
    def exponential(x: float | list | np.ndarray, A: float, B: float, C: float):
        '''Returns exponential A * np.exp(B * x) + C'''
        return A * np.exp(B * x) + C

    @staticmethod
    def exponential_initial_guess(x_array: list | np.ndarray, y_array: list | np.ndarray) -> tuple[float]:
        """Generates an rough initial guess for an exponential fitting."""
        _n = len(y_array)
        _end_idx = 1 if int(_n * 2 / 100) < 1 else int(_n * 2 / 100)
        C: float = np.average(y_array[-_end_idx])
        A: float = y_array[0] - C
        B: float = -1 / x_array[int(_n / 2)]
        return A, B, C

    @staticmethod
    def joint_model(x, mmt_relax, thermal_pop, std0, v0, std1, v1, N):
        """Model used to fit the data."""
        prep0_m0 = 1 - thermal_pop
        prep1_m0 = mmt_relax
        cdf_prep0 = N * FittingClass.two_gaussians(x, prep0_m0, std0, v0, std1, v1)
        cdf_prep1 = N * FittingClass.two_gaussians(x, prep1_m0, std0, v0, std1, v1)

        return np.concatenate((cdf_prep0, cdf_prep1))

    @staticmethod
    def lorentzian_fit(y_values, x_values):
        """Fits the data to a lorentzian"""

        def lorentzian(x, amplitude, center, width, offset):
            return amplitude / (1 + ((x - center) / (0.5 * width)) ** 2) + offset

        # Fit signal
        fit_signal = y_values

        # Lorentzian fit
        mod = Model(lorentzian)

        # Set initial parameter values
        initial_amp = np.max(fit_signal) - np.min(fit_signal)
        if (np.mean(fit_signal) - np.min(fit_signal)) ** 2 > (np.mean(fit_signal) - np.max(fit_signal)) ** 2:
            # if the mean of the values is closer to the maximum, it means we have negative amplitude!
            initial_amp = -initial_amp

        mod.set_param_hint(
            "amplitude",
            value=initial_amp,
            min=-abs(np.max(fit_signal) - np.min(fit_signal)),
            max=abs(np.max(fit_signal) - np.min(fit_signal)),
        )
        mod.set_param_hint("center", value=np.mean(x_values), min=np.min(x_values), max=np.max(x_values))
        mod.set_param_hint(
            "width",
            value=(np.max(x_values) - np.min(x_values)) / 10,
            min=(np.max(x_values) - np.min(x_values)) / 100,
            max=(np.max(x_values) - np.min(x_values)),
        )
        mod.set_param_hint("offset", value=np.mean(fit_signal), min=np.min(fit_signal), max=np.max(fit_signal))

        params = mod.make_params()
        fit = mod.fit(data=fit_signal, params=params, x=x_values, method="differential_evolution")

        # Update r_squared value
        r_squared = fit.rsquared

        fitted_amplitude = fit.params["amplitude"].value
        fitted_center = fit.params["center"].value
        fitted_width = fit.params["width"].value
        fitted_offset = fit.params["offset"].value

        popt = [fitted_amplitude, fitted_center, fitted_width, fitted_offset]

        if fitted_amplitude < 0:
            fitted_if = x_values[
                np.argmin(lorentzian(x_values, fitted_amplitude, fitted_center, fitted_width, fitted_offset))
            ]
        else:
            fitted_if = x_values[
                np.argmax(lorentzian(x_values, fitted_amplitude, fitted_center, fitted_width, fitted_offset))
            ]

        fit_values = lorentzian(x_values, *popt)

        return fitted_if, fit_values, r_squared

    @staticmethod
    def sinus(x, a, b, c, d):
        return a * np.sin(b * np.array(x) - c) + d

    @staticmethod
    def sinus_abs(x, a, b, c, d):
        return abs(a) * np.sin(b * np.array(x) - c) + d

    @staticmethod
    def two_gaussians(x, a0, std0, v0, std1, v1):
        a1 = 1 - a0
        cdf_0 = a0 * (1 + _erf((x - v0) / (std0 * 2**0.5))) / 2
        cdf_1 = a1 * (1 + _erf((x - v1) / (std1 * 2**0.5))) / 2

        return cdf_0 + cdf_1

    @staticmethod
    def sinus_exp(x, a, b, c, d, f):
        # Define the decaying exponential function
        return a * np.cos(2 * np.pi * b * x + c) * np.exp(-1 * x / d) + f

    @staticmethod
    def line(x, a, b):
        return a * x + b

    @staticmethod
    def cosfunc(phi, A, omega, offset, phase_offset):
        return offset + A * np.cos(omega * phi + phase_offset)