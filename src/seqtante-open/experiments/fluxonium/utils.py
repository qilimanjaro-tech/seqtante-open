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

import qcodes as qc
import lmfit
import numpy as np
from qilitools.analysis.lorentzian_fit import lorentzian
from qililab.instruments.voltage_source import VoltageSource


def generate_qdac_voltage_param(qdac_channel, param_name):
    setter = qdac_channel.dc_constant_V
    getter = qdac_channel.dc_constant_V

    return qc.Parameter(name=param_name,
                        label=param_name,
                        set_cmd=setter,
                        get_cmd=getter,
                        unit="V")


def set_all_flux_channels_to_zero(voltage_source: VoltageSource):
    for channel_id in voltage_source.dacs:
        channel = voltage_source.device.channel(channel_id)
        channel.dc_constant_V(0.0)


def lorentzian_fit_custom(y_values, x_values, peak_pos_guess=False):
    """
    Fit a Lorentzian model to spectroscopy data using non-linear optimization.

    Parameters
    ----------
    Y_VALUES : array_like
        Measured signal values (e.g., I or Q amplitude).
    X_VALUES : array_like
        Independent variable values (e.g., frequency sweep points).

    Returns
    -------
    fitted_if : float
        The frequency (or x-value) at which the Lorentzian reaches its minimum or maximum.
        This is used as the "fitted IF" (intermediate frequency).
    fit_values : np.ndarray
        The fitted Lorentzian curve evaluated at all X_VALUES.
    r_squared : float
        Coefficient of determination (R²) for the fit, indicating quality.

    Notes
    -----
    - Uses the `lmfit` package with the `differential_evolution` global optimization method.
    - Automatically detects whether the peak is a dip or bump by checking amplitude sign.
    - Suitable for resonator or qubit spectroscopy data where a Lorentzian line shape is expected.
    """
    fit_signal = y_values

    # Create model
    mod = lmfit.Model(lorentzian)

    # Estimate initial amplitude sign (dip or peak)
    initial_amp = np.max(fit_signal) - np.min(fit_signal)
    if (np.mean(fit_signal) - np.min(fit_signal)) ** 2 > (np.mean(fit_signal) - np.max(fit_signal)) ** 2:
        initial_amp = -initial_amp

    # Set parameter hints
    mod.set_param_hint("amplitude", value=initial_amp,
                       min=-abs(np.max(fit_signal) - np.min(fit_signal)),
                       max=abs(np.max(fit_signal) - np.min(fit_signal)))
    if peak_pos_guess:
        mod.set_param_hint("center", value=x_values[peak_pos_guess],
                           min=np.min(x_values), max=np.max(x_values))
    else:
        mod.set_param_hint("center", value=np.mean(x_values),
                           min=np.min(x_values), max=np.max(x_values))

    mod.set_param_hint("width", value=(np.max(x_values) - np.min(x_values)) / 10,
                       min=(np.max(x_values) - np.min(x_values)) / 100,
                       max=(np.max(x_values) - np.min(x_values)))
    mod.set_param_hint("offset", value=np.mean(fit_signal),
                       min=np.min(fit_signal), max=np.max(fit_signal))

    # Fit the model
    params = mod.make_params()
    fit = mod.fit(data=fit_signal, params=params, x=x_values, method="differential_evolution")
    r_squared = fit.rsquared

    # Extract parameters
    fitted_amplitude = fit.params["amplitude"].value
    fitted_center = fit.params["center"].value
    fitted_width = fit.params["width"].value
    fitted_offset = fit.params["offset"].value
    popt = [fitted_amplitude, fitted_center, fitted_width, fitted_offset]

    # Determine peak/dip position
    if fitted_amplitude < 0:
        fitted_if = x_values[np.argmin(lorentzian(x_values, *popt))]
    else:
        fitted_if = x_values[np.argmax(lorentzian(x_values, *popt))]

    fit_values = lorentzian(x_values, *popt)

    return fitted_if, fit_values, r_squared