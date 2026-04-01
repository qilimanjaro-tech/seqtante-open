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
from warnings import warn

import numpy as np
import qcodes as qc
from qililab import Parameter
from qililab.platform import Platform
from qililab.result import DatabaseManager, StreamArray
from qililab.waveforms import Arbitrary
from tqdm.auto import tqdm

from .analysis import XTalk, sss_from_array
from analysis.analysis import compensate_round
from experiments import run_single_point_reset_debug
from qprogram import (
    raw_trace_two_tone,
    resonator_spectroscopy_single_point_reset,
    two_tone_spectroscopy,
    two_tone_spectroscopy_reset,
)


def two_tone_frequency_vs_flux_pulsed_dc_update_freq(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_if_freq: int | np.ndarray,
    averages: int,
    r_duration: int,
    r_amp: float,
    d_duration: int,
    d_amp: float,
    overlap_time: int,
    ringup_time: int,
    relax_duration: int,
    readout_bus: str,
    drive_bus: str,
    drive_gain: float,
    drive_IF_sweep: np.ndarray,
    flux_parameter: qc.Parameter,
    flux_sweep: np.ndarray,
    drive_LO: int | None = None,
    readout_LO: int | None = None,
    optional_identifier: str | None = None,
):

    qprogram = two_tone_spectroscopy(
        *sss_from_array(drive_IF_sweep),
        averages=averages,
        r_duration=r_duration,
        r_amp=r_amp,
        d_duration=d_duration,
        d_amp=d_amp,
        relax_duration=relax_duration,
        overlap_time=overlap_time,
        ringup_time=ringup_time,
    )

    if drive_LO is not None:
        platform.set_parameter(
            alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_LO
        )  # NOTE should this be in the experiment?
    if readout_LO is not None:
        platform.set_parameter(
            alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_LO
        )  # NOTE should this be in the experiment?

    update_readout = False
    if isinstance(readout_if_freq, np.ndarray) or isinstance(readout_if_freq, list):
        update_readout = True
        if len(readout_if_freq) != len(flux_sweep):
            raise ValueError("readout_if_freq and flux_sweep has to be the same length")
        #NOTE find a way to save these frequencies without breaking auto-plotting
        #NOTE should we tag measurements differently to indicate that readout is fixed or updated?
    else:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.IF, value=readout_if_freq)
        

    for instrument_platform in platform.get_element(drive_bus).instruments:
        if instrument_platform.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif instrument_platform.name.name == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
        else:
            warn(f"Not setting voltage to this parameter: {instrument_platform.name.name}")
    
    stream_array = StreamArray(
        shape=(len(flux_sweep), len(drive_IF_sweep), 2),
        loops={
            "flux": {
                "array": flux_sweep,
                "units": flux_parameter.unit,
                "bus": flux_parameter.label,
                "parameter": "Flux",
            },
            "IF_frequency": {"array": drive_IF_sweep, "units": "Hz", "bus": drive_bus, "parameter": "IF_frequency"},
        },
        platform=platform,
        experiment_name="two_tone__frequency_vs_flux_pulsed_dc",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
    )

    with stream_array:
        for ii, flux in enumerate(tqdm(flux_sweep)):
            flux_parameter(flux)
            if update_readout:
                platform.set_parameter(alias=readout_bus, parameter=Parameter.IF, value=readout_if_freq[ii])
            

            results = platform.execute_qprogram(
                qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
            ).results
            stream_array[ii,] = results[readout_bus][0].array.T

    for instrument in platform.get_element(drive_bus).instruments:
        if instrument.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)
    id = stream_array.measurement.measurement_id

    return id
from qilitools.plotting import get_xarray_from_meas
import qilitools.plotting
import xarray as xr
import qilitools


from warnings import warn

import numpy as np
import qcodes as qc
from qililab import Parameter
from qililab.platform import Platform
from qililab.result import DatabaseManager, StreamArray
from qililab.waveforms import Arbitrary
from tqdm.auto import tqdm

from qilitools.analysis import XTalk, sss_from_array
from qilitools.analysis.analysis import compensate_round
from qilitools.experiments import run_single_point_reset_debug
from qilitools.qprogram import (
    raw_trace_two_tone,
    resonator_spectroscopy_single_point_reset,
    two_tone_spectroscopy,
    two_tone_spectroscopy_reset,
)

import lmfit
from .analysis.lorentzian_fit import lorentzian
def lorentzian_fit_custom(y_values, x_values, peak_pos_guess = False):
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

from scipy.signal import find_peaks
from tqdm.auto import tqdm
from copy import deepcopy
from scipy.signal import find_peaks, savgol_filter
from qilitools.plotting import convert_plot_units, rotated_IQ_divide_by_median_col
from qilitools.plotting import get_xarray_from_meas


def find_peaks_from_arrays(
    data: np.ndarray,
    x_vals: np.ndarray,
    filter: bool = False,
    filter_window_length: int = 21,
    filter_polyorder: int = 3,
    fit_lorentzian: bool = False,
):
    """Find peaks in a 2D array (num_traces, trace_length)"""
    peak_indices = []
    fitted_ifs = []

    for y in tqdm(data):
        if filter and len(y) >= filter_window_length:
            y_smooth = savgol_filter(y, window_length=filter_window_length, polyorder=filter_polyorder)
        else:
            y_smooth = y

        peaks, props = find_peaks(-y_smooth, prominence=0.01)
        if len(peaks) > 0:
            best_peak_index = peaks[np.argmax(props["prominences"])]
        else:
            best_peak_index = np.argmax(y_smooth)

        peak_indices.append(best_peak_index)

        if fit_lorentzian:
            fitted_if, _, _ = lorentzian_fit_custom(
                y_values=y_smooth,
                x_values=x_vals,
                peak_pos_guess=best_peak_index,
            )
            fitted_ifs.append(fitted_if)

    if fit_lorentzian:
        return np.array(fitted_ifs)
    else:
        return x_vals[np.array(peak_indices)]


def find_peaks_in_meas(
    measurement,
    peak_axis: int | str,
    filter: bool = False,
    filter_window_length: int = 21,
    filter_polyorder: int = 3,
    fit_lorentzian: bool = False,
    dataprocessing=rotated_IQ_divide_by_median_col,
):
    """Wrapper to handle xarray from a Qililab measurement."""
    xarr = get_xarray_from_meas(measurement)
    xarr = convert_plot_units(xarr)
    xarr = xr.apply_ufunc(dataprocessing, xarr.T)

    # Determine axis name
    if isinstance(peak_axis, str):
        axis_name = peak_axis
    elif isinstance(peak_axis, int):
        axis_name = xarr.dims[peak_axis]
    else:
        raise ValueError("peak_axis must be int or str")

    # Prepare data
    data = xarr.transpose(..., axis_name).to_numpy()
    x_vals = xarr[axis_name].to_numpy()

    return find_peaks_from_arrays(
        data=data,
        x_vals=x_vals,
        filter=filter,
        filter_window_length=filter_window_length,
        filter_polyorder=filter_polyorder,
        fit_lorentzian=fit_lorentzian,
    )

dm = ql.get_db_manager()

readout_bus = "readoutB"
if_sweep = np.linspace(154, 156, 101)*1e6

xtalk_bot.set_all_fluxes_to_zero()
xtalk_bot.phi_z(0.51402)
xtalk_bot.phi_x(0.415)

flux_sweep = array_from_center_span_npoints(0.514, 0.1, 51)

id = single_tone_frequency_vs_flux_cw_dc(platform=platform,
                                  db_manager=dm,
                                  readout_bus=readout_bus,
                                  IF_sweep=if_sweep,
                                  amplitude=0.2,
                                  averages=200,
                                  duration=6000,
                                  flux_parameter=xtalk_bot.phi_z,
                                  flux_sweep=flux_sweep,
                                  )
set_all_to_zero()

meas = dm.load_by_id(id)

readout_if_list = find_peaks_in_meas(meas, peak_axis = "IF_frequency readoutB (MHz)",  filter=True, filter_window_length=5, filter_polyorder=3, fit_lorentzian=False, dataprocessing=np.abs)
### plot fit  #############
fig = auto_plot(meas, x = "Flux phi_z (phi_0)")

_, loops = meas.load_old_h5()
x_flux = loops["flux"]["array"]

# Add fitted points as a scatter
import plotly.graph_objects as go
fig.add_trace(go.Scatter(
    x=x_flux,
    y=readout_if_list,
    mode="markers+lines",
    name="Peak IF freq",
    marker=dict(color="red", size=6),
    line=dict(dash="dot", color="red"),
))
fig.show()
#################



readout_bus = "readoutB"
drive_bus = "drive_QCM_RS"

xtalk_bot.set_all_fluxes_to_zero()
xtalk_bot.phi_z(0.514)
xtalk_bot.phi_x(0.415)

id = two_tone_frequency_vs_flux_pulsed_dc_update_freq(platform = platform, 
                          db_manager = dm, 
                          readout_bus = readout_bus, 
                          drive_bus = drive_bus, 
                          readout_if_freq = readout_if_list*1e6, 
                          averages = 1000,
                          r_duration = 4000, 
                          r_amp = 0.2,
                          d_duration = 4000, 
                          d_amp = 0.1, 
                          relax_duration = 0, 
                          overlap_time = 4000, 
                          ringup_time =0,
                          drive_IF_sweep = np.linspace(-300, 300, 301)*1e6,
                          drive_gain= -25,
                          flux_parameter=xtalk_bot.phi_z,
                          flux_sweep=flux_sweep,
                          drive_LO = 1.8e9,
                          optional_identifier=f"readout_if_list: {readout_if_list}")

set_all_to_zero()
meas = dm.load_by_id(id)
fig = auto_plot(meas, x = "Flux phi_z (phi_0)", dataprocessing=qilitools.plotting.center_phase_around_median)
fig.show()