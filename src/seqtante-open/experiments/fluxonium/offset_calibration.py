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
import qcodes as qc
import qililab as ql
import tqdm
from qililab import Platform
from qilitools.analysis import XTalk, array_from_center_span_npoints
from seqtante.experiments.fluxoniums.experiment_classes.offset_calibration import single_tone_frequency_vs_flux_cw_dc as single_tone_frequency_flux_experiment
from tqdm.auto import tqdm

from seqtante.experiments.fluxoniums.fit.resonator_fit import ResonatorSpectroscopyFit
from seqtante.experiments.fluxoniums.utils import set_all_flux_channels_to_zero


def single_tone_frequency_vs_flux_cw_dc(platform_path: str, platform: Platform, parameters: dict):
    SAMPLE_NUMBER = 3.7
    db_manager = ql.get_db_manager()
    db_manager.set_sample_and_cooldown(sample=f"ConscienceWF02-{SAMPLE_NUMBER}", cooldown="CD79-Mimas")
    qubit_idx = parameters["qubit_idx"]
    readout_bus = parameters["readout_bus"]
    if_sweep_params = parameters["frequency_sweep_values"]
    if_sweep = np.linspace(if_sweep_params[0], if_sweep_params[1], if_sweep_params[2])
    flux_sweep_params = parameters["flux_sweep_values"]
    flux_sweep = array_from_center_span_npoints(flux_sweep_params[0], flux_sweep_params[1], flux_sweep_params[2])
    flux_parameter_id = parameters["flux_parameter_id"]
    xtalk_bot = parameters.get("xtalk_bot")
    voltage_source = parameters.get("voltage_source")

    flux_parameter = xtalk_bot.phi_z if flux_parameter_id == "flux_z" else xtalk_bot.phi_x

    averages = parameters["hw_avg"]
    duration = parameters["repetition_duration"]

    set_all_flux_channels_to_zero(voltage_source)
    exp_id = single_tone_frequency_vs_flux_cw_dc(platform=platform,
        db_manager=db_manager,
        readout_bus=readout_bus,
        if_sweep=if_sweep,
        amplitude=0.1,
        averages=averages,
        duration=duration,
        optional_identifier="",
        flux_parameter=flux_parameter,
        flux_sweep=np.linspace(-1, 1, 1001),
    )
    set_all_flux_channels_to_zero(voltage_source)

    meas = db_manager.load_by_id(exp_id)
    data_folder = parameters["data_folder"]
    x_axis = parameters["x_axis"]
    peak_axis = parameters["peak_axis"]
    filter_on = parameters["filter_on"]
    filter_window_length = parameters["filter_window_length"]
    filter_polyorder = parameters["filter_polyorder"]
    fit_lorentzian = parameters["fit_lorentzian"]
    dataprocessing = np.abs if parameters["dataprocessing"] == "absolute" else None
    resonator_model = ResonatorSpectroscopyFit(qubit_idx=qubit_idx, measurement=meas, path=data_folder)
    resonator_model.fit(
        peak_axis=peak_axis,
        filter_on=filter_on,
        filter_window_length=filter_window_length,
        filter_polyorder=filter_polyorder,
        fit_lorentzian=fit_lorentzian,
        dataprocessing=dataprocessing,
    )
    resonator_model.plot(x_axis=x_axis)

    return {
        "readout_if_list": resonator_model.readout_if_list,
        "xtalk_bot": xtalk_bot,
        "voltage_source": voltage_source,
    }