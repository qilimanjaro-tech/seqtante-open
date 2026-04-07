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
import tqdm
from qililab import Platform
from qililab.result import StreamArray

from seqtante_open.experiments.fit import ResonatorSpectroscopyFit
from seqtante_open.experiments.qprogram import resonator_spectroscopy_cw
from seqtante_open.experiments.utils import from_parameters_to_calibration, set_all_flux_channels_to_zero
from seqtante_open.outputs import output_controller

SAMPLE_NUMBER = 3.7


def single_tone_frequency_vs_flux_cw_dc(platform_path: str, platform: Platform, parameters: dict):
    db_manager = output_controller.db_manager
    qubit_idx = parameters["targets"]
    parameters = parameters.copy()
    parameters.update(parameters[qubit_idx])
    readout_bus = parameters["readout_bus"]
    if_sweep = np.linspace(*parameters["frequency_sweep_values"])
    flux_sweep = np.linspace(*parameters["flux_sweep_values"])
    flux_parameter_id = parameters["flux_parameter_id"]
    xtalk_bot = parameters.get("xtalk_bot")
    voltage_source = parameters.get("voltage_source")

    flux_parameter = xtalk_bot.phi_z if flux_parameter_id == "flux_z" else xtalk_bot.phi_x

    averages = parameters["hw_avg"]
    duration = parameters["repetition_duration"]

    set_all_flux_channels_to_zero(voltage_source)

    qprogram = resonator_spectroscopy_cw(
        if_sweep[0],
        if_sweep[-1],
        if_sweep[1] - if_sweep[0],
        averages=averages,
        integration_time=duration,
        r_amp=0.1,
    )

    stream_array = StreamArray(
        shape=(len(flux_sweep), len(if_sweep), 2),
        loops={
            "flux": {
                "array": flux_sweep,
                "units": flux_parameter.unit,
                "bus": flux_parameter.label,
                "parameter": "Flux",
            },
            "frequency": {"array": if_sweep, "units": "Hz", "bus": readout_bus, "parameter": "IF_frequency"},
        },
        platform=platform,
        experiment_name="single_tone__frequency_vs_flux_cw_dc",
        autocalibration=True,
        calibration=from_parameters_to_calibration(parameters),
        db_manager=db_manager,
        qprogram=qprogram,
    )

    with stream_array:
        for ii, v in enumerate(tqdm(flux_sweep)):
            flux_parameter(v)
            results = platform.execute_qprogram(qprogram, bus_mapping={"readout": readout_bus}).results

            stream_array[ii,] = results[readout_bus][0].array.T

    exp_id = stream_array.measurement.measurement_id

    set_all_flux_channels_to_zero(voltage_source)

    meas = db_manager.load_by_id(exp_id)
    data_folder = parameters["data_folder"]
    resonator_model = ResonatorSpectroscopyFit(qubit_idx=qubit_idx, measurement=meas, path=data_folder)
    resonator_model.fit()
    resonator_model.plot()

    return {
        "readout_if_list": resonator_model.readout_if_list,
        "xtalk_bot": xtalk_bot,
        "voltage_source": voltage_source,
    }
