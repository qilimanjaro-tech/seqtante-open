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

from seqtante_open.experiments.fit import RabiFit
from seqtante_open.experiments.qprogram import rabi_amp_square_drive
from seqtante_open.experiments.utils import from_parameters_to_calibration, set_all_flux_channels_to_zero
from seqtante_open.outputs import output_controller
from seqtante_open.experiments.utils import sine

SAMPLE_NUMBER = 3.7


def rabi_amplitude_square(platform_path: str, platform: Platform, parameters: dict):
    db_manager = output_controller.db_manager
    qubit_idx = parameters["targets"]
    readout_bus = parameters["readout_bus"]
    drive_bus = parameters["drive_bus"]
    amplitude_sweep = np.linspace(*parameters["amplitude_sweep_values"])
    flux_parameter_id = parameters["flux_parameter_id"]
    xtalk_bot = parameters["xtalk_bot"]
    flux_parameter = xtalk_bot.phi_z if flux_parameter_id == "flux_z" else xtalk_bot.phi_x
    averages = parameters.get("hw_avg", 5_000)
    readout_duration = parameters.get("readout_duration", 4_000)
    readout_amp = parameters.get("readout_amp", 0.1)
    drive_duration = parameters.get("drive_duration", 4_000)
    relax_duration = parameters.get("relax_duration",0)
    overlap_time = parameters.get("overlap_time", 4_000)
    ringup_time = parameters.get("ringup_time", 0)
    voltage_source = parameters.get("voltage_source")

    set_all_flux_channels_to_zero(voltage_source)
    
    qprogram = rabi_amp_square_drive(
        amp_start=amplitude_sweep[0],
        amp_stop=amplitude_sweep[-1],
        amp_step=amplitude_sweep[1] - amplitude_sweep[0],
        averages=averages,
        readout_duration=readout_duration,
        readout_amplitude=readout_amp,
        d_duration=drive_duration,
        relax_duration=relax_duration,
        overlap_time=overlap_time,
        ringup_time=ringup_time,
    )

    stream_array = StreamArray(
        shape=(len(amplitude_sweep), 2),
        loops={
            "amplitude": {"array": amplitude_sweep, "units": "dB", "bus": drive_bus, "parameter": "amplitude"},
        },
        platform=platform,
        experiment_name="rabi_amp_square_drive",
        autocalibration=True,
        calibration=from_parameters_to_calibration(parameters),
        db_manager=db_manager,
        qprogram=qprogram,
    )

    with stream_array:
        for ii, v in enumerate(tqdm(amplitude_sweep)):
            flux_parameter(v)
            results = platform.execute_qprogram(qprogram, bus_mapping={"readout": readout_bus}).results

            stream_array[ii,] = results[readout_bus][0].array.T

    exp_id = stream_array.measurement.measurement_id

    set_all_flux_channels_to_zero(voltage_source)

    meas = db_manager.load_by_id(exp_id)
    results, loop = meas.load_old_h5()
    data_folder = parameters["data_folder"]
    qubit_model = RabiFit(qubit_idx=qubit_idx, measurement=meas, loop=loop, path=data_folder)
    qubit_model.fit()
    qubit_model.plot()

    return {
        "qubit_if": qubit_model.qubit_if,
        "xtalk_bot": xtalk_bot,
        "voltage_source": voltage_source,
    }
