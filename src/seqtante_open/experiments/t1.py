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

from seqtante_open.experiments.fit import T1Fit
from seqtante_open.experiments.qprogram import t1_single_square
from seqtante_open.experiments.utils import from_parameters_to_calibration, set_all_flux_channels_to_zero
from seqtante_open.outputs import output_controller
from seqtante_open.experiments.utils import sine

SAMPLE_NUMBER = 3.7


def t1(platform_path: str, platform: Platform, parameters: dict):
    db_manager = output_controller.db_manager
    qubit_idx = parameters["targets"]
    readout_bus = parameters["readout_bus"]
    drive_bus = parameters["drive_bus"]
    t1_time_sweep = np.linspace(*parameters["t1_time_sweep_values"])
    flux_parameter_id = parameters["flux_parameter_id"]
    q1_phi_x = parameters.get("q1_phi_x", 0)
    q1_phi_z = parameters.get("q1_phi_z", 0)
    xtalk_bot = parameters["xtalk_bot"]
    flux_parameter = xtalk_bot.phi_z if flux_parameter_id == "flux_z" else xtalk_bot.phi_x
    averages = parameters.get("hw_avg", 5_000)
    readout_duration = parameters.get("readout_duration", 4_000)
    readout_amp = parameters.get("readout_amp", 0.1)
    drive_duration = parameters.get("drive_duration", 4_000)
    drive_amplitude = parameters.get("drive_amplitude", 0.1)
    relax_duration = parameters.get("relax_duration",0)
    overlap_time = parameters.get("overlap_time", 4_000)
    ringup_time = parameters.get("ringup_time", 0)
    voltage_source = parameters.get("voltage_source")

    set_all_flux_channels_to_zero(voltage_source)
    flux_parameter.q1_phi_x(q1_phi_x)
    flux_parameter.q1_phi_z(q1_phi_z)

    qprogram = t1_single_square(
        averages=averages,
        r_duration=readout_duration,
        pi_pulse_duration=drive_duration,
        r_amp=readout_amp,
        wait_time=t1_time_sweep[0],
        pi_pulse_amp=drive_amplitude,
        ringup_time=ringup_time,
        relax_duration=relax_duration,
    )

    stream_array = StreamArray(
        shape=(len(t1_time_sweep), 2),
        loops={"t1_time": {"array": t1_time_sweep, "units": "ns", "bus": drive_bus, "parameter": "t1_time"}},
        platform=platform,
        experiment_name="T1",
        autocalibration=True,
        calibration=from_parameters_to_calibration(parameters),
        db_manager=db_manager,
        qprogram=qprogram,
    )

    with stream_array:
        for ii, t in enumerate(tqdm(t1_time_sweep)):
            qprogram = t1_single_square(
                averages=averages,
                r_duration=readout_duration,
                pi_pulse_duration=drive_duration,
                r_amp=readout_amp,
                wait_time=t1_time_sweep[0],
                pi_pulse_amp=drive_amplitude,
                ringup_time=ringup_time,
                relax_duration=relax_duration,
            )

            results = platform.execute_qprogram(qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus})
            stream_array[ii,] = results.results[readout_bus][0].array.T

    exp_id = stream_array.measurement.measurement_id

    set_all_flux_channels_to_zero(voltage_source)

    meas = db_manager.load_by_id(exp_id)
    results, loop = meas.load_old_h5()
    data_folder = parameters["data_folder"]
    coherence_model = T1Fit(qubit_idx=qubit_idx, measurement=meas, loop=loop, path=data_folder)
    coherence_model.fit()
    coherence_model.plot()

    return {
        "qubit_coherence": coherence_model.t1,
        "xtalk_bot": xtalk_bot,
        "voltage_source": voltage_source,
    }
