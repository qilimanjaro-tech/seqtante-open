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

from copy import deepcopy
from typing import Any, cast

import numpy as np
from qililab import save_platform
from qililab.platform.platform import Platform
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter
from qililab.utils.serialization import deserialize_from, serialize_to

from seqtante_open.experiments.experiment_classes import t1_saturation as t1_experiment
from seqtante_open.experiments.fitting import T1Fit
from seqtante_open.experiments.utils import get_lo_multiple_sources
from seqtante_open.outputs import output_controller

_DEFAULTS = {
    "drive_gain": 1,
    "overlap_time": 0,
    "q_relative_amplitude": 0,
    "drive_rise_time": 2_000,
    "n_sigmas": 4,
}


def t1_node(platform: Platform, platform_path: str, parameters: dict[str, Any]):
    targets = parameters["targets"]
    qubits = [target for target in targets if target.startswith("q")]

    db_manager = output_controller.db_manager

    calibration: Calibration = deserialize_from(parameters["calibration_path"], Calibration)
    if not isinstance(crosstalk := calibration.crosstalk_matrix, CrosstalkMatrix):
        raise ValueError(
            "To execute single_tone_vs_flux_fluxonium experiment, the Calibration needs to have a CrosstalkMatrix"
        )
    platform.set_crosstalk(crosstalk=crosstalk)

    platform.set_flux_to_zero()

    try:
        for qubit in qubits:
            readout_bus = f"readout_{qubit}"
            drive_bus = f"drive_{qubit}"
            target_params = {**parameters, **parameters[qubit]}
            drive_lo = get_lo_multiple_sources(
                bus=drive_bus,
                target=qubit,
                platform=platform,
                calibration=calibration,
            )
            wait_sweep = np.linspace(*target_params["wait_sweep"])
            readout_if = platform.get_parameter(alias=readout_bus, parameter=Parameter.IF)
            drive_if = platform.get_parameter(alias=drive_bus, parameter=Parameter.IF)
            calibration_copy = deepcopy(calibration)
            calibration_copy.parameters["data_folder"] = target_params["data_folder"]

            measurement_id = t1_experiment(
                platform=platform,
                db_manager=db_manager,
                readout_bus=readout_bus,
                drive_bus=drive_bus,
                wait_sweep=wait_sweep,
                readout_if=readout_if,
                drive_if=drive_if,
                drive_lo=drive_lo,
                averages=target_params["averages"],
                relax_duration=target_params["relax_duration"],
                drive_step_duration=target_params["drive_duration"],
                drive_amplitude=target_params["drive_amplitude"],
                drive_gain=target_params.get("drive_gain", _DEFAULTS["drive_gain"]),
                drive_rise_time=target_params.get("drive_rise_time", _DEFAULTS["drive_rise_time"]),
                q_relative_amplitude=target_params.get("q_relative_amplitude", _DEFAULTS["q_relative_amplitude"]),
                n_sigmas=target_params.get("n_sigmas", _DEFAULTS["n_sigmas"]),
                readout_amplitude=target_params["readout_amplitude"],
                readout_duration=target_params["readout_duration"],
                overlap=target_params.get("overlap_time", _DEFAULTS["overlap_time"]),
                calibration=calibration_copy,
                target=qubit,
                autocalibration=True,
            )

            model = T1Fit(measurement_id=cast("int", measurement_id), target=qubit, path=target_params["data_folder"])
            model.fit()
            model.plot()

    finally:
        platform.set_bias_to_zero()
        save_platform(path=platform_path, platform=platform)
        serialize_to(calibration, parameters["calibration_path"])
