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
from typing import Any

import numpy as np
from qililab import save_platform
from qililab.platform.platform import Platform
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter
from qililab.utils.serialization import deserialize_from, serialize_to

from seqtante_open.experiments.experiment_classes import two_tone_frequency as two_tone_experiment
from seqtante_open.experiments.fitting import FluxoniumTwoToneModel
from seqtante_open.experiments.utils import get_operating_point, save_parameters_to_operating_point
from seqtante_open.outputs import output_controller

_DEFAULTS = {
    "drive_gain": 1,
    "overlap_time": 0,
    "ringup_time": 0,
}


def two_tone_node(platform: Platform, platform_path: str, parameters: dict[str, Any]):
    targets = parameters["targets"]
    qubits = [target for target in targets if target.startswith("q")]

    db_manager = output_controller.db_manager

    calibration: Calibration = deserialize_from(parameters["calibration_path"], Calibration)
    if not isinstance(crosstalk := calibration.crosstalk_matrix, CrosstalkMatrix):
        raise ValueError(
            "To execute single_tone_vs_flux_fluxonium experiment, the Calibration needs to have a CrosstalkMatrix"
        )
    platform.set_crosstalk(crosstalk=crosstalk)

    try:
        for qubit in qubits:
            platform.set_flux_to_zero()
            readout_bus = f"readout_{qubit}"
            drive_bus = f"drive_{qubit}"
            target_params = {**parameters, **parameters[qubit]}
            if (operating_point_name := target_params.get("operating_point")) is not None:
                get_operating_point(
                    calibration,
                    target=qubit,
                    operating_point_name=operating_point_name,
                    platform=platform,
                )
            freq_sweep = np.linspace(*target_params["freq_sweep"]) + platform.get_parameter(
                alias=drive_bus, parameter=Parameter.IF
            )
            drive_LO = platform.get_parameter(drive_bus, parameter=Parameter.LO_FREQUENCY)
            calibration_copy = deepcopy(calibration)
            calibration_copy.parameters["data_folder"] = target_params["data_folder"]

            measurement_id = two_tone_experiment(
                platform=platform,
                db_manager=db_manager,
                readout_bus=readout_bus,
                drive_bus=drive_bus,
                drive_IF_sweep=freq_sweep,
                averages=target_params["averages"],
                relax_duration=target_params["relax_duration"],
                d_duration=target_params["drive_duration"],
                d_amp=target_params["drive_amplitude"],
                r_amp=target_params["readout_amplitude"],
                r_duration=target_params["readout_duration"],
                drive_gain=target_params.get("drive_gain", _DEFAULTS["drive_gain"]),
                ringup_time=target_params.get("ringup_time", _DEFAULTS["ringup_time"]),
                overlap_time=target_params.get("overlap_time", _DEFAULTS["overlap_time"]),
                calibration=calibration_copy,
                target=qubit,
                autocalibration=True,
            )

            model = FluxoniumTwoToneModel(measurement_id, target=qubit, path=target_params["data_folder"], lo=drive_LO)
            model.fit()
            model.plot()

            if operating_point_name is not None:
                save_parameters_to_operating_point(
                    (drive_bus, Parameter.IF, model.results["signal"]["fitted_if"]),
                    calibration=calibration,
                    target=qubit,
                    operating_point_name=operating_point_name,
                )
            else:
                platform.set_parameter(drive_bus, Parameter.IF, model.results["signal"]["fitted_if"])

    finally:
        platform.set_bias_to_zero()
        serialize_to(calibration, parameters["calibration_path"])
        save_platform(path=platform_path, platform=platform)
