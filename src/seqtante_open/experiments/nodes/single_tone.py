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
from qililab.utils.serialization import deserialize_from

from seqtante_open.experiments.experiment_classes import single_tone__frequency_sweep as single_tone_experiment
from seqtante_open.experiments.fitting import FluxoniumSingleToneModel
from seqtante_open.outputs import output_controller

_DEFAULTS = {
    "ringup_time": 0,
}


def single_tone_node(platform: Platform, platform_path: str, parameters: dict[str, Any]):
    targets = parameters["targets"]
    qubits = [target for target in targets if target.startswith("q")]

    db_manager = output_controller.db_manager

    calibration: Calibration = deserialize_from(parameters["calibration_path"], Calibration)
    if not isinstance(crosstalk := calibration.crosstalk_matrix, CrosstalkMatrix):
        raise ValueError("To execute single_tone_node experiment, the Calibration needs to have a CrosstalkMatrix")
    platform.set_crosstalk(crosstalk=crosstalk)

    platform.set_flux_to_zero()

    try:
        for qubit in qubits:
            readout_bus = f"readout_{qubit}"
            target_params = {**parameters, **parameters[qubit]}
            LO = platform.get_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY)
            if_sweep = np.linspace(*target_params["if_sweep"]) + platform.get_parameter(
                alias=readout_bus, parameter=Parameter.IF
            )
            calibration_copy = deepcopy(calibration)
            calibration_copy.parameters["data_folder"] = target_params["data_folder"]

            measurement_id = single_tone_experiment(
                platform=platform,
                db_manager=db_manager,
                readout_bus=readout_bus,
                if_sweep=if_sweep,
                readout_amplitude=target_params["readout_amplitude"],
                averages=target_params["averages"],
                readout_duration=target_params["readout_duration"],
                relax_duration=target_params["relax_duration"],
                ringup_time=target_params.get("ringup_time", _DEFAULTS["ringup_time"]),
                calibration=calibration_copy,
                qubit_idx=qubit,
                autocalibration=True,
            )

            model = FluxoniumSingleToneModel(
                cast("int", measurement_id), target=qubit, path=target_params["data_folder"], lo=LO
            )
            model.fit()
            model.plot()
            platform.set_parameter(
                alias=readout_bus, parameter=Parameter.IF, value=float(model.results["signal"]["fitted_if"])
            )

    finally:
        platform.set_bias_to_zero()
        save_platform(path=platform_path, platform=platform)
