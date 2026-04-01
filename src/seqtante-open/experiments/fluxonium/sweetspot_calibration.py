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

from typing import Any

import numpy as np
from qililab import Parameter, save_platform
from qililab.platform.platform import Platform
from qilitools.experiments.two_tone import two_tone__frequency_d_drag
from qilitools.experiments.utils import get_gate_params

from seqtante.experiments.utils import from_parameters_to_calibration
from seqtante.outputs import output_controller

from .fit import TwoToneFit

_DEFAULTS = {
    "averages": 2000,
    "relax_duration": 200_000,
}


def two_tone_ex(platform: Platform, platform_path: str, parameters: dict[str, Any]):
    qubits = parameters["targets"]
    db_manager = output_controller.db_manager
    ex_p = get_gate_params(platform, qubits, **{"drive_q{}_bus": [Parameter.IF, Parameter.LO_FREQUENCY]})
    for qubit in qubits:
        ex_p[qubit].update(parameters[qubit])
        ex_p[qubit]["freq_sweep"] = np.arange(*parameters[qubit].get("freq_sweep", parameters["freq_sweep"]))\
                                        + ex_p[qubit]["intermediate_frequency"]

    measurement_ids = two_tone__frequency_d_drag(
        platform=platform,
        db_manager=db_manager,
        parameters=ex_p,
        qubit_idx=qubits,
        calibration=from_parameters_to_calibration(parameters, targets=qubits),
        autocalibration=True,
        averages=parameters.get("averages", _DEFAULTS["averages"]),
        relax_duration=parameters.get("relax_duration", _DEFAULTS["relax_duration"]),
    )

    for qubit, measurement_id in zip(qubits, measurement_ids):

        two_tone_model = TwoToneFit(
            qubit_idx=qubit,
            measurement_id=measurement_id,
            lo=ex_p[qubit]["frequency"],
            path=parameters[qubit]["data_folder"]
        )
        two_tone_model.fit()
        two_tone_model.plot()
        i_r_squared = two_tone_model.results["r_squared"][0]
        q_r_squared = two_tone_model.results["r_squared"][1]
        if i_r_squared > q_r_squared:
            fitted_if = two_tone_model.results["fitted_ifs"][0]
        else:
            fitted_if = two_tone_model.results["fitted_ifs"][1]
        platform.set_parameter(alias=ex_p[qubit]["bus_mapping"]["drive"], parameter=Parameter.IF, value=fitted_if)
    save_platform(platform_path, platform)
    db_manager.update_platform(platform)