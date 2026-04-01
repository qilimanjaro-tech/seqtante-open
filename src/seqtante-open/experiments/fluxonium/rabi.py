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
import qililab as ql
from qililab import Parameter, Platform
from seqtante.experiments.fluxoniums.experiment_classes import rabi__amplitude_gaussian_hw_loop
from qilitools.experiments.utils import get_gate_params

from seqtante.experiments.utils import from_parameters_to_calibration
from seqtante.outputs import output_controller

from .fit import RabiFit

_DEFAULTS = {
    "averages": 2000,
    "relax_duration": 200_000,
}


def rabi_ex(platform: Platform, platform_path: str, parameters: dict[str, Any]):
    qubits = parameters["targets"]
    db_manager = output_controller.db_manager
    ex_p = get_gate_params(platform, qubits, **{"drive_q{}_bus": Parameter.IF})
    for qubit in qubits:
        ex_p[qubit].update(parameters[qubit])
        if "drag_amplitude_sweep" in ex_p[qubit]:
            ex_p[qubit]["drag_amplitude_sweep"] = np.linspace(*parameters[qubit].get("drag_amplitude_sweep"))

    measurement_ids = rabi__amplitude_gaussian_hw_loop(
        platform=platform,
        db_manager=db_manager,
        parameters=ex_p,
        qubit_idx=qubits,
        calibration=from_parameters_to_calibration(parameters, targets=qubits),
        autocalibration=True,
        averages=parameters.get("averages", _DEFAULTS["averages"]),
        relax_duration=parameters.get("relax_duration", _DEFAULTS["relax_duration"]),
        drag_amplitude_sweep=np.linspace(*parameters["drag_amplitude_sweep"])
        )

    for qubit, measurement_id in zip(qubits, measurement_ids):
        rabi_model = RabiFit(
            qubit_idx=qubit,
            measurement_id=measurement_id,
            path=parameters[qubit]["data_folder"]
        )
        rabi_model.fit()
        rabi_model.plot()
        platform.set_parameter(alias=f"Drag({qubit})",
                            parameter=Parameter.AMPLITUDE,
                            value=float(rabi_model.fitted_amplitude))

    ql.save_platform(platform_path, platform)
    db_manager.update_platform(platform)