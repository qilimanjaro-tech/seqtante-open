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

from itertools import product
from typing import Any, cast
from copy import deepcopy

import numpy as np
from qililab.platform.platform import Platform
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter
from qililab.utils.serialization import deserialize_from, serialize_to

from seqtante_open.experiments.experiment_classes import (
    single_tone__frequency_vs_flux as single_tone_vs_flux_experiment,
)
from seqtante_open.experiments.fitting import FluxoniumSingleToneFluxModel
from seqtante_open.experiments.utils import coupler_readout_qubit, x_loop_readout_flux
from seqtante_open.outputs import output_controller


def single_tone_vs_flux(platform: Platform, platform_path: str, parameters: dict[str, Any]):
    targets = parameters["targets"]
    qubits = [target for target in targets if target.startswith("q")]
    couplers = [target for target in targets if target.startswith("c")]

    readout_x_couplers = coupler_readout_qubit(couplers=couplers, coupler_readout_overwrite=parameters.get("coupler_readout_qubit", {}))
    qubit_loops = acs.qubit_loops if (acs := platform.analog_compilation_settings) else 1
    coupler_loops = acs.coupler_loops if acs else 1
    db_manager = output_controller.db_manager

    calibration: Calibration = deserialize_from(parameters["calibration_path"], Calibration)
    if not isinstance(crosstalk := calibration.crosstalk_matrix, CrosstalkMatrix):
        raise ValueError(
            "To execute single_tone_vs_flux experiment, the Calibration needs to have a CrosstalkMatrix"
        )
    platform.set_crosstalk(crosstalk=crosstalk)

    def _run_experiment(target: str, flux_bus: str, readout_bus: str, readout_flux: tuple[str, float] | None = None):
        platform.set_flux_to_zero()
        if readout_flux:
            platform.set_parameter(readout_flux[0], Parameter.FLUX, readout_flux[1])
        target_params = {**parameters, **parameters[target]}
        LO = platform.get_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY)
        if_sweep = np.linspace(*target_params["if_sweep"]) + platform.get_parameter(
            alias=readout_bus, parameter=Parameter.IF
        )
        flux_sweep = np.linspace(*target_params["flux_sweep"])
        calibration_copy = deepcopy(calibration)
        calibration_copy.parameters["data_folder"] = target_params["data_folder"] + flux_bus

        measurement_id = single_tone_vs_flux_experiment(
            platform=platform,
            db_manager=db_manager,
            readout_bus=readout_bus,
            if_sweep=if_sweep,
            r_amp=target_params["readout_amp"],
            averages=target_params["averages"],
            duration=target_params["duration"],
            flux_bus=flux_bus,
            flux_sweep=flux_sweep,
            minimum_wait_after_step_override=target_params.get("minimum_wait_after_step"),
            qdac_stop_ro_before_step_override=target_params.get("qdac_stop_ro_before_step"),
            lo=LO,
            calibration=calibration_copy,
            flux_parameter=Parameter.FLUX,
            qubit_idx=target,
            autocalibration=True
        )

        model = FluxoniumSingleToneFluxModel(cast("int", measurement_id), target=target, path=target_params["data_folder"] + flux_bus, lo=LO)
        model.fit()
        model.plot()
        crosstalk.flux_offsets[flux_bus] += float(model.offset)

    try:
        for qubit, loop in product(qubits, ["z", "x"][:qubit_loops][::-1]):
            readout_bus = f"readout_{qubit}"
            flux_bus = f"flux_{qubit}_{loop}"
            readout_flux = x_loop_readout_flux(qubit, qubit_loops, parameters) if loop != "x" else None
            _run_experiment(target=qubit, flux_bus=flux_bus, readout_bus=readout_bus, readout_flux=readout_flux)

        for coupler, loop in product(couplers, ["z", "x"][:coupler_loops][::-1]):
            readout_qubit = readout_x_couplers[coupler]
            readout_bus = f"readout_{readout_qubit}"
            flux_bus = f"flux_{coupler}_{loop}"
            _run_experiment(
                target=coupler,
                flux_bus=flux_bus,
                readout_bus=readout_bus,
                readout_flux=x_loop_readout_flux(f"{readout_qubit}", qubit_loops, parameters),
            )

    finally:
        platform.set_bias_to_zero()
        serialize_to(calibration, parameters["calibration_path"])
