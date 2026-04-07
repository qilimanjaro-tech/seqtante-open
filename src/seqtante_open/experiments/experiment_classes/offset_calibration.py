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
from typing import overload

import numpy as np
import qcodes as qc
from qililab import Parameter
from qililab.platform import Platform
from qililab.result import DatabaseManager, StreamArray
from seqtante_open.experiments.fluxonium.analysis import sss_from_array
from seqtante_open.experiments.fluxonium.qprogram import resonator_spectroscopy_cw
from tqdm.auto import tqdm


def single_tone_frequency_vs_flux_cw_dc(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    if_sweep: np.ndarray,
    amplitude: float,
    averages: int,
    duration: int,
    flux_parameter: qc.Parameter,
    flux_sweep: np.ndarray,
    lo: int | None = None,
    optional_identifier: str | None = None,
) -> int | None:

    if lo is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=lo)

    qprogram = resonator_spectroscopy_cw(
        *sss_from_array(if_sweep),
        averages=averages,
        integration_time=duration,
        r_amp=amplitude,
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
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
    )

    with stream_array:
        for ii, v in enumerate(tqdm(flux_sweep)):
            flux_parameter(v)
            results = platform.execute_qprogram(qprogram, bus_mapping={"readout": readout_bus}).results

            stream_array[ii,] = results[readout_bus][0].array.T

    return stream_array.measurement.measurement_id if stream_array.measurement is not None else None  # type:ignore [return-value]
