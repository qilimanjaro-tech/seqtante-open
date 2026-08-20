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

from warnings import warn

import numpy as np
from qililab import Calibration, Parameter
from qililab.platform import Platform
from qililab.result import DatabaseManager, StreamArray

from seqtante_open.experiments.analysis import sss_from_array
from seqtante_open.experiments.qprogram import two_tone_spectroscopy


def two_tone__frequency_vs_flux(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_if_freq: int,
    averages: int,
    r_duration: int,
    r_amp: float,
    d_duration: int,
    d_amp: float,
    overlap_time: int,
    ringup_time: int,
    relax_duration: int,
    readout_bus: str,
    drive_bus: str,
    drive_gain: float,
    drive_IF_sweep: np.ndarray,
    flux_bus: str,
    flux_parameter: Parameter,
    flux_sweep: np.ndarray,
    drive_LO: int | None = None,
    readout_LO: int | None = None,
    target: str | None = None,
    calibration: Calibration | None = None,
    optional_identifier: str | None = None,
    autocalibration: bool = False
):
    qprogram = two_tone_spectroscopy(  # type: ignore [misc]
        *sss_from_array(drive_IF_sweep),
        averages=averages,
        r_duration=r_duration,
        r_amp=r_amp,
        d_duration=d_duration,
        d_amp=d_amp,
        relax_duration=relax_duration,
        overlap_time=overlap_time,
        ringup_time=ringup_time,
    )

    if drive_LO is not None:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_LO)
    if readout_LO is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_LO)
    platform.set_parameter(alias=readout_bus, parameter=Parameter.IF, value=readout_if_freq)
    instrument_platform = next(
        instrument_name for instrument_platform in platform.get_element(drive_bus).instruments
        if (instrument_name := instrument_platform.name.name) in ["ROHDE_SCHWARZ", "QCMRF"]
    )
    if instrument_platform is not None:
        if instrument_platform == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif instrument_platform == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
    else:
        warn("No instrument to set power to.")

    if flux_parameter == Parameter.FLUX:
        unit = "phi_0"
    elif flux_parameter == Parameter.VOLTAGE:
        unit = "V"
    else:
        raise ValueError("choose either FLUX or VOLTAGE")

    stream_array = StreamArray(
        shape=(len(flux_sweep), len(drive_IF_sweep), 2),
        loops={
            "flux": {
                "array": flux_sweep,
                "units": unit,
                "bus": flux_bus,
                "parameter": flux_parameter.value,
            },
            "IF_frequency": {"array": drive_IF_sweep, "units": "Hz", "bus": drive_bus, "parameter": "IF_frequency"},
        },
        platform=platform,
        experiment_name="two_tone__frequency_vs_flux_pulsed_dc",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
        calibration=calibration,
        autocalibration=autocalibration,
        qubit_idx=target,
    )

    with stream_array:
        for ii, flux in enumerate(flux_sweep):
            platform.set_parameter(alias=flux_bus, parameter=flux_parameter, value=flux)

            results = platform.execute_qprogram(
                qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
            ).results
            stream_array[ii,] = results[readout_bus][0].array.T

    for instrument in platform.get_element(drive_bus).instruments:
        if instrument.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)
    id = stream_array.measurement.measurement_id if stream_array.measurement is not None else None

    return id
