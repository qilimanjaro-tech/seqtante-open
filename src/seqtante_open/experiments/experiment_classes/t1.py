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

from typing import cast
from warnings import warn

import numpy as np
from qililab import Calibration, Parameter
from qililab.platform import Platform
from qililab.result import DatabaseManager, StreamArray

from seqtante_open.experiments.qprogram import t1_saturation as t1_qprogram


def t1_saturation(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    drive_bus: str,
    wait_sweep: np.ndarray,
    drive_if: float,
    drive_step_duration: int,
    drive_amplitude: float,
    drive_gain: float,
    readout_if: int,
    readout_duration: int,
    readout_amplitude: float,
    overlap: int,
    relax_duration: int,
    averages: int,
    drive_rise_time: int = 2_000,
    n_sigmas: int = 4,
    q_relative_amplitude: float = 0,
    drive_lo: int | None = None,
    iq_modulation: bool | None = None,
    readout_lo: int | None = None,
    readout_attenuation: int | None = None,
    optional_identifier: str | None = None,
    target: str | None = None,
    autocalibration: bool = False,
    calibration: Calibration | None = None,
) -> int | None:
    """T1 measured via a saturation pulse.

    Args:
        platform: Qililab platform to execute on.
        db_manager: Database manager for result storage.
        readout_bus: Physical alias of the readout bus.
        drive_bus: Physical alias of the drive bus.
        wait_sweep: Idle (wait) between exiting the qubit and the readout.
        drive_if: Intermediate frequency (Hz) for the drive tone.
        drive_step_duration: Duration (ns) of the drive step.
        drive_amplitude: Amplitude of the drive pulse.
        drive_gain: Power/gain to set on the drive bus before executing.
        readout_if: Intermediate frequency (Hz) for the readout tone.
        readout_duration: Duration of the readout pulse.
        readout_amplitude: Amplitude of the readout pulse.
        overlap: Overlap (ns) between the drive and readout windows.
        relax_duration: Cooldown time (ns) between repetitions.
        averages: Number of hardware averages.
        drive_rise_time: Rise time (ns) of the drive pulse envelope. Defaults to 2_000.
        n_sigmas: Number of Gaussian sigmas in the drive pulse rise/fall. Defaults to 4.
        q_relative_amplitude: Relative amplitude of the drive pulse's Q component. Defaults to 0.
        drive_lo: If provided, set the drive LO frequency before executing. Defaults to None.
        iq_modulation: If provided, set IQ modulation on the drive bus before executing.
            Defaults to None.
        readout_lo: If provided, set the readout LO frequency before executing. Defaults to None.
        readout_attenuation: If provided, set the readout output attenuation before executing.
            Defaults to None.
        optional_identifier: Identifier for the measurement in the database. Defaults to None.

    Returns:
        int | None: ID of the measurement in the database.
    """
    qprogram = t1_qprogram(
        wait_sweep=wait_sweep,
        drive_if=drive_if,
        drive_amplitude=drive_amplitude,
        drive_step_duration=drive_step_duration,
        readout_if=readout_if,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
        relax_duration=relax_duration,
        averages=averages,
        overlap=overlap,
        drive_rise_time=drive_rise_time,
        n_sigmas=n_sigmas,
        q_relative_amplitude=q_relative_amplitude,
    )

    if readout_attenuation is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.OUT0_ATT, value=readout_attenuation)
    if iq_modulation is not None:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.IQ_MODULATION, value=iq_modulation)
    if drive_lo is not None:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_lo)
    if readout_lo is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_lo)

    instrument_platform = next(
        (
            instrument_name
            for instrument_platform in platform.get_element(drive_bus).instruments
            if (instrument_name := instrument_platform.name.name) in ["ROHDE_SCHWARZ", "QCMRF"]
        ),
        None,
    )
    if instrument_platform is not None:
        if instrument_platform == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif instrument_platform == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
    else:
        warn("No instrument to set power to.")

    stream_array = StreamArray(
        shape=(len(wait_sweep), 2),
        loops={"time": {"array": wait_sweep.astype(int), "units": "ns", "bus": drive_bus, "parameter": "time"}},
        platform=platform,
        experiment_name="T1_saturation",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
        calibration=calibration,
        autocalibration=autocalibration,
        qubit_idx=target,
    )
    try:
        with stream_array:
            results = platform.execute_qprogram(
                qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
            ).results
            for ii, _ in enumerate(wait_sweep):
                stream_array[ii,] = results[readout_bus][ii].array.T

    finally:
        for instrument in platform.get_element(drive_bus).instruments:
            if instrument.name.name == "ROHDE_SCHWARZ":
                platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return cast("int", stream_array.measurement.measurement_id) if stream_array.measurement is not None else None
