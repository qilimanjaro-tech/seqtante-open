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

"""T1 saturation execution driver around ``qili_qprograms``' shared ``t1_saturation`` QProgram."""

from typing import cast
from warnings import warn

import numpy as np
from qili_qprograms.experiments import t1_saturation
from qililab import Calibration, Parameter
from qililab.platform import Platform
from qililab.result import AutocalMeasurement, DatabaseManager, Measurement, StreamArray

_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."


def _measurement_id(stream_array: StreamArray) -> int:
    """The measurement id of a completed stream, narrowing the measurement union."""
    return cast("Measurement | AutocalMeasurement", stream_array.measurement).measurement_id


def t1_pulsed_saturation_experiment(
    platform: Platform,
    db_manager: DatabaseManager,
    calibration: Calibration,
    target: str,
    operating_point: str,
    readout_bus: str,
    drive_bus: str,
    durations_idle: np.ndarray,
    averages: int,
    relax_duration: int,
    drive_duration: int,
    drive_amplitude: float,
    drive_gain: float,
    drive_LO: int,
    readout_LO: int,
    overlap_time: int = 0,
    readout_ringup_time: int = 0,
    drive_ringup_time: int = 200,
    drive_ringdown_time: int = 200,
    drive_n_sigmas: float = 4,
    optional_identifier: str | None = None,
) -> int:
    """Run one T1 saturation sweep over ``durations_idle`` and stream the readout result to the database.

    The QProgram is built by :func:`qili_qprograms.experiments.t1_saturation`, the shared
    production experiment: it reads the readout pulse, its weights and the drive/readout
    intermediate frequencies for ``operating_point`` from ``calibration`` and plays the saturation
    drive on the target's buses. This driver owns only what qili-qprograms does not: it sets the
    drive/readout LOs and the drive power/RF state on the platform (Rohde & Schwarz or QCMRF),
    executes the sweep, streams the per-idle readout result, and turns a Rohde & Schwarz drive output
    off afterwards.

    Args:
        platform (Platform): Platform the QProgram runs on.
        db_manager (DatabaseManager): Database manager the measurement is stored through.
        calibration (Calibration): Calibration whose operating point and readout pulse the QProgram reads.
        target (str): Qubit token the measurement belongs to, e.g. ``"q14"``; also names its buses.
        operating_point (str): Operating point the readout pulse and the drive/readout IFs are taken from.
        readout_bus (str): Readout bus alias.
        drive_bus (str): Drive bus alias.
        durations_idle (np.ndarray): Idle times in ns to sweep between drive and readout (the T1 curve).
        averages (int): Number of averages per measurement point.
        relax_duration (int): Relaxation wait after each measurement, in ns.
        drive_duration (int): Square saturation drive pulse duration, in ns.
        drive_amplitude (float): Drive amplitude for ring-up, saturation and ring-down.
        drive_gain (float): Drive power (``POWER`` for Rohde & Schwarz, ``GAIN`` for QCMRF).
        drive_LO (int): Drive LO frequency, in Hz.
        readout_LO (int): Readout LO frequency, in Hz.
        overlap_time (int, optional): Overlap in ns between the drive ring-down and the readout. Defaults to 0.
        readout_ringup_time (int, optional): Readout ring-up before the measurement, in ns (0 = none). Defaults to 0.
        drive_ringup_time (int, optional): Smooth drive ring-up duration, in ns. Defaults to 200.
        drive_ringdown_time (int, optional): Smooth drive ring-down duration, in ns. Defaults to 200.
        drive_n_sigmas (float, optional): Number of sigmas of the ring-up/ring-down envelopes. Defaults to 4.
        optional_identifier (str | None, optional): Optional measurement identifier. Defaults to None.

    Returns:
        int: Measurement id assigned by the database.
    """
    qprogram = t1_saturation(
        calibration=calibration,
        target=target,
        operating_point=operating_point,
        idle_duration_sweep=durations_idle,
        averages=averages,
        relax_duration=relax_duration,
        drive_duration=drive_duration,
        drive_amplitude=drive_amplitude,
        drive_ringup_time=drive_ringup_time,
        drive_ringdown_time=drive_ringdown_time,
        drive_n_sigmas=drive_n_sigmas,
        overlap_time=overlap_time,
        readout_ringup_time=readout_ringup_time,
    )

    platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_LO)
    platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_LO)

    drive_instruments = [instrument.name.name for instrument in platform.get_element(drive_bus).instruments]
    rohde_schwarz = "ROHDE_SCHWARZ" in drive_instruments
    if rohde_schwarz:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
        platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
    elif "QCMRF" in drive_instruments:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
    else:
        warn(_NO_DRIVE_INSTRUMENT_MSG)

    stream_array = StreamArray(
        shape=(len(durations_idle), 2),
        loops={"time": {"array": durations_idle.astype(int), "units": "ns", "bus": drive_bus, "parameter": "time"}},
        platform=platform,
        experiment_name="t1_pulsed_saturation",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
        calibration=calibration,
        qubit_idx=target,
    )
    try:
        with stream_array:
            results = platform.execute_qprogram(qprogram).results
            for idx in range(len(durations_idle)):
                stream_array[idx,] = results[readout_bus][idx].array.T
    finally:
        if rohde_schwarz:
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return _measurement_id(stream_array)
