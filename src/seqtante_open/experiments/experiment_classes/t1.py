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

"""T1 saturation execution driver, ported from ``qilitools.experiments.t1.t1__saturation`` (QHC-1667)."""

from typing import cast
from warnings import warn

import numpy as np
from qililab import Calibration, Parameter
from qililab.platform import Platform
from qililab.result import AutocalMeasurement, DatabaseManager, Measurement, StreamArray

from seqtante_open.experiments.qprogram import t1_saturation

_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."


def _measurement_id(stream_array: StreamArray) -> int:
    """The measurement id of a completed stream, narrowing the measurement union."""
    return cast("Measurement | AutocalMeasurement", stream_array.measurement).measurement_id


def t1_pulsed_saturation_experiment(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    drive_bus: str,
    durations_idle: np.ndarray,
    drive_if_freq: float,
    drive_duration: int,
    drive_amplitude: float,
    drive_gain: float,
    readout_if_freq: int,
    readout_duration: int,
    readout_amplitude: float,
    overlap_time: int,
    relax_duration: int,
    averages: int,
    ringup_time: int = 0,
    duration_ringup_drive: int = 200,
    drive_sigmas: int = 4,
    drive_LO: int | None = None,
    readout_LO: int | None = None,
    target: str | None = None,
    calibration: Calibration | None = None,
    optional_identifier: str | None = None,
) -> int:
    """Run one T1 saturation sweep over ``durations_idle`` and stream the readout result to the database.

    Builds the QProgram via :func:`~seqtante_open.experiments.qprogram.t1_saturation`, sets the
    drive/readout hardware (LO/IF, drive power/gain, RF state) according to the drive instrument
    (Rohde & Schwarz or QCMRF), executes the sweep and streams the per-idle readout result; the RF
    output of a Rohde & Schwarz drive instrument is turned off afterwards. Ported from
    ``qilitools.experiments.t1.t1__saturation`` (QHC-1667).

    Args:
        platform (Platform): Platform the QProgram runs on.
        db_manager (DatabaseManager): Database manager the measurement is stored through.
        readout_bus (str): Readout bus alias.
        drive_bus (str): Drive bus alias.
        durations_idle (np.ndarray): Idle times in ns to sweep between drive and readout (the T1 curve).
        drive_if_freq (float): Drive intermediate frequency, in Hz.
        drive_duration (int): Square saturation drive pulse duration, in ns.
        drive_amplitude (float): Drive amplitude for ring-up, saturation and ring-down.
        drive_gain (float): Drive power (``POWER`` for Rohde & Schwarz, ``GAIN`` for QCMRF).
        readout_if_freq (int): Readout intermediate frequency, in Hz.
        readout_duration (int): Readout measurement pulse duration, in ns.
        readout_amplitude (float): Readout pulse amplitude.
        overlap_time (int): Overlap in ns between the drive ring-down and the readout pulse.
        relax_duration (int): Relaxation wait after each measurement, in ns.
        averages (int): Number of averages per measurement point.
        ringup_time (int, optional): Readout ring-up duration before the measurement, in ns (0 = none). Defaults to 0.
        duration_ringup_drive (int, optional): Smooth drive ring-up duration, in ns. Defaults to 200.
        drive_sigmas (int, optional): Number of sigmas of the ring-up/ring-down envelopes. Defaults to 4.
        drive_LO (int | None, optional): Drive LO frequency, in Hz; unchanged when None. Defaults to None.
        readout_LO (int | None, optional): Readout LO frequency, in Hz; unchanged when None. Defaults to None.
        target (str | None, optional): Qubit token the measurement belongs to. Defaults to None.
        calibration (Calibration | None, optional): Calibration played with the QProgram. Defaults to None.
        optional_identifier (str | None, optional): Optional measurement identifier. Defaults to None.

    Returns:
        int: Measurement id assigned by the database.
    """
    qprogram = t1_saturation(
        durations_idle=durations_idle,
        drive_if_freq=drive_if_freq,
        drive_amplitude=drive_amplitude,
        drive_duration=drive_duration,
        readout_if_freq=readout_if_freq,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
        relax_duration=relax_duration,
        averages=averages,
        overlap_time=overlap_time,
        duration_ringup_drive=duration_ringup_drive,
        drive_sigmas=drive_sigmas,
        ringup_time=ringup_time,
    )

    if drive_LO is not None:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_LO)
    if readout_LO is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_LO)
    platform.set_parameter(alias=readout_bus, parameter=Parameter.IF, value=readout_if_freq)

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
            results = platform.execute_qprogram(
                qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
            ).results
            for idx in range(len(durations_idle)):
                stream_array[idx,] = results[readout_bus][idx].array.T
    finally:
        if rohde_schwarz:
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return _measurement_id(stream_array)
