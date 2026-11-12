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
from qililab.result import AutocalMeasurement, DatabaseManager, Measurement, StreamArray

from seqtante_open.experiments.qprogram import rabi_amplitude_drag


def rabi_amplitude(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    readout_if_freq: float,
    readout_duration: int,
    readout_amplitude: float,
    drive_bus: str,
    drive_power: float,
    drive_if_freq: float,
    drive_amplitude_sweep: np.ndarray,
    drive_duration: int,
    averages: int,
    relax_duration: int,
    ringup_time: int = 0,
    drive_sigmas: float = 4,
    drive_drag: float = 0,
    drive_LO: float | None = None,
    readout_LO: float | None = None,
    target: str | None = None,
    calibration: Calibration | None = None,
    optional_identifier: str | None = None,
) -> int:
    """Run a Rabi amplitude sweep with a DRAG drive pulse and stream the result to the database.

    Builds the QProgram via :func:`~seqtante_open.experiments.qprogram.rabi_amplitude_drag`, sets the
    drive and readout hardware (LO frequencies, drive power/gain and RF state) according to the drive
    instrument (Rohde & Schwarz or QCMRF), executes the amplitude sweep and streams the readout result.
    The RF output of a Rohde & Schwarz drive instrument is turned off again afterwards.

    Copied from ``qilitools.experiments.rabi.rabi__amplitude_drag_drive`` (fixes go back through
    QHC-1668) and aligned to the seqtante-open driver conventions.

    Args:
        platform (Platform): Platform the QProgram runs on.
        db_manager (DatabaseManager): Database manager the measurement is stored through.
        readout_bus (str): Readout bus alias.
        readout_if_freq (float): Readout intermediate frequency, in Hz.
        readout_duration (int): Readout measurement pulse duration, in ns.
        readout_amplitude (float): Readout pulse amplitude.
        drive_bus (str): Drive bus alias.
        drive_power (float): Drive power (``POWER`` for Rohde & Schwarz, ``GAIN`` for QCMRF).
        drive_if_freq (float): Drive intermediate frequency, in Hz.
        drive_amplitude_sweep (np.ndarray): Linearly spaced drive amplitude (gain) values to sweep.
        drive_duration (int): DRAG drive pulse duration, in ns.
        averages (int): Number of averages per measurement point.
        relax_duration (int): Relaxation wait after each measurement, in ns.
        ringup_time (int, optional): Ring-up pulse duration before the readout, in ns. Defaults to 0.
        drive_sigmas (float, optional): Number of sigmas of the DRAG Gaussian envelope. Defaults to 4.
        drive_drag (float, optional): DRAG coefficient. Defaults to 0.
        drive_LO (float | None, optional): Drive LO frequency, in Hz; unchanged when None. Defaults to None.
        readout_LO (float | None, optional): Readout LO frequency, in Hz; unchanged when None. Defaults to None.
        target (str | None, optional): Qubit token the measurement belongs to, e.g. ``"q1"``. Defaults to None.
        calibration (Calibration | None, optional): Calibration played with the QProgram. Defaults to None.
        optional_identifier (str | None, optional): Optional measurement identifier. Defaults to None.

    Returns:
        int: Measurement id assigned by the database.
    """
    qprogram = rabi_amplitude_drag(
        readout_if_freq=readout_if_freq,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
        ringup_time=ringup_time,
        drive_amplitude_sweep=drive_amplitude_sweep,
        drive_if_freq=drive_if_freq,
        drive_duration=drive_duration,
        drive_sigmas=drive_sigmas,
        drag_coefficient=drive_drag,
        relax_duration=relax_duration,
        averages=averages,
    )

    if drive_LO is not None:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_LO)
    if readout_LO is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_LO)

    drive_instruments = [instrument.name.name for instrument in platform.get_element(drive_bus).instruments]
    rohde_schwarz = "ROHDE_SCHWARZ" in drive_instruments
    if rohde_schwarz:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_power)
        platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
    elif "QCMRF" in drive_instruments:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_power)
    else:
        warn("No instrument to set power to.")

    stream_array = StreamArray(
        shape=(len(drive_amplitude_sweep), 2),
        loops={
            "drive_amplitude": {
                "array": drive_amplitude_sweep,
                "units": "amp",
                "bus": drive_bus,
                "parameter": "drive_amplitude",
            }
        },
        platform=platform,
        experiment_name="rabi_amplitude_drag",
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
            stream_array[()] = results[readout_bus][0].array.T
    finally:
        if rohde_schwarz:
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return cast("Measurement | AutocalMeasurement", stream_array.measurement).measurement_id
