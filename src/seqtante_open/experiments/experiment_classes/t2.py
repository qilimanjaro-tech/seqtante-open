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

"""T2 (Ramsey) DRAG execution driver, ported from ``qilitools.experiments.t2.t2__drag`` (QHC-1683)."""

from typing import cast
from warnings import warn

import numpy as np
from qililab import Calibration, Parameter
from qililab.platform import Platform
from qililab.result import AutocalMeasurement, DatabaseManager, Measurement, StreamArray

from seqtante_open.experiments.qprogram import t2_single

_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."


def _measurement_id(stream_array: StreamArray) -> int:
    """The measurement id of a completed stream, narrowing the measurement union."""
    return cast("Measurement | AutocalMeasurement", stream_array.measurement).measurement_id


def t2_drag_experiment(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    drive_bus: str,
    elapsed_time_sweep: np.ndarray,
    drive_if_freq: float,
    drive_duration: int,
    drive_amplitude: float,
    drive_power: float,
    readout_if_freq: float,
    readout_duration: int,
    readout_amplitude: float,
    relax_duration: int,
    averages: int,
    ringup_time: int = 0,
    drive_sigmas: float = 4.0,
    drag_coefficient: float = 0.0,
    drive_LO: int | None = None,
    readout_LO: int | None = None,
    target: str | None = None,
    calibration: Calibration | None = None,
    optional_identifier: str | None = None,
) -> int:
    """Run one T2 (Ramsey) DRAG sweep over ``elapsed_time_sweep`` and stream the readout to the database.

    For each free-evolution time it builds the :func:`~seqtante_open.experiments.qprogram.t2_single`
    QProgram (two DRAG pi-half pulses separated by the wait, then a readout), sets the drive/readout
    hardware (LO/IF, drive power/gain, RF state) according to the drive instrument (Rohde & Schwarz or
    QCMRF), executes the point and streams its readout; the RF output of a Rohde & Schwarz drive
    instrument is turned off afterwards. Ported from ``qilitools.experiments.t2.t2__drag`` (QHC-1683).

    Args:
        platform (Platform): Platform the QProgram runs on.
        db_manager (DatabaseManager): Database manager the measurement is stored through.
        readout_bus (str): Readout bus alias.
        drive_bus (str): Drive bus alias.
        elapsed_time_sweep (np.ndarray): Free-evolution times to sweep, in ns (the T2 curve).
        drive_if_freq (float): Drive intermediate frequency, in Hz.
        drive_duration (int): DRAG pi-half pulse duration, in ns.
        drive_amplitude (float): Amplitude of the DRAG pi-half pulse as played (the pi/2 amplitude); it
            is passed to ``t2_single`` as ``pi_pulse_amp`` doubled, since ``t2_single`` plays
            ``pi_pulse_amp / 2``.
        drive_power (float): Drive power (``POWER`` for Rohde & Schwarz, ``GAIN`` for QCMRF).
        readout_if_freq (float): Readout intermediate frequency, in Hz.
        readout_duration (int): Readout measurement pulse duration, in ns.
        readout_amplitude (float): Readout pulse amplitude.
        relax_duration (int): Relaxation wait after each measurement, in ns.
        averages (int): Number of averages per measurement point.
        ringup_time (int, optional): Readout ring-up duration before the measurement, in ns (0 = none). Defaults to 0.
        drive_sigmas (float, optional): Number of sigmas of the DRAG envelope. Defaults to 4.0.
        drag_coefficient (float, optional): DRAG coefficient of the pi-half pulse. Defaults to 0.0.
        drive_LO (int | None, optional): Drive LO frequency, in Hz; unchanged when None. Defaults to None.
        readout_LO (int | None, optional): Readout LO frequency, in Hz; unchanged when None. Defaults to None.
        target (str | None, optional): Qubit token the measurement belongs to. Defaults to None.
        calibration (Calibration | None, optional): Calibration played with the QProgram. Defaults to None.
        optional_identifier (str | None, optional): Optional measurement identifier. Defaults to None.

    Returns:
        int: Measurement id assigned by the database.
    """

    def build_qprogram(elapsed_time: int):
        # drive_amplitude is the pi/2 amplitude as played; t2_single halves pi_pulse_amp.
        return t2_single(
            wait_time=elapsed_time,
            averages=averages,
            r_duration=readout_duration,
            pi_pulse_duration=drive_duration,
            r_amp=readout_amplitude,
            pi_pulse_amp=drive_amplitude * 2,
            pi_pulse_sigmas=drive_sigmas,
            pi_drag_coef=drag_coefficient,
            relax_duration=relax_duration,
            ringup_time=ringup_time,
            readout_if_freq=readout_if_freq,
            drive_if_freq=drive_if_freq,
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
        warn(_NO_DRIVE_INSTRUMENT_MSG)

    qprogram = build_qprogram(int(elapsed_time_sweep[0]))
    stream_array = StreamArray(
        shape=(len(elapsed_time_sweep), 2),
        loops={
            "t2_time": {
                "array": elapsed_time_sweep.astype(int),
                "units": "ns",
                "bus": drive_bus,
                "parameter": "t2_elapsed_time",
            }
        },
        platform=platform,
        experiment_name="t2_drag",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
        calibration=calibration,
        qubit_idx=target,
    )
    try:
        with stream_array:
            for idx, elapsed_time in enumerate(elapsed_time_sweep):
                qprogram = build_qprogram(int(elapsed_time))
                results = platform.execute_qprogram(
                    qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
                ).results
                stream_array[idx,] = results[readout_bus][0].array.T
    finally:
        if rohde_schwarz:
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return _measurement_id(stream_array)
