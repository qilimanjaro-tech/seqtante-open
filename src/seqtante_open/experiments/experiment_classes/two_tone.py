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

from seqtante_open.experiments.analysis import sss_from_array
from seqtante_open.experiments.qprogram import two_tone_pulsed_saturation, two_tone_spectroscopy

#: Warning raised when a drive bus has no instrument whose power/gain can be set.
_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."


def _measurement_id(stream_array: StreamArray) -> int:
    """The measurement id of a completed stream, narrowing the measurement union."""
    return cast("Measurement | AutocalMeasurement", stream_array.measurement).measurement_id


def two_tone_pulsed_saturation_experiment(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    drive_bus: str,
    readout_if_freq: int,
    averages: int,
    readout_duration: int,
    readout_amplitude: float,
    drive_duration: int,
    drive_amplitude: float,
    relax_duration: int,
    overlap_time: int,
    ringup_time: int,
    drive_if_sweep: np.ndarray,
    drive_gain: float,
    drive_sigmas: int = 4,
    duration_ringup_drive: int = 200,
    drive_LO: int | None = None,
    readout_LO: int | None = None,
    target: str | None = None,
    calibration: Calibration | None = None,
    optional_identifier: str | None = None,
) -> int:
    """Run one two-tone pulsed-saturation sweep over ``drive_if_sweep`` and stream the result to the database.

    Builds the QProgram via :func:`~seqtante_open.experiments.qprogram.two_tone_pulsed_saturation`, sets
    the drive/readout hardware (LO/IF, drive power/gain, RF state) according to the drive instrument
    (Rohde & Schwarz or QCMRF), executes the sweep and streams the readout result; the RF output of a
    Rohde & Schwarz drive instrument is turned off afterwards. This is the measurement the reset-point
    search loops (QHC-1807), ported from the experimentalists' calibration script.

    Args:
        platform (Platform): Platform the QProgram runs on.
        db_manager (DatabaseManager): Database manager the measurement is stored through.
        readout_bus (str): Readout bus alias.
        drive_bus (str): Drive bus alias.
        readout_if_freq (int): Readout intermediate frequency, in Hz.
        averages (int): Number of averages per measurement point.
        readout_duration (int): Readout measurement pulse duration, in ns.
        readout_amplitude (float): Readout pulse amplitude.
        drive_duration (int): Square saturation drive pulse duration, in ns.
        drive_amplitude (float): Drive amplitude for ring-up, saturation and ring-down.
        relax_duration (int): Relaxation wait after each measurement, in ns.
        overlap_time (int): Overlap in ns between the drive ring-down and the readout pulse.
        ringup_time (int): Readout ring-up duration before the measurement, in ns (0 = none).
        drive_if_sweep (np.ndarray): Linearly spaced drive IF values to sweep, in Hz.
        drive_gain (float): Drive power (``POWER`` for Rohde & Schwarz, ``GAIN`` for QCMRF).
        drive_sigmas (int, optional): Number of sigmas of the ring-up/ring-down envelopes. Defaults to 4.
        duration_ringup_drive (int, optional): Smooth drive ring-up duration, in ns. Defaults to 200.
        drive_LO (int | None, optional): Drive LO frequency, in Hz; unchanged when None. Defaults to None.
        readout_LO (int | None, optional): Readout LO frequency, in Hz; unchanged when None. Defaults to None.
        target (str | None, optional): Qubit token the measurement belongs to. Defaults to None.
        calibration (Calibration | None, optional): Calibration played with the QProgram. Defaults to None.
        optional_identifier (str | None, optional): Optional measurement identifier. Defaults to None.

    Returns:
        int: Measurement id assigned by the database.
    """
    qprogram = two_tone_pulsed_saturation(  # type: ignore [misc]
        readout_if_freq=readout_if_freq,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
        drive_if_sweep=drive_if_sweep,
        drive_duration=drive_duration,
        drive_amplitude=drive_amplitude,
        averages=averages,
        duration_ringup_drive=duration_ringup_drive,
        drive_sigmas=drive_sigmas,
        relax_duration=relax_duration,
        overlap_time=overlap_time,
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
        shape=(len(drive_if_sweep), 2),
        loops={
            "IF_frequency": {"array": drive_if_sweep, "units": "Hz", "bus": drive_bus, "parameter": "IF_frequency"},
        },
        platform=platform,
        experiment_name="two_tone_pulsed_saturation",
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

    return _measurement_id(stream_array)


def two_tone_frequency(
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
    drive_LO: int | None = None,
    readout_LO: int | None = None,
    target: str | None = None,
    calibration: Calibration | None = None,
    optional_identifier: str | None = None,
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
        instrument_name
        for instrument_platform in platform.get_element(drive_bus).instruments
        if (instrument_name := instrument_platform.name.name) in ["ROHDE_SCHWARZ", "QCMRF"]
    )
    if instrument_platform is not None:
        if instrument_platform == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif instrument_platform == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
    else:
        warn(_NO_DRIVE_INSTRUMENT_MSG)

    stream_array = StreamArray(
        shape=(len(drive_IF_sweep), 2),
        loops={
            "IF_frequency": {"array": drive_IF_sweep, "units": "Hz", "bus": drive_bus, "parameter": "IF_frequency"},
        },
        platform=platform,
        experiment_name="two_tone_frequency",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
        calibration=calibration,
        qubit_idx=target,
    )

    with stream_array:
        results = platform.execute_qprogram(qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}).results
        stream_array[:,] = results[readout_bus][0].array.T

    for instrument in platform.get_element(drive_bus).instruments:
        if instrument.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return _measurement_id(stream_array)


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
        instrument_name
        for instrument_platform in platform.get_element(drive_bus).instruments
        if (instrument_name := instrument_platform.name.name) in ["ROHDE_SCHWARZ", "QCMRF"]
    )
    if instrument_platform is not None:
        if instrument_platform == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif instrument_platform == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
    else:
        warn(_NO_DRIVE_INSTRUMENT_MSG)

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

    return _measurement_id(stream_array)
