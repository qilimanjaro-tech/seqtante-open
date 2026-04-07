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
from qililab import Calibration, Parameter, Platform
from qililab.result import DatabaseManager, StreamArray
from qililab.waveforms import Arbitrary
from tqdm.auto import tqdm

from seqtante_open.experiments.analysis import XTalk
from seqtante_open.experiments.analysis.analysis import compensate_round
from seqtante_open.experiments.qprogram import (
    rabi_amplitude_gaussian,
    rabi_single_gaussian_ac,
    rabi_single_gaussian_drive,
    rabi_single_reset,
    rabi_single_square_drive,
)


def rabi__time(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_if_freq: int,
    averages: int,
    r_amp: float,
    d_amp: float,
    drive_gain: float,
    relax_duration: int,
    ringup_time: int,
    readout_duration: int,
    readout_bus: str,
    drive_bus: str,
    drive_if_freq: float,
    drive_lo_freq: int,
    drive_time_sweep: np.ndarray,
    readout_lo_freq: int | None = None,
    optional_identifier: str | None = None,
):
    qprogram = rabi_single_square_drive(
        readout_if_freq=readout_if_freq,
        drive_if_freq=drive_if_freq,
        averages=averages,
        r_duration=readout_duration,
        d_duration=drive_time_sweep[0],
        r_amp=r_amp,
        d_amp=d_amp,
        ringup_time=ringup_time,
        relax_duration=relax_duration,
    )

    if drive_lo_freq:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_lo_freq)
    if readout_lo_freq:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_lo_freq)

    for instrument_platform in platform.get_element(drive_bus).instruments:
        if instrument_platform.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)

        elif instrument_platform.name.name == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
        else:
            warn(f"Not setting voltage to this parameter: {instrument_platform.name.name}")

    stream_array = StreamArray(
        shape=(len(drive_time_sweep), 2),
        loops={"drive_time": {"array": drive_time_sweep, "units": "ns", "bus": drive_bus, "parameter": "drive_time"}},
        platform=platform,
        experiment_name="rabi_time",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
    )

    with stream_array:
        for ii, d_dur in enumerate(tqdm(drive_time_sweep)):
            qprogram = rabi_single_square_drive(
                readout_if_freq=readout_if_freq,
                drive_if_freq=drive_if_freq,
                averages=averages,
                r_duration=readout_duration,
                d_duration=d_dur,
                r_amp=r_amp,
                d_amp=d_amp,
                ringup_time=ringup_time,
                relax_duration=relax_duration,
            )

            results = platform.execute_qprogram(
                qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
            ).results
            stream_array[ii,] = results[readout_bus][0].array.T

    for instrument_platform in platform.get_element(drive_bus).instruments:
        if instrument_platform.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return stream_array.measurement.measurement_id if stream_array.measurement is not None else None


def rabi__time_gain(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_if_freq: int,
    averages: int,
    r_amp: float,
    d_amp: float,
    drive_gain_sweep: np.ndarray,
    relax_duration: int,
    ringup_time: int,
    readout_duration: int,
    readout_bus: str,
    drive_bus: str,
    drive_if_freq: float,
    drive_lo_freq: int,
    drive_time_sweep: np.ndarray,
    readout_lo_freq: int | None = None,
    optional_identifier: str | None = None,
):
    qprogram = rabi_single_square_drive(
        readout_if_freq=readout_if_freq,
        drive_if_freq=drive_if_freq,
        averages=averages,
        r_duration=readout_duration,
        d_duration=drive_time_sweep[0],
        r_amp=r_amp,
        d_amp=d_amp,
        ringup_time=ringup_time,
        relax_duration=relax_duration,
    )

    if drive_lo_freq:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_lo_freq)
    if readout_lo_freq:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_lo_freq)

    for instrument_platform in platform.get_element(drive_bus).instruments:
        if instrument_platform.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain_sweep[0])
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)

            loops_ = {
                "drive_gain": {
                    "array": drive_gain_sweep,
                    "units": "dB",
                    "bus": drive_bus,
                    "parameter": "drive_gain",
                },  # different units for gain
                "drive_time": {"array": drive_time_sweep, "units": "ns", "bus": drive_bus, "parameter": "drive_time"},
            }

        elif instrument_platform.name.name == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain_sweep[0])

            loops_ = {
                "drive_gain": {
                    "array": drive_gain_sweep,
                    "units": "a.u.",
                    "bus": drive_bus,
                    "parameter": "drive_gain",
                },  # different units for gain
                "drive_time": {"array": drive_time_sweep, "units": "ns", "bus": drive_bus, "parameter": "drive_time"},
            }

        else:
            warn(f"Not setting voltage to this parameter: {instrument_platform.name.name}")

    stream_array = StreamArray(
        shape=(len(drive_gain_sweep), len(drive_time_sweep), 2),
        loops=loops_,
        platform=platform,
        experiment_name="rabi_time_gain",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
    )

    with stream_array:
        for jj, d_gain in enumerate(tqdm(drive_gain_sweep)):
            if instrument_platform.name.name == "ROHDE_SCHWARZ":
                platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=d_gain)
            elif instrument_platform.name.name == "QCMRF":
                platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=d_gain)

            for ii, d_dur in enumerate(tqdm(drive_time_sweep)):
                qprogram = rabi_single_square_drive(
                    readout_if_freq=readout_if_freq,
                    drive_if_freq=drive_if_freq,
                    averages=averages,
                    r_duration=readout_duration,
                    d_duration=d_dur,
                    r_amp=r_amp,
                    d_amp=d_amp,
                    ringup_time=ringup_time,
                    relax_duration=relax_duration,
                )

                results = platform.execute_qprogram(
                    qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
                ).results
                stream_array[
                    jj,
                    ii,
                ] = results[readout_bus][0].array.T

    for instrument_platform in platform.get_element(drive_bus).instruments:
        if instrument_platform.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return stream_array.measurement.measurement_id if stream_array.measurement is not None else None


def rabi__time_frequency(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_if_freq: int,
    averages: int,
    r_amp: float,
    d_amp: float,
    drive_gain: float,
    relax_duration: int,
    ringup_time: int,
    readout_duration: int,
    readout_bus: str,
    drive_bus: str,
    drive_if_array: np.ndarray,
    drive_lo_freq: int,
    drive_time_sweep: np.ndarray,
    readout_lo_freq: int | None = None,
    optional_identifier: str | None = None,
):
    qprogram = rabi_single_square_drive(
        readout_if_freq=readout_if_freq,
        drive_if_freq=drive_if_array[0],
        averages=averages,
        r_duration=readout_duration,
        d_duration=drive_time_sweep[0],
        r_amp=r_amp,
        d_amp=d_amp,
        ringup_time=ringup_time,
        relax_duration=relax_duration,
    )

    if drive_lo_freq:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_lo_freq)
    if readout_lo_freq:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_lo_freq)

    for instrument_platform in platform.get_element(drive_bus).instruments:
        if instrument_platform.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif instrument_platform.name.name == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
        else:
            warn(f"Not setting voltage to this parameter: {instrument_platform.name.name}")

    stream_array = StreamArray(
        shape=(len(drive_if_array), len(drive_time_sweep), 2),
        loops={
            "drive_if": {"array": drive_if_array, "units": "Hz", "bus": drive_bus, "parameter": "drive_if"},
            "drive_time": {"array": drive_time_sweep, "units": "ns", "bus": drive_bus, "parameter": "drive_time"},
        },
        platform=platform,
        experiment_name="rabi_time_frequency",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
    )

    with stream_array:
        for jj, d_if in enumerate(tqdm(drive_if_array)):
            for ii, d_dur in enumerate(tqdm(drive_time_sweep)):
                qprogram = rabi_single_square_drive(
                    readout_if_freq=readout_if_freq,
                    drive_if_freq=drive_if_array[jj],
                    averages=averages,
                    r_duration=readout_duration,
                    d_duration=d_dur,
                    r_amp=r_amp,
                    d_amp=d_amp,
                    ringup_time=ringup_time,
                    relax_duration=relax_duration,
                )

                results = platform.execute_qprogram(
                    qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
                ).results
                stream_array[
                    jj,
                    ii,
                ] = results[readout_bus][0].array.T

    for instrument_platform in platform.get_element(drive_bus).instruments:
        if instrument_platform.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return stream_array.measurement.measurement_id if stream_array.measurement is not None else None


def rabi__amplitude_gaussian_drive(
    platform,
    db_manager,
    readout_if_freq,
    averages,
    ringup_time,
    readout_duration,
    r_amp,
    d_duration,
    readout_bus,
    drive_bus,
    drive_gain,
    drive_if_freq,
    drive_lo_freq,
    drive_amp_sweep,
    relax_duration,
    d_sigmas=4,
    d_drag=0,
    readout_lo_freq=None,
    optional_identifier=None,
):
    qprogram = rabi_single_gaussian_drive(
        readout_if_freq=readout_if_freq,
        drive_if_freq=drive_if_freq,
        averages=averages,
        r_duration=readout_duration,
        d_duration=d_duration,
        r_amp=r_amp,
        d_amp=drive_amp_sweep[0],
        d_sigmas=d_sigmas,
        d_drag=d_drag,
        ringup_time=ringup_time,
        relax_duration=relax_duration,
    )

    if drive_lo_freq is not None:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_lo_freq)
    if readout_lo_freq is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_lo_freq)

    for instrument in platform.get_element(drive_bus).instruments:
        if instrument.name.name == "ROHDE_SCHWARZ":
            warn("initializing RS")
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif instrument.name.name == "QCMRF":
            warn("initializing QCMRF")
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
        else:
            warn(f"doing nothing to this bus: {instrument.name.name}")

    stream_array = StreamArray(
        shape=(len(drive_amp_sweep), 2),
        loops={
            "drive_amplitude": {
                "array": drive_amp_sweep,
                "units": "amp",
                "bus": drive_bus,
                "parameter": "drive_amplitude",
            }
        },
        platform=platform,
        experiment_name="rabi_amplitude",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
    )

    with stream_array:
        for ii, d_amp in enumerate(tqdm(drive_amp_sweep)):
            qprogram = rabi_single_gaussian_drive(
                readout_if_freq=readout_if_freq,
                drive_if_freq=drive_if_freq,
                averages=averages,
                r_duration=readout_duration,
                d_duration=d_duration,
                r_amp=r_amp,
                d_amp=d_amp,
                d_sigmas=d_sigmas,
                d_drag=d_drag,
                ringup_time=ringup_time,
                relax_duration=relax_duration,
            )

            results = platform.execute_qprogram(
                qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}
            ).results
            stream_array[ii,] = results[readout_bus][0].array.T

    for instrument in platform.get_element(drive_bus).instruments:
        if instrument.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)
    id = stream_array.measurement.measurement_id if stream_array.measurement is not None else None

    return id


def rabi__amplitude_gaussian_hw_loop(
    platform: Platform,
    db_manager: DatabaseManager,
    averages: int,
    readout_duration: int,
    readout_amplitude: float,
    drag_duration: int,
    readout_bus: str,
    drive_bus: str,
    drag_amplitude_sweep: np.ndarray,
    relax_duration: int,
    drag_sigmas: float,
    drag_coefficient: float,
    ringup_time: int = 0,
    optional_identifier: str | None = None,
    qubit_idx: int | None = None,
    calibration: Calibration | None = None,
    autocalibration: bool = False,
) -> int:
    """Executes the "*Rabi vs Amplitude*" experiment and saves it to the database.

    Args:
        platform (Platform): Platform object.
        db_manager (DatabaseManager): DatabaseManager object.
        averages (int): Avereges.
        readout_duration (int): Calibrated duration of a readout waveform for the qubit.
        readout_amplitude (float): Calibrated amplitude of a readout waveform for the qubit.
        drag_duration (int): Calibrated duration of a pi rotation drag waveform for the qubit.
        readout_bus (str): Identifier for the qubit's readout bus.
        drive_bus (str): Identifier for the qubit's drive bus.
        drag_amplitude_sweep (np.ndarray): _description_
        relax_duration (int): Time needed to ensure the qubit loses it state.
        drag_sigmas (float): Calibrated sigmas of a pi rotation drag waveform for the qubit.
        drag_coefficient (float): Calibrated drag coefficient of a pi rotation drag waveform for the qubit.
        ringup_time (int, optional): Time of the pulse needed to exite the resonator for readout. Defaults to 0.
        optional_identifier (str | None, optional): Identifier for the measurement in the database. Defaults to None.
        qubit_idx (int | None, optional): Index of the qubit. Defaults to None.
        calibration (Calibration | None, optional): Calibration object for the database . Defaults to None.
        autocalibration (bool, optional): If True it saves the measurement to the autocalibration database. Defaults to False.

    Returns:
        int: ID of the measurement in the database.
    """

    qprogram = rabi_amplitude_gaussian(
        averages=averages,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
        drag_amplitude_sweep=drag_amplitude_sweep,
        drag_duration=drag_duration,
        drag_sigmas=drag_sigmas,
        drag_coefficient=drag_coefficient,
        relax_duration=relax_duration,
        ringup_time=ringup_time,
    )

    stream_array = StreamArray(
        shape=(len(drag_amplitude_sweep), 2),
        loops={
            "drive_amplitude": {
                "array": drag_amplitude_sweep,
                "units": "",
                "bus": drive_bus,
                "parameter": "drive_amplitude",
            }
        },
        platform=platform,
        experiment_name="rabi_amplitude",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
        qubit_idx=qubit_idx,
        calibration=calibration,
        autocalibration=autocalibration,
    )

    with stream_array:
        results = platform.execute_qprogram(qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus}).results
        stream_array[()] = results[readout_bus][0].array.T

    return stream_array.measurement.measurement_id if stream_array.measurement is not None else None  # type:ignore [return-value]


def rabi__amplitude_gaussian_ac(
    platform,
    db_manager,
    readout_bus,
    drive_bus,
    flux_bus,
    readout_if_freq,
    averages,
    r_amp,
    r_duration,
    r_wait_time,
    d_duration,
    d_wait_time,
    drive_if_freq,
    drive_lo_freq,
    drive_amp_sweep,
    b_amp,
    b_duration,
    repetition_time,
    ringup_time: int = 0,
    d_sigmas=4.0,
    d_drag=0.0,
    RS_drive_gain=1,
    readout_lo_freq=None,
    optional_identifier=None,
):
    qprogram = rabi_single_gaussian_ac(
        readout_if_freq=readout_if_freq,
        drive_if_freq=drive_if_freq,
        averages=averages,
        r_duration=r_duration,
        r_wait_time=r_wait_time,
        d_amp=drive_amp_sweep[0],
        d_duration=d_duration,
        d_wait_time=d_wait_time,
        b_amp=b_amp,
        b_duration=b_duration,
        repetition_time=repetition_time,
        r_amp=r_amp,
        d_sigmas=d_sigmas,
        d_drag=d_drag,
        ringup_time=ringup_time,
    )

    if drive_lo_freq is not None:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_lo_freq)
    if readout_lo_freq is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_lo_freq)

    for instrument in platform.get_element(drive_bus).instruments:
        if instrument.name.name == "ROHDE_SCHWARZ":
            warn(f"{instrument.name.name}: initializing RS")
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=RS_drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        else:
            warn(f"doing nothing to this bus: {instrument.name.name}")

    stream_array = StreamArray(
        shape=(len(drive_amp_sweep), 2),
        loops={"drive_amp": {"array": drive_amp_sweep, "units": "", "bus": drive_bus, "parameter": "drive_amp"}},
        platform=platform,
        experiment_name="rabi_amplitude_ac",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
    )

    with stream_array:
        for ii, amp in enumerate(tqdm(drive_amp_sweep)):
            qprogram = rabi_single_gaussian_ac(
                readout_if_freq=readout_if_freq,
                drive_if_freq=drive_if_freq,
                averages=averages,
                r_duration=r_duration,
                r_wait_time=r_wait_time,
                d_amp=amp,
                d_duration=d_duration,
                d_wait_time=d_wait_time,
                b_amp=b_amp,
                b_duration=b_duration,
                repetition_time=repetition_time,
                r_amp=r_amp,
                d_sigmas=d_sigmas,
                d_drag=d_drag,
                ringup_time=ringup_time,
            )

            results = platform.execute_qprogram(
                qprogram, bus_mapping={"readout": readout_bus, "drive": drive_bus, "flux": flux_bus}
            ).results
            stream_array[ii,] = results[readout_bus][0].array.T

    for instrument in platform.get_element(drive_bus).instruments:
        if instrument.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)
    id = stream_array.measurement.measurement_id if stream_array.measurement is not None else None

    return id


def rabi__time_reset(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_if_freq: int,
    averages: int,
    r_amp: float,
    d_amp: float,
    drive_gain: float,
    ramp_up_duration: int,
    wait_after_ramp_up: int,  # in ns
    ramp_down_duration: int,
    dwell_us: int,
    reset_time: int,
    reset_point: list[float],
    operating_point: list[float],
    xtalk_object: XTalk,
    trigger_output_channel: int,
    ringup_time: int,
    readout_duration: int,
    readout_bus: str,
    readout_bus_debug: str,
    drive_bus: str,
    drive_if_freq: float,
    drive_lo_freq: int,
    drive_time_sweep: np.ndarray,
    d_sigmas: float = 4.0,
    d_drag: float = 0.0,
    readout_lo_freq: int | None = None,
    optional_identifier: str | None = None,
):
    params_to_check = {
        "ramp_up_duration": ramp_up_duration,
        "wait_after_ramp_up": wait_after_ramp_up * 1e-3,
        "ramp_down_duration": ramp_down_duration,
        "reset_time": reset_time,
    }

    not_aligned = {name: val for name, val in params_to_check.items() if val % dwell_us != 0}

    if not_aligned:
        raise ValueError(
            f"The following parameters are not divisible by dwell_us={dwell_us}:\n"
            + "\n".join([f"  {k}: {v}" for k, v in not_aligned.items()])
        )

    qprogram = rabi_single_reset(
        readout_if_freq=readout_if_freq,
        drive_if_freq=drive_if_freq,
        averages=averages,
        r_duration=readout_duration,
        d_duration=drive_time_sweep[0],
        ramp_up_duration=int(ramp_up_duration * 1e3),  # us , convert to ns
        reset_time=int(reset_time * 1e3),  # us , convert to ns
        wait_after_ramp_up=wait_after_ramp_up,
        r_amp=r_amp,
        d_amp=d_amp,
        ringup_time=ringup_time,
        d_sigmas=d_sigmas,
        d_drag=d_drag,
    )

    for dim in range(xtalk_object.N):  # set voltage to the reset point
        xtalk_object.set_flux(dim, reset_point[dim])

    if drive_lo_freq is not None:
        platform.set_parameter(alias=drive_bus, parameter=Parameter.LO_FREQUENCY, value=drive_lo_freq)
    if readout_lo_freq is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_lo_freq)

    for instrument_platform in platform.get_element(drive_bus).instruments:
        if instrument_platform.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.POWER, value=drive_gain)
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=True)
        elif instrument_platform.name.name == "QCMRF":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.GAIN, value=drive_gain)
        else:
            warn(f"Not setting voltage to this parameter: {instrument_platform.name.name}")

    stream_array = StreamArray(
        shape=(len(drive_time_sweep), 2),
        loops={"drive_time": {"array": drive_time_sweep, "units": "ns", "bus": drive_bus, "parameter": "drive_time"}},
        platform=platform,
        experiment_name="rabi_time_reset",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
    )

    with stream_array:
        for ii, d_dur in enumerate(tqdm(drive_time_sweep)):
            time_at_operting_point = compensate_round(
                (d_dur + ringup_time + readout_duration + wait_after_ramp_up) * 1e-3 + dwell_us
            )  # length of the reset pulse, excluding ramps, must be multiple of dwell times
            ramps = xtalk_object.get_reset_list(
                reset_point,
                operating_point,
                reset_time,
                ramp_up_duration,
                time_at_operting_point,
                ramp_down_duration,
                dwell_us,
            )  # NOTE Operating point needs to be given in the same basis as the xtalk class is defined in

            qp = rabi_single_reset(
                readout_if_freq=readout_if_freq,
                drive_if_freq=drive_if_freq,
                averages=averages,
                r_duration=readout_duration,
                d_duration=d_dur,
                ramp_up_duration=int(ramp_up_duration * 1e3),  # us , convert to ns
                reset_time=int(reset_time * 1e3),  # us , convert to ns
                wait_after_ramp_up=wait_after_ramp_up,
                r_amp=r_amp,
                d_amp=d_amp,
                ringup_time=ringup_time,
                d_sigmas=d_sigmas,
                d_drag=d_drag,
            )

            # repetitions = averages
            qdac2 = platform.get_element("qdac2")  # this one gets the qililab version

            if xtalk_object.qdac_channels:
                for qdac_ch, ramp in zip(xtalk_object.qdac_channels, ramps):
                    qdac_ch_int = int(qdac_ch.label.lstrip("ch"))

                    qdac2.upload_voltage_list(
                        waveform=Arbitrary(samples=ramp),
                        channel_id=qdac_ch_int,
                        dwell_us=dwell_us,
                        sync_delay_s=0 * 1e-6,
                        repetitions=averages,
                    )

                # Ext trigger only from 1 voltage list is enough
                qdac2.set_start_marker_external_trigger(
                    channel_id=int(xtalk_object.qdac_channels[0].label.lstrip("ch")),
                    out_port=trigger_output_channel,
                    trigger=f"ext_trigger_{trigger_output_channel}",
                    width_s=ramp_up_duration * 1e-6,
                )

            # TODO: Remove execute_qblox_qdac_triggers replacing it with the correct qprogram with QdacCompiler
            results = platform.execute_qblox_qdac_triggers(  # type:ignore [attr-defined]
                qdac2, qp, bus_mapping={"readout": readout_bus, "readout_debug": readout_bus_debug, "drive": drive_bus}
            )

            stream_array[ii,] = results.results[readout_bus][0].array.T

    for instrument in platform.get_element(drive_bus).instruments:
        if instrument.name.name == "ROHDE_SCHWARZ":
            platform.set_parameter(alias=drive_bus, parameter=Parameter.RF_ON, value=False)

    return stream_array.measurement.measurement_id if stream_array.measurement is not None else None