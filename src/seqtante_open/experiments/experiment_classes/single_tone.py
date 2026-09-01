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

"""Single-tone drivers, copied from ``qilitools.experiments.single_tone``."""

import numpy as np
from qililab import Calibration, Parameter
from qililab.platform import Platform
from qililab.result import DatabaseManager, StreamArray

from seqtante_open.experiments.analysis import sss_from_array
from seqtante_open.experiments.qprogram import resonator_spectroscopy, single_tone_vs_flux
from seqtante_open.experiments.utils import get_qdac_out_trigger, qdac_step_timings


def single_tone__frequency_vs_flux(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    if_sweep: np.ndarray,
    r_amp: float,
    averages: int,
    duration: int,
    flux_bus: str,
    flux_sweep: np.ndarray,
    minimum_wait_after_step_override=None,
    qdac_stop_ro_before_step_override=None,
    lo: int | None = None,
    calibration=None,
    optional_identifier: str | None = None,
    flux_parameter: Parameter = Parameter.FLUX,
    qubit_idx: str | None = None,
    secondary_idx: str | None = None,
    autocalibration: bool = False,
    readout_attenuation: int | None = None,
) -> int | None:
    """Continuous-wave single-tone spectroscopy vs. flux, hardware-looping over the readout IF
    while the flux is ramped via QDAC soft steps.

    Args:
        platform: Qililab platform to execute on.
        db_manager: Database manager for result storage.
        readout_bus: Physical alias of the readout bus.
        if_sweep: Readout intermediate frequencies to hardware-loop over.
        r_amp: Amplitude of the readout pulse.
        averages: Number of hardware averages.
        duration: Duration of the readout pulse.
        flux_bus: Physical alias of the flux bus being ramped.
        flux_sweep: Flux bias points to ramp the QDAC through.
        minimum_wait_after_step_override: If provided (together with
            `qdac_stop_ro_before_step_override`), used instead of the QDAC low-pass-filter
            settling time computed from the platform. Defaults to None.
        qdac_stop_ro_before_step_override: If provided (together with
            `minimum_wait_after_step_override`), used instead of the default
            `QDAC_TRIGGER_TO_VOLTAGE_PADDING` readout cutoff before each QDAC step. Defaults to
            None.
        lo: If provided, set the readout LO frequency before executing. Defaults to None.
        calibration: Calibration to use when executing the qprogram. Defaults to None.
        optional_identifier: Identifier for the measurement in the database. Defaults to None.
        flux_parameter: Whether `flux_sweep` sets flux or voltage. Defaults to `Parameter.FLUX`.
        qubit_idx: Qubit index to associate with the measurement. Defaults to None.
        secondary_idx: Secondary qubit/coupler index to associate with the measurement. Defaults
            to None.
        readout_attenuation: If provided, set the readout output attenuation before executing.
            Defaults to None.

    Returns:
        int | None: ID of the measurement in the database.
    """
    qdac_min_wait_after_step, qdac_stop_ro_before_step = qdac_step_timings(
        platform, minimum_wait_after_step_override, qdac_stop_ro_before_step_override
    )

    if lo is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=lo)
    if readout_attenuation is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.OUT0_ATT, value=readout_attenuation)

    qprogram = single_tone_vs_flux(
        if_sweep,
        averages=averages,
        time_per_avg=duration,
        r_amp=r_amp,
        ramp_array=flux_sweep,
        minimum_wait_after_step=qdac_min_wait_after_step,
        stop_ro_before_step=qdac_stop_ro_before_step,
        trigger_channel=get_qdac_out_trigger(platform),
    )

    if flux_parameter is Parameter.FLUX:
        units = "phi_0"
    if flux_parameter is Parameter.VOLTAGE:
        units = "V"

    stream_array = StreamArray(
        shape=(len(flux_sweep), len(if_sweep), 2),
        loops={
            "flux_outer": {
                "array": flux_sweep,
                "units": units,
                "bus": flux_bus,
                "parameter": flux_parameter.title(),
            },
            "flux_inner": {
                "array": if_sweep,
                "units": "Hz",
                "bus": readout_bus,
                "parameter": "IF_frequency",
            },
        },
        platform=platform,
        experiment_name="single_tone__frequency_vs_flux",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
        calibration=calibration,
        qubit_idx=qubit_idx,
        secondary_idx=secondary_idx,
        autocalibration=autocalibration,
    )

    with stream_array:
        if flux_parameter is Parameter.VOLTAGE:
            results = platform.execute_qprogram(
                qprogram,
                bus_mapping={"readout": readout_bus, "flux": flux_bus},
                calibration=calibration,
                crosstalk=False,
            ).results
        else:
            results = platform.execute_qprogram(
                qprogram, bus_mapping={"readout": readout_bus, "flux": flux_bus}, calibration=calibration
            ).results
        stream_array[()] = results[readout_bus][0].array.transpose(1, 2, 0)
    return stream_array.measurement.measurement_id if stream_array.measurement is not None else None


def single_tone__frequency_sweep(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    if_sweep: np.ndarray,
    readout_amplitude: float,
    averages: int,
    readout_duration: int,
    relax_duration: int,
    ringup_time: int = 0,
    readout_LO: int | None = None,
    qubit_idx: str | None = None,
    calibration: Calibration | None = None,
    optional_identifier: str | None = None,
    autocalibration: bool = False,
) -> int | None:
    """Pulsed single-tone spectroscopy (resonator spectroscopy), hardware-looping over the readout IF.

    Args:
        platform: Qililab platform to execute on.
        db_manager: Database manager for result storage.
        readout_bus: Physical alias of the readout bus.
        if_sweep: Readout intermediate frequencies to hardware-loop over.
        readout_amplitude: Amplitude of the readout pulse.
        averages: Number of hardware averages.
        readout_duration: Duration of the readout pulse.
        relax_duration: Resonator relaxation time between repetitions, in ns.
        ringup_time: Time of the pulse needed to excite the resonator for readout. Defaults to 0.
        readout_LO: If provided, set the readout LO frequency before executing. Defaults to None.
        qubit_idx: Qubit index to associate with the measurement. Defaults to None.
        calibration: Calibration to use when executing the qprogram. Defaults to None.
        optional_identifier: Identifier for the measurement in the database. Defaults to None.
        autocalibration: If True the measurement is saved in the autocalibration database. Defaults to False.

    Returns:
        int | None: ID of the measurement in the database.
    """
    qprogram = resonator_spectroscopy(
        *sss_from_array(if_sweep),
        averages=averages,
        r_duration=readout_duration,
        r_amp=readout_amplitude,
        relax_duration=relax_duration,
        ringup_time=ringup_time,
    )

    if readout_LO is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_LO)

    stream_array = StreamArray(
        shape=(len(if_sweep), 2),
        loops={
            "frequency": {"array": if_sweep, "units": "Hz", "bus": readout_bus, "parameter": "IF_frequency"},
        },
        platform=platform,
        experiment_name="single_tone__frequency_sweep",
        db_manager=db_manager,
        qprogram=qprogram,
        optional_identifier=optional_identifier,
        calibration=calibration,
        qubit_idx=qubit_idx,
        autocalibration=autocalibration,
    )

    with stream_array:
        results = platform.execute_qprogram(qprogram, bus_mapping={"readout": readout_bus}).results
        stream_array[()] = results[readout_bus][0].array.T
    return stream_array.measurement.measurement_id if stream_array.measurement is not None else None
