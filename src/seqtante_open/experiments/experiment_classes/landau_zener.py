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

"""Compartmentalized Landau-Zener transition experiment with a user-provided AC crosstalk matrix.

Vendored from ``qilitools.experiments.landau_zener`` (QHC-1822); only the compartmentalized
non-idling-reset driver the ``landau_zener`` node runs (QHC-1808).

This sweeps a set of annealing paths and, for each, builds the compartmentalized
non-idling-reset QProgram (see
:func:`~seqtante_open.experiments.qprogram.generate_lz_qprogram_not_idling_reset`) and executes it,
streaming the readout quadratures and the single-shot threshold to the database.

The AC crosstalk matrix is supplied by the caller as a qililab ``CrosstalkMatrix`` and baked into
the played waveforms, so qililab's own crosstalk compensation is switched off at execution time
(``crosstalk=False``) to avoid correcting twice. This is the variant for BSC, where qililab does
not yet drive the AC crosstalk from the calibration. A caller holding a raw array can wrap it with
``CrosstalkMatrix.from_array(buses=flux_buses, matrix_array=xtalk_ac)``.
"""

from typing import TYPE_CHECKING, cast

from qililab import CrosstalkMatrix, Parameter, Platform
from qililab.result import DatabaseManager, StreamArray

from seqtante_open.experiments.qprogram import generate_lz_qprogram_not_idling_reset

if TYPE_CHECKING:
    from qililab import Measurement
    from qililab.result.database import AutocalMeasurement


def lz_transition_not_idling_reset__custom_linear_ramp(
    platform: Platform,
    db_manager: DatabaseManager,
    readout_bus: str,
    readout_if_freq: int,
    flux_buses: list[str],
    flux_readout: list[float],
    flux_reset: list[float],
    idle_point_dc: list[float],
    reset_dur: int,
    annealing_paths_list: list[list[list]],
    landau_zener_annealing_time: list,
    readout_dur: int,
    crosstalk: CrosstalkMatrix,
    averages: int,
    ramp_reset_duration: list[int],
    ramp_readout_duration: list[int],
    readout_amp: float = 0.1,
    ringup_time: int = 0,
    readout_LO: float | None = None,
    optional_identifier: str | None = None,
) -> tuple[int, object]:
    """Run a compartmentalized Landau-Zener transition experiment with a custom AC crosstalk matrix.

    The slow flux lines are expected biased to ``idle_point_dc``; every played flux level is taken
    relative to it. For each annealing path in ``annealing_paths_list`` the function builds the
    compartmentalized QProgram -- reset hold, reset ramp, Landau-Zener path, readout ramp, readout
    hold -- and executes it, sweeping the loop over ``landau_zener_annealing_time`` and streaming
    the readout quadratures and the single-shot threshold to the database.

    Args:
        platform (Platform): Hardware platform used to execute the QProgram.
        db_manager (DatabaseManager): Database manager for storing the results.
        readout_bus (str): Alias of the readout bus on the platform.
        readout_if_freq (int): Intermediate frequency of the readout bus, in Hz.
        flux_buses (list[str]): Exactly two flux bus aliases ``[z_bus, x_bus]``, bound to ``flux_1``
            and ``flux_2`` in this order. The crosstalk matrix must cover exactly these two buses;
            the inverse crosstalk is keyed by bus name, so the matrix's own bus order does not matter.
        flux_readout (list[float]): Readout point ``[phi_z, phi_x]`` in absolute flux units.
        flux_reset (list[float]): Reset point ``[phi_z, phi_x]`` in absolute flux units (the
            "not idling" reset).
        idle_point_dc (list[float]): Idle point ``[phi_z, phi_x]`` every flux level is taken
            relative to.
        reset_dur (int): Duration of the reset hold before each ramp, in ns.
        annealing_paths_list (list[list[list]]): Annealing paths. Each path is a list of waypoints
            ``[phi_z, phi_x, n_points]``. One path is played per sweep point.
        landau_zener_annealing_time (list): Total annealing times in ns, the sweep loop variable
            stored as metadata. One per annealing path.
        readout_dur (int): Duration of the readout measurement pulse, in ns.
        crosstalk (CrosstalkMatrix): qililab AC crosstalk matrix whose buses cover ``flux_buses``
            in the same order. Its inverse is applied to the flux waveforms via ``flux_to_bias``.
        averages (int): Number of averages per measurement point.
        ramp_reset_duration (list[int]): Reset-ramp stage durations ``[longitudinal, transverse]``,
            in ns.
        ramp_readout_duration (list[int]): Readout-ramp stage durations, read the same way; the
            readout ramp moves both axes at once, so both entries must be equal.
        readout_amp (float, optional): Amplitude of the readout pulse. Defaults to 0.1.
        ringup_time (int, optional): Ring-up pre-pulse before the acquisition, in ns. Defaults to 0.
        readout_LO (float, optional): Local oscillator frequency for the readout bus, in Hz. If
            given, it is set on the platform before the experiment. Defaults to None.
        optional_identifier (str | None, optional): Optional string identifier for the
            measurement. Defaults to None.

    Returns:
        tuple[int, QProgram]: The measurement ID assigned by the database, and the last QProgram
            executed during the sweep.

    Raises:
        ValueError: If the crosstalk matrix buses do not match ``flux_buses``, if the sweep lists
            have different lengths, or if a path's ramp overruns the Qblox arbitrary-waveform
            budget (raised by the QProgram builder).
    """
    if len(flux_buses) != 2 or set(crosstalk.matrix) != set(flux_buses):
        raise ValueError(
            "The compartmentalized Landau-Zener experiment drives exactly two flux buses, and the AC "
            f"crosstalk matrix must cover exactly those. Matrix buses: {sorted(crosstalk.matrix)}; "
            f"flux_buses: {flux_buses}."
        )

    if any(offset != 0 for offset in crosstalk.flux_offsets.values()):
        raise ValueError(
            "The AC crosstalk matrix must carry zero flux offsets: this additive AC layer does not use "
            "them, but CrosstalkMatrix.flux_to_bias would still subtract them from the played waveforms. "
            "Pass a zero-offset matrix (e.g. CrosstalkMatrix.from_array(buses=flux_buses, matrix_array=...))."
        )

    if not annealing_paths_list:
        raise ValueError("annealing_paths_list is empty; there is nothing to sweep.")

    if len(annealing_paths_list) != len(landau_zener_annealing_time):
        raise ValueError(
            f"annealing_paths_list ({len(annealing_paths_list)}) and landau_zener_annealing_time "
            f"({len(landau_zener_annealing_time)}) must have one entry per sweep point."
        )

    if readout_LO is not None:
        platform.set_parameter(alias=readout_bus, parameter=Parameter.LO_FREQUENCY, value=readout_LO)

    flux_buses_pair = (flux_buses[0], flux_buses[1])

    def build(flux_path_sequence: list[list]):
        return generate_lz_qprogram_not_idling_reset(
            readout_if_freq=readout_if_freq,
            reset_dur=reset_dur,
            flux_path_sequence=flux_path_sequence,
            flux_idle_dc=(idle_point_dc[0], idle_point_dc[1]),
            flux_readout=(flux_readout[0], flux_readout[1]),
            flux_reset=(flux_reset[0], flux_reset[1]),
            readout_dur=readout_dur,
            averages=averages,
            readout_amp=readout_amp,
            ringup_time=ringup_time,
            ramp_reset_duration=(ramp_reset_duration[0], ramp_reset_duration[1]),
            ramp_readout_duration=(ramp_readout_duration[0], ramp_readout_duration[1]),
            crosstalk=crosstalk,
            flux_buses=flux_buses_pair,
        )

    # Build every QProgram up front so a malformed or over-long path fails before any hardware
    # runs, rather than part-way through the sweep after earlier points have streamed to the DB.
    qprograms = [build(annealing_paths) for annealing_paths in annealing_paths_list]

    stream_array = StreamArray(
        shape=(len(landau_zener_annealing_time), 3),
        loops={
            "total_annealing_time": {
                "array": landau_zener_annealing_time,
                "units": "ns",
                "bus": flux_buses[0],
                "parameter": "Total_annealing_time",
            },
        },
        db_manager=db_manager,
        platform=platform,
        experiment_name="Landau Zener transition experiment (compartmentalized)",
        qprogram=qprograms[0],
        optional_identifier=optional_identifier,
    )

    bus_mapping = {"readout": readout_bus}
    for i, bus_name in enumerate(flux_buses):
        bus_mapping[f"flux_{i + 1}"] = bus_name

    with stream_array:
        for ii, qprogram in enumerate(qprograms):
            # Crosstalk is already baked into the waveforms, so disable qililab's bias-level
            # crosstalk compensation to avoid correcting twice.
            results = platform.execute_qprogram(qprogram, bus_mapping=bus_mapping, crosstalk=False).results
            stream_array[ii, :2] = results[readout_bus][0].array.T
            stream_array[ii, 2] = results[readout_bus][0].threshold[0]

    # Inside the ``with`` block the stream was opened, so ``measurement`` is set; its concrete type
    # is ``Measurement | AutocalMeasurement``.
    measurement_id = cast("Measurement | AutocalMeasurement", stream_array.measurement).measurement_id

    return measurement_id, qprograms[-1]
