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

"""Compartmentalized flux-pulsed QProgram builders for the Landau-Zener experiment.

Vendored from ``qilitools.qprogram.annealing`` (QHC-1822); only the compartmentalized non-idling-reset
builders the ``landau_zener`` node needs are kept here (QHC-1808).

The earlier Landau-Zener QProgram registered the whole flux trajectory -- reset hold, ramps
and readout hold -- as one arbitrary waveform per flux bus, which hits the Qblox sequencer's
arbitrary-waveform memory limit. These builders are compartmentalized: the reset and readout
holds are ``Square`` pulses (which cost no arbitrary-waveform memory) and only the ramps are
``Arbitrary`` waveforms, so the budget is spent on the parts of the trajectory that vary.

The sequence a built QProgram plays, inside one ``average`` block, is:

1. a ``Square`` hold at the reset point;
2. a reset ramp (``Arbitrary``) from the reset point to the start of the path;
3. the Landau-Zener path itself (``Arbitrary``);
4. a readout ramp (``Arbitrary``) from the end of the path to the readout point;
5. a ``Square`` hold at the readout point, with the readout measurement.

Flux bus ``i`` is played on the placeholder alias ``flux_{i + 1}`` (two buses: ``flux_1`` for
the longitudinal axis Z and ``flux_2`` for the transverse axis X), bound to physical buses in
the same order by the caller's ``bus_mapping``; the readout goes to ``readout``. The two flux
buses are passed through as ``flux_buses = (z_bus, x_bus)`` and the inverse crosstalk is keyed by
bus *name*, so the played waveforms stay correct whatever order the caller built the matrix in.

The AC crosstalk matrix is supplied by the caller as a qililab ``CrosstalkMatrix`` rather than
read from the calibration, because qililab at BSC does not yet drive the AC crosstalk from the
calibration. ``CrosstalkMatrix.flux_to_bias`` applies the inverse of the matrix to the flux
waveforms, so the caller must execute with qililab's own crosstalk compensation switched off to
avoid correcting twice. A caller holding a raw array can wrap it with
``CrosstalkMatrix.from_array(buses=flux_buses, matrix_array=xtalk_ac)``.
"""

import numpy as np
from qililab import Arbitrary, CrosstalkMatrix, IQPair, QProgram, Square

MAX_ARBITRARY_WAVEFORM_SAMPLES = 16_384
"""Samples a Qblox sequencer holds for *all* arbitrary waveforms together (one sample per ns).

Neither qililab nor the Qblox compiler checks this budget: an over-long set of arbitrary
waveforms compiles and only fails at upload, so the builder guards the total explicitly.
"""


def _as_int(value: float, name: str) -> int:
    """Coerce an integer-valued number to ``int``, or raise if it has a fractional part.

    ``np.linspace(num=...)`` and ``Square(duration=...)`` reject non-integers, so waypoint
    ``n_points`` and ramp durations written as floats (``100.0``, or a ``np.float64`` from a
    ``np.array`` waypoint) would otherwise raise a bare ``TypeError`` deep inside ``linspace``.

    Args:
        value (float): The number of samples / ns.
        name (str): Name of the quantity, for the error message.

    Returns:
        int: ``value`` as an ``int``.

    Raises:
        ValueError: If ``value`` is not integer-valued.
    """
    as_int = int(value)
    if as_int != value:
        raise ValueError(f"{name} must be a whole number of samples/ns, got {value!r}.")
    return as_int


def _validate_flux_buses(crosstalk: CrosstalkMatrix, flux_buses: tuple[str, str]) -> None:
    """Require the crosstalk matrix to cover exactly the two flux buses.

    ``CrosstalkMatrix.flux_to_bias`` reads one flux entry per matrix bus and returns one bias per
    matrix bus, so a matrix that does not cover exactly ``flux_buses`` would otherwise raise a
    bare ``KeyError`` (extra/renamed bus) or ``IndexError`` (too few buses) at play time.

    The played order is fixed by keying ``flux_to_bias`` on the bus name, not by the matrix's
    insertion order.

    Args:
        crosstalk (CrosstalkMatrix): AC crosstalk matrix supplied by the caller.
        flux_buses (tuple[str, str]): The two flux bus aliases ``(z_bus, x_bus)``.

    Raises:
        ValueError: If the matrix does not cover exactly the two flux buses.
    """
    matrix_buses = set(crosstalk.matrix.keys())
    if len(flux_buses) != 2 or set(flux_buses) != matrix_buses:
        raise ValueError(
            "The compartmentalized Landau-Zener builders drive exactly two flux buses, and the AC "
            f"crosstalk matrix must cover exactly those. Matrix buses: {sorted(matrix_buses)}; "
            f"flux_buses: {list(flux_buses)}."
        )


def offset_flux_point(point: tuple[float, float], flux_idle_dc: tuple[float, float]) -> tuple[float, float]:
    """Take a flux point relative to the idle point, axis by axis.

    Args:
        point (tuple[float, float]): A flux point ``(phi_z, phi_x)`` in absolute flux units.
        flux_idle_dc (tuple[float, float]): The idle point ``(phi_z, phi_x)`` to subtract.

    Returns:
        tuple[float, float]: ``point - flux_idle_dc``, axis by axis.
    """
    return (point[0] - flux_idle_dc[0], point[1] - flux_idle_dc[1])


def path_interpolation_linear(path: list[list], start_point: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a sequence of 2D waypoints, as one concatenated array per axis.

    Starting from ``start_point``, each segment of the path is interpolated linearly between
    consecutive waypoints, using the waypoint's own ``n_points`` steps, and all segments are
    concatenated. The result is the two per-axis sample arrays ``(phi_z, phi_x)`` ready to key
    into ``CrosstalkMatrix.flux_to_bias`` -- the only form any caller uses.

    Args:
        path (list[list]): Waypoints, each ``[phi_z, phi_x, n_points]``. ``phi_z`` and ``phi_x``
            are the target coordinates and ``n_points`` the number of interpolation steps.
        start_point (tuple[float, float]): Starting point ``(phi_z, phi_x)``. The first segment is
            interpolated from this point to the first waypoint.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(phi_z_samples, phi_x_samples)``, the whole path
            concatenated, one array per axis.
    """
    z_segments: list[np.ndarray] = []
    x_segments: list[np.ndarray] = []

    previous = start_point
    for waypoint in path:
        n_points = _as_int(waypoint[2], "n_points")
        z_segments.append(np.linspace(previous[0], waypoint[0], n_points))
        x_segments.append(np.linspace(previous[1], waypoint[1], n_points))
        previous = (waypoint[0], waypoint[1])

    return np.concatenate(z_segments), np.concatenate(x_segments)


def flux_to_bias_ac(
    path: list[list],
    flux_initial_point: tuple[float, float],
    crosstalk: CrosstalkMatrix,
    flux_buses: tuple[str, str],
) -> list[np.ndarray]:
    """Interpolate a path and apply the inverse AC crosstalk matrix, one bias array per flux bus.

    The path is expected already relative to the idle point. ``CrosstalkMatrix.flux_to_bias``
    applies the inverse of the matrix (matching ``inv(xtalk_ac) @ flux_path``), so the returned
    arrays are the hardware bias -- the voltage the flux line plays -- not amperes. They come back
    in ``flux_buses`` play order: ``flux_buses[0]`` on ``flux_1`` and ``flux_buses[1]`` on
    ``flux_2``, keyed by bus name rather than the matrix's insertion order.

    Args:
        path (list[list]): Waypoints ``[phi_z, phi_x, n_points]``, relative to the idle point.
        flux_initial_point (tuple[float, float]): Point ``(phi_z, phi_x)`` the interpolation
            starts from, relative to the idle point.
        crosstalk (CrosstalkMatrix): AC crosstalk matrix covering exactly ``flux_buses``.
        flux_buses (tuple[str, str]): Flux bus aliases ``(z_bus, x_bus)`` in play order.

    Returns:
        list[np.ndarray]: ``[z_bias_samples, x_bias_samples]`` in play order.

    Raises:
        ValueError: If the matrix does not cover exactly ``flux_buses``.
    """
    _validate_flux_buses(crosstalk, flux_buses)
    z_samples, x_samples = path_interpolation_linear(path, flux_initial_point)
    flux_by_bus = {flux_buses[0]: z_samples, flux_buses[1]: x_samples}
    bias_by_bus = crosstalk.flux_to_bias(flux_by_bus)
    return [
        np.asarray(bias_by_bus[flux_buses[0]], dtype=float),
        np.asarray(bias_by_bus[flux_buses[1]], dtype=float),
    ]


def add_reset_square(qp: QProgram, flux_reset: tuple[float, float], reset_dur: int) -> None:
    """Hold each flux bus at its reset level for the reset duration.

    Args:
        qp (QProgram): The QProgram to append to.
        flux_reset (tuple[float, float]): Per-bus reset level ``(z, x)``, relative to the idle point.
        reset_dur (int): Duration of the reset hold, in ns.
    """
    qp.play(bus="flux_1", waveform=Square(amplitude=float(flux_reset[0]), duration=reset_dur))
    qp.play(bus="flux_2", waveform=Square(amplitude=float(flux_reset[1]), duration=reset_dur))


def add_flux_ramp(
    qp: QProgram,
    ramp_duration: tuple[int, int],
    crosstalk: CrosstalkMatrix,
    flux_initial_point: tuple[float, float],
    flux_end_point: tuple[float, float],
    flux_buses: tuple[str, str],
    move_first: str = "Z",
) -> int:
    """Append one calibrated flux ramp between two points, moving the axes in the given order.

    ``move_first="Z"`` moves the longitudinal axis first with the transverse parked, then the
    transverse; ``"X"`` is the mirror image; ``"both"`` moves the two axes together in one leg
    (both ``ramp_duration`` entries must then be equal). The ramp is played as one ``Arbitrary``
    waveform per flux bus, followed by a ``sync``.

    Args:
        qp (QProgram): The QProgram to append to.
        ramp_duration (tuple[int, int]): Stage durations ``(longitudinal, transverse)``, in ns.
        crosstalk (CrosstalkMatrix): AC crosstalk matrix covering exactly ``flux_buses``.
        flux_initial_point (tuple[float, float]): Point ``(z, x)`` the ramp leaves from, idle-relative.
        flux_end_point (tuple[float, float]): Point ``(z, x)`` the ramp lands on, idle-relative.
        flux_buses (tuple[str, str]): Flux bus aliases ``(z_bus, x_bus)`` in play order.
        move_first (str): ``"Z"``, ``"X"`` or ``"both"``. Defaults to ``"Z"``.

    Returns:
        int: Number of arbitrary samples the ramp plays per flux bus (for the sequencer-memory guard).

    Raises:
        ValueError: If ``move_first`` is not one of the three options, or if ``"both"`` is asked
            with unequal stage durations.
    """
    if move_first == "X":
        ramp_points = [
            [flux_initial_point[0], flux_end_point[1], ramp_duration[1]],
            [flux_end_point[0], flux_end_point[1], ramp_duration[0]],
        ]
    elif move_first == "Z":
        ramp_points = [
            [flux_end_point[0], flux_initial_point[1], ramp_duration[0]],
            [flux_end_point[0], flux_end_point[1], ramp_duration[1]],
        ]
    elif move_first == "both":
        if ramp_duration[0] != ramp_duration[1]:
            raise ValueError("To move both axes at once, both entries of ramp_duration must be equal.")
        ramp_points = [[flux_end_point[0], flux_end_point[1], ramp_duration[0]]]
    else:
        raise ValueError(f"move_first must be 'Z', 'X' or 'both'; got {move_first!r}.")

    bias_flux = flux_to_bias_ac(ramp_points, flux_initial_point, crosstalk, flux_buses)
    for i, bias in enumerate(bias_flux):
        qp.play(bus=f"flux_{i + 1}", waveform=Arbitrary(bias))
    qp.sync()
    return len(bias_flux[0])


def add_flux_pulsed_sequence(
    qp: QProgram,
    flux_path_sequence: list[list],
    crosstalk: CrosstalkMatrix,
    flux_buses: tuple[str, str],
) -> int:
    """Append the Landau-Zener path itself as one arbitrary waveform per flux bus.

    The path is interpolated from its first waypoint through its waypoints, then corrected with
    the inverse crosstalk matrix and played, followed by a ``sync``.

    Args:
        qp (QProgram): The QProgram to append to.
        flux_path_sequence (list[list]): Waypoints ``[phi_z, phi_x, n_points]``, idle-relative.
        crosstalk (CrosstalkMatrix): AC crosstalk matrix covering exactly ``flux_buses``.
        flux_buses (tuple[str, str]): Flux bus aliases ``(z_bus, x_bus)`` in play order.

    Returns:
        int: Number of arbitrary samples the path plays per flux bus (for the sequencer-memory guard).
    """
    start_point = (flux_path_sequence[0][0], flux_path_sequence[0][1])
    bias_flux = flux_to_bias_ac(flux_path_sequence, start_point, crosstalk, flux_buses)
    for i, bias in enumerate(bias_flux):
        qp.play(bus=f"flux_{i + 1}", waveform=Arbitrary(bias))
    qp.sync()
    return len(bias_flux[0])


def add_readout_square(
    qp: QProgram,
    readout_if_freq: float,
    flux_readout: tuple[float, float],
    readout_amp: float,
    readout_dur: int,
    ringup_time: int = 0,
) -> None:
    """Hold the flux at the readout level and measure, with an optional ring-up pre-pulse.

    Args:
        qp (QProgram): The QProgram to append to.
        readout_if_freq (float): Readout intermediate frequency, in Hz.
        flux_readout (tuple[float, float]): Per-bus readout level ``(z, x)``, relative to the idle point.
        readout_amp (float): Amplitude of the readout pulse.
        readout_dur (int): Duration of the readout measurement pulse, in ns.
        ringup_time (int): Ring-up pre-pulse before the acquisition, in ns. 0 means none.
    """
    qp.set_frequency(bus="readout", frequency=readout_if_freq)

    r_wf_I = Square(amplitude=readout_amp, duration=readout_dur)
    r_wf_Q = Square(amplitude=0.0, duration=readout_dur)
    weights_shape = Square(amplitude=1, duration=readout_dur)
    flux_hold_dur = readout_dur + ringup_time

    qp.play(bus="flux_1", waveform=Square(amplitude=float(flux_readout[0]), duration=flux_hold_dur))
    qp.play(bus="flux_2", waveform=Square(amplitude=float(flux_readout[1]), duration=flux_hold_dur))

    if ringup_time > 0:
        # Single-path waveform: the compiler pairs it with the zero Q-path on the readout bus.
        qp.play(bus="readout", waveform=Square(amplitude=readout_amp, duration=ringup_time))

    qp.measure(
        bus="readout",
        waveform=IQPair(I=r_wf_I, Q=r_wf_Q),
        weights=IQPair(I=weights_shape, Q=weights_shape),
    )
    qp.sync()


def _guard_sequencer_memory(reset_ramp_samples: int, path_samples: int, readout_ramp_samples: int) -> None:
    """Raise if the arbitrary waveforms together overrun one Qblox sequencer's budget.

    The reset ramp, the path and the readout ramp all live in the same flux sequencer's
    arbitrary-waveform memory, so the limit applies to their *sum*, not to each ramp on its own.

    Args:
        reset_ramp_samples (int): Arbitrary samples the reset ramp plays per flux bus.
        path_samples (int): Arbitrary samples the Landau-Zener path plays per flux bus.
        readout_ramp_samples (int): Arbitrary samples the readout ramp plays per flux bus.

    Raises:
        ValueError: If the total exceeds :data:`MAX_ARBITRARY_WAVEFORM_SAMPLES`.
    """
    total = reset_ramp_samples + path_samples + readout_ramp_samples
    if total > MAX_ARBITRARY_WAVEFORM_SAMPLES:
        raise ValueError(
            f"The flux trajectory plays {total} arbitrary samples per flux bus (reset ramp "
            f"{reset_ramp_samples} + path {path_samples} + readout ramp {readout_ramp_samples}), "
            f"past the {MAX_ARBITRARY_WAVEFORM_SAMPLES} a Qblox sequencer holds for all arbitrary "
            "waveforms together; it would compile and then fail at upload. Shorten the ramps or the path."
        )


def generate_lz_qprogram_not_idling_reset(
    readout_if_freq: float,
    reset_dur: int,
    flux_path_sequence: list[list],
    flux_reset: tuple[float, float],
    flux_readout: tuple[float, float],
    readout_dur: int,
    averages: int,
    ramp_reset_duration: tuple[int, int],
    ramp_readout_duration: tuple[int, int],
    flux_idle_dc: tuple[float, float],
    crosstalk: CrosstalkMatrix,
    flux_buses: tuple[str, str],
    readout_amp: float = 0.1,
    ringup_time: int = 0,
) -> QProgram:
    """Build the compartmentalized Landau-Zener QProgram with a non-idling reset.

    All flux points are taken relative to ``flux_idle_dc`` and then played, so the slow flux
    lines must be biased to the idle point before the QProgram runs. The sequence is the one
    documented in the module: reset hold, reset ramp, Landau-Zener path, readout ramp, readout
    hold and measurement. The combined reset ramp + path + readout ramp is guarded against the
    sequencer's arbitrary-waveform budget before the program is returned.

    Args:
        readout_if_freq (float): Readout intermediate frequency, in Hz.
        reset_dur (int): Duration of the reset hold, in ns.
        flux_path_sequence (list[list]): Landau-Zener waypoints ``[phi_z, phi_x, n_points]`` in
            absolute flux units. The second waypoint's ``n_points`` is the annealing time swept.
        flux_reset (tuple[float, float]): Reset point ``(phi_z, phi_x)`` in absolute flux units.
        flux_readout (tuple[float, float]): Readout point ``(phi_z, phi_x)`` in absolute flux units.
        readout_dur (int): Duration of the readout measurement pulse, in ns.
        averages (int): Number of averages.
        ramp_reset_duration (tuple[int, int]): Reset-ramp stage durations
            ``(longitudinal, transverse)``, in ns.
        ramp_readout_duration (tuple[int, int]): Readout-ramp stage durations, read the same way;
            the readout ramp moves both axes at once, so both entries must be equal.
        flux_idle_dc (tuple[float, float]): Idle point ``(phi_z, phi_x)`` every flux point is taken
            relative to.
        crosstalk (CrosstalkMatrix): AC crosstalk matrix covering exactly ``flux_buses``.
        flux_buses (tuple[str, str]): Flux bus aliases ``(z_bus, x_bus)`` in play order.
        readout_amp (float, optional): Amplitude of the readout pulse. Defaults to 0.1.
        ringup_time (int, optional): Ring-up pre-pulse before the acquisition, in ns. Defaults to 0.

    Returns:
        QProgram: The compartmentalized Landau-Zener sequence.

    Raises:
        ValueError: If the crosstalk matrix does not cover exactly ``flux_buses``, or if the
            combined arbitrary waveforms overrun the Qblox sequencer budget.
    """
    flux_readout = offset_flux_point(flux_readout, flux_idle_dc)
    flux_reset = offset_flux_point(flux_reset, flux_idle_dc)
    flux_path_sequence = [[*offset_flux_point((p[0], p[1]), flux_idle_dc), p[2]] for p in flux_path_sequence]
    path_start = (flux_path_sequence[0][0], flux_path_sequence[0][1])
    path_end = (flux_path_sequence[-1][0], flux_path_sequence[-1][1])

    qp = QProgram()

    with qp.average(averages):
        add_reset_square(qp, flux_reset=flux_reset, reset_dur=reset_dur)
        reset_ramp_samples = add_flux_ramp(
            qp,
            ramp_duration=ramp_reset_duration,
            crosstalk=crosstalk,
            flux_initial_point=flux_reset,
            flux_end_point=path_start,
            flux_buses=flux_buses,
        )
        path_samples = add_flux_pulsed_sequence(qp, flux_path_sequence, crosstalk=crosstalk, flux_buses=flux_buses)
        readout_ramp_samples = add_flux_ramp(
            qp,
            ramp_duration=ramp_readout_duration,
            crosstalk=crosstalk,
            flux_initial_point=path_end,
            flux_end_point=flux_readout,
            flux_buses=flux_buses,
            move_first="both",
        )
        add_readout_square(
            qp,
            readout_if_freq=readout_if_freq,
            flux_readout=flux_readout,
            readout_amp=readout_amp,
            readout_dur=readout_dur,
            ringup_time=ringup_time,
        )

    _guard_sequencer_memory(reset_ramp_samples, path_samples, readout_ramp_samples)

    return qp
