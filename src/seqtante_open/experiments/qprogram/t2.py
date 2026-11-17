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

"""T2 (Ramsey) DRAG QProgram, copied from ``qilitools.qprogram.t2.t2_single`` (QHC-1683)."""

from qililab import IQDrag, IQPair, QProgram, Square


def t2_single(
    wait_time: int,
    averages: int,
    r_duration: int,
    pi_pulse_duration: int,
    r_amp: float = 0.1,
    pi_pulse_amp: float = 0.1,
    pi_pulse_sigmas: float = 4.0,
    pi_drag_coef: float = 0.0,
    relax_duration: int = 2000,
    ringup_time: int = 0,
    readout_if_freq: float | None = None,
    drive_if_freq: float | None = None,
) -> QProgram:
    """Build one T2 (Ramsey) point: two DRAG pi-half pulses separated by ``wait_time``, then a readout.

    The drive pulses are DRAG pi-half pulses: the played amplitude is ``pi_pulse_amp / 2`` (i.e.
    ``pi_pulse_amp`` is the full pi-pulse amplitude, the usual convention). Copied from
    ``qilitools.qprogram.t2.t2_single`` (QHC-1683); the free-evolution sweep is the driver's software
    loop over this single-point program.

    Args:
        wait_time (int): Free-evolution wait between the two pi-half pulses, in ns.
        averages (int): Number of averages per measurement point.
        r_duration (int): Duration of the readout pulse, in ns.
        pi_pulse_duration (int): Duration of the pi pulse, in ns.
        r_amp (float, optional): Amplitude of the readout pulse. Defaults to 0.1.
        pi_pulse_amp (float, optional): Full pi-pulse amplitude; the played pi-half amplitude is
            ``pi_pulse_amp / 2``. Defaults to 0.1.
        pi_pulse_sigmas (float, optional): Gaussian sigma of the DRAG pi pulse. Defaults to 4.0.
        pi_drag_coef (float, optional): DRAG coefficient of the pi pulse. Defaults to 0.0.
        relax_duration (int, optional): Qubit relaxation wait after the measurement, in ns. Defaults to 2000.
        ringup_time (int, optional): Readout ring-up duration, in ns; 0 applies none. Defaults to 0.
        readout_if_freq (float | None, optional): Readout bus intermediate frequency, in Hz; left
            unchanged when None. Defaults to None.
        drive_if_freq (float | None, optional): Drive bus intermediate frequency, in Hz; left unchanged
            when None. Defaults to None.

    Returns:
        QProgram: the single-point T2 (Ramsey) sequence.
    """
    r_wf_I = Square(amplitude=r_amp, duration=r_duration)
    r_wf_Q = Square(amplitude=0.0, duration=r_duration)
    # Pi-half pulse: the played amplitude is pi_pulse_amp / 2.
    pihalf_wf = IQDrag(
        amplitude=pi_pulse_amp / 2,
        duration=pi_pulse_duration,
        num_sigmas=pi_pulse_sigmas,
        drag_coefficient=pi_drag_coef,
    )
    if ringup_time > 0:
        ringup_wf = Square(amplitude=r_amp, duration=ringup_time)

    weights_shape = Square(amplitude=1, duration=r_duration)

    qp_t2 = QProgram()

    if readout_if_freq is not None:
        qp_t2.set_frequency(bus="readout", frequency=readout_if_freq)
    if drive_if_freq is not None:
        qp_t2.set_frequency(bus="drive", frequency=drive_if_freq)

    with qp_t2.average(averages):
        # HALF PI PULSE, WAIT AND HALF PI PULSE
        qp_t2.play(bus="drive", waveform=pihalf_wf)
        qp_t2.wait(bus="drive", duration=wait_time)
        qp_t2.play(bus="drive", waveform=pihalf_wf)

        qp_t2.sync()

        # READOUT PULSE
        if ringup_time > 0:
            qp_t2.play(bus="readout", waveform=ringup_wf)
        qp_t2.measure(
            bus="readout",
            waveform=IQPair(I=r_wf_I, Q=r_wf_Q),
            weights=IQPair(I=weights_shape, Q=weights_shape),
        )

        qp_t2.wait(bus="readout", duration=relax_duration)
    return qp_t2
