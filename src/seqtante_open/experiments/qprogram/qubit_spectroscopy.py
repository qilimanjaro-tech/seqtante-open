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

from qililab import Domain, IQPair, QProgram, Square

def two_tone_spectroscopy(
    freq_start: float,
    freq_stop: float,
    freq_step: float,
    averages: int,
    r_duration: int,
    d_duration: int,
    r_amp=0.1,
    d_amp=0.1,
    relax_duration=2000,
    overlap_time=0,
    ringup_time: int = 0,
) -> QProgram:
    """Two tone experiment defined in Qprogram for both QM and Qblox.
    The experiment does a hardware loop over the drive frequecy.

    Args:
        freq_start (float): initial point of the frequency range.
        freq_stop (float): final point of the frequency range.
        freq_step (float): step size of the frequency range.
        averages (int): number of averages in the experiment.
        r_duration (int): duration of the readout pulse.
        d_duration (int): duration of the drive pulse.
        r_amp (float, optional): amplitude of the readout pulse. Defaults to
        0.1.
        d_amp (float, optional): amplitude of the drive pulse. Defaults to 0.1.
        relax_duration (int, optional): time of Qbit relaxation (in ns).
        Defaults to 2000.
        overlap_time (int, optional): intersection time between drive and readout.
        Defaults to 0.
        ringup_time (int): ringup duration time. Defaults to 0.

    Returns:
        QProgram: defined qprogram for the experiment
    """
    r_wf_I = Square(amplitude=r_amp, duration=r_duration)
    r_wf_Q = Square(amplitude=0.0, duration=r_duration)
    if ringup_time > 0:
        ringup_wf_I = Square(amplitude=r_amp, duration=ringup_time)
        ringup_wf_Q = Square(amplitude=0.0, duration=ringup_time)

    d_wf_I = Square(amplitude=d_amp, duration=d_duration)
    d_wf_Q = Square(amplitude=0.0, duration=d_duration)

    weights_shape = Square(amplitude=1, duration=r_duration)

    qp_2tone = QProgram()

    freq = qp_2tone.variable(label="frequency", domain=Domain.Frequency)

    with qp_2tone.average(averages):
        with qp_2tone.for_loop(variable=freq, start=freq_start, stop=freq_stop, step=freq_step):
            qp_2tone.set_frequency(bus="drive", frequency=freq)
            qp_2tone.play(bus="drive", waveform=IQPair(I=d_wf_I, Q=d_wf_Q))

            qp_2tone.wait(bus="readout", duration=d_duration - overlap_time)

            if ringup_time > 0:
                qp_2tone.play(
                    bus="readout",
                    waveform=IQPair(I=ringup_wf_I, Q=ringup_wf_Q),
                )
            qp_2tone.measure(
                bus="readout",
                waveform=IQPair(I=r_wf_I, Q=r_wf_Q),
                weights=IQPair(I=weights_shape, Q=weights_shape),
            )

            qp_2tone.wait(bus="readout", duration=relax_duration)

    return qp_2tone
