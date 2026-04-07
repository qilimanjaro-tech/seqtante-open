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

import numpy as np
from qililab import IQPair, QProgram, Square

def t1_single_square(
    wait_time: int,
    averages: int,
    r_duration: int,
    pi_pulse_duration: int,
    readout_if_freq: int,
    drive_if_freq: int,
    r_amp: float = 0.1,
    pi_pulse_amp: float = 0.1,
    relax_duration: int = 2000,
    ringup_time: int = 0,
) -> QProgram:
    """T1 experiment defined in Qprogram for both QM and Qblox.
    Args:
        wait_time (int): wait time value.
        averages (int): number of averages in the experiment.
        r_duration (int): duration of the readout pulse.
        pi_pulse_duration (int): duration of the pi pulse.
        r_amp (float, optional): amplitude of the readout pulse. Defaults to
        0.1.
        pi_pulse_amp (float, optional): amplitude of the pi pulse. Defaults to
        0.1.
        pi_pulse_sigmas (float, optional): Gaussian sigma of the pi pulse.
        Defaults to 4.0.
        pi_drag_coef (float, optional): Drag coefficient of the pi pulse.
        Defaults to 0.
        relax_duration (int, optional): time of Qbit relaxation (in ns).
        Defaults to 2000.
        ringup_time (int): ringup duration time. Defaults to 0.
    Returns:
        QProgram: defined qprogram for the experiment.
    """

    r_wf_I = Square(amplitude=r_amp, duration=r_duration)
    r_wf_Q = Square(amplitude=0.0, duration=r_duration)

    pi_wf_I = Square(amplitude=pi_pulse_amp, duration=int(pi_pulse_duration))
    pi_wf_Q = Square(amplitude=0.0, duration=int(pi_pulse_duration))

    if ringup_time > 0:
        ringup_wf_I = Square(amplitude=r_amp, duration=ringup_time)
        ringup_wf_Q = Square(amplitude=0.0, duration=ringup_time)

    weights_shape = Square(amplitude=1, duration=r_duration)

    qp_t1 = QProgram()
    qp_t1.set_frequency(bus="readout", frequency=readout_if_freq)
    qp_t1.set_frequency(bus="drive", frequency=drive_if_freq)

    with qp_t1.average(averages):
        # PI PULSE AND WAIT
        qp_t1.play(bus="drive", waveform=IQPair(I=pi_wf_I, Q=pi_wf_Q))
        qp_t1.wait(bus="drive", duration=wait_time)

        qp_t1.sync()

        # READOUT PULSE
        if ringup_time > 0:
            qp_t1.play(
                bus="readout",
                waveform=IQPair(I=ringup_wf_I, Q=ringup_wf_Q),
            )
        qp_t1.measure(
            bus="readout",
            waveform=IQPair(I=r_wf_I, Q=r_wf_Q),
            weights=IQPair(I=weights_shape, Q=weights_shape),
        )
        qp_t1.wait(bus="readout", duration=relax_duration)
    return qp_t1
