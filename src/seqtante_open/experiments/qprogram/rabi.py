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
from qililab import Domain, IQDrag, IQPair, QProgram, Square


def rabi_amplitude_drag(
    readout_if_freq: float,
    readout_amplitude: float,
    readout_duration: int,
    ringup_time: int,
    drive_amplitude_sweep: np.ndarray,
    drive_if_freq: float,
    drive_duration: int,
    drive_sigmas: float,
    drag_coefficient: float,
    relax_duration: int,
    averages: int,
) -> QProgram:
    """Rabi amplitude sweep with a DRAG drive pulse, for both QM and Qblox.

    For each drive amplitude in the sweep a DRAG pulse is played on the drive bus through a gain
    sweep, followed by a readout measurement. An optional ring-up pulse pre-charges the readout
    resonator before the measurement, and a relaxation wait after each measurement lets the qubit
    return to its ground state.

    Copied from ``qilitools.qprogram.rabi.rabi_amplitude_drag`` (fixes go back through QHC-1668).

    Args:
        readout_if_freq (float): Intermediate frequency of the readout bus, in Hz.
        readout_amplitude (float): Amplitude of the readout pulse.
        readout_duration (int): Duration of the readout measurement pulse, in ns.
        ringup_time (int): Duration of the ring-up pulse before the readout measurement, in ns. No
            ring-up pulse is played when 0.
        drive_amplitude_sweep (np.ndarray): Linearly spaced drive amplitude (gain) values to sweep;
            start, stop and step are inferred from it.
        drive_if_freq (float): Intermediate frequency of the drive bus, in Hz.
        drive_duration (int): Duration of the DRAG drive pulse, in ns.
        drive_sigmas (float): Number of sigmas of the Gaussian envelope of the DRAG pulse.
        drag_coefficient (float): DRAG coefficient of the quadrature correction term that suppresses
            leakage to higher qubit levels.
        relax_duration (int): Duration of the relaxation wait after each measurement, in ns.
        averages (int): Number of averages per measurement point.

    Returns:
        QProgram: The Rabi amplitude-sweep sequence.
    """
    r_wf_I = Square(amplitude=readout_amplitude, duration=readout_duration)
    r_wf_Q = Square(amplitude=0.0, duration=readout_duration)
    d_wf = IQDrag(amplitude=1.0, duration=drive_duration, num_sigmas=drive_sigmas, drag_coefficient=drag_coefficient)

    if ringup_time > 0:
        ringup_wf = Square(amplitude=readout_amplitude, duration=ringup_time)

    weights_shape = Square(amplitude=1, duration=readout_duration)

    qp_rabi = QProgram()
    qp_rabi.set_frequency(bus="readout", frequency=readout_if_freq)
    qp_rabi.set_frequency(bus="drive", frequency=drive_if_freq)

    gain = qp_rabi.variable(domain=Domain.Voltage, label="gain")

    with qp_rabi.average(averages):
        with qp_rabi.for_loop(
            variable=gain,
            start=drive_amplitude_sweep[0],
            stop=drive_amplitude_sweep[-1],
            step=drive_amplitude_sweep[1] - drive_amplitude_sweep[0],
        ):
            # AMPLITUDE-DEPENDENT DRAG PULSE
            qp_rabi.set_gain(bus="drive", gain=gain)
            qp_rabi.play(bus="drive", waveform=d_wf)

            qp_rabi.sync()

            # READOUT PULSE
            if ringup_time > 0:
                qp_rabi.play(bus="readout", waveform=ringup_wf)
            qp_rabi.measure(
                bus="readout",
                waveform=IQPair(I=r_wf_I, Q=r_wf_Q),
                weights=IQPair(I=weights_shape, Q=weights_shape),
            )
            qp_rabi.wait(bus="readout", duration=relax_duration)

    return qp_rabi
