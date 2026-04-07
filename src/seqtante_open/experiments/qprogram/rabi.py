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

def rabi_amp_square_drive(
    amp_start: float,
    amp_stop: float,
    amp_step: float,
    averages: int,
    readout_amplitude: float,
    readout_duration: int,
    drive_duration: int,
    relax_duration: int,
    ringup_time: int = 0,
) -> QProgram:
    """Generates a QProgram for the "rabi amplitude" experiment.

    Args:
        averages (int): Averages.
        readout_amplitude (float): Calibrated amplitude of a readout waveform for the qubit.
        readout_duration (int): Calibrated duration of a readout waveform for the qubit.
        drag_amplitude_sweep (np.ndarray): Amplitude values for the rabi experiment.
        drag_duration (int): Calibrated duration of a pi rotation drag waveform for the qubit.
        drag_sigmas (float): Calibrated sigmas of a pi rotation drag waveform for the qubit.
        drag_coefficient (float): Calibrated drag coefficient of a pi rotation drag waveform for the qubit.
        relax_duration (int): Time needed to ensure the qubit loses it state.
        ringup_time (int, optional): Duration of the pulse used to exite the readout resonator. Defaults to 0.

    Returns:
        QProgram
    """
    r_wf_I = Square(amplitude=readout_amplitude, duration=readout_duration)
    r_wf_Q = Square(amplitude=0.0, duration=readout_duration)
    if ringup_time > 0:
        ringup_wf_I = Square(amplitude=readout_amplitude, duration=ringup_time)
        ringup_wf_Q = Square(amplitude=0.0, duration=ringup_time)

    weights_shape = Square(amplitude=1, duration=readout_duration)
    
    d_wf_I = Square(amplitude=readout_amplitude, duration=drive_duration)
    d_wf_Q = Square(amplitude=0.0, duration=drive_duration)
    d_wf = IQPair(I=d_wf_I, Q=d_wf_Q)

    qp_rabi = QProgram()

    gain = qp_rabi.variable(domain=Domain.Voltage, label="gain")

    with qp_rabi.average(averages):
        # Loop over all amplitudes
        qp_rabi.set_gain(bus="readout", gain=1)
        with qp_rabi.for_loop(
            variable=gain,
            start=amp_start,
            stop=amp_stop,
            step=amp_step,
        ):
            # AMPLITUDE DEPENDENT DRAG PULSE
            qp_rabi.set_gain(bus="drive", gain=gain)
            qp_rabi.play(bus="drive", waveform=d_wf)

            qp_rabi.sync()

            # READOUT PULSE
            if ringup_time > 0:
                qp_rabi.play(
                    bus="readout",
                    waveform=IQPair(I=ringup_wf_I, Q=ringup_wf_Q),
                )
            qp_rabi.measure(
                bus="readout",
                waveform=IQPair(I=r_wf_I, Q=r_wf_Q),
                weights=IQPair(I=weights_shape, Q=weights_shape),
            )
            qp_rabi.wait(bus="readout", duration=relax_duration)

    return qp_rabi
