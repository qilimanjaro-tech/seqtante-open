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


def resonator_spectroscopy_cw(
    freq_start: float,
    freq_stop: float,
    freq_step: float,
    averages: int,
    integration_time: int,
    r_amp: int,
) -> QProgram:
    """Resonator spectroscopy experiment defined in Qprogram for both QM and
    Qblox.
    The experiment does a hardware loop over the readout frequecy.

    Args:
        freq_start (float): initial point of the frequency range.
        freq_stop (float): final point of the frequency range.
        freq_step (float): step size of the frequency range.
        averages (int): number of averages in the experiment.
        r_duration (int): duration of the readout pulse.
        r_amp (float, optional): amplitude of the readout pulse. Defaults
        to 0.1.
        relax_duration (int, optional): resonator relaxation time between repetitions.
        Defaults to 2000.
        ringup_time (int): ringup duration time. Defaults to 0.

    Returns:
        QProgram: defined qprogram for the experiment
    """
    square_wf_I = Square(amplitude=0.0, duration=integration_time)  # we send a 0 amplitude pulse
    square_wf_Q = Square(amplitude=0.0, duration=integration_time)
    weights_shape = Square(amplitude=1, duration=integration_time)

    qp_reson = QProgram()

    freq = qp_reson.variable(domain=Domain.Frequency, label="freq")

    qp_reson.set_offset(bus="readout", offset_path0=r_amp, offset_path1=0)

    # Loop over all frequenciess

    with qp_reson.for_loop(variable=freq, start=freq_start, stop=freq_stop, step=freq_step):
        qp_reson.set_frequency(bus="readout", frequency=freq)
        with qp_reson.average(averages):

            qp_reson.measure(
                bus="readout",
                waveform=IQPair(I=square_wf_I, Q=square_wf_Q),
                weights=IQPair(I=weights_shape, Q=weights_shape),
            )
    qp_reson.set_offset(bus="readout", offset_path0=0, offset_path1=0)

    return qp_reson
