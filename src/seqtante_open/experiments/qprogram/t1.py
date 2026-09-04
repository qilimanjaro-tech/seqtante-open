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

from typing import Iterable

from loguru import logger
from qililab.qprogram import QProgram
from qililab.waveforms import Arbitrary, IQPair, Square

from .utils import smooth_ringup_wf


def t1_saturation(
    wait_sweep: Iterable[int],
    drive_if: float,
    drive_amplitude: float,
    drive_step_duration: int,
    readout_if: float,
    readout_amplitude: float,
    readout_duration: int,
    relax_duration: int,
    averages: int,
    overlap: int = 0,
    drive_rise_time: int = 2_000,
    n_sigmas: int = 4,
    q_relative_amplitude: float = 0,
) -> QProgram:
    """T1 measured via a saturation pulse, software-unrolling the idle-time sweep into the
    compiled qprogram (one drive/readout block per entry of `wait_sweep`).

    Args:
        wait_sweep: Idle (wait) durations to sweep over.
        drive_if: Intermediate frequency (Hz) for the drive tone.
        drive_amplitude: Amplitude of the drive pulse.
        drive_step_duration: Duration (ns) of the drive step.
        readout_if: Intermediate frequency (Hz) for the readout tone.
        readout_amplitude: Amplitude of the readout pulse.
        readout_duration: Duration of the readout pulse.
        relax_duration: Cooldown time (ns) between repetitions. Defaults to 150_000.
        averages: Number of hardware averages. Defaults to 200.
        overlap: Overlap (ns) between the drive and readout windows; capped to
            `min(readout_duration, drive_step_duration - 4)` if it would otherwise overrun.
            Defaults to 0.
        drive_rise_time: Rise time (ns) of the drive pulse envelope. Defaults to 2_000.
        n_sigmas: Number of Gaussian sigmas in the drive pulse rise/fall. Defaults to 4.
        q_relative_amplitude: Relative amplitude of the drive pulse's Q component. Defaults to 0.

    Returns:
        QProgram: defined qprogram for the experiment.
    """
    max_overlap = int(min(readout_duration, drive_step_duration - 4))
    if overlap > max_overlap:
        logger.warning(f"Positive overlap capped to max_overlap = {max_overlap:.0f} ns")
    eff_overlap = min(overlap, max_overlap)

    smooth_rup, d_smooth_rup, _ = smooth_ringup_wf(duration=4 * drive_rise_time, n_sigmas=n_sigmas, amplitude=drive_amplitude)
    # The shape of the pulses is introduced here
    wf_I = Square(amplitude=readout_amplitude, duration=readout_duration)
    wf_Q = Square(amplitude=0, duration=readout_duration)
    wf_pulse = IQPair(I=wf_I, Q=wf_Q)

    weights_shape = Square(amplitude=1, duration=readout_duration)  # uniform weights
    weights = IQPair(I=weights_shape, Q=weights_shape)

    wf_i_drive_rup = Arbitrary(samples=smooth_rup)
    wf_q_drive_rup = Arbitrary(samples=q_relative_amplitude * d_smooth_rup)
    wf_drive_rup = IQPair(wf_i_drive_rup, wf_q_drive_rup)

    qp = QProgram()

    qp.set_frequency(bus="readout", frequency=readout_if)
    qp.set_frequency(bus="drive", frequency=drive_if)

    with qp.average(averages):
        for duration_idle in wait_sweep:
            qp.reset_phase(bus="drive")
            qp.reset_phase(bus="readout")

            qp.play(bus="drive", waveform=wf_drive_rup)
            qp.set_offset(bus="drive", offset_path0=drive_amplitude, offset_path1=0)
            qp.sync()

            qp.wait(bus="drive", duration=drive_step_duration)
            qp.set_offset(bus="drive", offset_path0=0, offset_path1=0)
            qp.wait(bus="readout", duration=int(drive_step_duration - eff_overlap + duration_idle))
            qp.measure(bus="readout", waveform=wf_pulse, weights=weights)

            qp.sync()
            qp.wait(bus="readout", duration=relax_duration)
            qp.sync()

    return qp
