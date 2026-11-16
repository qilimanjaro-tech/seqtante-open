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

"""T1 saturation QProgram, copied from ``qilitools.qprogram.t1.t1_saturation`` (QHC-1667)."""

import numpy as np
from qililab import Arbitrary, IQPair, QProgram, Square

from .utils import smooth_ringup_wf

#: Fraction of the drive amplitude the ring-down must drop below for the idle wait to start counting.
_RINGDOWN_THRESHOLD = 0.02


def t1_saturation(
    durations_idle: np.ndarray,
    drive_if_freq: float,
    drive_amplitude: float,
    drive_duration: int,
    readout_if_freq: float,
    readout_amplitude: float,
    readout_duration: int,
    relax_duration: int,
    averages: int,
    overlap_time: int,
    duration_ringup_drive: int,
    drive_sigmas: int,
    ringup_time: int,
    drive_ringdown_time: int = 200,
) -> QProgram:
    """Build the T1 saturation sequence: a saturation drive, an idle wait, then a readout, per idle time.

    For each idle duration in ``durations_idle`` the drive bus plays a smooth ring-up, a square
    saturation pulse and a smooth ring-down to excite the qubit; the idle time is counted from where the
    ring-down has dropped below ``_RINGDOWN_THRESHOLD`` of ``drive_amplitude``, after which the readout
    bus measures. An optional ring-up pulse pre-charges the readout resonator, and a relaxation wait
    follows each measurement. Copied from ``qilitools.qprogram.t1.t1_saturation`` (QHC-1667).

    Args:
        durations_idle (np.ndarray): Idle times in ns between the drive and the readout (the T1 sweep).
        drive_if_freq (float): Intermediate frequency of the drive bus, in Hz.
        drive_amplitude (float): Amplitude of the ring-up, saturation and ring-down waveforms.
        drive_duration (int): Duration of the square saturation drive pulse, in ns.
        readout_if_freq (float): Intermediate frequency of the readout bus, in Hz.
        readout_amplitude (float): Amplitude of the readout pulse.
        readout_duration (int): Duration of the readout measurement pulse, in ns.
        relax_duration (int): Relaxation wait after each measurement, in ns.
        averages (int): Number of averages per measurement point.
        overlap_time (int): Overlap in ns between the drive ring-down and the start of the readout.
        duration_ringup_drive (int): Duration of the smooth drive ring-up pulse, in ns.
        drive_sigmas (int): Sigmas of the Gaussian envelope of the smooth ring-up and ring-down.
        ringup_time (int): Duration of the readout ring-up pulse, in ns; 0 applies none.
        drive_ringdown_time (int, optional): Duration of the smooth drive ring-down, in ns. Defaults to 200.

    Returns:
        QProgram: the full T1 saturation sequence.
    """
    r_wf_I = Square(amplitude=readout_amplitude, duration=readout_duration)
    r_wf_Q = Square(amplitude=0.0, duration=readout_duration)
    if ringup_time > 0:
        ringup_wf = Square(amplitude=readout_amplitude, duration=ringup_time)
    weights_shape = Square(amplitude=1, duration=readout_duration)

    smooth_rup, d_smooth_rup, _ = smooth_ringup_wf(duration_ringup_drive, drive_sigmas)
    smooth_rdown, d_smooth_rdown, _ = smooth_ringup_wf(drive_ringdown_time, drive_sigmas)
    smooth_rdown = smooth_rdown[::-1]
    d_smooth_rdown = d_smooth_rdown[::-1]

    # Time until the drive ring-down drops below _RINGDOWN_THRESHOLD of drive_amplitude.
    below = np.hypot(smooth_rdown, d_smooth_rdown) < _RINGDOWN_THRESHOLD
    wait_after_drive = int(np.argmax(below)) if below.any() else drive_ringdown_time

    d_rup_wf_I = Arbitrary(samples=smooth_rup * drive_amplitude)
    d_rup_wf_Q = Arbitrary(samples=d_smooth_rup * drive_amplitude)
    d_rdown_wf_I = Arbitrary(samples=smooth_rdown * drive_amplitude)
    d_rdown_wf_Q = Arbitrary(samples=d_smooth_rdown * drive_amplitude)
    d_wf = Square(amplitude=drive_amplitude, duration=drive_duration)

    qp_t1 = QProgram()
    qp_t1.set_frequency(bus="readout", frequency=readout_if_freq)
    qp_t1.set_frequency(bus="drive", frequency=drive_if_freq)
    with qp_t1.average(averages):
        for duration_idle in durations_idle:
            qp_t1.sync()
            qp_t1.reset_phase(bus="drive")
            qp_t1.play(bus="drive", waveform=IQPair(d_rup_wf_I, d_rup_wf_Q))
            qp_t1.sync()
            qp_t1.play(bus="drive", waveform=d_wf)
            qp_t1.play(bus="drive", waveform=IQPair(d_rdown_wf_I, d_rdown_wf_Q))
            if ringup_time > 0:
                qp_t1.wait(
                    bus="readout",
                    duration=drive_duration + duration_idle + wait_after_drive - ringup_time - overlap_time,
                )
                qp_t1.play(bus="readout", waveform=ringup_wf)
            else:
                qp_t1.wait(bus="readout", duration=drive_duration + duration_idle + wait_after_drive - overlap_time)
            qp_t1.measure(
                bus="readout",
                waveform=IQPair(I=r_wf_I, Q=r_wf_Q),
                weights=IQPair(I=weights_shape, Q=weights_shape),
            )
            qp_t1.sync()
            qp_t1.wait(bus="readout", duration=relax_duration)
    return qp_t1
