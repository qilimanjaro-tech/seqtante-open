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

from qililab import Arbitrary, Domain, IQPair, QProgram, Square

from seqtante_open.experiments.qprogram.utils import smooth_ringup_wf


def two_tone_pulsed_saturation(
    readout_if_freq: float,
    readout_amplitude: float,
    readout_duration: int,
    drive_if_sweep,
    drive_duration: int,
    drive_amplitude: float,
    averages: int,
    duration_ringup_drive: int,
    drive_sigmas: int,
    relax_duration: int,
    overlap_time: int,
    ringup_time: int,
) -> QProgram:
    """Two-tone pulsed-saturation spectroscopy sweeping the drive IF (the reset-point search's measurement).

    For each drive frequency the drive bus plays a smooth ring-up, a square saturation pulse and a
    smooth ring-down, then the readout bus measures (with an optional readout ring-up). Copied from the
    experimentalists' calibration script (``two_tone_frequency_pulsed_saturation``; QHC-1807, fixes go
    back through QHC-1665/QHC-1669).

    Args:
        readout_if_freq (float): Readout intermediate frequency, in Hz.
        readout_amplitude (float): Readout pulse amplitude.
        readout_duration (int): Readout measurement pulse duration, in ns.
        drive_if_sweep (np.ndarray): Linearly spaced drive IF values to sweep, in Hz.
        drive_duration (int): Square saturation drive pulse duration, in ns.
        drive_amplitude (float): Drive amplitude, applied to ring-up, saturation and ring-down.
        averages (int): Number of averages per measurement point.
        duration_ringup_drive (int): Smooth drive ring-up duration, in ns.
        drive_sigmas (int): Number of sigmas of the Gaussian ring-up/ring-down envelopes.
        relax_duration (int): Relaxation wait after each measurement, in ns.
        overlap_time (int): Overlap in ns between the drive ring-down and the readout pulse.
        ringup_time (int): Readout ring-up duration before the measurement, in ns (0 = none).

    Returns:
        QProgram: The two-tone pulsed-saturation sequence.
    """
    r_wf_I = Square(amplitude=readout_amplitude, duration=readout_duration)
    r_wf_Q = Square(amplitude=0.0, duration=readout_duration)
    if ringup_time > 0:
        ringup_wf_I = Square(amplitude=readout_amplitude, duration=ringup_time)
        ringup_wf_Q = Square(amplitude=0.0, duration=ringup_time)

    weights_shape = Square(amplitude=1, duration=readout_duration)

    smooth_rup, d_smooth_rup, _ = smooth_ringup_wf(duration_ringup_drive, drive_sigmas)
    smooth_rdown, d_smooth_rdown, _ = smooth_ringup_wf(200, drive_sigmas)
    smooth_rdown = smooth_rdown[::-1]
    d_smooth_rdown = d_smooth_rdown[::-1]

    d_rup_wf_I = Arbitrary(samples=smooth_rup * drive_amplitude)
    d_rup_wf_Q = Arbitrary(samples=d_smooth_rup * drive_amplitude)
    d_rdown_wf_I = Arbitrary(samples=smooth_rdown * drive_amplitude)
    d_rdown_wf_Q = Arbitrary(samples=d_smooth_rdown * drive_amplitude)

    d_wf_I = Square(amplitude=drive_amplitude, duration=drive_duration)
    d_wf_Q = Square(amplitude=0.0, duration=drive_duration)

    qp_2tone = QProgram()
    qp_2tone.set_frequency(bus="readout", frequency=readout_if_freq)

    freq = qp_2tone.variable(label="frequency", domain=Domain.Frequency)

    with qp_2tone.average(averages):
        with qp_2tone.for_loop(
            variable=freq, start=drive_if_sweep[0], stop=drive_if_sweep[-1], step=drive_if_sweep[1] - drive_if_sweep[0]
        ):
            qp_2tone.set_frequency(bus="drive", frequency=freq)
            qp_2tone.reset_phase(bus="drive")
            qp_2tone.play(bus="drive", waveform=IQPair(I=d_rup_wf_I, Q=d_rup_wf_Q))
            qp_2tone.sync()
            qp_2tone.play(bus="drive", waveform=IQPair(I=d_wf_I, Q=d_wf_Q))
            qp_2tone.play(bus="drive", waveform=IQPair(I=d_rdown_wf_I, Q=d_rdown_wf_Q))
            qp_2tone.wait(bus="readout", duration=drive_duration + 38 - overlap_time)
            if ringup_time > 0:
                qp_2tone.play(bus="readout", waveform=IQPair(I=ringup_wf_I, Q=ringup_wf_Q))
            qp_2tone.measure(
                bus="readout",
                waveform=IQPair(I=r_wf_I, Q=r_wf_Q),
                weights=IQPair(I=weights_shape, Q=weights_shape),
            )
            qp_2tone.sync()
            qp_2tone.wait(bus="readout", duration=relax_duration)

    return qp_2tone


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
