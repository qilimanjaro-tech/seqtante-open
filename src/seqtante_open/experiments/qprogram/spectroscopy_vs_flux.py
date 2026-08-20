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

"""Flux-ramp spectroscopy qprograms, copied from ``qilitools.qprogram.spectroscopy_vs_flux``."""

import math

import numpy as np
from qililab import Arbitrary, Domain, IQPair, QProgram, Square

from seqtante_open.experiments.analysis import sss_from_array
from seqtante_open.experiments.qprogram.utils import multi_wait_for_trigger


def spectroscopy_vs_flux_qdac_ramp(
    if_sweep: np.ndarray,
    averages: int,
    time_per_avg: float,
    r_amp: float,
    ramp_array: np.ndarray,
    minimum_wait_after_step: float,
    stop_ro_before_step: float,
    trigger_channel: int = 2,
):
    freq_start, freq_stop, freq_step = sss_from_array(if_sweep)
    num_points = len(ramp_array)

    time_per_point_us = (time_per_avg * averages * len(if_sweep) + minimum_wait_after_step + stop_ro_before_step) * 1e-3
    time_per_point_rounded_up_us = math.ceil(time_per_point_us)

    extra_wait_ns = (time_per_point_rounded_up_us - time_per_point_us) * 1e3
    wait_after_step = int(minimum_wait_after_step + extra_wait_ns)

    dwell_time_us = time_per_point_rounded_up_us

    square_wf_I = Square(amplitude=0.0, duration=time_per_avg - 4)  # NOTE -4 due to the 4ns gap between acquires
    square_wf_Q = Square(amplitude=0.0, duration=time_per_avg - 4)
    weights_shape = Square(amplitude=1, duration=time_per_avg - 4)
    flux_wf = Arbitrary(ramp_array)

    qp_qdac = QProgram()

    bins_qblox = qp_qdac.variable(label="bins_qblox", domain=Domain.Scalar, type=int)
    freq = qp_qdac.variable(domain=Domain.Frequency, label="freq")

    qp_qdac.set_offset(bus="readout", offset_path0=r_amp, offset_path1=0)

    # Stepped here means it will trigger for every step in the DC-list
    qp_qdac.qdac.play(bus="flux", waveform=flux_wf, dwell=dwell_time_us, stepped=False)
    # NOTE: 10us is somewhat arbitrary but seems consistent
    qp_qdac.set_trigger(bus="flux", duration=10e-6, outputs=trigger_channel, position="step")

    with qp_qdac.for_loop(variable=bins_qblox, start=0, stop=num_points - 1, step=1):
        multi_wait_for_trigger(qp_qdac, "readout", wait_after_step)
        with qp_qdac.for_loop(variable=freq, start=freq_start, stop=freq_stop, step=freq_step):
            qp_qdac.set_frequency(bus="readout", frequency=freq)
            with qp_qdac.average(averages):
                qp_qdac.qblox.play(bus="readout", waveform=IQPair(I=square_wf_I, Q=square_wf_Q), wait_time=4)
                qp_qdac.qblox.acquire("readout", weights=IQPair(I=weights_shape, Q=weights_shape))
            # Here we effectively have a wait time = stop_ro_before_step, when the Qblox will be listening for the trigger

    return qp_qdac
