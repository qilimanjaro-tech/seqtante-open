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

"""QProgram helpers, copied from ``qilitools.qprogram.utils``."""

import numpy as np
from scipy.special import erf


def smooth_ringup_wf(
    duration: int, n_sigmas: float = 4, amplitude: float = 1
) -> tuple[
    np.ndarray[tuple, np.dtype[np.float64]],
    np.ndarray[tuple, np.dtype[np.float64]],
    np.ndarray[tuple, np.dtype[np.signedinteger[np._typing._32Bit]]],
]:
    t = np.array(range(duration // 4 + 1), dtype=np.int32)

    tau = (t[-1] + t[0]) / 2
    T = t - tau
    sigma = (T[-1] - T[0]) / (2 * n_sigmas)
    C = erf(n_sigmas / np.sqrt(2)) - erf(-n_sigmas / np.sqrt(2))
    WF: np.ndarray[tuple, np.dtype[np.float64]] = np.round(
        erf(T / (np.sqrt(2) * sigma)) / C - erf(-n_sigmas / np.sqrt(2)) / C, decimals=4
    )

    dWF: np.ndarray[tuple, np.dtype[np.float64]] = np.round(
        np.exp(-1 * T * T / (2 * sigma * sigma)) * (1 / np.sqrt(2 * np.pi * sigma)) * (1 / C), decimals=4
    )

    return WF * amplitude, dWF * amplitude, t


def multi_wait_for_trigger(qp, bus, total_duration):
    """Creates a series of waits after a wait trigger based on a maximum value of wait."""
    MAX_WAIT = 20_000
    if total_duration > MAX_WAIT:
        qp.wait_trigger(bus=bus, duration=MAX_WAIT)
        remaining = total_duration - MAX_WAIT
        while remaining > MAX_WAIT:
            qp.wait(bus=bus, duration=MAX_WAIT)
            remaining -= MAX_WAIT
        qp.wait(bus=bus, duration=remaining)
    else:
        qp.wait_trigger(bus=bus, duration=total_duration)
    qp.sync()
