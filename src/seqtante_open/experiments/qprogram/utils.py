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


def smooth_ringup_wf(duration: int, n_sigmas: float = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ramp from 0 to 1 over ``duration`` samples through a centred integrated Gaussian, with its derivative.

    The derivative is per sample, so its amplitude scales with ``1/duration``: a ramp built at one
    duration and sliced is not the same as one built at the sliced duration. Copied from
    ``qilitools.qprogram.utils.smooth_ringup_wf``.

    Args:
        duration (int): Number of samples of the resulting waveforms.
        n_sigmas (float, optional): Number of sigmas the ramp spans. Defaults to 4.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: the integrated-Gaussian ramp, its derivative, and a
        range-like array the length of the duration.

    Raises:
        ValueError: if ``duration`` is smaller than 4, or ``n_sigmas`` is not positive.
    """
    if duration < 4:
        raise ValueError(f"'duration' must be at least 4 samples, got {duration}")
    if n_sigmas <= 0:
        raise ValueError(f"'n_sigmas' must be larger than 0, got {n_sigmas}")
    t = np.arange(duration, dtype=np.int32)

    tau = (t[-1] + t[0]) / 2
    centred = t - tau
    sigma = (centred[-1] - centred[0]) / (2 * n_sigmas)
    normalisation = erf(n_sigmas / np.sqrt(2)) - erf(-n_sigmas / np.sqrt(2))
    ramp = np.round(
        erf(centred / (np.sqrt(2) * sigma)) / normalisation - erf(-n_sigmas / np.sqrt(2)) / normalisation,
        decimals=6,
    )
    derivative = np.round(
        2 * np.exp(-1 * centred * centred / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma * normalisation),
        decimals=6,
    )
    return ramp, derivative, t


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
