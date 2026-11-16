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

"""``LandauZenerFit`` normalises the measured signal with the readout levels and plots it (QHC-1808).

The MVP fit does not override ``fit`` (no agreement metric yet, QHC-1921), so the harness exercises
construction and plotting: the measurement is a three-column ``[I, Q, threshold]`` stream over the
annealing-time loop, and the readout levels normalise the rotated signal to a transition probability.
"""

import numpy as np

from seqtante_open.experiments.fitting.landau_zener_fit import LandauZenerFit
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop


def make_landau_zener_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A transition probability swept over the total annealing time, as the LZ stream writes it.

    The experiment streams three columns per annealing time: the two readout quadratures and the
    single-shot threshold. The signal is placed on a rotated IQ axis (what ``rotate_iq`` undoes) and
    the threshold column is carried along, so the fixture matches the real ``(n, 3)`` stream shape.
    """
    annealing_times = np.array([100, 200, 300, 400, 500])
    probability = 0.5 * (1.0 - np.cos(annealing_times / 150.0))
    iq = as_iq(probability, rng, sigma=0.01)  # (n, 2): rotated I/Q
    results = np.concatenate([iq, probability[:, None]], axis=-1)  # (n, 3): I, Q, threshold
    loops = {
        "total_annealing_time": loop(annealing_times, units="ns", bus="flux_z_q1", parameter="Total_annealing_time"),
    }
    return results, loops


class TestLandauZenerFit(FittingTestCase):
    FIT_CLASS = LandauZenerFit
    DATA = make_landau_zener_data
    INIT = {"measurement_id": 1, "target": "q1", "level_0": 0.0, "level_1": 1.0}
