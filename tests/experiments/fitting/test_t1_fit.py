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

"""``T1Fit`` against a synthetic decay of known lifetime.

The plainest case there is, and the reference to copy: one sweep, one exponential,
data checked into ``data/`` and regenerated from the builder below.
"""

import numpy as np
import pytest

from seqtante_open.experiments.fitting.fit_base import FittingClass
from seqtante_open.experiments.fitting.t1_fit import T1Fit
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

T1_NS = 12_400
"""Relaxation time the data decays with, in ns."""

DECAY_RATE = -1.0 / T1_NS
"""``B`` of ``FittingClass.exponential``, which is what curve_fit reports."""


def make_t1_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Exponential relaxation, placed on a rotated axis of the IQ plane.

    ``T1Fit`` rotates the trace back before fitting, so the decay constant has to
    survive that round trip. Wait times are integer ns, as the sequencer sweeps them.
    """
    wait = np.arange(0, 40_001, 500)
    decay = FittingClass.exponential(wait, 1.0, DECAY_RATE, 0.0)
    results = as_iq(decay, rng, sigma=0.01)

    return results, {"wait": loop(wait, units="ns", bus="drive_q1", parameter="duration")}


class TestT1Fit(FittingTestCase):
    FIT_CLASS = T1Fit
    DATA = "t1_fit.h5"
    BUILDER = make_t1_data
    INIT = {"measurement_id": 1, "target": "q1"}
    PLOTS = ["q1_T1.png"]
    EXPECTED = {
        "results.1": pytest.approx(DECAY_RATE, rel=0.02),
        "T1": pytest.approx(T1_NS, rel=0.02),
    }
