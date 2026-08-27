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

"""``FluxoniumTwoToneModel`` against a qubit peak at a known IF.

Like its single-tone sibling, this fit plots with plotly and reads the drive LO
off a platform built from ``measurement.platform_before`` (``lo`` is left unset),
so the ``drive_q1`` alias its loop metadata names has to exist in the test
runcard.

The fit is a Lorentzian on the rotated signal quadrature, so it can land between
sweep points; ``FREQ_STEP`` is a generous tolerance rather than the resolution.
Only ``signal`` is asserted on: the orthogonal quadrature holds noise by
construction, so what a Lorentzian makes of it is meaningless.
"""

import numpy as np
import pytest

from seqtante_open.experiments.fitting.two_tone_fit import FluxoniumTwoToneModel
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

QUBIT_IF = 4.0e6
"""IF the qubit peak is centred on, in Hz."""

FREQ_STEP = 0.5e6
"""Step of the IF sweep."""


def make_two_tone_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A single qubit peak: a Lorentzian centred on ``QUBIT_IF``.

    The peak lives on one quadrature in the rotated IQ plane, which is what
    ``rotate_iq`` recovers before the Lorentzian fit locates its centre.
    """
    frequencies = np.arange(-30_000_000, 30_000_001, int(FREQ_STEP))
    width = 4.0e6
    detuning = (frequencies - QUBIT_IF) / (0.5 * width)
    signal = 0.8 / (1.0 + detuning**2)

    results = as_iq(signal, rng, sigma=0.002)

    loops = {
        "frequency": loop(frequencies, units="Hz", bus="drive_q1", parameter="IF_frequency"),
    }
    return results, loops


class TestFluxoniumTwoToneModel(FittingTestCase):
    FIT_CLASS = FluxoniumTwoToneModel
    DATA = make_two_tone_data
    INIT = {"measurement_id": 1, "target": "q1"}
    EXPECTED = {
        "results.signal.fitted_if": pytest.approx(QUBIT_IF, abs=FREQ_STEP),
        "results.signal.r_squared": lambda r: r > 0.9,
    }
