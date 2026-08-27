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

"""``FluxoniumSingleToneModel`` against a resonator dip at a known IF.

Like the flux-swept sibling, this fit plots with plotly and reads the readout LO
off a platform built from ``measurement.platform_before`` (``lo`` is left unset),
so the ``readout_q1`` alias its loop metadata names has to exist in the test
runcard, where it resolves to ``QRM-RF1`` at 6.3 GHz.

The fit takes the deepest swept point rather than fitting a curve, so the dip is
placed exactly on a sweep point and ``FREQ_STEP`` is the tolerance it can be
found to.
"""

import numpy as np
import pytest

from seqtante_open.experiments.fitting.single_tone_fit import FluxoniumSingleToneModel
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

RESONANCE_IF = -5.0e6
"""IF the resonator dip is centred on, in Hz."""

FREQ_STEP = 1.0e6
"""Step of the IF sweep the fit reports its answer on."""


def make_single_tone_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A single resonator dip: a Lorentzian in |S21| centred on ``RESONANCE_IF``.

    Only the magnitude matters to this fit, but the trace is still packed onto a
    rotated IQ axis the way the acquisition writes it.
    """
    frequencies = np.arange(-30_000_000, 30_000_001, int(FREQ_STEP))
    width = 4.0e6
    detuning = (frequencies - RESONANCE_IF) / (0.5 * width)
    magnitude = 1.0 - 0.8 / (1.0 + detuning**2)

    results = as_iq(magnitude, rng, sigma=0.002)

    loops = {
        "frequency": loop(frequencies, units="Hz", bus="readout_q1", parameter="IF_frequency"),
    }
    return results, loops


class TestFluxoniumSingleToneModel(FittingTestCase):
    FIT_CLASS = FluxoniumSingleToneModel
    DATA = make_single_tone_data
    INIT = {"measurement_id": 1, "target": "q1"}
    EXPECTED = {
        "results.signal.fitted_if": pytest.approx(RESONANCE_IF, abs=FREQ_STEP),
    }
