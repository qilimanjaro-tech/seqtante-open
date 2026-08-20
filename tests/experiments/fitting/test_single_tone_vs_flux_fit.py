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

"""``FluxoniumSingleToneFluxModel`` against a resonator arc with a known sweet spot.

The awkward case, and the reason the harness stays backend-agnostic: this fit
plots with plotly rather than matplotlib, and it writes to the database from
inside ``plot()`` rather than ``fit()``.

``lo`` is left unset on purpose, so ``plot()`` goes and reads the readout LO off
a platform built from ``measurement.platform_before``. The bus aliases below
therefore have to exist in the test runcard: ``readout_q1`` resolves to
``QRM-RF1`` at 6.3 GHz there.
"""

import numpy as np
import pytest

from seqtante_open.experiments.fitting.single_tone_vs_flux_fit import FluxoniumSingleToneFluxModel
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

SWEET_SPOT = 0.15
"""Flux bias the resonator response is symmetric about, in V."""

FLUX_HALF_SPAN = 0.5
FLUX_POINTS = 41

FLUX_STEP = 2 * FLUX_HALF_SPAN / (2 * FLUX_POINTS - 2)
"""Step of the auto-convolution axis the fit reports its answer on."""


def make_single_tone_vs_flux_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A resonator dip tracing a parabola in flux, centred on the sweet spot.

    Every cut of the image at fixed frequency is symmetric about ``SWEET_SPOT``,
    because the resonance depends on ``(flux - sweet_spot) ** 2``. That symmetry
    is what the auto-convolution inside the fit locates.
    """
    fluxes = np.linspace(SWEET_SPOT - FLUX_HALF_SPAN, SWEET_SPOT + FLUX_HALF_SPAN, FLUX_POINTS)
    frequencies = np.arange(-50_000_000, 10_000_001, 1_000_000)

    resonance = -10.0e6 - 30.0e6 * ((fluxes - SWEET_SPOT) / FLUX_HALF_SPAN) ** 2
    width = 4.0e6
    detuning = (frequencies[None, :] - resonance[:, None]) / (0.5 * width)
    magnitude = 1.0 - 0.8 / (1.0 + detuning**2)

    results = as_iq(magnitude, rng, sigma=0.002)

    loops = {
        "flux": loop(fluxes, units="V", bus="flux_q1_z", parameter="Flux"),
        "frequency": loop(frequencies, units="Hz", bus="readout_q1", parameter="IF_frequency"),
    }
    return results, loops


class TestFluxoniumSingleToneFluxModel(FittingTestCase):
    FIT_CLASS = FluxoniumSingleToneFluxModel
    DATA = "single_tone_vs_flux_fit.h5"
    BUILDER = make_single_tone_vs_flux_data
    INIT = {"measurement_id": 1, "target": "q1"}
    EXPECTED = {
        # The auto-convolution reports a point on its own axis, so one step of that
        # axis is the tightest tolerance that means anything. The fit lands dead on.
        "center": pytest.approx(SWEET_SPOT, abs=FLUX_STEP),
        "offset": pytest.approx(-SWEET_SPOT, abs=FLUX_STEP),
    }
