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

"""``FluxoniumTwoToneFluxModel`` against a qubit arc with a known sweet spot.

The fit works in two stages, so the data has to satisfy both: every flux row is a
Lorentzian the per-trace fit can find, and the fitted IFs together trace a
parabola whose vertex sits inside the swept flux range.

``lo`` is left unset on purpose, so ``plot()`` reads the drive LO off a platform
built from ``measurement.platform_before``. The bus aliases below therefore have
to exist in the test runcard: ``drive_q1`` resolves to ``QCM-RF1`` at 4.5 GHz
there.

The per-trace Lorentzian is fitted with ``differential_evolution``, which is
stochastic, and the fitted IF it reports is snapped to the frequency grid. Both
put a floor under how tight the tolerances below can be.
"""

import numpy as np
import pytest

from seqtante_open.experiments.fitting.two_tone_vs_flux_fit import FluxoniumTwoToneFluxModel
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

SWEET_SPOT = 0.15
"""Flux bias the qubit frequency is stationary at, in phi_0."""

FLUX_HALF_SPAN = 0.4
FLUX_POINTS = 15

FREQUENCY_STEP = 1.0e6
CURVATURE = -20.0e6 / FLUX_HALF_SPAN**2
"""Hz per phi_0², so the arc drops 20 MHz between the sweet spot and the edge."""

CENTER_TOLERANCE = 0.02
"""Flux, generous next to the 0.057 phi_0 step of the sweep itself."""


def make_two_tone_vs_flux_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A qubit peak tracing a downward parabola in flux, apex at the sweet spot."""
    fluxes = np.linspace(SWEET_SPOT - FLUX_HALF_SPAN, SWEET_SPOT + FLUX_HALF_SPAN, FLUX_POINTS)
    frequencies = np.arange(-30.0e6, 30.0e6 + FREQUENCY_STEP, FREQUENCY_STEP)

    qubit_if = -5.0e6 + CURVATURE * (fluxes - SWEET_SPOT) ** 2
    width = 4.0e6
    detuning = (frequencies[None, :] - qubit_if[:, None]) / (0.5 * width)
    magnitude = 1.0 / (1.0 + detuning**2)

    results = as_iq(magnitude, rng, sigma=0.01)

    loops = {
        "flux": loop(fluxes, units="phi_0", bus="flux_q1_z", parameter="Flux"),
        "IF_frequency": loop(frequencies, units="Hz", bus="drive_q1", parameter="IF_frequency"),
    }
    return results, loops


class TestFluxoniumTwoToneFluxModel(FittingTestCase):
    FIT_CLASS = FluxoniumTwoToneFluxModel
    DATA = "two_tone_vs_flux_fit.h5"
    BUILDER = make_two_tone_vs_flux_data
    INIT = {"measurement_id": 1, "target": "q1"}
    EXPECTED = {
        "center": pytest.approx(SWEET_SPOT, abs=CENTER_TOLERANCE),
        "offset": pytest.approx(-SWEET_SPOT, abs=CENTER_TOLERANCE),
        "r_squared": lambda r_squared: r_squared > 0.99,
    }

    def test_every_trace_is_fitted(self, fitted):
        """Clean data means no row is dropped, so the parabola is accepted."""
        assert fitted.fitted
        assert fitted.mask.all(), f"{(~fitted.mask).sum()} of {len(fitted.mask)} traces were dropped"
        assert np.isfinite(fitted.fitted_if).all()

    def test_fitted_ifs_follow_the_arc(self, fitted):
        """Each row's fitted IF lands on the qubit peak that row was built around."""
        expected = -5.0e6 + CURVATURE * (fitted.fluxes - SWEET_SPOT) ** 2
        np.testing.assert_allclose(fitted.fitted_if, expected, atol=2 * FREQUENCY_STEP)
