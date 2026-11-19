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

"""``LorentzianPeakModel`` against a two-tone sweep with a qubit peak at a known IF (QHC-1807)."""

import numpy as np
import pytest

from seqtante_open.experiments.fitting.lorentzian_peak_fit import LorentzianPeakModel, _lorentzian
from seqtante_open.outputs import FittedParameter
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

QUBIT_IF = -1.0e8
"""IF the qubit peak is centred on, in Hz (absolute drive frequency minus the LO, as in the search)."""

FREQ_STEP = 0.4e6
"""Step of the IF sweep (the script's value)."""


def make_lorentzian_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A single qubit peak: a Lorentzian in the two-tone magnitude, centred on ``QUBIT_IF``."""
    frequencies = np.arange(-1.3e8, -0.7e8 + 1, int(FREQ_STEP))
    linewidth = 4.0e6
    # _lorentzian(x, x0, a, gamma, c): a positive peak of linewidth gamma on a flat baseline.
    signal = _lorentzian(frequencies, QUBIT_IF, 0.8 * np.pi * linewidth, linewidth, 0.2)
    results = as_iq(signal, rng, sigma=0.002)
    loops = {"IF_frequency": loop(frequencies, units="Hz", bus="drive_q1", parameter="IF_frequency")}
    return results, loops


class TestLorentzianPeakModel(FittingTestCase):
    FIT_CLASS = LorentzianPeakModel
    DATA = make_lorentzian_data
    INIT = {"measurement_id": 1, "target": "q1"}
    EXPECTED = {
        "results.fitted_if": pytest.approx(QUBIT_IF, abs=FREQ_STEP),
        "results.accepted": True,
        "results.r_squared": lambda r: r > 0.9,
    }

    def test_save_fitting(self, environment, fitted) -> None:
        """``save_fitting`` writes the fitted drive IF and the fit R² under their QHC-1497 keys, once."""
        environment.add_fitting.reset_mock()
        fitted.save_fitting()
        assert environment.add_fitting.call_count == 1
        _, kwargs = environment.add_fitting.call_args
        parameters = kwargs["parameters"]
        assert parameters[FittedParameter.DRIVE_IF.value] == pytest.approx(QUBIT_IF, abs=FREQ_STEP)
        assert FittedParameter.FIT_R2.value in parameters
