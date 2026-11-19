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

"""``RabiAmplitudeModel`` against a Rabi amplitude sweep with a known pi amplitude.

The signal is an undamped cosine of the drive amplitude, placed on one quadrature in the IQ plane
(which ``rotate_iq`` recovers). The ground state sits at the first sweep point and the pi amplitude at
the first excited-state extremum half a Rabi period on; the sweep below puts that at ``PI_AMPLITUDE``.
"""

import numpy as np
import pytest

from seqtante_open.experiments.fitting.rabi_fit import RabiAmplitudeModel, _cosine
from seqtante_open.outputs import FittedParameter
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

PI_AMPLITUDE = 0.5
"""Drive amplitude of the first excited-state extremum (the pi pulse)."""


def make_rabi_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A full Rabi oscillation: ground (|0>) at amplitude 0, the pi extremum at ``PI_AMPLITUDE``."""
    amplitudes = np.linspace(0.0, 1.0, 81)
    angular_frequency = np.pi / PI_AMPLITUDE  # half a period lands on PI_AMPLITUDE
    signal = _cosine(amplitudes, offset=0.0, amplitude=1.0, angular_frequency=angular_frequency, phase=0.0)
    results = as_iq(signal, rng, sigma=0.02)
    loops = {"drive_amplitude": loop(amplitudes, units="amp", bus="drive_q1", parameter="drive_amplitude")}
    return results, loops


class TestRabiAmplitudeModel(FittingTestCase):
    FIT_CLASS = RabiAmplitudeModel
    DATA = make_rabi_data
    INIT = {"measurement_id": 1, "target": "q1"}
    EXPECTED = {
        "results.pi_amplitude": pytest.approx(PI_AMPLITUDE, abs=0.03),
        "results.r_squared": lambda r: r > 0.9,
        # The two readout levels are the cosine's extrema, |offset ± amplitude| ~ 1 here.
        "results.level_0": lambda v: abs(abs(v) - 1.0) < 0.1,
        "results.level_1": lambda v: abs(abs(v) - 1.0) < 0.1,
    }

    def test_readout_levels(self, fitted) -> None:
        """The two levels are |0>/|1>, ~2 apart, with |0> the extremum at the first sweep point."""
        level_0, level_0_std, level_1, level_1_std = fitted.readout_levels
        assert abs(level_1 - level_0) == pytest.approx(2.0, abs=0.1)  # the |0>-|1> contrast
        assert level_0_std > 0  # uncertainties propagated from the fit
        assert level_1_std > 0
        # |0> is the extremum the first sweep point (ground state) sits closest to.
        first = fitted.signal[0]
        assert abs(first - level_0) < abs(first - level_1)

    def test_save_fitting(self, environment, fitted) -> None:
        """``save_fitting`` writes the pi amplitude and the fit R² under their QHC-1497 keys, once."""
        environment.add_fitting.reset_mock()
        fitted.save_fitting()
        assert environment.add_fitting.call_count == 1
        _, kwargs = environment.add_fitting.call_args
        parameters = kwargs["parameters"]
        assert parameters[FittedParameter.RABI_AMPLITUDE.value] == pytest.approx(PI_AMPLITUDE, abs=0.03)
        assert FittedParameter.FIT_R2.value in parameters
