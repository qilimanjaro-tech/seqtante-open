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

from seqtante_open.experiments.fitting.fit_base import FittingClass
from seqtante_open.experiments.fitting.lorentzian_peak_fit import LorentzianPeakModel
from seqtante_open.outputs import FittedParameter
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

QUBIT_IF = -1.0e8
"""IF the qubit peak is centred on, in Hz (absolute drive frequency minus the LO, as in the search)."""

FREQ_STEP = 0.4e6
"""Step of the IF sweep (the script's value)."""


def make_lorentzian_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A single qubit peak: a Lorentzian on one quadrature, centred on ``QUBIT_IF``."""
    frequencies = np.arange(-1.3e8, -0.7e8 + 1, int(FREQ_STEP))
    linewidth = 4.0e6
    # A peak of height 0.4 and FWHM linewidth on a flat baseline.
    signal = FittingClass.lorentzian(frequencies, 0.4, QUBIT_IF, linewidth, 0.2)
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
    # The measurement id is in the filename, so a search's sweeps of one target don't overwrite one file.
    PLOTS = ["Two tone q1 ID 1.png"]

    def test_save_fitting(self, environment, fitted) -> None:
        """``save_fitting`` writes the fitted drive IF and the fit R² under their QHC-1497 keys, once."""
        environment.add_fitting.reset_mock()
        fitted.save_fitting()
        assert environment.add_fitting.call_count == 1
        _, kwargs = environment.add_fitting.call_args
        parameters = kwargs["parameters"]
        assert parameters[FittedParameter.DRIVE_IF.value] == pytest.approx(QUBIT_IF, abs=FREQ_STEP)
        assert FittedParameter.FIT_R2.value in parameters

    def test_save_fitting_refuses_an_unfitted_model(self, fit_obj) -> None:
        """Before ``fit()`` there is no drive IF to store, so it raises the shared ``FittingClass`` error."""
        with pytest.raises(RuntimeError, match="LorentzianPeakModel has no fit result"):
            fit_obj.save_fitting()

    def test_flat_trace_is_rejected_with_an_undefined_r_squared(self, fitted, monkeypatch) -> None:
        """A flat trace has no variance: R² is ``nan``, not a division by zero, and the fit is rejected.

        Every comparison with ``nan`` is False, so without the explicit check a flat trace could be accepted.
        """
        monkeypatch.setattr(fitted, "signal", np.ones(len(fitted.frequencies)))
        fitted.fit()

        assert np.isnan(fitted.r_squared)
        assert not fitted.accepted
        assert fitted.fitted_if is None

    @pytest.mark.parametrize("stored_if", [0.0, -3e8])
    @pytest.mark.parametrize(("height", "scale"), [(1.0, 1.0), (-1.0, 1e-3)], ids=["peak", "small_dip"])
    def test_the_line_is_found_wherever_the_window_sits_and_at_any_scale(
        self, fitted, monkeypatch, stored_if, height, scale
    ) -> None:
        """The start comes from the data, not the window's position or a fixed amplitude (Ely's #45 review).

        The script's start took the width from the sum of the sweep ends, so a window centred on IF 0
        started at zero width and never fit.
        """
        frequencies = stored_if + np.linspace(-5e6, 5e6, 101)
        centre = stored_if + 1.3e6
        noise = np.random.default_rng(0).normal(0.0, scale / 50, frequencies.size)
        signal = scale * (0.5 + FittingClass.lorentzian(frequencies, height, centre, 0.6e6, 0.0)) + noise
        monkeypatch.setattr(fitted, "frequencies", frequencies)
        monkeypatch.setattr(fitted, "signal", signal)

        fitted.fit()

        assert fitted.accepted
        assert fitted.fitted_if == pytest.approx(centre, abs=50e3)
