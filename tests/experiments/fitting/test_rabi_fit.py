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

from seqtante_open.experiments.fitting import RabiAmplitudeModel, cosfunc
from seqtante_open.outputs import FittedParameter
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

PI_AMPLITUDE = 0.5
"""Drive amplitude of the first excited-state extremum (the pi pulse)."""


def make_rabi_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A full Rabi oscillation: ground (|0>) at amplitude 0, the pi extremum at ``PI_AMPLITUDE``."""
    amplitudes = np.linspace(0.0, 1.0, 81)
    angular_frequency = np.pi / PI_AMPLITUDE  # half a period lands on PI_AMPLITUDE
    signal = cosfunc(amplitudes, A=1.0, omega=angular_frequency, offset=0.0, phase_offset=0.0)
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

    def test_save_fitting_refuses_an_unfitted_model(self, fit_obj) -> None:
        """Before ``fit()`` there is no pi amplitude to store, so it raises the shared ``FittingClass`` error."""
        with pytest.raises(RuntimeError, match="RabiAmplitudeModel has no fit result"):
            fit_obj.save_fitting()

    def test_flat_trace_leaves_r_squared_undefined(self, fitted, monkeypatch) -> None:
        """A flat trace (drive off, readout saturated) has no variance: R² is ``nan``, which the node rejects."""
        monkeypatch.setattr(fitted, "signal", np.ones(len(fitted.signal)))
        fitted.fit()

        assert np.isnan(fitted.r_squared)

    @staticmethod
    def _refit(model: RabiAmplitudeModel, monkeypatch, amplitudes: np.ndarray, signal: np.ndarray) -> None:
        """Refit ``model`` on a noiseless trace (the rotated signal, as the fit sees it)."""
        monkeypatch.setattr(model, "amplitudes", amplitudes)
        monkeypatch.setattr(model, "signal", signal)
        model.fit()

    def test_a_sweep_over_several_periods_reports_the_first_pi(self, fitted, monkeypatch) -> None:
        """Every excited extremum fits equally well; the pi amplitude is the first one, not the 3π one."""
        amplitudes = np.linspace(0.02, 1.6, 80)
        self._refit(fitted, monkeypatch, amplitudes, np.cos(np.pi * amplitudes / PI_AMPLITUDE))

        assert fitted.pi_amplitude == pytest.approx(PI_AMPLITUDE, abs=0.01)

    def test_a_trace_starting_at_its_minimum_still_fits(self, fitted, monkeypatch) -> None:
        """``rotate_iq`` fixes the trace's sign arbitrarily, so |0> may be the bottom of the trace."""
        amplitudes = np.linspace(0.0, 1.0, 81)
        self._refit(fitted, monkeypatch, amplitudes, -np.cos(np.pi * amplitudes / PI_AMPLITUDE))

        assert fitted.r_squared > 0.99
        assert fitted.pi_amplitude == pytest.approx(PI_AMPLITUDE, abs=0.01)
        level_0, _, level_1, _ = fitted.readout_levels
        assert level_0 == pytest.approx(-1.0, abs=0.01)  # the value at zero drive
        assert level_1 == pytest.approx(1.0, abs=0.01)

    def test_ground_is_the_fit_at_zero_drive_not_the_first_sample(self, fitted, monkeypatch) -> None:
        """A window starting past the pi amplitude still takes |0> from zero drive and the first pi after it."""
        pi = 0.4
        amplitudes = np.linspace(0.3, 1.0, 40)  # starts near the excited extremum
        self._refit(fitted, monkeypatch, amplitudes, np.cos(np.pi * amplitudes / pi))

        assert fitted.pi_amplitude == pytest.approx(pi, abs=0.01)
        level_0, _, level_1, _ = fitted.readout_levels
        assert level_0 == pytest.approx(1.0, abs=0.01)
        assert level_1 == pytest.approx(-1.0, abs=0.01)

    def test_a_sweep_ending_before_pi_puts_it_past_the_window(self, fitted, monkeypatch) -> None:
        """A partial cosine still fits well; the pi amplitude lands past the window for the node to reject."""
        amplitudes = np.linspace(0.02, 0.8, 40)
        self._refit(fitted, monkeypatch, amplitudes, np.cos(np.pi * amplitudes / 1.0))

        assert fitted.r_squared > 0.99
        assert fitted.pi_amplitude > amplitudes.max()

    @pytest.mark.parametrize("seed", range(20))
    def test_a_noisy_trace_of_under_two_periods_finds_the_right_frequency(self, fitted, monkeypatch, seed) -> None:
        """The example tree's window holds ~1.4 periods for pi = 0.28: an FFT-peak guess sat a bin too high."""
        pi = 0.28
        amplitudes = np.linspace(0.02, 0.8, 40)
        noise = np.random.default_rng(seed).normal(0.0, 0.15, amplitudes.size)
        self._refit(fitted, monkeypatch, amplitudes, np.cos(np.pi * amplitudes / pi) + noise)

        assert fitted.pi_amplitude == pytest.approx(pi, abs=0.02)
        assert fitted.r_squared > 0.8

    def test_a_fit_without_a_covariance_has_undefined_uncertainties(self, fitted, monkeypatch) -> None:
        """When lmfit cannot estimate the covariance the uncertainties are NaN, never a fake 0."""
        from lmfit import Model

        from seqtante_open.experiments.fitting import rabi_fit

        class NoCovarianceModel(Model):
            def fit(self, *args, **kwargs):
                result = super().fit(*args, **kwargs)
                for parameter in result.params.values():
                    parameter.stderr = None
                return result

        monkeypatch.setattr(rabi_fit, "Model", NoCovarianceModel)
        fitted.fit()

        _, level_0_std, _, level_1_std = fitted.readout_levels
        assert np.isnan(level_0_std)
        assert np.isnan(level_1_std)
        assert np.isnan(fitted.results["pi_amplitude_std"])
