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

"""``T2Fit`` against a synthetic Ramsey decay of known dephasing time.

One sweep, one damped cosine, built at test time from the model already on
``FittingClass`` so the recipe sits with the test.
"""

import numpy as np
import pytest

from seqtante_open.experiments.fitting.fit_base import FittingClass
from seqtante_open.experiments.fitting.t2_fit import T2Fit
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

T2_NS = 15_000
"""Dephasing time the data decays with, in ns."""

FRINGE_FREQ = 1.0 / 12_000
"""Ramsey fringe frequency, in 1/ns, i.e. ``b`` of ``FittingClass.sinus_exp``."""


WAIT = np.arange(0, 60_001, 500)


def make_t2_data(
    rng: np.random.Generator, *, wait: np.ndarray = WAIT, t2: float = T2_NS, sigma: float = 0.01
) -> tuple[np.ndarray, dict]:
    """A damped cosine of lifetime ``t2`` (ns) over ``wait``, placed on a rotated axis of the IQ plane.

    ``T2Fit`` rotates the trace back before fitting, so the decay has to survive that
    round trip. Wait times are integer ns, as the sequencer sweeps them.
    """
    signal = FittingClass.sinus_exp(wait, 1.0, FRINGE_FREQ, 0.0, t2, 0.0)
    results = as_iq(signal, rng, sigma=sigma)

    return results, {"wait": loop(wait, units="ns", bus="drive_q1", parameter="t2_elapsed_time")}


def _curve_fit_returning(damped_cosine: list[float], exponential: list[float]):
    """A ``curve_fit`` stub that returns fixed parameters, and an identity covariance, for each of T2Fit's two models."""

    def curve_fit(model, *args, **kwargs):
        params = damped_cosine if model is FittingClass.sinus_exp else exponential
        return np.array(params), np.eye(len(params))

    return curve_fit


class TestT2Fit(FittingTestCase):
    FIT_CLASS = T2Fit
    DATA = make_t2_data
    INIT = {"measurement_id": 1, "target": "q1"}
    PLOTS = ["q1_T2.png"]
    EXPECTED = {
        "T2": pytest.approx(T2_NS, rel=0.05),
        "r_squared": pytest.approx(1.0, abs=0.05),
    }

    def test_save_fitting(self, fit_obj, monkeypatch):
        """``save_fitting`` records ``t2`` (in us) and ``fit_r2``, then writes them to the database."""
        from seqtante_open.outputs import FittedParameter

        fit_obj.fit()
        recorded: dict = {}
        monkeypatch.setattr(
            fit_obj, "add_fitted_parameter", lambda parameter, value: recorded.__setitem__(parameter, value)
        )
        monkeypatch.setattr(fit_obj, "save_fitting_to_database", lambda: recorded.__setitem__("saved", True))

        fit_obj.save_fitting()

        assert recorded[FittedParameter.T2] == pytest.approx(T2_NS / 1000, rel=0.05)
        assert recorded[FittedParameter.FIT_R2] == pytest.approx(1.0, abs=0.05)
        assert recorded["saved"] is True

    def test_save_fitting_refuses_an_unfitted_model(self, fit_obj):
        """Before ``fit()`` there is no T2 to store, so it raises the shared ``FittingClass`` error."""
        with pytest.raises(RuntimeError, match="T2Fit has no fit result"):
            fit_obj.save_fitting()

    def test_flat_trace_leaves_r_squared_undefined(self, fit_obj, monkeypatch):
        """A flat trace (drive off, readout saturated) has no variance: R² is ``nan``, not a ZeroDivisionError.

        The node rejects a ``nan`` R². The trace and ``curve_fit`` are stubbed so the branch is tested
        without depending on the optimiser.
        """
        from seqtante_open.experiments.fitting import t2_fit as t2_fit_module

        monkeypatch.setattr(fit_obj, "arr", np.ones(len(fit_obj.arr)))
        monkeypatch.setattr(
            t2_fit_module.sp, "curve_fit", _curve_fit_returning([0.0, 1e-4, 0.0, 1e4, 1.0], [0.0, -1e-4, 1.0])
        )
        fit_obj.fit()

        assert np.isnan(fit_obj.r_squared)

    def test_a_growing_envelope_is_not_a_physical_decay(self, fit_obj, monkeypatch):
        """``d <= 0`` is a growing envelope: T2 and its uncertainty are left ``nan`` instead of reporting ``|d|``."""
        from seqtante_open.experiments.fitting import t2_fit as t2_fit_module

        # Both models grow: the damped cosine has d < 0 and the exponential a positive rate.
        monkeypatch.setattr(
            t2_fit_module.sp, "curve_fit", _curve_fit_returning([0.1, 1e-4, 0.0, -1e4, 0.0], [0.1, 1e-4, 0.0])
        )
        fit_obj.fit()

        assert not fit_obj.is_physical_decay
        assert np.isnan(fit_obj.T2)
        assert np.isnan(fit_obj.relative_uncertainty)

    def test_a_trace_with_no_detuning_is_fit_as_a_plain_decay(self, fit_obj, monkeypatch):
        """With no detuning the rFFT seeds a frequency from the noise and the damped cosine misreads T2.

        On this trace (noise 0.05) the damped cosine alone put T2 within 20% in 59% of 200 seeds; choosing
        by BIC puts it there in 98%. The rest are noise the cosine fits marginally better.
        """
        wait = np.linspace(0, 60_000, 61)
        monkeypatch.setattr(fit_obj, "wait_values", wait)
        within, plain = 0, 0
        for seed in range(100):
            noise = np.random.default_rng(seed).normal(0.0, 0.05, wait.size)
            monkeypatch.setattr(fit_obj, "arr", np.exp(-wait / T2_NS) + noise)
            fit_obj.fit()
            within += abs(fit_obj.T2 / T2_NS - 1) < 0.2
            # The plain decay is recorded with b set to exactly 0.
            plain += not fit_obj.results["params"][1]

        assert within >= 90
        assert plain >= 90

    def test_a_detuned_trace_keeps_the_damped_cosine(self, fitted):
        assert fitted.results["params"][1] == pytest.approx(FRINGE_FREQ, rel=0.05)
