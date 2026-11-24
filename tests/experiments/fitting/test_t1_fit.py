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
        "r_squared": pytest.approx(1.0, abs=0.05),
    }

    def test_save_fitting(self, fit_obj, monkeypatch):
        """``save_fitting`` records ``t1`` (in us) and ``fit_r2``, then writes them to the database."""
        from seqtante_open.outputs import FittedParameter

        fit_obj.fit()
        recorded: dict = {}
        monkeypatch.setattr(
            fit_obj, "add_fitted_parameter", lambda parameter, value: recorded.__setitem__(parameter, value)
        )
        monkeypatch.setattr(fit_obj, "save_fitting_to_database", lambda: recorded.__setitem__("saved", True))

        fit_obj.save_fitting()

        assert recorded[FittedParameter.T1] == pytest.approx(T1_NS / 1000, rel=0.02)
        assert recorded[FittedParameter.FIT_R2] == pytest.approx(1.0, abs=0.05)
        assert recorded["saved"] is True

    def test_save_fitting_refuses_an_unfitted_model(self, fit_obj):
        """Before ``fit()`` there is no T1 to store, so it raises the shared ``FittingClass`` error."""
        with pytest.raises(RuntimeError, match="T1Fit has no fit result"):
            fit_obj.save_fitting()

    def test_is_physical_decay(self, fitted):
        """A clean decay fits to rate < 0, so it reads as a physical decay with a small relative uncertainty."""
        assert fitted.is_physical_decay is True
        assert fitted.relative_uncertainty < 0.1

    def test_relative_uncertainty(self, fit_obj, monkeypatch):
        """A fit that lands on rate >= 0: T1 and its uncertainty are left non-finite and the fit is flagged.

        The node turns that flag into a failed target instead of scoring a negative or infinite lifetime
        (QHC-1803). ``relative_uncertainty`` is ``nan`` there, which the node also rejects. ``curve_fit``
        is stubbed to a non-negative rate so the branch is tested without depending on the optimiser.
        """
        from seqtante_open.experiments.fitting import t1_fit as t1_fit_module

        monkeypatch.setattr(
            t1_fit_module.sp, "curve_fit", lambda *args, **kwargs: (np.array([1.0, 1.0e-4, 0.0]), np.eye(3))
        )
        fit_obj.fit()

        assert fit_obj.rate >= 0.0
        assert fit_obj.is_physical_decay is False
        assert np.isnan(fit_obj.T1)
        assert np.isnan(fit_obj.t1_uncertainty)
        assert np.isnan(fit_obj.relative_uncertainty)

    def test_flat_trace_leaves_r_squared_undefined(self, fit_obj, monkeypatch):
        """A flat trace (drive off, readout saturated) has no variance: R² is ``nan``, not a ZeroDivisionError.

        The node rejects a ``nan`` R² (its check is ``not r_squared >= min_r2``). The trace and ``curve_fit``
        are stubbed so the branch is tested without depending on the optimiser.
        """
        from seqtante_open.experiments.fitting import t1_fit as t1_fit_module

        monkeypatch.setattr(fit_obj, "arr", np.ones(len(fit_obj.arr)))
        monkeypatch.setattr(
            t1_fit_module.sp, "curve_fit", lambda *args, **kwargs: (np.array([0.0, -1.0e-4, 1.0]), np.eye(3))
        )
        fit_obj.fit()

        assert np.isnan(fit_obj.r_squared)
