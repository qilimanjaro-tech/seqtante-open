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

"""``LandauZenerFit`` scores a phi_x-by-T surface against the closed-form theory curve (ADR-0012, QHC-1921).

The harness runs the fit on a surface placed on the theory curve with a little noise, so every variant of
S is close to 1. The score's pieces are checked on their own below: the weights, the references, the
decay fit and the noise estimate.
"""

import math

import numpy as np
import pytest

from seqtante_open.experiments.fitting import landau_zener_fit as lz
from seqtante_open.experiments.fitting.landau_zener_fit import LandauZenerFit
from seqtante_open.outputs import FittedParameter
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

PHI_X = np.array([0.20, 0.225, 0.25])
DURATIONS = np.geomspace(5, 400, 25).round()
MIN_GAP = [0.05, 0.0572, 0.065]
DETUNING_SPAN = [1.9, 1.887, 1.85]
NOISE = 0.01


def make_landau_zener_data(rng: np.random.Generator, *, sigma: float = NOISE) -> tuple[np.ndarray, dict]:
    """The theory surface with noise, as the LZ stream writes it: ``[I, Q, threshold]`` per (phi_x, T).

    The readout levels are 0 and 1, so the rotated signal is the transition probability itself.
    """
    probability = lz.lz_probability(DURATIONS, np.array(MIN_GAP), np.array(DETUNING_SPAN))
    iq = as_iq(probability, rng, sigma=sigma)  # (phi_x, T, 2): rotated I/Q
    results = np.concatenate([iq, probability[..., None]], axis=-1)
    loops = {
        lz.PHI_X_LOOP: loop(PHI_X, units="phi_0", bus="fast_flux_x_q1", parameter=lz.PHI_X_LOOP),
        lz.DURATION_LOOP: loop(DURATIONS, units="ns", bus="fast_flux_z_q1", parameter=lz.DURATION_LOOP),
    }
    return results, loops


class TestLandauZenerFit(FittingTestCase):
    FIT_CLASS = LandauZenerFit
    DATA = make_landau_zener_data
    INIT = {
        "measurement_id": 1,
        "target": "q1",
        "level_0": 0.0,
        "level_1": 1.0,
        "min_gap": MIN_GAP,
        "detuning_span": DETUNING_SPAN,
    }
    EXPECTED = {
        "results.scores.rmse.value": lambda s: s > 0.95,
        "results.scores.shape.value": lambda s: s > 0.9,
        "results.scores.rmse_excess.value": lambda s: s > 0.95,
        "results.noise": pytest.approx(NOISE, rel=0.3),
        "results.fit_r2": lambda r: r > 0.99,
        "results.gap_ratio": lambda ratios: np.allclose(ratios, 1.0, atol=0.05),
    }
    PLOTS = ["landau_zener_q1 ID 1.png"]

    def test_save_fitting(self, environment, fitted) -> None:
        """The database keys are ``score_s`` (the benchmark S) and ``rmse`` (its distance D), as Natàlia decided."""
        environment.add_fitting.reset_mock()
        fitted.save_fitting()
        _, kwargs = environment.add_fitting.call_args
        parameters = kwargs["parameters"]
        assert set(parameters) == {FittedParameter.SCORE_S.value, FittedParameter.RMSE.value}
        assert parameters[FittedParameter.SCORE_S.value] == pytest.approx(fitted.score.value, abs=1e-6)
        assert parameters[FittedParameter.RMSE.value] == pytest.approx(fitted.score.distance, abs=1e-6)

    def test_the_benchmark_score_is_the_rmse_variant(self, fitted) -> None:
        assert fitted.score is fitted.results["scores"]["rmse"]

    def test_one_level_and_gap_per_phi_x_is_required(self, environment) -> None:
        with pytest.raises(ValueError, match="3 phi_x were measured, but 2 min_gap"):
            LandauZenerFit(**{**self.INIT, "min_gap": MIN_GAP[:2]})

    def test_equal_readout_levels_cannot_normalise(self, environment) -> None:
        with pytest.raises(ValueError, match="readout levels are equal"):
            LandauZenerFit(**{**self.INIT, "level_1": 0.0})

    def test_fit_r2_comes_from_the_decay_fits_not_the_noise_fallback(self, environment) -> None:
        """sigma_hat falls back to the scatter where a decay is unresolved; fit_r2 stays the fits' own R²."""
        model = LandauZenerFit(**self.INIT)
        surface = model.transition_probability.copy()
        surface[0] = 0.5 + np.random.default_rng(11).normal(0, 0.02, surface.shape[1])  # no decay to see
        model.transition_probability = surface
        fits = [lz.fit_decay(model.durations, row) for row in surface]
        assert not fits[0].resolved
        assert all(fit.resolved for fit in fits[1:])

        results = model.fit()

        assert results["fit_r2"] == pytest.approx(1 - sum(f.ss_res for f in fits) / sum(f.ss_tot for f in fits))
        dof = model.durations.size - 3
        noise_ss = lz.successive_scatter(surface[0]) * dof + sum(f.ss_res for f in fits[1:])
        assert results["noise"] == pytest.approx(math.sqrt(noise_ss / (dof * len(fits))))

    def test_a_wrong_gap_lowers_s_and_shows_in_the_gap_ratio(self, environment) -> None:
        """With Delta 20% too large in the data, the gap ratio reads it (tau ~ 1/Delta^2) and S drops."""
        measured = lz.lz_probability(DURATIONS, 1.2 * np.array(MIN_GAP), np.array(DETUNING_SPAN))
        model = LandauZenerFit(**self.INIT)
        model.transition_probability = measured

        results = model.fit()

        np.testing.assert_allclose(results["gap_ratio"], 1.2, rtol=0.02)
        assert results["scores"]["rmse"].value < 0.9


# ------------------------------ the score's pieces ------------------------------


def test_the_theory_curve_is_the_closed_form():
    probability = lz.lz_probability(np.array([0.0, 100.0]), np.array([0.05]), np.array([2.0]))
    assert probability[0, 0] == pytest.approx(1.0)
    assert probability[0, 1] == pytest.approx(math.exp(-(math.pi**2) * 0.05**2 * 100 / 2.0))


@pytest.mark.parametrize(
    ("x", "expected"),
    [([0.3], [1.0]), ([0.0, 1.0], [0.5, 0.5]), ([0.0, 1.0, 3.0], [1 / 6, 1 / 2, 1 / 3])],
    ids=["one", "two", "uneven"],
)
def test_trapezoid_weights(x, expected):
    np.testing.assert_allclose(lz.trapezoid_weights(np.array(x)), expected)


def test_trapezoid_weights_need_increasing_points():
    with pytest.raises(ValueError, match="strictly increasing"):
        lz.trapezoid_weights(np.array([0.2, 0.1]))


def test_each_phi_x_weighs_by_its_trapezoid_weight_only():
    """``W = v_k w_i``: summing over T leaves ``v_k``, whatever the number of durations (ADR-0012)."""
    weights = lz.surface_weights(PHI_X, DURATIONS)
    np.testing.assert_allclose(weights.sum(axis=1), lz.trapezoid_weights(PHI_X))
    assert weights.sum() == pytest.approx(1.0)


def test_the_uniform_reference_is_reproducible_and_scores_uniform_data_near_zero():
    theory = lz.lz_probability(DURATIONS, np.array(MIN_GAP), np.array(DETUNING_SPAN))
    weights = lz.surface_weights(PHI_X, DURATIONS)
    reference = lz.uniform_reference(theory, weights)

    assert lz.uniform_reference(theory, weights) == reference  # fixed seed
    uniform = np.random.default_rng(7).uniform(0, 1, theory.shape)
    assert 1 - lz.weighted_distance(uniform - theory, weights) / reference == pytest.approx(0.0, abs=0.15)


def test_the_shape_reference_scores_a_flat_curve_at_zero():
    theory = lz.lz_probability(DURATIONS, np.array(MIN_GAP), np.array(DETUNING_SPAN))
    weights = lz.surface_weights(PHI_X, DURATIONS)
    row_weights = weights / weights.sum(axis=1, keepdims=True)
    flat = np.broadcast_to(np.sum(row_weights * theory, axis=1, keepdims=True), theory.shape)

    assert 1 - lz.weighted_distance(flat - theory, weights) / lz.flat_reference(theory, weights) == pytest.approx(0.0)


def test_the_excess_distance_removes_the_noise_and_stops_at_zero():
    assert lz.excess_distance(0.5, 0.3) == pytest.approx(0.4)
    assert lz.excess_distance(0.1, 0.3) == pytest.approx(0.0)


def test_the_decay_fit_recovers_tau_and_reports_an_unresolved_decay():
    durations = np.geomspace(5, 400, 25)
    decay = lz.fit_decay(durations, 0.1 + 0.8 * np.exp(-(durations - durations[0]) / 60.0))
    assert decay.resolved
    assert decay.tau == pytest.approx(60.0, rel=1e-3)

    flat = lz.fit_decay(durations, np.full(durations.size, 0.5))
    assert not flat.resolved
    assert math.isnan(flat.tau)


def test_an_unresolved_decay_keeps_its_fit_residuals():
    """A converged fit with no decay to see still has residuals: fit_r2 uses them; only tau is unknown."""
    durations = np.geomspace(5, 400, 25)
    flat = 0.5 + np.random.default_rng(5).normal(0, 0.02, durations.size)

    decay = lz.fit_decay(durations, flat)

    assert not decay.resolved
    assert math.isnan(decay.tau)
    assert math.isfinite(decay.ss_res)


def test_the_successive_scatter_estimates_white_noise():
    noise = np.random.default_rng(3).normal(0, 0.05, 20_000)
    assert math.sqrt(lz.successive_scatter(noise)) == pytest.approx(0.05, rel=0.03)
