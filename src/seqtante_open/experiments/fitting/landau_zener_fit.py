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

"""Score the compartmentalized Landau-Zener transition against the closed-form theory curve (ADR-0012, QHC-1921).

The measured φx-by-T surface is rotated, normalised with the operating point's readout levels to a
transition probability ``P'``, and compared with ``P_LZ(T) = exp(-π² Δ² T / Δε)`` for each φx's simulated
minimum gap Δ and detuning span Δε. The similarity score ``S = 1 - D/D_ref`` is computed three ways
(``rmse``, ``rmse_excess``, ``shape``); ``rmse`` is the benchmark value. The functions below are the
score's pieces, kept pure so they can be checked on their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict, get_args

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit

from seqtante_open.experiments.fitting.fit_base import FittingClass

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The loop names the ``landau_zener`` node streams the surface under, outer first.
PHI_X_LOOP = "phi_x"
DURATION_LOOP = "ramp_duration"

#: Uniform draws for the ``rmse`` and ``rmse_excess`` references (ADR-0012), with a fixed seed so S is reproducible.
UNIFORM_DRAWS = 20_000
UNIFORM_SEED = 0
_DRAW_CHUNK = 2_000

#: The decay fit's free parameters (floor, amplitude, τ); T₀ is fixed at the shortest duration.
_DECAY_PARAMETERS = 3
#: How many residual standard deviations the decay amplitude must exceed for τ to count as measured.
_MIN_DECAY_SIGNIFICANCE = 3

Variant = Literal["rmse", "rmse_excess", "shape"]
VARIANTS: tuple[Variant, ...] = get_args(Variant)
BENCHMARK_VARIANT: Variant = "rmse"


def lz_probability(durations: np.ndarray, min_gap: np.ndarray, detuning_span: np.ndarray) -> np.ndarray:
    """``P_LZ = exp(-π² Δ² T / Δε)``, one row per (Δ, Δε) pair and one column per duration.

    Args:
        durations: the ramp durations T, in ns.
        min_gap: Δ per row, in GHz.
        detuning_span: Δε per row, in GHz.
    """
    durations = np.asarray(durations, dtype=float)
    gap = np.asarray(min_gap, dtype=float)[:, None]
    span = np.asarray(detuning_span, dtype=float)[:, None]
    return np.exp(-(math.pi**2) * gap**2 * durations[None, :] / span)


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoid weights of the points ``x``, in their order, normalised to sum to 1; ``[1]`` for one point.

    Raises:
        ValueError: if ``x`` is empty or not strictly increasing.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("no points to weight")
    if x.size == 1:
        return np.ones(1)
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"the points must be strictly increasing, got {x.tolist()}")
    weights = np.empty_like(x)
    weights[0] = (x[1] - x[0]) / 2
    weights[-1] = (x[-1] - x[-2]) / 2
    weights[1:-1] = (x[2:] - x[:-2]) / 2
    return weights / weights.sum()


def surface_weights(phi_x: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """``W_ki = v_k · w_i``: trapezoid weights over φx times trapezoid weights in log T, summing to 1."""
    return trapezoid_weights(phi_x)[:, None] * trapezoid_weights(np.log(np.asarray(durations, dtype=float)))[None, :]


def weighted_distance(residuals: np.ndarray, weights: np.ndarray) -> float:
    """``D = √(Σ W r²)``, the pooled weighted RMS of ``residuals`` (ADR-0012)."""
    return float(np.sqrt(np.sum(weights * residuals**2)))


def excess_distance(distance: float, noise: float) -> float:
    """``D_ex = √max(D² - sigma_hat², 0)``: ``distance`` with the data's noise removed."""
    return math.sqrt(max(distance**2 - noise**2, 0.0))


def uniform_reference(theory: np.ndarray, weights: np.ndarray, noise: float | None = None) -> float:
    """The mean D of uniform values in [0, 1] against ``theory``: ``D_ref`` of ``rmse`` (``noise`` None) or ``rmse_excess``.

    :data:`UNIFORM_DRAWS` draws with the fixed :data:`UNIFORM_SEED`. With ``noise``, each draw's distance
    is the excess distance (:func:`excess_distance`), the same reduction ``rmse_excess`` applies to the data.
    """
    rng = np.random.default_rng(UNIFORM_SEED)
    total = 0.0
    remaining = UNIFORM_DRAWS
    while remaining:
        chunk = min(remaining, _DRAW_CHUNK)
        draws = rng.uniform(0.0, 1.0, size=(chunk, *theory.shape))
        distances = np.sqrt(np.sum(weights * (draws - theory) ** 2, axis=(1, 2)))
        if noise is not None:
            distances = np.sqrt(np.maximum(distances**2 - noise**2, 0.0))
        total += float(distances.sum())
        remaining -= chunk
    return total / UNIFORM_DRAWS


def flat_reference(theory: np.ndarray, weights: np.ndarray) -> float:
    """``D_ref`` of ``shape``: the distance from ``theory`` of the flat curve at each φx's weighted mean."""
    row_weights = weights / weights.sum(axis=1, keepdims=True)
    flat = np.sum(row_weights * theory, axis=1, keepdims=True)
    return weighted_distance(flat - theory, weights)


def _decay(durations: np.ndarray, floor: float, amplitude: float, tau: float, start: float) -> np.ndarray:
    return floor + amplitude * np.exp(-(durations - start) / tau)


@dataclass(frozen=True)
class DecayFit:
    """``floor + A·exp(-(T - T₀)/τ)`` fitted to one φx's curve, T₀ fixed at the shortest duration.

    ``ss_res`` is the residual sum of squares of the converged fit, NaN when the fit failed. ``resolved``
    is false when the fit failed, τ ended on a bound, or the amplitude is not clear of the fit's residual
    noise: then τ is not a measured decay time, and is NaN.
    """

    tau: float
    ss_res: float
    ss_tot: float
    resolved: bool


def fit_decay(durations: np.ndarray, values: np.ndarray) -> DecayFit:
    """Fit the LZ decay to one φx's curve (ADR-0012), τ bounded to [a tenth of the shortest step, 100 spans].

    The sums of squares are of the fitted curve, whether or not the decay is resolved.
    """
    durations = np.asarray(durations, dtype=float)
    values = np.asarray(values, dtype=float)
    start = float(durations[0])
    span = float(durations[-1] - start)
    tau_min, tau_max = float(np.min(np.diff(durations))) / 10, 100 * span
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    initial = (float(values[-1]), float(values[0] - values[-1]), min(max(span / 3, tau_min * 2), tau_max / 2))
    nan = float("nan")
    try:
        (floor, amplitude, tau), _ = curve_fit(
            lambda t, floor, amplitude, tau: _decay(t, floor, amplitude, tau, start),
            durations,
            values,
            p0=initial,
            bounds=([-np.inf, -np.inf, tau_min], [np.inf, np.inf, tau_max]),
        )
    except (RuntimeError, ValueError):
        return DecayFit(tau=nan, ss_res=nan, ss_tot=ss_tot, resolved=False)
    on_bound = math.isclose(tau, tau_min, rel_tol=1e-3) or math.isclose(tau, tau_max, rel_tol=1e-3)
    ss_res = float(np.sum((values - _decay(durations, floor, amplitude, tau, start)) ** 2))
    # With no decay to see, τ is free and stays wherever the fit left it.
    residual_noise = math.sqrt(ss_res / max(durations.size - _DECAY_PARAMETERS, 1))
    if not math.isfinite(ss_res):
        return DecayFit(tau=nan, ss_res=nan, ss_tot=ss_tot, resolved=False)
    no_decay = abs(amplitude) <= _MIN_DECAY_SIGNIFICANCE * residual_noise
    if on_bound or no_decay:
        return DecayFit(tau=nan, ss_res=ss_res, ss_tot=ss_tot, resolved=False)
    return DecayFit(tau=float(tau), ss_res=ss_res, ss_tot=ss_tot, resolved=True)


def successive_scatter(values: np.ndarray) -> float:
    """The noise variance estimated from successive differences, ``mean(Δy²)/2``: sigma_hat's fallback (ADR-0012)."""
    return float(np.mean(np.diff(np.asarray(values, dtype=float)) ** 2) / 2)


@dataclass(frozen=True)
class Score:
    """One variant of S: its value, its standard uncertainty, the distance and the reference it came from."""

    value: float
    uncertainty: float
    distance: float
    reference: float


class _LandauZenerResults(TypedDict):
    #: S per variant (``rmse``, ``rmse_excess``, ``shape``).
    scores: dict[Variant, Score]
    #: sigma_hat, the noise of the normalised data, pooled over φx.
    noise: float
    #: R² of the decay fits, pooled over φx.
    fit_r2: float
    #: ``√(τ_theory/τ_measured)`` per φx, NaN where either decay is unresolved.
    gap_ratio: np.ndarray


class LandauZenerFit(FittingClass[_LandauZenerResults]):
    """Score the Landau-Zener φx-by-T surface against the closed-form theory curve (ADR-0012, QHC-1921).

    The averaged readout quadratures are rotated to the real axis (:meth:`FittingClass.rotate_iq`, over the
    whole surface) and mapped to ``P' = (signal - level_0)/(level_1 - level_0)`` with the |0>/|1> readout
    levels of the step's operating point (ADR-0001). The levels and each φx's simulated Δ and Δε come from
    the node, which reads them from the calibration file and checks them before measuring.

    Args:
        measurement_id (int): Id of the measurement: an ``[I, Q, threshold]`` stream over the
            ``phi_x`` (outer) and ``ramp_duration`` (inner) loops.
        target (str): Measured target token, e.g. ``"q1"``.
        level_0 (float): The |0> readout level, in the rotated-signal units.
        level_1 (float): The |1> readout level, in the rotated-signal units.
        min_gap (Sequence[float]): Δ for each φx of the loop, in GHz.
        detuning_span (Sequence[float]): Δε for each φx of the loop, in GHz.
        path (str | None, optional): Directory the plot is saved into; if None the plot is shown.

    Raises:
        ValueError: if the levels are equal, or there is not one Δ and one Δε per φx.
    """

    def __init__(
        self,
        measurement_id: int,
        target: str,
        level_0: float,
        level_1: float,
        min_gap: Sequence[float],
        detuning_span: Sequence[float],
        path: str | None = None,
    ):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        if level_1 == level_0:
            raise ValueError(f"the |0> and |1> readout levels are equal ({level_0}); cannot normalise the signal")
        self.phi_x = np.asarray(self.loops[PHI_X_LOOP]["array"], dtype=float)
        self.durations = np.asarray(self.loops[DURATION_LOOP]["array"], dtype=float)
        if not len(min_gap) == len(detuning_span) == len(self.phi_x):
            raise ValueError(
                f"{len(self.phi_x)} phi_x were measured, but {len(min_gap)} min_gap and {len(detuning_span)} "
                "detuning_span were given"
            )
        iq = self.array[..., 0] + 1j * self.array[..., 1]
        signal = np.real(self.rotate_iq(iq.ravel())).reshape(iq.shape)
        self.transition_probability = (signal - level_0) / (level_1 - level_0)
        self.theory = lz_probability(self.durations, np.asarray(min_gap), np.asarray(detuning_span))

    def fit(self) -> _LandauZenerResults:
        """Compute the three S, sigma_hat, ``fit_r2`` and the gap ratios, and record them as :attr:`results`."""
        self._results = None
        measured, theory = self.transition_probability, self.theory
        weights = surface_weights(self.phi_x, self.durations)
        residuals = measured - theory

        fits = [fit_decay(self.durations, row) for row in measured]
        theory_fits = [fit_decay(self.durations, row) for row in theory]
        dof = self.durations.size - _DECAY_PARAMETERS
        # sigma_hat falls back to the successive-point scatter where the decay is not resolved (ADR-0012).
        noise_ss = sum(
            fit.ss_res if fit.resolved else successive_scatter(row) * dof
            for fit, row in zip(fits, measured, strict=True)
        )
        noise = math.sqrt(noise_ss / (dof * len(fits)))
        # fit_r2 is the decay fits' own quality: every converged fit, and nothing from the fallback.
        converged = [fit for fit in fits if math.isfinite(fit.ss_res)]
        fit_ss_tot = sum(fit.ss_tot for fit in converged)
        fit_r2 = 1 - sum(fit.ss_res for fit in converged) / fit_ss_tot if fit_ss_tot > 0 else float("nan")
        gap_ratio = np.array(
            [
                math.sqrt(model.tau / fit.tau) if fit.resolved and model.resolved else float("nan")
                for fit, model in zip(fits, theory_fits, strict=True)
            ]
        )

        distance = weighted_distance(residuals, weights)
        excess = excess_distance(distance, noise)
        spread = noise * math.sqrt(float(np.sum(weights**2 * residuals**2)))
        references = {
            "rmse": uniform_reference(theory, weights),
            "rmse_excess": uniform_reference(theory, weights, noise),
            "shape": flat_reference(theory, weights),
        }
        distances = {"rmse": distance, "rmse_excess": excess, "shape": distance}
        scores: dict[Variant, Score] = {}
        for variant in VARIANTS:
            d, d_ref = distances[variant], references[variant]
            # sigma_D is undefined at D = 0 (ADR-0012), and S at D_ref = 0 (a theory curve flat at its mean).
            sigma_d = spread / d if d > 0 else float("nan")
            if d_ref > 0:
                value, uncertainty = 1 - d / d_ref, sigma_d / d_ref
            else:
                value = uncertainty = float("nan")
            scores[variant] = Score(value=value, uncertainty=uncertainty, distance=d, reference=d_ref)

        self.results = _LandauZenerResults(scores=scores, noise=noise, fit_r2=fit_r2, gap_ratio=gap_ratio)
        return self.results

    @property
    def score(self) -> Score:
        """The benchmark variant of S (:data:`BENCHMARK_VARIANT`, ADR-0012)."""
        return self.results["scores"][BENCHMARK_VARIANT]

    def save_fitting(self) -> None:
        """Write ``score_s`` (the benchmark S) and ``rmse`` (its distance D) to the measurement entry (ADR-0012)."""
        from seqtante_open.outputs import FittedParameter

        self.add_fitted_parameter(FittedParameter.SCORE_S, self.score.value)
        self.add_fitted_parameter(FittedParameter.RMSE, self.score.distance)
        self.save_fitting_to_database()

    def plot(self) -> str | None:
        """Plot the measured ``P'`` and the theory curve per φx against the ramp duration (log axis).

        Returns:
            str | None: Path the plot was written to, or None when it was shown instead.
        """
        fig = go.Figure()
        has_results = self._results is not None
        for k, phi_x in enumerate(self.phi_x):
            label = f"φx = {phi_x:.4g}"
            if has_results and math.isfinite(self.results["gap_ratio"][k]):
                label += f", gap ratio {self.results['gap_ratio'][k]:.3f}"
            fig.add_trace(go.Scatter(x=self.durations, y=self.transition_probability[k], mode="markers", name=label))
            fig.add_trace(
                go.Scatter(x=self.durations, y=self.theory[k], mode="lines", name=f"theory, φx = {phi_x:.4g}")
            )
        title = f"Landau-Zener {self.target}"
        if has_results:
            scores = self.results["scores"]
            title += ": " + ", ".join(f"S_{name} = {scores[name].value:.3f}" for name in VARIANTS)
            title += f", sigma_hat = {self.results['noise']:.3g}, fit R² = {self.results['fit_r2']:.3f}"
        fig.update_layout(
            title=f"{title}, ID: {self.id}",
            xaxis={"title": "Ramp duration (ns)", "type": "log"},
            yaxis_title="Transition probability",
        )
        return self.save_plot(fig, title=f"landau_zener_{self.target} ID {self.id}")
