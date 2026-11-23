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

import numpy as np
import plotly.graph_objects as go
import scipy.optimize as sp

from seqtante_open.experiments.fitting.fit_base import FittingClass

#: Nanoseconds per microsecond, the benchmark's reporting unit (QHC-1809).
_NS_PER_US = 1_000.0


class T2Fit(FittingClass):
    """Fit a T2 (Ramsey) decaying oscillation and report the dephasing time with its uncertainty (QHC-1809).

    The rotated real quadrature ``Re(rotate_iq(I + iQ))`` is fit to the damped cosine
    ``a * cos(2*pi*b*t + c) * exp(-t / d) + f`` over the free-evolution times (ns), so the dephasing
    time is ``T2 = d``. Its standard uncertainty is the square root of the ``d`` variance in the fit
    covariance, and the fit quality is reported as R squared. The benchmark value and the DB fitting
    parameter are in microseconds (QHC-1497: ``t2``, ``fit_r2``).

    Args:
        target (str): Measured target token, e.g. ``"q1"``.
        measurement_id (int): Id of the measurement to fit.
        path (str | None, optional): Directory the plot is saved into; if None the plot is shown.
    """

    results: np.ndarray | None
    #: Set by :meth:`fit`; left unset until then so a missing fit is an error, not a silent default.
    decay: float
    T2: float
    t2_uncertainty: float
    r_squared: float

    def __init__(self, target: str, measurement_id: int, path: str | None = None):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.results = None
        xarr = self.get_xarray()
        self.wait_values = xarr[xarr.dims[0]].data
        self.arr = np.real(self.rotate_iq(self.array[:, 0] + 1j * self.array[:, 1]))

    def _initial_guess(self) -> tuple[float, float, float, float, float]:
        """A rough guess for ``a * cos(2*pi*b*t + c) * exp(-t / d) + f`` from the data.

        Offset ``f`` and amplitude ``a`` come from the signal's mean and peak-to-peak; the decay ``d`` is
        guessed as half the swept window; the frequency ``b`` (in 1/ns) is the dominant non-DC component
        of the rFFT of the centred signal; the phase ``c`` starts at zero.
        """
        y = self.arr
        t = self.wait_values
        offset = float(np.mean(y))
        amplitude = float((np.max(y) - np.min(y)) / 2) or 1.0
        decay = float((t[-1] - t[0]) / 2) or 1.0
        dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
        spectrum = np.abs(np.fft.rfft(y - offset))
        freqs = np.fft.rfftfreq(len(t), d=dt)
        frequency = float(freqs[1 + int(np.argmax(spectrum[1:]))]) if len(spectrum) > 1 else 0.0
        return amplitude, frequency, 0.0, decay, offset

    def fit(self):
        """Fit the damped cosine, then record the decay, T2 (ns), its uncertainty (ns) and the fit R².

        ``T2 = d`` is physical only for a decaying envelope (``d > 0``). The raw ``d`` is always recorded,
        but T2 and its uncertainty are left non-finite when ``d <= 0`` (a growing or flat envelope):
        :attr:`is_physical_decay` then reports the failure so ``analyze`` fails the target instead of
        scoring a non-physical value, as for T1 (QHC-1803).
        """
        self.results, pcov = sp.curve_fit(
            self.sinus_exp,
            self.wait_values,
            self.arr,
            p0=self._initial_guess(),
            maxfev=10_000,
        )
        decay = float(self.results[3])
        self.decay = decay
        if decay > 0.0:
            self.T2 = decay
            self.t2_uncertainty = float(np.sqrt(pcov[3, 3]))
        else:
            # A growing or flat envelope has no physical T2; flag it rather than reporting |d|.
            self.T2 = float("nan")
            self.t2_uncertainty = float("nan")
        model = self.sinus_exp(self.wait_values, *self.results)
        ss_res = float(np.sum((self.arr - model) ** 2))
        ss_tot = float(np.sum((self.arr - np.mean(self.arr)) ** 2))
        # A flat trace (drive off, readout saturated) has no variance to explain: R² is undefined, not a crash.
        self.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    @property
    def is_physical_decay(self) -> bool:
        """Whether the envelope decays (``d > 0``), so ``T2 = d`` is positive and finite."""
        self._require_fit()
        return self.decay > 0.0

    @property
    def relative_uncertainty(self) -> float:
        """The standard uncertainty on T2 as a fraction of T2 (unit-independent, so us or ns give the same).

        Non-finite when ``curve_fit`` could not estimate the covariance (``inf``) or the fit is
        non-physical (``nan``); ``analyze`` rejects both.
        """
        self._require_fit()
        return abs(self.t2_uncertainty / self.T2)

    @property
    def t2_us(self) -> float:
        """The dephasing time in microseconds (the benchmark's unit)."""
        self._require_fit()
        return self.T2 / _NS_PER_US

    @property
    def t2_uncertainty_us(self) -> float:
        """The standard uncertainty on T2 in microseconds."""
        self._require_fit()
        return self.t2_uncertainty / _NS_PER_US

    def save_fitting(self):
        """Write the fitted T2 (us) and the fit R squared to the measurement entry (QHC-1497, QHC-1791)."""
        from seqtante_open.outputs import FittedParameter

        self._require_fit()
        self.add_fitted_parameter(FittedParameter.T2, self.t2_us)
        self.add_fitted_parameter(FittedParameter.FIT_R2, self.r_squared)
        self.save_fitting_to_database()

    def _require_fit(self) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("No fit results available, call fit() first.")
        return self.results

    def plot(self):
        """Plot the rotated oscillation and the fitted damped cosine."""
        results = self._require_fit()
        title = f"{self.target}_T2"

        fig = go.Figure(
            go.Scatter(
                x=self.wait_values,
                y=self.arr,
                mode="markers",
                name="Data",
                marker={"color": "royalblue", "size": 5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.wait_values,
                y=self.sinus_exp(self.wait_values, *results),
                mode="lines",
                name="Fitted damped cosine",
                line={"color": "red"},
            )
        )
        fig.update_layout(
            title=f"{self.target}, ID: {self.id}<br>Rotated T2 = {self.t2_us:.3f} us (R² = {self.r_squared:.3f})",
            xaxis_title="Wait Time (ns)",
            yaxis_title="Integrated Voltage (a.u.)",
            width=1000,
            height=600,
            margin={"t": 120},
            showlegend=True,
        )

        return self.save_plot(fig, title)
