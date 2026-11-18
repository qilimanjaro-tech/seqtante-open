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

#: Nanoseconds per microsecond, the benchmark's reporting unit (QHC-1803).
_NS_PER_US = 1_000.0


class T1Fit(FittingClass):
    """Fit a T1 saturation decay and report the relaxation time with its uncertainty (QHC-1803).

    The rotated real quadrature ``Re(rotate_iq(I + iQ))`` is fit to ``a * exp(b * t) + c`` over the idle
    times (ns), so the relaxation time is ``T1 = -1 / b``. Its standard uncertainty is propagated from
    the covariance of ``b`` (``std_T1 = std_b / b**2``), and the fit quality is reported as R squared.
    The benchmark value and the DB fitting parameter are in microseconds (QHC-1497: ``t1``, ``fit_r2``).

    Args:
        target (str): Measured target token, e.g. ``"q1"``.
        measurement_id (int): Id of the measurement to fit.
        path (str | None, optional): Directory the plot is saved into; if None the plot is shown.
    """

    results: np.ndarray | None
    #: Set by :meth:`fit`; left unset until then so a missing fit is an error, not a silent default.
    rate: float
    T1: float
    t1_uncertainty: float
    r_squared: float

    def __init__(self, target: str, measurement_id: int, path: str | None = None):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.results = None
        xarr = self.get_xarray()
        self.wait_values = xarr[xarr.dims[0]].data
        self.arr = np.real(self.rotate_iq(self.array[:, 0] + 1j * self.array[:, 1]))

    def fit(self):
        """Fit the exponential, then record the rate, T1 (ns), its uncertainty (ns) and the fit R².

        ``T1 = -1 / rate`` is physical only for a decaying exponential (``rate < 0``). The raw ``rate``
        is always recorded, but T1 and its uncertainty are left non-finite when ``rate >= 0``: dividing
        would give a negative lifetime, or raise on ``rate == 0``. :attr:`is_physical_decay` then reports
        the failure so ``analyze`` fails the target instead of scoring a non-physical value (QHC-1803).
        """
        self.results, pcov = sp.curve_fit(
            self.exponential,
            self.wait_values,
            self.arr,
            p0=self.exponential_initial_guess(self.wait_values, self.arr),
        )
        rate = float(self.results[1])
        self.rate = rate
        if rate < 0.0:
            self.T1 = -1.0 / rate
            rate_std = float(np.sqrt(pcov[1, 1]))
            # T1 = -1/rate, so dT1/drate = 1/rate²; propagate the rate's standard uncertainty.
            self.t1_uncertainty = rate_std / rate**2
        else:
            # A non-decaying (or flat) fit has no physical T1; flag it rather than dividing.
            self.T1 = float("nan")
            self.t1_uncertainty = float("nan")
        model = self.exponential(self.wait_values, *self.results)
        ss_res = float(np.sum((self.arr - model) ** 2))
        ss_tot = float(np.sum((self.arr - np.mean(self.arr)) ** 2))
        self.r_squared = 1.0 - ss_res / ss_tot

    @property
    def is_physical_decay(self) -> bool:
        """Whether the fit is a real decay (``rate < 0``), so ``T1 = -1/rate`` is positive and finite."""
        self._require_fit()
        return self.rate < 0.0

    @property
    def relative_uncertainty(self) -> float:
        """The standard uncertainty on T1 as a fraction of T1 (unit-independent, so us or ns give the same).

        Non-finite when ``curve_fit`` could not estimate the covariance (``inf``) or the fit is
        non-physical (``nan``); ``analyze`` rejects both.
        """
        self._require_fit()
        return abs(self.t1_uncertainty / self.T1)

    @property
    def t1_us(self) -> float:
        """The relaxation time in microseconds (the benchmark's unit)."""
        self._require_fit()
        return self.T1 / _NS_PER_US

    @property
    def t1_uncertainty_us(self) -> float:
        """The standard uncertainty on T1 in microseconds."""
        self._require_fit()
        return self.t1_uncertainty / _NS_PER_US

    def save_fitting(self):
        """Write the fitted T1 (us) and the fit R squared to the measurement entry (QHC-1497, QHC-1791)."""
        from seqtante_open.outputs import FittedParameter

        self._require_fit()
        self.add_fitted_parameter(FittedParameter.T1, self.t1_us)
        self.add_fitted_parameter(FittedParameter.FIT_R2, self.r_squared)
        self.save_fitting_to_database()

    def _require_fit(self) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("No fit results available, call fit() first.")
        return self.results

    def plot(self):
        """Plot the rotated decay and the fitted exponential."""
        results = self._require_fit()
        title = f"{self.target}_T1"

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
                y=self.exponential(self.wait_values, *results),
                mode="lines",
                name="Fitted exponential",
                line={"color": "red"},
            )
        )
        fig.update_layout(
            title=f"{self.target}, ID: {self.id}<br>Rotated T1 = {self.t1_us:.3f} us (R² = {self.r_squared:.3f})",
            xaxis_title="Wait Time (ns)",
            yaxis_title="Integrated Voltage (a.u.)",
            width=1000,
            height=600,
            margin={"t": 120},
            showlegend=True,
        )

        return self.save_plot(fig, title)
