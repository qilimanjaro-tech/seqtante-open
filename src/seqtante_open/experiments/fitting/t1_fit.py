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

from typing import TypedDict

import numpy as np
import plotly.graph_objects as go
import scipy.optimize as sp

from seqtante_open.experiments.fitting.fit_base import FittingClass

#: Nanoseconds per microsecond, the benchmark's reporting unit (QHC-1803).
_NS_PER_US = 1_000.0


class T1Results(TypedDict):
    """What :meth:`T1Fit.fit` produces; times in ns."""

    #: ``(a, b, c)`` of ``a * exp(b * t) + c``, as ``curve_fit`` returns them.
    params: np.ndarray
    #: ``b``, recorded even when it is not a decay.
    rate: float
    #: ``-1 / rate``; NaN unless ``rate < 0``.
    T1: float
    t1_uncertainty: float
    #: NaN for a flat trace.
    r_squared: float


class T1Fit(FittingClass[T1Results]):
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

    def __init__(self, target: str, measurement_id: int, path: str | None = None):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
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
        params, pcov = sp.curve_fit(
            self.exponential,
            self.wait_values,
            self.arr,
            p0=self.exponential_initial_guess(self.wait_values, self.arr),
        )
        rate = float(params[1])
        if rate < 0.0:
            t1 = -1.0 / rate
            rate_std = float(np.sqrt(pcov[1, 1]))
            # T1 = -1/rate, so dT1/drate = 1/rate²; propagate the rate's standard uncertainty.
            t1_uncertainty = rate_std / rate**2
        else:
            # A non-decaying (or flat) fit has no physical T1; flag it rather than dividing.
            t1 = float("nan")
            t1_uncertainty = float("nan")
        model = self.exponential(self.wait_values, *params)
        ss_res = float(np.sum((self.arr - model) ** 2))
        ss_tot = float(np.sum((self.arr - np.mean(self.arr)) ** 2))
        # A flat trace (drive off, readout saturated) has no variance to explain: R² is undefined, not a crash.
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.results = T1Results(params=params, rate=rate, T1=t1, t1_uncertainty=t1_uncertainty, r_squared=r_squared)

    @property
    def rate(self) -> float:
        """The fitted exponent ``b`` in 1/ns."""
        return self.results["rate"]

    @property
    def T1(self) -> float:
        """The relaxation time in ns; NaN when the fit is not a decay."""
        return self.results["T1"]

    @property
    def t1_uncertainty(self) -> float:
        """The standard uncertainty on T1 in ns."""
        return self.results["t1_uncertainty"]

    @property
    def r_squared(self) -> float:
        """The fit's R²; NaN for a flat trace."""
        return self.results["r_squared"]

    @property
    def is_physical_decay(self) -> bool:
        """Whether the fit is a real decay (``rate < 0``), so ``T1 = -1/rate`` is positive and finite."""
        return self.rate < 0.0

    @property
    def relative_uncertainty(self) -> float:
        """The standard uncertainty on T1 as a fraction of T1 (unit-independent, so us or ns give the same).

        Non-finite when ``curve_fit`` could not estimate the covariance (``inf``) or the fit is
        non-physical (``nan``); ``analyze`` rejects both.
        """
        return abs(self.t1_uncertainty / self.T1)

    @property
    def t1_us(self) -> float:
        """The relaxation time in microseconds (the benchmark's unit)."""
        return self.T1 / _NS_PER_US

    @property
    def t1_uncertainty_us(self) -> float:
        """The standard uncertainty on T1 in microseconds."""
        return self.t1_uncertainty / _NS_PER_US

    def save_fitting(self):
        """Write the fitted T1 (us) and the fit R squared to the measurement entry (QHC-1497, QHC-1791)."""
        from seqtante_open.outputs import FittedParameter

        self.add_fitted_parameter(FittedParameter.T1, self.t1_us)
        self.add_fitted_parameter(FittedParameter.FIT_R2, self.r_squared)
        self.save_fitting_to_database()

    def plot(self):
        """Plot the rotated decay and the fitted exponential."""
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
                y=self.exponential(self.wait_values, *self.results["params"]),
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
