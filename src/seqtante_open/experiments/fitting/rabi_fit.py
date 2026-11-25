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
from lmfit import Model

from seqtante_open.experiments.fitting.fit_base import FittingClass


class _RabiResults(TypedDict):
    offset: float
    amplitude: float
    angular_frequency: float
    r_squared: float
    pi_amplitude: float
    pi_amplitude_std: float
    level_0: float
    level_0_std: float
    level_1: float
    level_1_std: float


class RabiAmplitudeModel(FittingClass[_RabiResults]):
    """Fit a Rabi amplitude sweep and read the pi amplitude and the |0>/|1> readout levels.

    The measurement sweeps the drive amplitude. The rotation angle is linear in it and zero at zero
    drive, where the qubit sits in |0>, so the rotated readout signal is the undamped cosine
    ``offset + amplitude * cos(angular_frequency * x)`` (:meth:`FittingClass.cosfunc` with its phase
    fixed at 0). ``amplitude`` is signed: ``rotate_iq`` fixes the trace's sign arbitrarily, so |0> may
    be its top or its bottom. From the fit:

    - the two readout **levels** are the cosine at zero drive (|0>, ``offset + amplitude``) and half a
      period on (|1>, ``offset - amplitude``), whatever amplitude the window starts at. Each carries the
      uncertainty propagated from the fitted ``offset`` and ``amplitude``, NaN when lmfit cannot estimate
      the covariance. The levels fold in thermal population and readout SNR, an accepted caveat until
      single-shot readout calibration exists (QHC-1806).
    - the **pi amplitude** is the first excited extremum, ``pi / angular_frequency`` (NaN for no
      oscillation), with its propagated uncertainty. It may lie outside the swept window when the sweep
      stops short of it; the node rejects that.

    Fitted parameters (``rabi_amplitude``, ``fit_r2``; QHC-1497) are written to the measurement entry
    with :meth:`save_fitting`.

    Args:
        measurement_id: Autocalibration database id of the amplitude sweep to load.
        target: Swept target token, e.g. ``"q1"``.
        path: Folder to save the plot into; if ``None`` the plot is shown.
    """

    def __init__(self, measurement_id: int, target: str | None = None, path: str | None = None):
        super().__init__(measurement_id=measurement_id, target=target, path=path)
        xarr = self.get_xarray()
        amplitude_coord = xarr[xarr.dims[0]]
        self.amplitudes = np.asarray(amplitude_coord.data, dtype=float)
        self.signal = np.real(self.rotate_iq(xarr.to_numpy()))

    def fit(self) -> _RabiResults:
        """Fit the cosine with its phase fixed at zero drive and derive the pi amplitude and the two readout levels."""
        # A refit that finds nothing must not leave the previous result readable.
        self._results = None
        amplitudes, signal = self.amplitudes, self.signal

        # A Rabi rotation starts from |0> at zero drive: the phase is 0, not a free parameter. The signed
        # amplitude covers |0> being the top or the bottom of the rotated trace.
        guess = self.oscillation_initial_guess(amplitudes, signal, free_phase=False)
        model = Model(self.cosfunc)
        model.set_param_hint("omega", value=guess["omega"], min=0.0)
        model.set_param_hint("phase_offset", value=0.0, vary=False)
        fit = model.fit(data=signal, params=model.make_params(a=guess["a"], offset=guess["offset"]), phi=amplitudes)

        offset, amplitude, angular_frequency = (fit.params[name].value for name in ("offset", "a", "omega"))
        offset_std, amplitude_std, angular_frequency_std = (
            self._stderr(fit.params[name]) for name in ("offset", "a", "omega")
        )
        # Both levels carry the same std, propagated from offset and amplitude ignoring their covariance --
        # an accepted approximation for the level uncertainty.
        level_std = float(np.hypot(offset_std, amplitude_std))
        has_oscillation = angular_frequency > 0
        pi_amplitude = np.pi / angular_frequency if has_oscillation else float("nan")
        pi_amplitude_std = pi_amplitude * angular_frequency_std / angular_frequency if has_oscillation else float("nan")

        # A flat trace (drive off, readout saturated) has no variance to explain: R² is undefined (NaN),
        # and the node rejects it like a low R².
        has_variance = float(np.sum((signal - np.mean(signal)) ** 2)) > 0
        r_squared = float(fit.rsquared) if has_variance else float("nan")

        self.results = _RabiResults(
            offset=offset,
            amplitude=amplitude,
            angular_frequency=angular_frequency,
            r_squared=r_squared,
            pi_amplitude=float(pi_amplitude),
            pi_amplitude_std=float(pi_amplitude_std),
            level_0=float(offset + amplitude),
            level_0_std=level_std,
            level_1=float(offset - amplitude),
            level_1_std=level_std,
        )
        return self.results

    @staticmethod
    def _stderr(parameter) -> float:
        """A fitted parameter's standard error, NaN (never 0) when lmfit could not estimate the covariance."""
        return float(parameter.stderr) if parameter.stderr is not None else float("nan")

    @property
    def pi_amplitude(self) -> float:
        return self.results["pi_amplitude"]

    @property
    def readout_levels(self) -> tuple[float, float, float, float]:
        """``(level_0, level_0_std, level_1, level_1_std)`` of the |0>/|1> readout response."""
        return (
            self.results["level_0"],
            self.results["level_0_std"],
            self.results["level_1"],
            self.results["level_1_std"],
        )

    @property
    def r_squared(self) -> float:
        return self.results["r_squared"]

    def save_fitting(self):
        """Write the pi amplitude and the fit R² to the measurement entry (QHC-1497, QHC-1791)."""
        from seqtante_open.outputs import FittedParameter

        self.add_fitted_parameter(FittedParameter.RABI_AMPLITUDE, self.results["pi_amplitude"])
        self.add_fitted_parameter(FittedParameter.FIT_R2, self.results["r_squared"])
        self.save_fitting_to_database()

    def plot(self):
        """Plot the sweep, its cosine fit, the two readout levels and the pi amplitude."""
        results = self.results
        title = f"Rabi amplitude {self.target}"
        curve = self.cosfunc(
            self.amplitudes, results["amplitude"], results["angular_frequency"], results["offset"], 0.0
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.amplitudes,
                y=self.signal,
                mode="lines+markers",
                name="Data",
                line={"color": "royalblue", "dash": "dash"},
                marker={"color": "royalblue", "size": 5},
            )
        )
        fig.add_trace(
            go.Scatter(x=self.amplitudes, y=curve, mode="lines", name="Fit", line={"color": "red", "width": 2})
        )
        for level, name, colour in (
            (results["level_0"], "|0>", "green"),
            (results["level_1"], "|1>", "orange"),
        ):
            fig.add_hline(y=level, line={"color": colour, "dash": "dot"}, annotation_text=name)
        fig.add_vline(
            x=results["pi_amplitude"],
            line={"color": "purple", "dash": "dot"},
            annotation_text=f"pi amp = {results['pi_amplitude']:.4g}",
        )
        fig.update_layout(
            title=f"{title}, ID: {self.id} (r² = {results['r_squared']:.3f})",
            xaxis_title="Drive amplitude (a.u.)",
            yaxis_title="Integrated Voltage (a.u.)",
            width=1000,
            height=600,
        )
        return self.save_plot(fig, title)
