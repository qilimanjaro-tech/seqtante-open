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

import matplotlib.pyplot as plt
import numpy as np

from seqtante_open.experiments.fitting.fit_base import FittingClass

_QUADRATURE_LABELS = {
    "signal": "Signal quadrature (rotated)",
    "noise": "Noise quadrature (orthogonal)",
}


class FluxoniumSingleToneModel(FittingClass):
    """Fit and plot a single-tone (resonator spectroscopy) frequency sweep.

    Loads the measurement's 1D ``S21`` trace and rotates the IQ plane so the
    response collapses onto one quadrature (``signal``) while the orthogonal one
    holds only noise (``noise``), then fits a Lorentzian to each. :meth:`plot`
    renders both quadratures side by side, each with its data and its fit, and
    marks the fitted IF. All frequencies are handled in Hz and only converted
    for display.

    Args:
        measurement_id: Autocalibration database id of the sweep to load.
        target: Swept target token, e.g. ``"q1"``.
        path: Folder to save the plot into; if ``None`` the plot is shown.
        lo: Readout LO in Hz, used to report the absolute readout frequency; if
            ``None`` it is read from the runcard stored with the measurement.
    """

    results: dict[str, dict[str, float | np.ndarray]]

    def __init__(
        self, measurement_id: int, target: str | None = None, path: str | None = None, lo: float | None = None
    ):

        super().__init__(measurement_id=measurement_id, target=target, path=path)
        self.target = target
        self.frequencies = self.loops["frequency"]["array"]
        self.lo = lo
        self.i = self.array[:, 0]
        self.q = self.array[:, 1]
        self.path = path
        self.fitted_amplitude = None
        self.magnitude = None

    def fit(self):
        """Fits the experimental data to the corresponding function."""
        self.magnitude = 20 * np.log10(np.sqrt(self.i**2 + self.q**2))
        self.fitted_frequency = self.frequencies[np.argmin(self.magnitude)]

        return self.fitted_frequency

    def plot(self):
        title = f"q{self.target}_Resonator_Spectroscopy"
        plt.figure(figsize=(10, 6))
        plt.axvline((self.fitted_frequency + self.lo) * 1e-9, color="red", label=f"Resonator Freq\n {(self.fitted_frequency + self.lo) * 1e-9:.4f}GHz")
        plt.plot((self.frequencies + self.lo) * 1e-9, self.magnitude, ".--")
        plt.legend()
        plt.grid(which="both")
        plt.xlabel("Frequency (GHz)", fontsize=12)
        plt.ylabel("Integrated Voltage (dB)", fontsize=12)
        plt.title(title + f", ID: {self.id}\nFitted Intermediate Frequency = {self.fitted_frequency * 1e-6:.4f} MHz", fontsize=14)

        self.save_plot(title)
