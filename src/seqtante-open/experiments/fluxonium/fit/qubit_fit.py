# Copyright 2024 Qilimanjaro Quantum Tech
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
"""Module for handling correction and fitting of Resonators data.

This module provides classes and functions to correct
and fit resonator data.

Classes:
    ResonatorSpectroscopyFit: Handles the correction and fitting
    of Resonator Spectroscopy data.

Functions:
    fit: fit the resonator spectroscopy data.
    plot: plot the fitted resonator spectroscopy data.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import tqdm
import xarray as xr
from scipy.signal import find_peaks
from scipy.signal import find_peaks, savgol_filter
from qilitools.plotting import auto_plot, center_phase_around_median
from seqtante.experiments.fluxoniums.utils import lorentzian_fit_custom


class QubitSpectroscopyFit:
    """Handle the correction of the data and fitting
    of Qubit Spectroscopy data.
    """

    def __init__(
        self, qubit_idx: int,
        measurement: np.ndarray,
        path: str | None = None
    ):
        """Initialize the class

        Args:
            qubit_idx (int): qubit index
            measurement (): array with the measurement result
            path (str | None): path to save plots
        """
        self.qubit_idx = qubit_idx
        self.measurement = measurement
        self.path = path
        self.readout_if_list = None

    def fit(self):
        """Empty placeholder for now"""

        return None

    def plot(self, x_axis: str):
        title = "Two tone vs Flux"
        fig = auto_plot(self.measurement,
                        x=x_axis,
                        dataprocessing=center_phase_around_median)

        if self.path:
            qubit_path = os.path.join(self.path, f"q{self.qubit_idx}")
            os.makedirs(qubit_path, exist_ok=True)
            filepath = os.path.join(qubit_path, title + ".png")
            fig.write_image(filepath)
        else:
            fig.show()