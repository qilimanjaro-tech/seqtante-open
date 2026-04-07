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

"""Module providing functions for handling crosstalk matrices.

This module contains functions to calculate crosstalk matrices from
input vectors, normalize crosstalk matrices, and apply crosstalk matrices
to coordinates for mesh plotting.

Functions:
    crosstalk_matrix_from_vectors(v1: np.ndarray,
                                  v2: np.ndarray) -> np.ndarray:
        Calculates crosstalk matrix from input vectors.

    normalize_crosstalk_matrix(crosstalk_matrix: np.ndarray)
    -> Tuple[np.ndarray, np.ndarray]:
        Normalizes crosstalk matrix and calculates inductances.

    crosstalk_mesh(axis1: np.ndarray,
                   axis2: np.ndarray,
                   crosstalk_matrix: np.ndarray)
    -> Tuple[np.ndarray, np.ndarray]:
    Applies crosstalk matrix to coordinates for mesh plotting.
"""

from functools import partial
from typing import Sequence

import numpy as np
import qcodes as qc
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2Channel


def crosstalk_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray):
    """Calculates XT matrix from vectors

    Args:
        v1 (np.array): Vector defining direction and period 1
        v2 (np.array): Vector defining direction and period 2

    Returns:
        np.array: returns crosstalk matrix
    """

    return np.linalg.inv(np.array([v1, v2]).T)


def normalize_crosstalk_matrix(crosstalk_matrix: np.ndarray):
    """Calculates the relative crosstalk matrix and inductances
        from the total matrix
    Args:
        crosstalk_matrix (np.ndarray): crosstalk matrix

    Returns:
        normalized_crosstalk_matrix (np.ndarray)
        inductances (np.ndarray): array with the inductances
    """

    inductances = np.diag(crosstalk_matrix)
    normalized_crosstalk_matrix = np.linalg.inv(np.diag(inductances)) @ crosstalk_matrix

    return normalized_crosstalk_matrix, inductances


def crosstalk_mesh(axis_z: np.ndarray, axis_x: np.ndarray, crosstalk_matrix: np.ndarray):
    """Applies the crosstalk matrix to the coordinates in axis1 and axis2
        to use in mesh plotting
    Args:
        axis1 (np.ndarray): coordinates of the first axis
        axis2 (np.ndarray): coordinates of the second axis
        crosstalk_matrix (np.ndarray): crosstalk matrix

    Returns:
        np.ndarray: mesh with the rescaled coordinates
        np.ndarray: mesh with the rescaled coordinates
    """

    # S21 is defined as (len(x), len(z))
    # but vectors are defind as (z, x)
    shape = (len(axis_x), len(axis_z))  # x, then z
    new_coordinates = np.zeros([shape[0], shape[1], 2])
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_coordinates[i, j, :] = crosstalk_matrix @ np.array([axis_z[j], axis_x[i]])

    return new_coordinates[:, :, 0], new_coordinates[:, :, 1]


class XTalk(qc.Instrument):
    def __init__(
        self,
        name: str,
        channels: list[qc.Parameter],
        name_list: list[str],
        qdac_channels: list[QDac2Channel] | None = None,
        unit: str = "phi_0",
    ) -> None:
        super().__init__(name)

        self.N: int = len(channels)
        self.xtalk_matrix: np.ndarray = np.eye(self.N)
        self.flux_offsets: np.ndarray = np.zeros(self.N)
        self.mutual_inductance: np.ndarray = np.ones(self.N)
        self.channels = channels
        self.name_list = name_list
        self.channels = channels
        self.qdac_channels = qdac_channels
        self._flux_mem: np.ndarray = np.zeros(self.N, dtype=float)

        for ii in range(len(channels)):
            self.add_parameter(
                name_list[ii], set_cmd=partial(self.set_flux, ii), get_cmd=partial(self.get_flux, ii), unit=unit
            )

    def set_biases(self, biases: np.ndarray | Sequence[float]) -> None:
        for ii, bias in enumerate(biases):
            self.channels[ii](bias)

    def get_biases(self):
        """Read current hardware biases (not used for normal operation)."""
        return [channel() for channel in self.channels]

    def flux_to_bias(self, flux: np.ndarray | Sequence[float]) -> np.ndarray:
        """Convert target flux vector to hardware bias vector."""
        # TODO: do not calculate inverse unless it is needed
        flux_arr = np.array(flux, dtype=float)
        inverse_crosstalk_matrix = np.linalg.inv(np.diag(self.mutual_inductance) @ self.xtalk_matrix)
        return inverse_crosstalk_matrix @ (flux_arr - self.flux_offsets)

    def bias_to_flux(self, bias: np.ndarray | Sequence[float]) -> np.ndarray:
        """Convert hardware bias vector to flux vector."""
        bias_arr = np.array(bias, dtype=float)
        return np.diag(self.mutual_inductance) @ self.xtalk_matrix @ bias_arr + self.flux_offsets

    def set_flux(self, ii: int, value: float):
        self._flux_mem[ii] = value
        biases = self.flux_to_bias(self._flux_mem)
        self.set_biases(biases)

    def get_flux(self, ii: int):
        return self._flux_mem[ii]

    def set_all_fluxes_to_zero(self):
        self._flux_mem = np.zeros(self.N, dtype=float)
        self.set_biases(self.flux_to_bias(np.zeros(self.N)))

    def set_all_voltages_to_zero(self):
        zero_vec = np.zeros(self.N, dtype=float)
        self.set_biases(zero_vec)
        fluxes = self.bias_to_flux(zero_vec)
        self._flux_mem = fluxes

    def sync_from_hardware(self):
        self._flux_mem = self.bias_to_flux(self.get_biases())

    def get_index_from_name(self, name: str):
        """Get the index of the channel based on its name."""
        if name in self.name_list:
            return self.name_list.index(name)
        raise ValueError(f"Channel name '{name}' not found in the list of channels.")

    def get_ramps(self, flux_to_ramp: str, span: np.ndarray, period: int, duty_cycle_percent: float):
        if not self.qdac_channels:
            raise ValueError("No QDACII channels given to XTalk class.")

        ii = self.get_index_from_name(flux_to_ramp)
        flux_span_vec = np.zeros((self.N))
        flux_span_vec[ii] = span

        inverse_crosstalk_matrix = np.linalg.inv(np.diag(self.mutual_inductance) @ self.xtalk_matrix)
        V_span_vec = inverse_crosstalk_matrix @ flux_span_vec

        triangle_wave_list = []
        for jj, span_V in enumerate(V_span_vec):
            source = self.qdac_channels[jj]
            if span_V < 0:
                triangle = source.triangle_wave(
                    period_s=period,
                    duty_cycle_percent=100 - duty_cycle_percent,
                    offset_V=0,
                    span_V=np.abs(span_V),
                    delay_s=period / 2,
                )  # duty_cycle_percent: goes 1 to 99 voth included
            else:
                triangle = source.triangle_wave(
                    period_s=period, duty_cycle_percent=duty_cycle_percent, offset_V=0, span_V=np.abs(span_V)
                )
            triangle_wave_list.append(triangle)

        return triangle_wave_list

    def get_reset_list(
        self,
        reset_point: list[float],
        operating_point: list[float],
        reset_time: int,
        ramp_up_duration: int,
        time_at_operting_point: int,
        ramp_down_duration: int,
        dwell_us: int,
    ):
        ramp_list_phi_0 = np.array([])
        for ii in range(self.N):
            offset_ramp_up = np.linspace(reset_point[ii], operating_point[ii], int(ramp_up_duration / dwell_us))
            offset_ramp_down = np.linspace(operating_point[ii], reset_point[ii], int(ramp_down_duration / dwell_us))
            offset_pulse = np.ones(int(time_at_operting_point / dwell_us)) * operating_point[ii]
            reset_voltage = np.ones(int(reset_time / dwell_us)) * reset_point[ii]
            full_pulse = np.concatenate([reset_voltage, offset_ramp_up, offset_pulse, offset_ramp_down])

            ramp_list_phi_0 = np.append(ramp_list_phi_0, full_pulse)

        inverse_crosstalk_matrix = np.linalg.inv(np.diag(self.mutual_inductance) @ self.xtalk_matrix)
        flux_list = inverse_crosstalk_matrix @ (ramp_list_phi_0 - self.flux_offsets[:, np.newaxis])

        return flux_list
