# Copyright 2023 Qilimanjaro Quantum Tech
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

from seqtante_open.experiments.analysis import decibels
from seqtante_open.experiments.plotting import divide_by_median, subtract_median


def amplitude_dB(S21):
    return decibels(S21)


def amplitude_linear(S21):
    return np.abs(S21)


def amplitude_linear_divide_by_median_col(S21):
    return np.apply_along_axis(divide_by_median, axis=0, arr=amplitude_linear(S21))


def amplitude_linear_divide_by_median_col_row(S21):
    return np.apply_along_axis(divide_by_median, axis=1, arr=amplitude_linear_divide_by_median_col(S21))


def amplitude_linear_subtract_mean_col(S21):
    return np.apply_along_axis(subtract_median, axis=0, arr=amplitude_linear(S21))


def amplitude_linear_subtract_mean_col_row(S21):
    return np.apply_along_axis(subtract_median, axis=1, arr=amplitude_linear_subtract_mean_col(S21))
