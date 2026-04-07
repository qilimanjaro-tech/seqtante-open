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

from seqtante_open.experiments.analysis import rotate_iq
from seqtante_open.experiments.analysis.analysis import subtract_mean
from seqtante_open.experiments.plotting import divide_by_median, subtract_median


def rotated_IQ(S21):
    return np.real(np.apply_along_axis(rotate_iq, axis=0, arr=S21))


def rotated_IQ_divide_by_median_col(S21):
    return np.apply_along_axis(divide_by_median, axis=0, arr=rotated_IQ(S21))


def rotated_IQ_divide_by_median_col_row(S21):
    return np.apply_along_axis(divide_by_median, axis=1, arr=rotated_IQ_divide_by_median_col(S21))


def rotated_IQ_subtract_mean_col(S21):
    return np.apply_along_axis(subtract_mean, axis=0, arr=rotated_IQ(S21))


def rotated_IQ_subtract_median_col(S21):
    return np.apply_along_axis(subtract_median, axis=0, arr=rotated_IQ(S21))


def rotated_IQ_subtract_mean_col_row(S21):
    return np.apply_along_axis(subtract_mean, axis=1, arr=rotated_IQ_subtract_mean_col(S21))


def rotated_IQ_subtract_median_col_row(S21):
    return np.apply_along_axis(subtract_median, axis=1, arr=rotated_IQ_subtract_median_col(S21))
