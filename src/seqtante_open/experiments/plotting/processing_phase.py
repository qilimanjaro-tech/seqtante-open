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

from seqtante_open.experiments.analysis.analysis import normalize
from seqtante_open.experiments.plotting import flip_peak_up, subtract_median


def phase(S21):
    return np.angle(S21)


def unwrapped_phase(S21):
    return np.unwrap(np.angle(S21))


def center_phase_around_median(S21):
    centered = np.apply_along_axis(subtract_median, axis=0, arr=phase(S21))
    return ((centered + np.pi) % (2 * np.pi)) - np.pi


def center_phase_around_median_subtract_median_row(S21):
    return np.apply_along_axis(subtract_median, axis=1, arr=center_phase_around_median(S21))


def center_unwrapped_phase_around_median(S21):
    centered = np.apply_along_axis(subtract_median, axis=0, arr=unwrapped_phase(S21))
    return ((centered + np.pi) % (2 * np.pi)) - np.pi


def center_phase_around_median_normalized(S21):
    return np.apply_along_axis(normalize, axis=0, arr=center_phase_around_median(S21))


def center_phase_around_median_flip_peak(S21):
    no_flip = center_phase_around_median(S21)  # data centered around zero
    flipped = np.apply_along_axis(flip_peak_up, axis=0, arr=no_flip)
    return flipped


def center_phase_around_median_flip_peak_norm(S21):
    return np.apply_along_axis(normalize, axis=0, arr=center_phase_around_median(S21))
