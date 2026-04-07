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


def subtract_median(arr):
    return arr - np.median(arr)


def divide_by_median(arr):
    return arr / np.median(arr)


def flip_peak_up(arr):
    if np.abs(np.max(arr)) <= np.abs(
        np.min(arr)
    ):  # If the biggest peak is negative, flip the data. Assumed centered around 0
        return arr * -1
    return arr
