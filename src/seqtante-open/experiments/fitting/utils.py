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
"""Module for handling correction and fitting of Two Tone data.
This module provides classes and functions to correct
and fit Two Tone data.
Classes:
    TwoToneFit: Handles the correction and fitting
    of Two Tone data.
Functions:
    fit: Fit the two tone data.
    plot: plot the fitted data.
"""

import numpy as np


def cosfunc(phi, A, omega, offset, phase_offset):
    return offset + A * np.cos(omega * phi + phase_offset)