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

"""This module contains the experiments built with QProgram.

.. currentmodule:: presets

"""
from .rabi import (
    rabi__amplitude_gaussian_ac,
    rabi__amplitude_gaussian_drive,
    rabi__amplitude_gaussian_hw_loop,
    rabi__time,
    rabi__time_frequency,
    rabi__time_gain,
    rabi__time_reset,
)
from .offset_calibration import (
    single_tone__frequency_vs_flux_cw_dc,
)
from .t1 import t1, t1_hardware_loop, t1_reset_experiment
from .t2 import t2, t2_echo, t2_echo_hardware_loop, t2_hardware_loop
from .two_tone import (
    raw_trace_two_tone_experiment,
    raw_trace_two_tone_readout_optimization,
    two_tone__frequency_cw,
    two_tone__frequency_d_drag,
    two_tone__frequency_pulsed,
    two_tone__frequency_pulsed_ac,
    two_tone__frequency_pulsed_ChangeLo,
    two_tone__frequency_pulsed_repeat_in_time,
    two_tone__frequency_pulsed_reset,
    two_tone__frequency_r_pulsed_d_cw_reset,
    two_tone__frequency_vs_d_amplitude_pulsed,
    two_tone__frequency_vs_d_gain_pulsed,
    two_tone__frequency_vs_flux_cw_dc,
    two_tone__frequency_vs_flux_cw_dc_change_readout_if,
    two_tone__frequency_vs_flux_pulsed_dc,
    two_tone__frequency_vs_flux_pulsed_dc_change_readout_if,
    two_tone__frequency_vs_flux_pulsed_dc_ChangeLo,
    two_tone__frequency_vs_flux_pulsed_reset,
    two_tone__frequency_vs_flux_r_pulsed_d_cw_reset,
    two_tone__frequency_vs_gain_cw,
    two_tone__frequency_vs_r_amplitude_pulsed,
    two_tone__frequency_vs_wait_time_pulsed_ac_change_drive_lo,
    two_tone__ro_frequency_pulsed,
)
