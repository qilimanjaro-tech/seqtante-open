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

import numpy as np
import qcodes as qc
import qililab as ql
from qililab.platform.platform import Platform

from seqtante_open.experiments.analysis import XTalk


def generate_voltage_param(qdac_channel, param_name):
    setter = qdac_channel.dc_constant_V
    getter = qdac_channel.dc_constant_V

    return qc.Parameter(name=param_name, label=param_name, set_cmd=setter, get_cmd=getter, unit="V")


def qdac_flux_channels_setup(platform_path: str, platform: Platform, parameters: dict):
    voltage_source_alias = parameters['voltage_source_alias']
    qdac = platform.get_element(alias=voltage_source_alias)
    xtalk_params = parameters['xtalk_params']

    if 'flux_z' not in xtalk_params:
        raise Exception("Z loop Xtalk parameters not defined in the configuration")
    if 'flux_x' not in xtalk_params:
        raise Exception("X loop Xtalk parameters not defined in the configuration")

    flux_z_alias = xtalk_params['flux_z']['alias']
    flux_z_name = xtalk_params['flux_z']['name']
    flux_z_dac = xtalk_params['flux_z']['dac']
    flux_z_offset = xtalk_params['flux_z']['offset']

    flux_x_alias = xtalk_params['flux_x']['alias']
    flux_x_name = xtalk_params['flux_x']['name']
    flux_x_dac = xtalk_params['flux_x']['dac']
    flux_x_offset = xtalk_params['flux_x']['offset']

    flux_z_channel = qdac.device.channel(flux_z_dac)
    flux_x_channel = qdac.device.channel(flux_x_dac)

    flux_z_parameter = generate_voltage_param(flux_z_channel, "Fz")
    flux_x_parameter = generate_voltage_param(flux_x_channel, "Fx")

    xtalk_matrix = [[xtalk_params['matrix'][0], xtalk_params['matrix'][1]],
                    [xtalk_params['matrix'][2], xtalk_params['matrix'][3]]]

    xtalk_bot = XTalk(name="xtalk_bot",
                      channels=[flux_z_parameter, flux_x_parameter],
                      name_list=[flux_z_name, flux_x_name],
                      qdac_channels=[flux_z_channel,
                                     flux_x_channel])

    xtalk_bot.xtalk_matrix = np.array(xtalk_matrix)
    xtalk_bot.flux_offsets = np.array([flux_z_offset, flux_x_offset])
    xtalk_bot.mutual_inductance = np.array([xtalk_params['mutual_inductance'][0],
                                            xtalk_params['mutual_inductance'][1]])

    return {
        "xtalk_bot": xtalk_bot,
        "voltage_source": qdac
        }