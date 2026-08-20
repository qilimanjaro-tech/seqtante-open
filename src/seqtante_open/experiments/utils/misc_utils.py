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
from loguru import logger
from qililab import Calibration, Parameter, Platform
from qililab.typings.enums import InstrumentName


def get_lo_multiple_sources(bus: str, platform: Platform, calibration: Calibration, parameters: dict = {}) -> int:
    if (lo := parameters.get("LO") or calibration.parameters.get("LO", {}).get(bus)) is None:
        bus_object = platform.get_element(alias=bus)
        lo = bus_object.get_parameter(parameter=Parameter.LO_FREQUENCY)
        if (
            next(
                (instrument for instrument in bus_object.instruments if instrument.name == InstrumentName.RSWU_SP16TR),
                None,
            )
            is not None
        ):
            logger.opt(colors=True).warning(
                "{bus} uses an RSWU-SP16TR, but no LO has been provided trough Calibration or the Experiment settings. Continuing with {lo} Hz",
                bus=bus,
                lo=lo,
            )
    return lo
