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
from collections.abc import MutableMapping
from typing import TypeAlias

from loguru import logger
from qililab import Calibration, Parameter, Platform
from qililab.typings.enums import InstrumentName

_Bus: TypeAlias = str
_ParameterName: TypeAlias = str
_OperatingPoint: TypeAlias = MutableMapping[_ParameterName, MutableMapping[_Bus, float]]

_OPERATING_POINT_KEY = "operating_point"


def get_lo_multiple_sources(bus: str, target: str, platform: Platform, calibration: Calibration) -> int:
    if (lo := calibration.parameters.get("LO", {}).get(target)) is None:
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


def get_operating_point(
    calibration: Calibration, target: str, operating_point_name: str, platform: Platform | None = None
) -> _OperatingPoint:
    """Retrieve an operating point from the calibration, optionally applying it to the hardware.

    Args:
        calibration (Calibration): Calibration holding the operating point.
        target (str): Qubit or coupler the operating point belongs to.
        operating_point_name (str): Name of the operating point to retrieve.
        platform (Platform | None): When given, the operating point is applied to the hardware before
            being returned. Defaults to None.

    Raises:
        KeyError: If the target or the operating point is not in the calibration.

    Returns:
        _OperatingPoint: The stored operating point.
    """
    operating_point: _OperatingPoint = calibration.parameters[_OPERATING_POINT_KEY][target][operating_point_name]
    if platform is not None:
        set_to_operating_point(platform=platform, operating_point=operating_point)
    return operating_point


def set_to_operating_point(platform: Platform, operating_point: _OperatingPoint) -> None:
    """Apply every parameter value of an operating point to the hardware.

    Parameters and buses are applied in insertion order, so the order they were saved in is the order
    they are set.

    Args:
        platform (Platform): Platform to set the parameters on.
        operating_point (_OperatingPoint): Operating point to apply.

    Raises:
        KeyError: If a stored key is not a ``Parameter`` name.
    """
    for parameter_name, changes in operating_point.items():
        for bus, value in changes.items():
            platform.set_parameter(alias=bus, parameter=Parameter[parameter_name], value=value)


def save_parameters_to_operating_point(
    *settings: tuple[_Bus, Parameter, float], calibration: Calibration, target: str, operating_point_name: str
) -> _OperatingPoint:
    """Record bus parameter values inside an operating point of the calibration.

    Parameters are keyed by enum name (``FLUX``, never the ``flux`` value), the single spelling
    ``set_to_operating_point`` reads back. If the operating point does not exist yet, a new one is
    created. A value already stored for the same parameter and bus is updated.

    Writing the calibration back to disk is left to the caller.

    Args:
        *settings (tuple[_Bus, Parameter, float]): Bus alias, parameter and value to store, in that order.
        calibration (Calibration): Calibration whose parameters are updated in place.
        target (str): Qubit or coupler the operating point belongs to.
        operating_point_name (str): Name of the operating point to write into.

    Returns:
        _OperatingPoint: The updated operating point.
    """
    operating_point: _OperatingPoint = (
        calibration.parameters.setdefault(_OPERATING_POINT_KEY, {})
        .setdefault(target, {})
        .setdefault(operating_point_name, {})
    )
    for bus, parameter, value in settings:
        operating_point.setdefault(parameter.name, {})[bus] = value
    return operating_point
