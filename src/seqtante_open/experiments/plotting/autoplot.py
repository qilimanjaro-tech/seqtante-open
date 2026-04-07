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

from typing import Callable

import numpy as np
from qililab.data_management import build_platform
from qililab.platform import Platform
from qililab.result import Measurement
from qililab.typings.enums import InstrumentName, Parameter
from xarray import DataArray

from seqtante_open.experiments.analysis import decibels
from seqtante_open.experiments.plotting import (
    convert_plot_units,
    get_xarray_from_meas,
    plot_measurement_1d_freq_updated,
    plot_measurement_1d_line_updated,
    plot_measurement_2d_heatmap_updated,
    plot_measurement_2d_line_updated,
    plot_measurement_3d_heatmap_grid_updated,
    plot_measurement_3d_heatmap_slider_updated,
    plot_two_tone_readout_optimization,
)


def correct_tof(xarr: DataArray, platform: Platform, tof: float | None = None):
    """
    Apply a phase correction to the data in `xarr` based on time-of-flight (tof)
    compensation along IF or LO frequency axes.

    Parameters:
    - xarr: xarray.DataArray
    - platform: Qililab platform instance
    - tof: Optional override in seconds (float). If None, read from hardware.

    Returns:
    - success: True if correction was applied, False otherwise
    """
    for ii, coord in enumerate(xarr.coords.values()):
        bus_name = coord.bus
        if bus_name in [bus.alias for bus in platform.buses]:

            instrument = platform.buses.get(alias=bus_name).instruments[0]

            # Skip if this isn't a readout bus
            if coord.parameter in ("IF_frequency", "LO_frequency", "frequency") and (
                instrument.name == InstrumentName.QBLOX_QRM
                or instrument.name == InstrumentName.QRMRF or instrument.name == InstrumentName.KEYSIGHT_E5080B
                # TODO: add or instrument.name == InstrumentName.QUANTUM_MACHINES_CLUSTER
            ):
                # Get time of flight in seconds
                if tof is None and instrument.name != InstrumentName.KEYSIGHT_E5080B:  # TODO VNA does not have TOF parameter in runcard yet, when added in Qililab, update the functionality below accortdingly.
                    tof = platform.get_parameter(bus_name, parameter=Parameter.TIME_OF_FLIGHT) * 1e-9
                elif tof is None:
                    tof = 0.0  # No TOF correction for VNA
                else:
                    tof = tof * 1e-9  # Convert from ns to seconds if user-provided

                # Get frequency array
                freq = xarr.coords[xarr.dims[ii]].values

                if coord.parameter == "LO_frequency" or coord.parameter == "frequency":
                    phase = np.exp(2j * np.pi * freq * tof)
                else:  # IF frequency
                    LO_freq = platform.get_parameter(bus_name, parameter=Parameter.LO_FREQUENCY)
                    phase = np.exp(2j * np.pi * (freq + LO_freq) * tof)

                # Apply the phase correction along the appropriate axis
                xarr.values *= phase.reshape([-1 if jj == ii else 1 for jj in range(xarr.ndim)])

    return xarr


def auto_plot(
    measurement: Measurement,
    plot_type: str | None = None,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    xarr: DataArray | None = None,
    dataprocessing: Callable | None = decibels,
    tof: int | None = None,
    unwrap: bool = False,
):
    """
    Should return the correct plot based on what is in the measurement
    """
    if not xarr:
        # The default is to read it automatically here. But if you want to do some post-processing you can pass an xarray as input as well
        xarr = get_xarray_from_meas(measurement)
    if measurement.platform:
        xarr = correct_tof(xarr, platform=build_platform(measurement.platform), tof=tof)  # type: ignore

    xarr = convert_plot_units(xarr)
    coords = xarr.coords  # type: ignore

    title = (
        f"{measurement.experiment_name}, id = {measurement.measurement_id}"
        + f"<br>{measurement.sample_name},  {measurement.cooldown}"
    )

    if measurement.experiment_name == "raw_trace_two_tone_readout_optimization":
        return plot_two_tone_readout_optimization(measurement=measurement, title=title)

    fixed_LO_freq = None
    if measurement.platform:
        for coord in coords.values():
            if coord.parameter == "IF_frequency":
                bus = coord.bus
                fixed_LO_freq = build_platform(measurement.platform).get_parameter(bus, parameter=Parameter.LO_FREQUENCY)  # type:ignore [arg-type]

    if len(coords) == 1:
        if plot_type == "line":
            return plot_measurement_1d_line_updated(
                xarr=xarr, title=title, fixed_LO_freq=fixed_LO_freq, dataprocessing=dataprocessing
            )
        return plot_measurement_1d_freq_updated(
            xarr=xarr,  # NOTE fix logix here to work with VNA (LO) and when your frequency is not a readout.
            title=title,
            fixed_LO_freq=fixed_LO_freq,
            mag_phase_IQ_unwrap=unwrap,
        )

    if len(coords) == 2:

        if x:
            xarr = xarr.transpose(..., x)  # type: ignore
        elif y:
            xarr = xarr.transpose(y, ...)  # type: ignore
        else:
            for dim, coord in xarr.coords.items():  # type: ignore
                if plot_type == "line":
                    # We want different default dims for frequency, depending on iof it is a line plot or heatmap
                    xarr = xarr.transpose(..., dim)  # type: ignore
                else:
                    xarr = xarr.transpose(dim, ...)  # type: ignore
                break

        if plot_type == "line":
            xarr = xarr.transpose()  # type: ignore
            return plot_measurement_2d_line_updated(
                xarr=xarr, title=title, fixed_LO_freq=fixed_LO_freq, dataprocessing=dataprocessing
            )
        return plot_measurement_2d_heatmap_updated(
            xarr=xarr, title=title, fixed_LO_freq=fixed_LO_freq, dataprocessing=dataprocessing
        )
    if len(coords) == 3:

        if not any([x, y, z]):
            # We need to transpose here because x is used as the first dim for this type.
            for dim, coord in xarr.coords.items():  # type: ignore
                if getattr(coord, "parameter", None) in ("IF_frequency", "LO_frequency"):
                    y_dim = dim
                    z_dim = min(xarr.sizes, key=xarr.sizes.get)  # type: ignore
                    remaining = [d for d in xarr.dims if d not in (y_dim, z_dim)]  # type: ignore
                    if len(remaining) != 1:
                        raise ValueError(f"Cannot determine unique x dimension. Remaining: {remaining}")
                    x_dim = remaining[0]
                    # Transpose in correct order: y (rows), x (cols), z (slider)
                    xarr = xarr.transpose(y_dim, x_dim, z_dim)  # type: ignore
                    break

        elif [x, y, z].count(None) > 1:
            raise (Exception("specify at least two variables"))

        else:
            t_list = []
            for dim in [y, x, z]:  # The order here is not x,y,z because of how px.imshow works
                if not dim:
                    dim = ...
                t_list.append(dim)
            xarr = xarr.transpose(*t_list)  # type: ignore

        if plot_type == "slider":
            return plot_measurement_3d_heatmap_slider_updated(
                xarr=xarr, title=title, fixed_LO_freq=fixed_LO_freq, dataprocessing=dataprocessing
            )
        return plot_measurement_3d_heatmap_grid_updated(
            xarr=xarr, title=title, dataprocessing=dataprocessing
        )
    raise (Exception("4-dim and higher data is not supported for automatic plotting"))
