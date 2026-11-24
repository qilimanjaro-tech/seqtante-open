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

"""The ``two_tone_vs_flux`` node: the flux sweet spot of one loop, from a two-tone map (ADR-0007, QHC-1804).

The second vertical slice and the root of every pipeline in the EuroQCS tree (``sweet_spot_flux``). One
two-tone saturation map runs per target: a QDAC steps the flux of the step's ``loop`` (``z`` or ``x``)
across a window around the operating point's stored bias, and at every flux the drive IF is swept around
the operating point's stored drive IF. A Lorentzian per flux and a parabola through the qubit IFs give
the sweet spot as the parabola's vertex (``parabola_vertex``).

``apply`` writes that vertex, as a measured flux bias, into the operating point's flux on the swept loop
(ADR-0011). It does not touch the crosstalk matrix or its offsets. The fitted ``sweet_spot_flux_<loop>``
and ``fit_r2`` go to the DB (QHC-1497, QHC-1791).

The measurement is the shared ``two_tone_vs_flux_saturation`` experiment from ``qili-qprograms``
(SDK-403), consumed as a dependency, as the ``t1`` node does (QHC-1803).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qili_qprograms.experiments import two_tone_vs_flux_saturation
from qililab import Parameter
from qililab.result import StreamArray

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import AnalysisResult, FluxSweepNode
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import FluxoniumTwoToneFluxModel
from seqtante_open.experiments.results import measurement_id
from seqtante_open.experiments.utils.qdac_utils import get_qdac_out_trigger, qdac_step_timings
from seqtante_open.outputs import output_controller
from seqtante_open.targets import QubitTarget

if TYPE_CHECKING:
    from collections.abc import Mapping

    from seqtante_open.calibration.model import FluxLoop
    from seqtante_open.targets import Target

PARABOLA_VERTEX = "parabola_vertex"

#: Stream/experiment name the map is stored under.
_EXPERIMENT_NAME = "two_tone_vs_flux_saturation"

#: Defaults for the optional step parameters (the node's ``defaults``); every one is overridable in the
#: tree. The measurement keys are named like the ``two_tone_vs_flux_saturation`` arguments. ``flux_sweep``,
#: ``drive_if_sweep``, ``averages``, ``relax_duration``, ``drive_duration`` and ``drive_amplitude`` have no
#: default, and the drive instrument decides which of ``drive_power`` (Rohde & Schwarz, dBm) or
#: ``drive_gain`` (QCM-RF, [-1, 1]) the step must set.
TWO_TONE_VS_FLUX_DEFAULTS: Mapping[str, Any] = {
    "loop": "z",
    "drive_ringup_time": 200,
    "drive_ringdown_time": 200,
    "drive_n_sigmas": 4,
    "overlap_time": 0,
    "readout_ringup_time": 0,
    # Times the whole map is measured; the stored map is the running average over them.
    "repetitions": 1,
    # QDAC settle wait after each step and idle time before the next, in ns. ``None`` computes them from
    # the QDAC low-pass filters (``qdac_step_timings``); set both or neither.
    "flux_settle_duration": None,
    "flux_pre_step_duration": None,
    # Minimum R² the parabola through the per-flux qubit IFs must reach to pass the step.
    "min_r2": 0.8,
}


@dataclass(frozen=True)
class SweetSpotFlux:
    """What ``analyze`` hands ``apply``: the measured sweet-spot bias on one loop, in Φ0."""

    loop: FluxLoop
    flux: float


class TwoToneVsFluxNode(FluxSweepNode[SweetSpotFlux]):
    """A two-tone map against one flux loop, fit to a parabola whose vertex is the sweet spot (QHC-1804).

    A :class:`FluxSweepNode`: ``run`` does not place the operating point, the node does, right before the
    sweep, so the loop it does not sweep and the neighbours the point parks sit at their stored biases
    while the QDAC ramps the swept loop.
    """

    experiment = "two_tone_vs_flux"
    analyses = (PARABOLA_VERTEX,)
    defaults = TWO_TONE_VS_FLUX_DEFAULTS

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the two-tone map on each target around its operating point; return its measurement id."""
        return {target: [self._run_map(target)] for target in targets}

    def _run_map(self, target: Target) -> int:
        """Place the point, set its hardware, build the QDAC-ramped two-tone QProgram, run it and stream it.

        The flux and drive-IF windows are offsets from the operating point's stored flux on the swept loop
        and its stored drive IF, so the same tree recentres on every recalibration. The stored map holds
        the absolute fluxes (Φ0) and drive IFs (Hz) that were played.

        Raises:
            ValueError: if the target is a coupler, the loop is not ``z``/``x``, or the operating point
                stores no bias on the swept loop.
        """
        if not isinstance(target, QubitTarget):
            raise ValueError(
                f"{target}: two_tone_vs_flux runs on qubits only; a coupler has no drive line of its own "
                "and the shared experiment has no readout-qubit mapping"
            )
        loop = self._flux_loop(target)
        section = self.calibration_file.experiments[self.experiment][str(target)]
        operating_point = operating_point_for(self.calibration_file, self.experiment, target)
        fluxes = self._absolute_sweep(target, "flux_sweep", operating_point.flux.on(loop))
        drive_ifs = self._absolute_sweep(target, "drive_if_sweep", operating_point.drive.if_)

        platform = self.context.platform
        readout_bus = bus_alias(target, Role.readout)
        settle, pre_step = qdac_step_timings(
            platform,
            self._param(target, "flux_settle_duration"),
            self._param(target, "flux_pre_step_duration"),
        )
        qprogram = two_tone_vs_flux_saturation(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=section.operating_point,
            flux_sweep=fluxes,
            drive_if_sweep=drive_ifs,
            averages=self._param(target, "averages"),
            relax_duration=self._param(target, "relax_duration"),
            drive_duration=self._param(target, "drive_duration"),
            drive_amplitude=self._param(target, "drive_amplitude"),
            drive_ringup_time=self._param(target, "drive_ringup_time"),
            flux_settle_duration=math.ceil(settle),
            readout_time_of_flight=int(platform.get_parameter(alias=readout_bus, parameter=Parameter.TIME_OF_FLIGHT)),
            flux_axis=loop,
            drive_ringdown_time=self._param(target, "drive_ringdown_time"),
            drive_n_sigmas=self._param(target, "drive_n_sigmas"),
            overlap_time=self._param(target, "overlap_time"),
            flux_pre_step_duration=math.ceil(pre_step),
            qdac_trigger_channel=get_qdac_out_trigger(platform),
            readout_ringup_time=self._param(target, "readout_ringup_time"),
        )

        # The unswept loop and the parked neighbours sit at their stored biases; the QDAC ramps the swept one.
        self._place_operating_point(target)
        self._set_operating_point_hardware(target)
        return self._stream(target, qprogram, fluxes, drive_ifs, loop)

    def _absolute_sweep(self, target: Target, key: str, centre: float) -> np.ndarray:
        """The step's ``key`` window (``np.linspace`` arguments, as offsets) shifted onto ``centre``."""
        return centre + np.linspace(*self._param(target, key))

    def _stream(self, target: Target, qprogram: Any, fluxes: np.ndarray, drive_ifs: np.ndarray, loop: FluxLoop) -> int:
        """Run the map ``repetitions`` times and stream the running average; return the measurement id."""
        platform = self.context.platform
        readout_bus = bus_alias(target, Role.readout)
        flux_bus = bus_alias(target, Role.flux, loop)
        repetitions = self._param(target, "repetitions")
        stream_array = StreamArray(
            shape=(len(fluxes), len(drive_ifs), 2),
            # The loop names are the ones FluxoniumTwoToneFluxModel reads back.
            loops={
                "flux": {"array": fluxes, "units": "phi_0", "bus": flux_bus, "parameter": Parameter.FLUX.value},
                "IF_frequency": {
                    "array": drive_ifs,
                    "units": "Hz",
                    "bus": bus_alias(target, Role.drive),
                    "parameter": "IF_frequency",
                },
            },
            platform=platform,
            experiment_name=_EXPERIMENT_NAME,
            db_manager=output_controller.db_manager,
            qprogram=qprogram,
            calibration=self.context.calibration,
            # The swept bus keeps the z and x maps of one qubit apart in the DB (ADR-0007).
            optional_identifier=flux_bus,
            qubit_idx=str(target),
        )
        total = np.zeros((len(fluxes), len(drive_ifs), 2))
        with stream_array:
            for repetition in range(repetitions):
                results = platform.execute_qprogram(qprogram).results
                # One acquisition, shaped (IQ, flux, drive IF): reorder it to the stored (flux, drive IF, IQ).
                total += np.asarray(results[readout_bus][0].array).transpose(1, 2, 0)
                stream_array[()] = total / (repetition + 1)
        return measurement_id(stream_array)

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[SweetSpotFlux]:
        """Fit the map, check it, save ``sweet_spot_flux_<loop>``/``fit_r2``, plot, and return the vertex.

        Fit, then the acceptance checks, then save, then plot: a rejected fit never reaches the DB.

        Raises:
            AnalysisError: when fewer than three fluxes give a qubit line, the parabola has no vertex inside
                the swept window, or its R² is below the step's minimum or undefined. The target then
                fails the step.
        """
        loop = self._flux_loop(target)
        operating_point = operating_point_for(self.calibration_file, self.experiment, target)
        model = FluxoniumTwoToneFluxModel(
            measurements[-1],
            target=str(target),
            # The graph sets data_folder to <data_root>/<step>/<target>; None shows the plot instead.
            path=self.parameters[target].get("data_folder"),
            lo=operating_point.drive.lo,
            flux_bus=bus_alias(target, Role.flux, loop),
        )
        model.fit()
        if not model.fitted:
            raise AnalysisError(f"{target} two-tone vs flux on '{loop}': no sweet spot inside the swept window")

        min_r2 = self._param(target, "min_r2")
        # Every comparison with NaN is False, so an undefined R² is rejected explicitly.
        if math.isnan(model.r_squared) or model.r_squared < min_r2:
            raise AnalysisError(f"{target} parabola R² {model.r_squared:.3f} is below the minimum {min_r2}")

        model.save_fitting(loop)
        plot = model.plot()
        return AnalysisResult(fitted=SweetSpotFlux(loop=loop, flux=model.center), plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[SweetSpotFlux]) -> None:
        """Store the measured sweet-spot bias on the swept loop of the operating point (ADR-0011).

        The crosstalk matrix and its offsets are left untouched; no other operating point moves.
        """
        # analyze returns a vertex or raises, so apply always has one; the guard is a type narrow.
        if result.fitted is None:
            return
        operating_point = operating_point_for(self.calibration_file, self.experiment, target)
        if result.fitted.loop == "z":
            operating_point.flux.z = result.fitted.flux
        else:
            operating_point.flux.x = result.fitted.flux
