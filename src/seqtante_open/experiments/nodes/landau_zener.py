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

"""The ``landau_zener`` node: a compartmentalized Landau-Zener benchmark at the reset point (QHC-1808).

The measurement is the shared ``generalized_linear_ramps`` experiment from ``qili-qprograms`` (SDK-402),
consumed as a dependency, as the ``t1`` node does (QHC-1803). One QProgram is built per flux path in the
step's ``flux_paths``; each repetition holds the fast flux at the reset point, ramps onto the path, walks
it, ramps to the readout point and reads out. The base biases the slow flux lines to the step's operating
point (the idle point) before ``execute_node``; the fast-flux levels are relative to it.

``analyze`` normalises the measured signal to a transition probability with the readout levels and plots
it; ``apply`` writes nothing, as a benchmark does (ADR-0002). The benchmark is left **"Not evaluated"**:
the agreement metric and the theory curve that give it a verdict are a follow-up (QHC-1921), so
``analyze`` reports no value and the run does not score it.

Where the inputs come from:

- The calibration file: the idle point and the readout point are the step's operating point, which also
  gives the readout pulse (``readout-<operating_point>``), its weights, IF, LO and attenuation, and the
  |0>/|1> readout levels. The reset point is the operating point the step's ``reset_point`` names (the
  step's own by default).
- The step's tree parameters: the flux paths, the entry/exit ramp durations, the reset hold, the
  averages and the readout ring-up. Each path's total annealing time, the sweep's axis, is its duration.
- The AC crosstalk on the fast-flux lines: qililab corrects the levels for it when it compiles the
  QProgram. The step's ``ac_crosstalk`` (a raw 2x2 array over the ``z``/``x`` fast-flux buses) is used
  when set, as on BSC; otherwise the calibration's ``crosstalk_matrix_ac``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from qili_qprograms.bus_alias import Role, bus_alias
from qili_qprograms.experiments import generalized_linear_ramps
from qili_qprograms.flux_path import FluxPath
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.experiments.fitting.landau_zener_fit import LandauZenerFit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qili_qprograms.calibration.model import ReadoutLevels
    from qili_qprograms.targets import Target
    from qililab import QProgram
    from qililab.qprogram.calibration import Calibration

THEORY_AGREEMENT = "theory_agreement"

#: Stream/experiment name the sweep is stored under.
_EXPERIMENT_NAME = "generalized_linear_ramps"

#: Defaults for the optional step parameters (the node's ``defaults``); every one is overridable in the
#: tree. The measurement keys are named like the ``generalized_linear_ramps`` arguments. ``flux_paths``,
#: ``averages`` and ``relax_duration`` have no default. A ramp duration is needed only
#: when a path does not start at the reset point's flux (entry) or end at the readout point's (exit).
LANDAU_ZENER_DEFAULTS: Mapping[str, Any] = {
    "reset_point": None,
    "entry_ramp_duration": None,
    "entry_ramp_move_first": "z",
    "exit_ramp_duration": None,
    "exit_ramp_move_first": "both",
    "readout_ringup_time": 0,
    "ac_crosstalk": None,
}


class LandauZenerNode(FixedFluxNode[None]):
    """A compartmentalized Landau-Zener transition at the reset point, scored as a benchmark (QHC-1808).

    The benchmark unit (dimensionless) and direction (lower-is-better) come from the ``landau_zener``
    entry in ``NODE_REGISTRY``. Until the agreement metric and theory curve land (QHC-1921) the node
    reports no value, so the run leaves the benchmark "Not evaluated".
    """

    experiment = "landau_zener"
    analyses = (THEORY_AGREEMENT,)
    defaults = LANDAU_ZENER_DEFAULTS

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        #: Each target's |0>/|1> readout levels, checked by :meth:`_run_sweep` before it measures.
        self._levels: dict[Target, ReadoutLevels] = {}

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the Landau-Zener sweep on each target at its operating point; return its measurement id."""
        return {target: [self._run_sweep(target)] for target in targets}

    def _run_sweep(self, target: Target) -> int:
        """Build one QProgram per flux path, set the readout line, run them in turn and stream the results.

        The readout pulse and levels are checked and every QProgram is built before anything is measured
        (``run`` has already biased the slow flux lines), so a malformed or over-long path, or levels the
        signal cannot be normalised with, fail the target without measuring. It executes the QPrograms itself rather than through
        :meth:`_stream`, which repeats one QProgram compiled with the run's calibration: here each row is
        its own QProgram, compiled with the AC crosstalk of :meth:`_with_ac_crosstalk`, and stores the
        threshold. The stream array is the shared :meth:`_stream_array`.

        Raises:
            TypeError: if a path's waypoint is not a ``(phi_z, phi_x, samples)`` triple.
            ValueError: if there are no flux paths, a path is malformed or overruns the Qblox waveform
                memory (raised by the QProgram), the readout levels are missing or were taken with another
                pulse (:meth:`_readout_levels`), or no AC crosstalk is available for the fast-flux lines.
        """
        flux_paths = self._param(target, "flux_paths")
        if not flux_paths:
            raise ValueError(f"{target}: flux_paths is empty; there is nothing to sweep")
        annealing_times = [self._annealing_time(path) for path in flux_paths]

        # Before measuring: the readout pulse the QProgram plays must be stored, with levels taken with it.
        self._require_pulse(target, Role.readout, "readout")
        self._levels[target] = self._readout_levels(target)
        operating_point = self._operating_point_name(target)
        reset_point = self._param(target, "reset_point") or operating_point
        calibration = self._with_ac_crosstalk(target)
        qprograms = [self._build(target, operating_point, reset_point, path) for path in flux_paths]

        self._set_readout_line(target)

        platform = self.context.platform
        readout_bus = bus_alias(target, Role.readout)
        stream_array = self._stream_array(
            target,
            qprograms[0],
            experiment_name=_EXPERIMENT_NAME,
            shape=(len(annealing_times), 3),
            # The loop name is the one LandauZenerFit reads back.
            loops={
                "total_annealing_time": {
                    "array": annealing_times,
                    "units": "ns",
                    "bus": bus_alias(target, Role.fast_flux, "z"),
                    "parameter": "Total_annealing_time",
                },
            },
            calibration=calibration,
        )
        with stream_array:
            for idx, qprogram in enumerate(qprograms):
                result = platform.execute_qprogram(qprogram, calibration=calibration).results[readout_bus][0]
                stream_array[idx, :2] = result.array.T
                stream_array[idx, 2] = result.threshold[0]
        return self._measurement_id(stream_array)

    def _build(self, target: Target, operating_point: str, reset_point: str, flux_path: Any) -> QProgram:
        """The ``generalized_linear_ramps`` QProgram for one flux path, idle and read out at ``operating_point``."""
        return generalized_linear_ramps(
            calibration=self.context.calibration,
            target=str(target),
            idle_operating_point=operating_point,
            reset_point=reset_point,
            readout_point=operating_point,
            flux_path=flux_path,
            averages=self._param(target, "averages"),
            relax_duration=self._param(target, "relax_duration"),
            readout_ringup_time=self._param(target, "readout_ringup_time"),
            entry_ramp_duration=self._ramp_duration(target, "entry_ramp_duration"),
            entry_ramp_move_first=self._param(target, "entry_ramp_move_first"),
            exit_ramp_duration=self._ramp_duration(target, "exit_ramp_duration"),
            exit_ramp_move_first=self._param(target, "exit_ramp_move_first"),
        )

    @staticmethod
    def _annealing_time(flux_path: Any) -> int:
        """A path's total annealing time, in ns: the duration of the ``FluxPath`` it describes.

        Waypoints are read as ``generalized_linear_ramps`` reads them, starting at the first, so it holds.
        """
        if isinstance(flux_path, FluxPath):
            return flux_path.duration
        return FluxPath.from_waypoints(flux_path, start=(flux_path[0][0], flux_path[0][1])).duration

    def _readout_levels(self, target: Target) -> ReadoutLevels:
        """The operating point's |0>/|1> readout levels, checked to be taken with the pulse the QProgram plays.

        Raises:
            ValueError: if the operating point stores no readout levels, or stores them for another
                readout pulse (ADR-0001, D8).
        """
        point_name = self._operating_point_name(target)
        # The pulse generalized_linear_ramps plays at its readout point, the step's operating point.
        readout_pulse = self._pulse(target, Role.readout, "readout")
        levels = self._operating_point(target).readout.levels
        if levels is None:
            raise ValueError(f"{target}: operating point '{point_name}' stores no readout levels to normalise with")
        if levels.pulse != readout_pulse:
            raise ValueError(
                f"{target}: the readout levels at operating point '{point_name}' were taken with pulse "
                f"'{levels.pulse}', but the QProgram plays '{readout_pulse}'; refusing to use them"
            )
        return levels

    def _ramp_duration(self, target: Target, key: str) -> int | tuple[int, int] | None:
        """The step's ``key`` ramp duration; a YAML list ``[z, x]`` becomes the ``(z, x)`` pair the QProgram takes."""
        duration = self._param(target, key)
        return tuple(duration) if isinstance(duration, list) else duration

    def _with_ac_crosstalk(self, target: Target) -> Calibration:
        """The calibration the QProgram is compiled with, holding the AC crosstalk of the fast-flux lines.

        With the step's ``ac_crosstalk`` set, a shallow copy of the run's calibration carrying it as
        ``crosstalk_matrix_ac`` (the run's calibration is left as it was); otherwise the run's own, which
        must hold a ``crosstalk_matrix_ac``: without one qililab would fall back to the DC matrix, which
        does not cover the fast-flux buses.

        Raises:
            ValueError: if there is no AC crosstalk, or it has non-zero flux offsets on the fast-flux buses
                (qililab would subtract them from levels already relative to the idle point).
        """
        calibration = self.context.calibration
        fast_flux_buses = [
            bus_alias(target, Role.fast_flux, "z"),
            bus_alias(target, Role.fast_flux, "x"),
        ]
        ac_crosstalk = self._param(target, "ac_crosstalk")
        if ac_crosstalk is not None:
            calibration = copy.copy(calibration)
            calibration.crosstalk_matrix_ac = CrosstalkMatrix.from_array(
                buses=fast_flux_buses, matrix_array=np.asarray(ac_crosstalk, dtype=float)
            )
        matrix = calibration.crosstalk_matrix_ac
        if matrix is None:
            raise ValueError(
                f"{target}: no AC crosstalk for the fast-flux lines; set the step's ac_crosstalk or the "
                "calibration's crosstalk_matrix_ac"
            )
        if any(matrix.flux_offsets.get(bus, 0) != 0 for bus in fast_flux_buses):
            raise ValueError(
                f"{target}: the AC crosstalk has non-zero flux offsets on {fast_flux_buses}; the fast-flux levels "
                "are already relative to the idle point, so they must be zero"
            )
        return calibration

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[None]:
        """Normalise the signal with the readout levels and plot it; report no value yet.

        The |0>/|1> readout levels are the operating point's, checked by :meth:`_run_sweep` before it
        measured (:meth:`_readout_levels`). The agreement metric that would give a value (and so a
        verdict) is a follow-up (QHC-1921), so ``value`` is ``None`` and the run leaves the benchmark unscored.
        """
        levels = self._levels[target]
        model = LandauZenerFit(
            measurement_id=measurements[-1],
            target=str(target),
            level_0=levels.level_0.nominal,
            level_1=levels.level_1.nominal,
            path=self.parameters[target].get("data_folder"),
        )
        plot = model.plot()
        # MVP: no agreement metric yet (QHC-1921), so report no value; the run leaves this unscored.
        return AnalysisResult(value=None, plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[None]) -> None:
        """A benchmark writes nothing to the calibration file; the run records the verdict (ADR-0002)."""
