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

"""The ``landau_zener`` node: a compartmentalized Landau-Zener benchmark, scored against theory (ADR-0012).

The measurement is the shared ``generalized_linear_ramps`` experiment from ``qili-qprograms`` (SDK-402),
consumed as a dependency, as the ``t1`` node does (QHC-1803). Each row of the target's simulated-points
table fixes a φx and a linear z ramp from ``z_start`` to ``z_end``; the step's ``ramp_durations`` are the
durations T that ramp is played in. One QProgram is built per (φx, T), as ``generalized_linear_ramps``
plays one path: each repetition holds the fast flux at the reset point, ramps onto the row's start,
plays the z ramp in T, ramps to the readout point and reads out. φx is the outer loop of the stored
measurement and T the inner one, so the data is one φx-by-T surface. The base biases the slow flux lines
to the step's operating point (the idle point) before ``execute_node``; the fast-flux levels are relative
to it.

``analyze`` normalises the surface with the readout levels and scores it against the closed-form theory
curve with :class:`~seqtante_open.experiments.fitting.landau_zener_fit.LandauZenerFit`. It returns the
``rmse`` similarity score S as the benchmark value, which the run scores against the tree's thresholds;
``apply`` writes nothing, as a benchmark does (ADR-0002).

Where the inputs come from:

- The calibration file: the idle point and the readout point are the step's operating point, which also
  gives the readout pulse (``readout-<operating_point>``), its weights, IF, LO and attenuation, and the
  |0>/|1> readout levels. The reset point is the operating point the step's ``reset_point`` names (the
  step's own by default). The simulated points are the ``simulated_points`` table of the target's
  ``landau_zener`` experiment section (ADR-0012), seeded by hand from offline simulations.
- The step's tree parameters: the ramp durations, the entry/exit ramp durations, the reset hold, the
  averages and the readout ring-up.
- The AC crosstalk on the fast-flux lines: qililab corrects the levels for it when it compiles the
  QProgram. The step's ``ac_crosstalk`` (a raw 2x2 array over the ``z``/``x`` fast-flux buses) is used
  when set, as on BSC; otherwise the calibration's ``crosstalk_matrix_ac``.
"""

from __future__ import annotations

import copy
import itertools
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, TypeAdapter, ValidationError, model_validator
from qili_qprograms.bus_alias import Role, bus_alias
from qili_qprograms.experiments import generalized_linear_ramps
from qili_qprograms.flux_path import FluxPath
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting.landau_zener_fit import (
    DURATION_LOOP,
    PHI_X_LOOP,
    LandauZenerFit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qili_qprograms.calibration.model import ReadoutLevels
    from qili_qprograms.targets import Target
    from qililab import QProgram
    from qililab.qprogram.calibration import Calibration

THEORY_AGREEMENT = "theory_agreement"

#: Stream/experiment name the sweep is stored under.
_EXPERIMENT_NAME = "generalized_linear_ramps"

#: The fewest ramp durations a step may sweep: the decay fit behind sigma_hat has three free parameters (ADR-0012).
_MIN_RAMP_DURATIONS = 4
#: The shortest ramp a QProgram can play, in ns.
_MIN_RAMP_DURATION = 4

#: Defaults for the optional step parameters (the node's ``defaults``); every one is overridable in the
#: tree. The measurement keys are named like the ``generalized_linear_ramps`` arguments.
#: ``ramp_durations``, ``averages`` and ``relax_duration`` have no default. The entry ramp is needed unless
#: a row's ramp starts at the reset point's flux, and the exit ramp unless it ends at the readout point's.
LANDAU_ZENER_DEFAULTS: Mapping[str, Any] = {
    "reset_point": None,
    "entry_ramp_duration": None,
    "entry_ramp_move_first": "z",
    "exit_ramp_duration": None,
    "exit_ramp_move_first": "both",
    "readout_ringup_time": 0,
    "ac_crosstalk": None,
}


class SimulatedPoint(BaseModel):
    """One row of a target's ``simulated_points`` table (ADR-0012): where the ramp runs, and Δ and Δε for P_LZ.

    ``phi_x``, ``z_start`` and ``z_end`` are in Φ₀, in the calibration file's ``flux.x``/``flux.z``
    convention; ``min_gap`` (Δ) and ``detuning_span`` (Δε) are in GHz. Every value must be finite: an
    infinite Δ or Δε would still give a finite S, so a seeding mistake would read as a Failed verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    phi_x: float
    z_start: float
    z_end: float
    min_gap: PositiveFloat
    detuning_span: PositiveFloat

    @model_validator(mode="after")
    def _ramps(self) -> SimulatedPoint:
        if self.z_start == self.z_end:
            raise ValueError(f"z_start and z_end are both {self.z_start}, so there is no ramp")
        return self


_SIMULATED_POINTS = TypeAdapter(list[SimulatedPoint])


class LandauZenerNode(FixedFluxNode[None]):
    """A compartmentalized Landau-Zener transition, scored against the closed-form theory curve (ADR-0012).

    The benchmark unit (dimensionless) and direction (higher-is-better) come from the ``landau_zener``
    entry in ``NODE_REGISTRY``. The value is the ``rmse`` similarity score S, with its uncertainty.
    """

    experiment = "landau_zener"
    analyses = (THEORY_AGREEMENT,)
    defaults = LANDAU_ZENER_DEFAULTS

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        #: Each target's |0>/|1> readout levels, checked by :meth:`_run_sweep` before it measures.
        self._levels: dict[Target, ReadoutLevels] = {}
        #: Each target's simulated points, in the order :meth:`_run_sweep` swept them (φx ascending).
        self._points: dict[Target, list[SimulatedPoint]] = {}

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the Landau-Zener sweep on each target at its operating point; return its measurement id."""
        return {target: [self._run_sweep(target)] for target in targets}

    def _run_sweep(self, target: Target) -> int:
        """Build one QProgram per (φx, ramp duration), set the readout line, run them and stream the surface.

        The table, the ramp durations, the readout pulse and levels are checked and every QProgram is built
        before anything is measured (``run`` has already biased the slow flux lines), so a missing or
        malformed table, an over-long ramp, or levels the signal cannot be normalised with fail the target
        without measuring. It executes the QPrograms itself rather than through :meth:`_stream`, which
        repeats one QProgram compiled with the run's calibration: here each point is its own QProgram,
        compiled with the AC crosstalk of :meth:`_with_ac_crosstalk`, and stores the threshold. The
        stream array is the shared :meth:`_stream_array`.

        Raises:
            ValueError: if the table is missing or malformed (:meth:`_simulated_points`), the ramp
                durations are not usable (:meth:`_ramp_durations`), a ramp overruns the Qblox waveform
                memory (raised by the QProgram), the readout levels are missing or were taken with another
                pulse (:meth:`_readout_levels`), or no AC crosstalk is available for the fast-flux lines.
        """
        points = self._simulated_points(target)
        durations = self._ramp_durations(target)

        # Before measuring: the readout pulse the QProgram plays must be stored, with levels taken with it.
        self._require_pulse(target, Role.readout, "readout")
        self._levels[target] = self._readout_levels(target)
        operating_point = self._operating_point_name(target)
        reset_point = self._param(target, "reset_point") or operating_point
        calibration = self._with_ac_crosstalk(target)
        qprograms = [
            [self._build(target, operating_point, reset_point, self._ramp(point, duration)) for duration in durations]
            for point in points
        ]

        self._set_readout_line(target)

        platform = self.context.platform
        readout_bus = bus_alias(target, Role.readout)
        stream_array = self._stream_array(
            target,
            qprograms[0][0],
            experiment_name=_EXPERIMENT_NAME,
            shape=(len(points), len(durations), 3),
            # The loop names are the ones LandauZenerFit reads back; φx is the outer loop (ADR-0012).
            loops={
                PHI_X_LOOP: {
                    "array": [point.phi_x for point in points],
                    "units": "phi_0",
                    "bus": bus_alias(target, Role.fast_flux, "x"),
                    "parameter": PHI_X_LOOP,
                },
                DURATION_LOOP: {
                    "array": durations,
                    "units": "ns",
                    "bus": bus_alias(target, Role.fast_flux, "z"),
                    "parameter": DURATION_LOOP,
                },
            },
            calibration=calibration,
        )
        with stream_array:
            for k, row in enumerate(qprograms):
                for i, qprogram in enumerate(row):
                    result = platform.execute_qprogram(qprogram, calibration=calibration).results[readout_bus][0]
                    stream_array[k, i, :2] = result.array.T
                    stream_array[k, i, 2] = result.threshold[0]
        self._points[target] = points
        return self._measurement_id(stream_array)

    def _simulated_points(self, target: Target) -> list[SimulatedPoint]:
        """The target's ``simulated_points`` table (ADR-0012), sorted by φx.

        It is a field of the target's ``landau_zener`` experiment section, kept by qili-qprograms as a
        section-specific extra and validated here.

        Raises:
            ValueError: if the section or its table is missing or empty, a row is malformed, or two rows
                share a φx.
        """
        section = self.calibration_file.experiments.get(self.experiment, {}).get(str(target))
        raw = (section.model_extra or {}).get("simulated_points") if section is not None else None
        if not raw:
            raise ValueError(
                f"{target}: the calibration file has no simulated_points table under "
                f"experiments.{self.experiment}.{target}, so there is no ramp to sweep or theory to score "
                "against (ADR-0012)"
            )
        try:
            points = _SIMULATED_POINTS.validate_python(raw)
        except ValidationError as error:
            raise ValueError(f"{target}: the simulated_points table is malformed: {error}") from error
        points.sort(key=lambda point: point.phi_x)
        phi_x = [point.phi_x for point in points]
        if len(set(phi_x)) != len(phi_x):
            raise ValueError(f"{target}: the simulated_points table has more than one row per phi_x: {phi_x}")
        return points

    def _ramp_durations(self, target: Target) -> list[int]:
        """The step's ``ramp_durations``: the durations T of the z ramp, in ns, as whole numbers.

        Raises:
            ValueError: if there are fewer than four (the decay fit has three free parameters), one is not
                a whole number of ns or is shorter than 4 ns, or they are not strictly increasing.
        """
        raw = list(self._param(target, "ramp_durations"))
        if len(raw) < _MIN_RAMP_DURATIONS:
            raise ValueError(
                f"{target}: ramp_durations has {len(raw)} values; the score needs at least {_MIN_RAMP_DURATIONS}"
            )
        if any(not float(duration).is_integer() or duration < _MIN_RAMP_DURATION for duration in raw):
            raise ValueError(f"{target}: ramp_durations must be whole numbers of ns, at least 4 ns; got {raw}")
        durations = [int(duration) for duration in raw]
        if any(later <= earlier for earlier, later in itertools.pairwise(durations)):
            raise ValueError(f"{target}: ramp_durations must be strictly increasing; got {durations}")
        return durations

    @staticmethod
    def _ramp(point: SimulatedPoint, duration: int) -> FluxPath:
        """The z ramp of ``point`` played in ``duration`` ns at its φx: T is this ramp alone (ADR-0012)."""
        return FluxPath.from_waypoints([(point.z_end, point.phi_x, duration)], start=(point.z_start, point.phi_x))

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
        """Score the surface against the theory curve, save ``score_s``/``rmse``, plot, and report the ``rmse`` S.

        The |0>/|1> readout levels are the operating point's and the Δ, Δε the table's, both checked by
        :meth:`_run_sweep` before it measured. Fit, then the check, then save, then plot: an undefined
        score never reaches the DB.

        Raises:
            AnalysisError: when the ``rmse`` S is undefined: a non-finite point in the data, or a zero
                reference distance. The target then fails the step.
        """
        levels, points = self._levels[target], self._points[target]
        model = LandauZenerFit(
            measurement_id=measurements[-1],
            target=str(target),
            level_0=levels.level_0.nominal,
            level_1=levels.level_1.nominal,
            min_gap=[point.min_gap for point in points],
            detuning_span=[point.detuning_span for point in points],
            path=self.parameters[target].get("data_folder"),
        )
        model.fit()
        score = model.score
        if not math.isfinite(score.value):
            raise AnalysisError(
                f"{target}: the Landau-Zener score is undefined (S = {score.value}, D = {score.distance}, "
                f"D_ref = {score.reference}); a non-finite point in the data or a zero reference distance"
            )
        model.save_fitting()
        plot = model.plot()
        uncertainty = score.uncertainty if math.isfinite(score.uncertainty) else None
        return AnalysisResult(value=score.value, uncertainty=uncertainty, plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[None]) -> None:
        """A benchmark writes nothing to the calibration file; the run records the verdict (ADR-0002)."""
