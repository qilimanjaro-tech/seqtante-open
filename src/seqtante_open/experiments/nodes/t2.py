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

"""The ``t2`` node: a T2 (Ramsey) DRAG benchmark at the sweet spot (ADR-0002, ADR-0006, ADR-0007, QHC-1809).

One T2 (Ramsey) DRAG sweep runs at the step's operating point (the base places it), a decaying-oscillation
fit gives the dephasing time with its uncertainty, and ``analyze`` reports that value in microseconds --
never a verdict: the run scores it against the tree's thresholds (QHC-1848). T2 is higher-is-better.
``apply`` writes nothing, as a benchmark does (ADR-0002). The fitted ``t2`` and ``fit_r2`` go to the DB
(QHC-1497, QHC-1791). Operationally the T2 pipeline runs after a calibrated pi pulse (rabi), but this
node's code does not depend on the rabi node: the QProgram reads the pi-half pulse from the calibration
file (or the pi pulse at half amplitude).

The T2 (Ramsey) QProgram is the shared ``qili_qprograms`` experiment (QHC-1928): the node builds it and
runs it directly -- no ``experiment_class`` driver layer -- setting the operating point's hardware through
the restore handle (ADR-0006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from warnings import warn

import numpy as np
from qili_qprograms.experiments import t2_ramsey
from qililab import Parameter
from qililab.result import StreamArray

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.fitting import T2Fit
from seqtante_open.experiments.results import measurement_id
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from seqtante_open.controllers.parameter_restore import ParameterRestore
    from seqtante_open.targets import Target

DECAYING_OSCILLATION = "decaying_oscillation"

#: Stream/experiment name the T2 measurement is stored under.
_EXPERIMENT_NAME = "t2_drag"

#: Warned when the drive bus has no instrument whose power this node knows how to set.
_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."

#: Default minimum R² a T2 fit must reach to pass the step (overridable per step).
DEFAULT_MIN_R2 = 0.8


class T2Node(FixedFluxNode[None]):
    """One T2 (Ramsey) DRAG sweep at the sweet spot, fit to a decaying oscillation and scored as a benchmark (QHC-1809).

    The benchmark unit (``us``) and direction (higher-is-better) come from the ``t2`` entry in
    ``NODE_REGISTRY``; the run reads them to score the reported value against the tree's thresholds.
    """

    experiment: ClassVar[str | None] = "t2"
    analyses: ClassVar[tuple[str, ...]] = (DECAYING_OSCILLATION,)

    def execute_node(self, targets: list[Target]) -> dict[Target, list[int]]:
        """Run the T2 (Ramsey) DRAG sweep on each target at its operating point; return its measurement id."""
        return {target: [self._run_ramsey_sweep(target)] for target in targets}

    def _run_ramsey_sweep(self, target: Target) -> int:
        """Build the T2 (Ramsey) QProgram, set the operating point's hardware, run the sweep and stream it.

        The QProgram is the shared ``qili_qprograms`` experiment: it plays the calibrated ``pi_half-<op>``
        pulse (or ``pi-<op>`` at half amplitude) twice around an idle wait and reads the ``readout-<op>``
        pulse, its weights and the drive/readout IFs from the calibration file. This node sets what the
        QProgram does not -- the drive/readout LOs, the readout attenuation and the drive power/RF -- from
        the operating point, through the restore handle so the run undoes them (ADR-0006).
        """
        experiment = self.experiment
        # A registered leaf node always sets experiment (enforced in __init_subclass__); narrow for the type checker.
        if experiment is None:
            raise RuntimeError(f"{type(self).__name__} declares no experiment")
        params = self.parameters[target]
        section = self.calibration_file.experiments[experiment][str(target)]
        operating_point_name = section.operating_point
        point = self.calibration_file.operating_points[str(target)][operating_point_name]
        drive_bus = bus_alias(target, Role.drive)
        readout_bus = bus_alias(target, Role.readout)
        idle_sweep = np.linspace(*params["elapsed_sweep"]).astype(int)

        qprogram = t2_ramsey(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=operating_point_name,
            idle_duration_sweep=idle_sweep,
            averages=params["averages"],
            relax_duration=params["relax_duration"],
            readout_ringup_time=params.get("ringup_time", 0),
        )

        restore = self.restore
        restore.set(drive_bus, Parameter.LO_FREQUENCY, int(point.drive.lo))
        restore.set(readout_bus, Parameter.LO_FREQUENCY, int(point.readout.lo))
        restore.set(readout_bus, Parameter.OUT0_ATT, int(point.readout.attenuation))
        self._set_drive_source(restore, drive_bus, params["drive_power"])

        platform = self.context.platform
        stream_array = StreamArray(
            shape=(len(idle_sweep), 2),
            loops={
                "t2_time": {
                    "array": idle_sweep.astype(int),
                    "units": "ns",
                    "bus": drive_bus,
                    "parameter": "t2_elapsed_time",
                }
            },
            platform=platform,
            experiment_name=_EXPERIMENT_NAME,
            db_manager=output_controller.db_manager,
            qprogram=qprogram,
            calibration=self.context.calibration,
            qubit_idx=str(target),
        )
        with stream_array:
            results = platform.execute_qprogram(qprogram).results
            for idx in range(len(idle_sweep)):
                stream_array[idx,] = results[readout_bus][idx].array.T
        return measurement_id(stream_array)

    def _set_drive_source(self, restore: ParameterRestore, drive_bus: str, drive_power: float) -> None:
        """Set the drive power, and arm a Rohde & Schwarz source to play the IQ drive.

        The power goes through the parameter the bus's drive instrument uses (R&S ``POWER``, QCMRF
        ``GAIN``). A Rohde & Schwarz source also needs IQ modulation switched on -- otherwise it emits a
        bare carrier instead of the pi-half pulses' IQ waveform -- and its RF output switched on. Both,
        like the power, go through the restore handle, so the node's run turns them back off at the end.
        """
        drive_instruments = [
            instrument.name.name for instrument in self.context.platform.get_element(drive_bus).instruments
        ]
        if "ROHDE_SCHWARZ" in drive_instruments:
            restore.set(drive_bus, Parameter.IQ_MODULATION, True)
            restore.set(drive_bus, Parameter.POWER, drive_power)
            restore.set(drive_bus, Parameter.RF_ON, True)
        elif "QCMRF" in drive_instruments:
            restore.set(drive_bus, Parameter.GAIN, drive_power)
        else:
            warn(_NO_DRIVE_INSTRUMENT_MSG, stacklevel=2)

    def analyze(self, target: Target, measurements: list[int]) -> AnalysisResult[None]:
        """Fit the oscillation, save ``t2``/``fit_r2`` to the DB, and report T2 and its uncertainty in us; read-only.

        Raises:
            AnalysisError: when the fit's R² is below the step's minimum (the target fails the step).
        """
        params = self.parameters[target]
        model = T2Fit(target=str(target), measurement_id=measurements[-1], path=params.get("data_folder"))
        model.fit()
        plot = model.plot()
        model.save_fitting()

        min_r2 = params.get("min_r2", DEFAULT_MIN_R2)
        if model.r_squared < min_r2:
            raise AnalysisError(f"{target} T2 fit R² {model.r_squared:.3f} is below the minimum {min_r2}")

        return AnalysisResult(value=model.t2_us, uncertainty=model.t2_uncertainty_us, plots=(plot,) if plot else ())

    def apply(self, target: Target, result: AnalysisResult[None]) -> None:
        """A benchmark writes nothing to the calibration file; the run records the scored verdict (ADR-0002)."""
