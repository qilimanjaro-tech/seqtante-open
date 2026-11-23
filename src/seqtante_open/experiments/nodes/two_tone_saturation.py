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

"""The two-tone pulsed-saturation measurement the fixed-flux two-tone nodes share (QHC-1807, QHC-1805).

The QProgram is the shared ``qili_qprograms`` experiment ``two_tone_saturation`` (QHC-1916): it reads the
``readout-<op>`` pulse, its integration weights and the readout IF from the calibration file and sweeps the
drive IF with a smooth ring-up / square / ring-down saturation drive. :class:`TwoToneSaturationNode` keeps
the execution half -- the drive/readout LOs, the readout attenuation and the drive power (plus IQ
modulation and RF for a Rohde & Schwarz source), all through the restore handle -- then runs the sweep and
streams it. It is a spine, not an experiment: it sets no ``experiment`` and does not register.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import warn

from qili_qprograms.experiments import two_tone_saturation
from qililab import Parameter
from qililab.result import StreamArray

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import FixedFluxNode, ResultT
from seqtante_open.controllers.operating_point import operating_point_for
from seqtante_open.experiments.results import measurement_id
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    import numpy as np

    from seqtante_open.controllers.parameter_restore import ParameterRestore
    from seqtante_open.targets import Target

#: Stream/experiment name the measurement is stored under.
EXPERIMENT_NAME = "two_tone_saturation"

#: Warned when the drive bus has no instrument whose power this node knows how to set.
_NO_DRIVE_INSTRUMENT_MSG = "No instrument to set power to."

#: Measurement defaults (the experimentalists' calibration-script values); every one is a tree parameter.
#: The readout pulse (amplitude, duration, weights) comes from the calibration file, not from parameters.
TWO_TONE_SATURATION_DEFAULTS: dict[str, Any] = {
    "averages": 5000,
    "relax_duration": 50_000,
    "ringup_time": 600,
    "drive_duration": 30_000,
    "drive_amplitude": 0.8,
    "drive_power": -40,
    "overlap_time": 0,
    "drive_sigmas": 4,
    "duration_ringup_drive": 200,
    "drive_ringdown_time": 200,
}


class TwoToneSaturationNode(FixedFluxNode[ResultT]):
    """A fixed-flux node that measures with the qili two-tone pulsed-saturation QProgram."""

    def _two_tone_param(self, target: Target, key: str) -> Any:
        return self.parameters[target].get(key, TWO_TONE_SATURATION_DEFAULTS[key])

    def _measure_two_tone_saturation(self, target: Target, drive_if_sweep: np.ndarray) -> int:
        """Build the qili two-tone QProgram, set the operating point's hardware, run the sweep; return its id.

        The QProgram plays the ``readout-<op>`` pulse with its weights and sets the readout IF, all from
        the calibration file, and sweeps ``drive_if_sweep`` (Hz) itself. This node sets what the QProgram
        does not -- the drive/readout LOs, the readout attenuation and the drive power/RF -- from the
        operating point, through the restore handle so the run undoes them (ADR-0006).

        Raises:
            TypeError: if called on a node that sets no ``experiment`` (a spine, not a leaf).
            ValueError: if ``drive_if_sweep`` has fewer than 2 points.
        """
        experiment = self.experiment
        if experiment is None:
            raise TypeError(f"{type(self).__name__} sets no experiment; only a leaf node can measure")
        if len(drive_if_sweep) < 2:
            raise ValueError(f"{target}: the drive-IF sweep must have at least 2 points, got {len(drive_if_sweep)}")
        section = self.calibration_file.experiments[experiment][str(target)]
        point = operating_point_for(self.calibration_file, experiment, target)
        drive_bus = bus_alias(target, Role.drive)
        readout_bus = bus_alias(target, Role.readout)

        qprogram = two_tone_saturation(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=section.operating_point,
            drive_if_sweep=drive_if_sweep,
            averages=self._two_tone_param(target, "averages"),
            relax_duration=self._two_tone_param(target, "relax_duration"),
            drive_duration=self._two_tone_param(target, "drive_duration"),
            drive_amplitude=self._two_tone_param(target, "drive_amplitude"),
            drive_ringup_time=self._two_tone_param(target, "duration_ringup_drive"),
            drive_ringdown_time=self._two_tone_param(target, "drive_ringdown_time"),
            drive_n_sigmas=self._two_tone_param(target, "drive_sigmas"),
            overlap_time=self._two_tone_param(target, "overlap_time"),
            readout_ringup_time=self._two_tone_param(target, "ringup_time"),
        )

        restore = self.restore
        restore.set(drive_bus, Parameter.LO_FREQUENCY, int(point.drive.lo))
        restore.set(readout_bus, Parameter.LO_FREQUENCY, int(point.readout.lo))
        restore.set(readout_bus, Parameter.OUT0_ATT, int(point.readout.attenuation))
        self._set_drive_source(restore, drive_bus, self._two_tone_param(target, "drive_power"))

        platform = self.context.platform
        stream_array = StreamArray(
            shape=(len(drive_if_sweep), 2),
            loops={
                "IF_frequency": {"array": drive_if_sweep, "units": "Hz", "bus": drive_bus, "parameter": "IF_frequency"}
            },
            platform=platform,
            experiment_name=EXPERIMENT_NAME,
            db_manager=output_controller.db_manager,
            qprogram=qprogram,
            calibration=self.context.calibration,
            qubit_idx=str(target),
        )
        with stream_array:
            results = platform.execute_qprogram(qprogram).results
            stream_array[()] = results[readout_bus][0].array.T
        return measurement_id(stream_array)

    def _set_drive_source(self, restore: ParameterRestore, drive_bus: str, drive_power: float) -> None:
        """Set the drive power, and arm a Rohde & Schwarz source to play the IQ drive.

        The power goes through the parameter the bus's drive instrument uses (R&S ``POWER``, QCMRF
        ``GAIN``). A Rohde & Schwarz source also needs IQ modulation switched on -- otherwise it emits a
        bare carrier instead of the saturation drive's IQ waveform -- and its RF output switched on. Both,
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
