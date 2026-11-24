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
the execution half: the base's :meth:`~seqtante_open.controllers.calibration_node.CalibrationNode._set_operating_point_hardware`
sets the drive/readout LOs, the readout attenuation and the drive source through the restore handle, then
the node runs the sweep and streams it. It is a spine, not an experiment: it sets no ``experiment`` and
does not register.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qili_qprograms.experiments import two_tone_saturation

from seqtante_open.calibration.bus_alias import Role, bus_alias
from seqtante_open.controllers.calibration_node import FixedFluxNode, ResultT, write_whole

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from seqtante_open.targets import Target

#: Stream/experiment name the measurement is stored under.
EXPERIMENT_NAME = "two_tone_saturation"

#: Measurement defaults (the experimentalists' calibration-script values), the spine's ``defaults``; every
#: one is a tree parameter, named like the ``two_tone_saturation`` argument it feeds. The readout pulse
#: (amplitude, duration, weights) comes from the calibration file, not from parameters. ``drive_power``
#: is the Rohde & Schwarz power in dBm; a QCM-RF drive needs ``drive_gain`` set in the step.
TWO_TONE_SATURATION_DEFAULTS: Mapping[str, Any] = {
    "averages": 5000,
    "relax_duration": 50_000,
    "readout_ringup_time": 600,
    "drive_duration": 30_000,
    "drive_amplitude": 0.8,
    "drive_power": -40,
    "overlap_time": 0,
    "drive_n_sigmas": 4,
    "drive_ringup_time": 200,
    "drive_ringdown_time": 200,
}


class TwoToneSaturationNode(FixedFluxNode[ResultT]):
    """A fixed-flux node that measures with the qili two-tone pulsed-saturation QProgram."""

    defaults = TWO_TONE_SATURATION_DEFAULTS

    def _measure_two_tone_saturation(self, target: Target, drive_if_sweep: np.ndarray) -> int:
        """Build the qili two-tone QProgram, set the operating point's hardware, run the sweep; return its id.

        The QProgram plays the ``readout-<op>`` pulse with its weights and sets the readout IF, all from
        the calibration file, and sweeps ``drive_if_sweep`` (Hz) itself. What it does not set -- the
        drive/readout LOs, the readout attenuation and the drive source -- the shared
        :meth:`~seqtante_open.controllers.calibration_node.CalibrationNode._set_operating_point_hardware`
        sets from the operating point, through the restore handle so the run undoes it (ADR-0006).

        Raises:
            TypeError: if called on a node that sets no ``experiment`` (a spine, not a leaf).
        """
        operating_point = self._section(target).operating_point
        qprogram = two_tone_saturation(
            calibration=self.context.calibration,
            target=str(target),
            operating_point=operating_point,
            drive_if_sweep=drive_if_sweep,
            averages=self._param(target, "averages"),
            relax_duration=self._param(target, "relax_duration"),
            drive_duration=self._param(target, "drive_duration"),
            drive_amplitude=self._param(target, "drive_amplitude"),
            drive_ringup_time=self._param(target, "drive_ringup_time"),
            drive_ringdown_time=self._param(target, "drive_ringdown_time"),
            drive_n_sigmas=self._param(target, "drive_n_sigmas"),
            overlap_time=self._param(target, "overlap_time"),
            readout_ringup_time=self._param(target, "readout_ringup_time"),
        )

        self._set_operating_point_hardware(target)
        # The drive-IF sweep runs in hardware: one acquisition holds the whole sweep.
        return self._stream(
            target,
            qprogram,
            experiment_name=EXPERIMENT_NAME,
            loops={
                "IF_frequency": {
                    "array": drive_if_sweep,
                    "units": "Hz",
                    "bus": bus_alias(target, Role.drive),
                    "parameter": "IF_frequency",
                }
            },
            shape=(len(drive_if_sweep), 2),
            write=write_whole,
        )
