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
"""The calibration-file model: operating points per target and experiment sections (ADR-0001).

Operating points are stored once per target; an experiment section names the point it runs at
and the pulses it plays. Pulses themselves are named waveforms in qililab's ``Calibration.waveforms``
(not here), referenced by name (ADR-0008). Fluxes are plain measured values (ADR-0011); an operating
point also parks the fluxes of its neighbours (ADR-0010).

This module is pure: it validates the structure and the naming grammar. Whether a referenced pulse
name actually exists in ``Calibration.waveforms`` is checked in :mod:`seqtante_open.calibration.io`,
which has the ``Calibration`` to check against.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from seqtante_open.calibration.bus_alias import Role
from seqtante_open.calibration.pulses import parse_pulse_name
from seqtante_open.targets import ExperimentTarget


class _Base(BaseModel):
    """Forbid unknown keys (a one-way-door contract catches typos) and accept field names or aliases."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Measured(_Base):
    """A measured value with its standard uncertainty."""

    nominal: float
    std: float


class Flux(_Base):
    """A flux setpoint on a target's loops, as a measured bias (ADR-0011). ``x`` is absent on z-only loops."""

    z: float
    x: float | None = None


class DriveFreqs(_Base):
    """The drive line's local-oscillator and intermediate frequencies."""

    lo: float
    if_: float = Field(alias="if")


class ReadoutLevels(_Base):
    """The |0> and |1> readout levels, tagged with the readout pulse they were taken with (ADR-0001, D8)."""

    pulse: str
    level_0: Measured
    level_1: Measured

    @field_validator("pulse")
    @classmethod
    def _grammar(cls, value: str) -> str:
        parse_pulse_name(value)
        return value


class ReadoutFreqs(_Base):
    """The readout line's frequencies and attenuation, with readout levels where measured."""

    lo: float
    if_: float = Field(alias="if")
    attenuation: float
    levels: ReadoutLevels | None = None


class OperatingPoint(_Base):
    """A target at one operating point: its own flux, the neighbours it parks, and its frequencies."""

    flux: Flux
    parked_flux: dict[str, Flux] = Field(default_factory=dict)
    drive: DriveFreqs
    readout: ReadoutFreqs

    @field_validator("parked_flux")
    @classmethod
    def _neighbour_targets_valid(cls, value: dict[str, Flux]) -> dict[str, Flux]:
        for target in value:
            ExperimentTarget.from_str(target)
        return value


class ExperimentSection(_Base):
    """An experiment at one target: the operating point it runs at and the pulses it plays (by name)."""

    operating_point: str
    pulses: dict[Role, str] = Field(default_factory=dict)

    @field_validator("pulses")
    @classmethod
    def _grammar(cls, value: dict[Role, str]) -> dict[Role, str]:
        for name in value.values():
            parse_pulse_name(name)
        return value


class CalibrationFile(_Base):
    """The structured calibration-file content: operating points per target and experiment sections.

    Stored in qililab's ``Calibration.parameters`` (ADR-0001); pulses live in ``Calibration.waveforms``.
    """

    operating_points_version: Literal[1]
    operating_points: dict[str, dict[str, OperatingPoint]] = Field(default_factory=dict)
    experiments: dict[str, dict[str, ExperimentSection]] = Field(default_factory=dict)

    @field_validator("operating_points")
    @classmethod
    def _targets_valid(cls, value: dict[str, dict[str, OperatingPoint]]) -> dict[str, dict[str, OperatingPoint]]:
        for target in value:
            ExperimentTarget.from_str(target)
        return value

    @field_validator("experiments")
    @classmethod
    def _section_targets_valid(
        cls, value: dict[str, dict[str, ExperimentSection]]
    ) -> dict[str, dict[str, ExperimentSection]]:
        for by_target in value.values():
            for target in by_target:
                ExperimentTarget.from_str(target)
        return value

    @model_validator(mode="after")
    def _no_self_parking(self) -> Self:
        for target, points in self.operating_points.items():
            for name, point in points.items():
                if target in point.parked_flux:
                    raise ValueError(
                        f"operating_points['{target}']['{name}']: a point cannot park its own target "
                        f"'{target}' (its own flux belongs in `flux`, not `parked_flux`)"
                    )
        return self

    @model_validator(mode="after")
    def _sections_name_known_points(self) -> Self:
        for experiment, by_target in self.experiments.items():
            for target, section in by_target.items():
                points = self.operating_points.get(target, {})
                if section.operating_point not in points:
                    raise ValueError(
                        f"experiment '{experiment}' target '{target}': operating_point "
                        f"'{section.operating_point}' is not defined under operating_points['{target}']"
                    )
        return self
