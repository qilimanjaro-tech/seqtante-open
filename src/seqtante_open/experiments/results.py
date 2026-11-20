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

"""Helpers for reading back a stream's measurement after a QProgram has run."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from qililab.result import AutocalMeasurement, Measurement, StreamArray


def measurement_id(stream_array: StreamArray) -> int:
    """The measurement id of a completed stream, narrowing the stored measurement union."""
    return cast("Measurement | AutocalMeasurement", stream_array.measurement).measurement_id
