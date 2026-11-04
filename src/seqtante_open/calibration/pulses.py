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
"""Pulse-name grammar for the calibration file (ADR-0008).

A pulse is named by its purpose, in ``lower_snake``, and carries its operating
point as a ``-``-joined suffix where the same purpose is calibrated at more than
one point: ``<purpose>-<operating_point>`` (``readout-reset``, ``pi-reset``,
``pi_half-reset``). ``-`` separates purpose from point because ``_`` already
joins words within a purpose.
"""

import re

#: A ``lower_snake`` purpose, optionally followed by ``-<operating_point>``.
PULSE_NAME_PATTERN = r"^[a-z][a-z0-9_]*(-[a-z0-9_]+)?$"

_PULSE_NAME_RE = re.compile(PULSE_NAME_PATTERN)


def parse_pulse_name(name: str) -> tuple[str, str | None]:
    """Split a pulse name into its purpose and operating point.

    Args:
        name: the pulse name, e.g. ``readout-reset`` or ``pi``.

    Returns:
        ``(purpose, operating_point)``, where ``operating_point`` is ``None`` when
        the name carries no ``-`` suffix.

    Raises:
        ValueError: if ``name`` does not match :data:`PULSE_NAME_PATTERN`.
    """
    if _PULSE_NAME_RE.match(name) is None:
        raise ValueError(f"invalid pulse name {name!r}: expected {PULSE_NAME_PATTERN}")
    purpose, _, operating_point = name.partition("-")
    return purpose, operating_point or None
