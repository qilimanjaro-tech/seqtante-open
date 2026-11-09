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

"""The one place a run touches qililab's crosstalk API (the swap point for QHC-1735).

Installing the crosstalk matrix, zeroing flux at the start, zeroing bias at the end, and writing a
measured flux offset back into the matrix are the four things a run does with crosstalk, and the
crosstalk refactor (QHC-1735) changes all of them. Keeping every one of them behind
:func:`crosstalk_session` and :class:`CrosstalkSession` means that refactor edits this file alone,
not every node. Nodes use the session; they never call ``platform.set_crosstalk``,
``set_flux_to_zero``, ``set_bias_to_zero`` or touch ``flux_offsets`` directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qililab import Platform
    from qililab.qprogram.calibration import Calibration


class CrosstalkSession:
    """A handle to the run's crosstalk matrix, the only writer of flux offsets (ADR-0006).

    Held for the duration of :func:`crosstalk_session`. The ``apply`` phase of an offset-calibration
    node writes a measured offset through :meth:`add_flux_offset`; nothing else mutates the matrix.
    """

    def __init__(self, crosstalk: CrosstalkMatrix) -> None:
        self._crosstalk = crosstalk

    def add_flux_offset(self, flux_bus: str, offset: float) -> None:
        """Add ``offset`` to the crosstalk matrix's flux offset for ``flux_bus`` (ADR-0006, ``apply``)."""
        self._crosstalk.flux_offsets[flux_bus] += float(offset)


@contextmanager
def crosstalk_session(platform: Platform, calibration: Calibration) -> Iterator[CrosstalkSession]:
    """Install the run's crosstalk matrix and zero flux, then zero bias on exit.

    Yields a :class:`CrosstalkSession` for writing measured flux offsets. The bias is zeroed in a
    ``finally``, so it is reset whether the body returns or raises.

    Raises:
        ValueError: if the calibration carries no :class:`CrosstalkMatrix`; a run needs one to set flux.
    """
    crosstalk = calibration.crosstalk_matrix
    if not isinstance(crosstalk, CrosstalkMatrix):
        raise ValueError("the calibration has no CrosstalkMatrix; a run needs one to set flux")
    platform.set_crosstalk(crosstalk=crosstalk)
    try:
        platform.set_flux_to_zero()
        yield CrosstalkSession(crosstalk)
    finally:
        platform.set_bias_to_zero()
