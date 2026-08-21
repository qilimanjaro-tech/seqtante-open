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

"""QProgram helpers, copied from ``qilitools.qprogram.utils``."""


def multi_wait_for_trigger(qp, bus, total_duration):
    """Creates a series of waits after a wait trigger based on a maximum value of wait."""
    MAX_WAIT = 20_000
    if total_duration > MAX_WAIT:
        qp.wait_trigger(bus=bus, duration=MAX_WAIT)
        remaining = total_duration - MAX_WAIT
        while remaining > MAX_WAIT:
            qp.wait(bus=bus, duration=MAX_WAIT)
            remaining -= MAX_WAIT
        qp.wait(bus=bus, duration=remaining)
    else:
        qp.wait_trigger(bus=bus, duration=total_duration)
    qp.sync()
