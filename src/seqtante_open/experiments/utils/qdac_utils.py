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

"""QDAC helpers, copied from ``qilitools.utils.platform``."""

import math
from typing import Iterable
from warnings import warn

from qililab import Platform
from qililab.instrument_controllers import QDevilQDac2Controller

QDAC_TRIGGER_TO_VOLTAGE_PADDING = 1400


def get_qdac_lp_filters(platform: Platform):
    """
    Retruns list of all LP-filters used, for all qdac sources
    """
    lp_filters = []
    for instrument_controller in platform.instrument_controllers.elements:
        if isinstance(instrument_controller, QDevilQDac2Controller):
            lp_filters.extend(instrument_controller.modules[0].low_pass_filter)

    return lp_filters


def get_qdac_out_trigger(platform: Platform, default: int = 2) -> int:
    """External trigger output the QDAC drives at each ramp step.

    Takes the first QDAC that declares one; ``out_trigger`` is unset by default.

    Args:
        platform: Platform to read the QDAC settings from.
        default: Trigger output to use when no QDAC declares one.

    Returns:
        int: The declared ``out_trigger``, or ``default``.
    """
    for instrument_controller in platform.instrument_controllers.elements:
        if isinstance(instrument_controller, QDevilQDac2Controller):
            out_trigger = instrument_controller.modules[0].out_trigger
            if out_trigger is not None:
                return out_trigger

    return default


def wait_time_to_settle_from_filters(
    filters: Iterable[str],
    settle_percent: float = 99.9,
    safety_factor: float = 1.0,
    extra_ns: float = 4000.0,
    fridge_filter_hz: float = 100e3,
) -> int:
    """
    Compute recommended wait time (seconds) to settle to `settle_percent`,
    using the slowest (lowest cutoff) filter among `filters` AND a fridge filter in series
    (default 100 kHz), plus a fixed extra delay to compensate for desync between QDAC trigger and signal.

    Returned value in ns

    Effective cutoff is taken as the minimum cutoff in the chain.
    """
    QDACL_FILTERS_HZ = {"dc": 10.0, "med": 10e3, "high": 230e3}

    filt_list = [f.strip().lower() for f in filters if f is not None]
    if not filt_list:
        raise ValueError("filters is empty")

    unknown = sorted({f for f in filt_list if f not in QDACL_FILTERS_HZ})
    if unknown:
        raise ValueError(f"Unknown filter setting(s): {unknown}. Allowed: {list(QDACL_FILTERS_HZ)}")

    if not (0.0 < settle_percent < 100.0):
        raise ValueError("settle_percent must be between 0 and 100 (exclusive).")

    slowest_qdac_hz = min(QDACL_FILTERS_HZ[f] for f in filt_list)
    # fridge always in series
    f_eff = min(slowest_qdac_hz, float(fridge_filter_hz))

    tau = 1.0 / (2.0 * math.pi * f_eff)
    eps = 1.0 - settle_percent / 100.0
    t = -tau * math.log(eps)

    return int((safety_factor * t + float(extra_ns) * 1e-9) * 1e9)


def all_qdacs_using_ext_clock(platform: Platform):
    qdacs_using_ext_clock = []
    for instrument_controller in platform.instrument_controllers.elements:
        if isinstance(instrument_controller, QDevilQDac2Controller):
            qdacs_using_ext_clock.append(instrument_controller.reference_clock == "external")  # type: ignore [attr-defined]

    if all(qdacs_using_ext_clock):
        return True
    return False


def qdac_step_timings(
    platform: Platform,
    minimum_wait_after_step_override: float | None = None,
    qdac_stop_ro_before_step_override: float | None = None,
) -> tuple[int, int]:
    """Wait times around each QDAC step, in ns.

    Warns if any QDAC runs off its internal clock, which risks missing a trigger.

    Args:
        platform: Platform whose QDAC low-pass filters set the settling time.
        minimum_wait_after_step_override: Settling wait after each step. Must be given
            together with ``qdac_stop_ro_before_step_override``.
        qdac_stop_ro_before_step_override: How long readout stops before a step. Must be
            given together with ``minimum_wait_after_step_override``.

    Returns:
        tuple[int, int]: ``(wait_after_step, stop_ro_before_step)``.
    """
    if not all_qdacs_using_ext_clock(platform):
        warn(
            "At least one QDAC is using the internal clock. Use a shared clock to guarantee not "
            "missing a trigger when using triggered measurements"
        )

    if minimum_wait_after_step_override is None and qdac_stop_ro_before_step_override is None:
        return wait_time_to_settle_from_filters(get_qdac_lp_filters(platform)), QDAC_TRIGGER_TO_VOLTAGE_PADDING
    if minimum_wait_after_step_override is not None and qdac_stop_ro_before_step_override is not None:
        return minimum_wait_after_step_override, qdac_stop_ro_before_step_override
    raise ValueError(
        "If overriding wait-times, you need to set both minimum_wait_after_step_override and "
        "qdac_stop_ro_before_step_override"
    )
