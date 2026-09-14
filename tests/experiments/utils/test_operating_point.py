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

"""Tests for the operating-point helpers in ``seqtante_open.experiments.utils.misc_utils``.

An operating point is the bias configuration a target is parked at while it is
measured. It lives in the calibration under
``parameters["operating_point"][target][name]``, keyed first by ``Parameter``
*name* and then by bus::

    {"FLUX": {"flux_q1_z": 0.35, "flux_q1_x": 0.1}, "IF": {"readout_q1": 1.2e8}}

``save_parameters_to_operating_point`` writes that structure,
``get_operating_point`` reads it back and ``set_to_operating_point`` applies it to
the hardware. The behaviour is pinned here; the nodes that use these helpers
assert only that they are called correctly.
"""

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from qililab.qprogram.calibration import Calibration
from qililab.typings.enums import Parameter

from seqtante_open.experiments.utils.misc_utils import (
    get_operating_point,
    save_parameters_to_operating_point,
    set_to_operating_point,
)

RUNCARD_PATH = Path(__file__).resolve().parents[2] / "runcards" / "test_AQPU_runcard.yml"

TARGET = "q1"
POINT = "park"


@pytest.fixture
def calibration() -> Calibration:
    """An empty calibration: the helpers create every level they need."""
    return Calibration()


def _stored(calibration: Calibration) -> dict:
    """The operating point as it sits in the calibration, reached the long way."""
    return calibration.parameters["operating_point"][TARGET][POINT]


def _save(calibration: Calibration, *settings, target: str = TARGET, point: str = POINT):
    return save_parameters_to_operating_point(
        *settings, calibration=calibration, target=target, operating_point_name=point
    )


def test_save_creates_every_level_from_scratch(calibration):
    """Nothing has to exist beforehand -- target and operating point are created."""
    _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.35))

    assert calibration.parameters == {"operating_point": {"q1": {"park": {"FLUX": {"flux_q1_z": 0.35}}}}}


def test_save_keys_parameters_by_enum_name(calibration):
    """``FLUX``, never the ``flux`` value, so a parameter has a single spelling."""
    _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.35))

    stored = _stored(calibration)
    assert Parameter.FLUX.name in stored
    assert Parameter.FLUX.value not in stored


def test_save_returns_the_live_operating_point(calibration):
    """The return value is the object held by the calibration, not a copy of it."""
    operating_point = _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.35))

    assert operating_point is _stored(calibration)


def test_save_stores_every_setting_of_one_call(calibration):
    """Several buses and several parameters can be written in a single call."""
    _save(
        calibration,
        ("flux_q1_z", Parameter.FLUX, 0.35),
        ("flux_q1_x", Parameter.FLUX, 0.1),
        ("readout_q1", Parameter.IF, 1.2e8),
    )

    assert _stored(calibration) == {
        "FLUX": {"flux_q1_z": 0.35, "flux_q1_x": 0.1},
        "IF": {"readout_q1": 1.2e8},
    }


def test_save_updates_only_the_bus_it_is_given(calibration):
    """Saving a bus again updates that one value and leaves its neighbours alone."""
    _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.35), ("flux_q1_x", Parameter.FLUX, 0.1))
    _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.5))

    assert _stored(calibration) == {"FLUX": {"flux_q1_z": 0.5, "flux_q1_x": 0.1}}


def test_save_keeps_targets_and_operating_points_apart(calibration):
    """Writing one point of one target disturbs no other point and no other target."""
    _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.1), target="q1", point="park")
    _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.2), target="q1", point="idle")
    _save(calibration, ("flux_q2_z", Parameter.FLUX, 0.3), target="q2", point="park")

    assert calibration.parameters["operating_point"] == {
        "q1": {"park": {"FLUX": {"flux_q1_z": 0.1}}, "idle": {"FLUX": {"flux_q1_z": 0.2}}},
        "q2": {"park": {"FLUX": {"flux_q2_z": 0.3}}},
    }


def test_save_without_settings_creates_an_empty_operating_point(calibration):
    """No settings still materialises the point, so it can be filled in later."""
    operating_point = _save(calibration)

    assert operating_point == {}
    assert _stored(calibration) == {}


def test_get_returns_the_stored_operating_point(calibration):
    """Without a platform it is a pure read, handing back the stored object."""
    saved = _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.35))

    assert get_operating_point(calibration, TARGET, POINT) is saved


@pytest.mark.parametrize(
    ("target", "point"),
    [("q2", POINT), (TARGET, "idle")],
    ids=["unknown target", "unknown operating point"],
)
def test_get_raises_for_a_point_that_was_never_saved(calibration, target, point):
    """A missing target or name is a ``KeyError``, not a silently empty point."""
    _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.35))

    with pytest.raises(KeyError):
        get_operating_point(calibration, target, point)


def test_get_applies_the_point_when_a_platform_is_given(calibration):
    """Passing a platform parks the target as a side effect of reading the point."""
    _save(calibration, ("flux_q1_z", Parameter.FLUX, 0.35))
    platform = MagicMock(name="platform")

    returned = get_operating_point(calibration, TARGET, POINT, platform=platform)

    platform.set_parameter.assert_called_once_with(alias="flux_q1_z", parameter=Parameter.FLUX, value=0.35)
    assert returned is _stored(calibration)


def test_set_applies_every_setting_in_insertion_order():
    """Buses are set in the order they were saved, not in some hash order."""
    platform = MagicMock(name="platform")
    operating_point = {
        "FLUX": {"flux_q1_z": 0.35, "flux_q1_x": 0.1},
        "IF": {"readout_q1": 1.2e8},
    }

    set_to_operating_point(platform=platform, operating_point=operating_point)

    assert platform.set_parameter.call_args_list == [
        call(alias="flux_q1_z", parameter=Parameter.FLUX, value=0.35),
        call(alias="flux_q1_x", parameter=Parameter.FLUX, value=0.1),
        call(alias="readout_q1", parameter=Parameter.IF, value=1.2e8),
    ]


def test_set_rejects_a_parameter_stored_by_value():
    """``flux`` is the enum value; operating points key by name, so it is no key at all."""
    platform = MagicMock(name="platform")

    with pytest.raises(KeyError):
        set_to_operating_point(platform=platform, operating_point={Parameter.FLUX.value: {"flux_q1_z": 0.35}})

    platform.set_parameter.assert_not_called()


def test_set_of_an_empty_operating_point_touches_nothing():
    """An operating point with no parameters is applied without reaching the hardware."""
    platform = MagicMock(name="platform")

    set_to_operating_point(platform=platform, operating_point={})

    platform.set_parameter.assert_not_called()


def test_round_trip_lands_the_saved_values_on_a_real_platform(platform, calibration):
    """Saved, read back and applied, the values reach a genuine ``Platform``."""
    _save(calibration, ("readout_q1", Parameter.IF, 1.11e8), ("drive_q1", Parameter.LO_FREQUENCY, 5.5e9))

    get_operating_point(calibration, TARGET, POINT, platform=platform)

    assert platform.get_parameter("readout_q1", Parameter.IF) == pytest.approx(1.11e8)
    assert platform.get_parameter("drive_q1", Parameter.LO_FREQUENCY) == pytest.approx(5.5e9)
