"""Tests for arming a drive bus's source, independent of any node or operating point.

The Rohde & Schwarz branch runs on a fake platform; the QCM-RF branch on a real offline qililab platform,
so its gain round-trips through the restore scope as on hardware.
"""

from pathlib import Path

import pytest
from qililab import Parameter

from seqtante_open.controllers.drive_source import NO_DRIVE_INSTRUMENT_MSG, drive_source_writes, set_drive_source
from seqtante_open.controllers.parameter_restore import restoring_parameters
from tests.controllers.fakes import InstrumentPlatform

#: The offline runcard the ``platform`` fixture builds a real qililab platform from (QCM-RF ``drive_q1``).
RUNCARD_PATH = Path(__file__).resolve().parents[1] / "runcards" / "test_AQPU_runcard.yml"

DRIVE_KEYS = [Parameter.GAIN, Parameter.HARDWARE_MODULATION]


def test_rohde_schwarz_gets_power_iq_modulation_and_rf_and_turns_off_after():
    platform = InstrumentPlatform("ROHDE_SCHWARZ")

    with restoring_parameters(platform) as restore:
        set_drive_source(restore, platform, "drive_q1", drive_power=-20)
        in_scope = dict(platform.values)

    assert in_scope["drive_q1", Parameter.IQ_MODULATION] is True
    assert in_scope["drive_q1", Parameter.POWER] == -20
    assert in_scope["drive_q1", Parameter.RF_ON] is True
    assert not platform.values["drive_q1", Parameter.RF_ON]


def test_qcm_rf_gets_the_gain_and_restores_it_on_a_real_platform(platform):
    before = {p: platform.get_parameter(alias="drive_q1", parameter=p) for p in DRIVE_KEYS}

    with restoring_parameters(platform) as restore:
        set_drive_source(restore, platform, "drive_q1", drive_power=-20, drive_gain=0.4)
        in_scope = {p: platform.get_parameter(alias="drive_q1", parameter=p) for p in DRIVE_KEYS}

    assert in_scope[Parameter.GAIN] == pytest.approx((0.4, 0.4))  # the gain, not the R&S power
    assert in_scope[Parameter.HARDWARE_MODULATION] is True
    assert {p: platform.get_parameter(alias="drive_q1", parameter=p) for p in DRIVE_KEYS} == before


@pytest.mark.parametrize(
    ("instrument", "levels", "message"),
    [
        ("ROHDE_SCHWARZ", {"drive_gain": 0.4}, "needs drive_power"),
        ("QCMRF", {"drive_power": -20}, "needs drive_gain"),
    ],
)
def test_the_instrument_needs_its_own_level(instrument, levels, message):
    platform = InstrumentPlatform(instrument)

    with pytest.raises(ValueError, match=message):
        drive_source_writes(platform, "drive_q1", **levels)


def test_an_unknown_instrument_warns_and_writes_nothing():
    platform = InstrumentPlatform("SOMETHING_ELSE")

    with pytest.warns(UserWarning, match=NO_DRIVE_INSTRUMENT_MSG):
        writes = drive_source_writes(platform, "drive_q1", drive_power=-20)

    assert writes == []
