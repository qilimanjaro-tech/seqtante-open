"""Tests for the parameter-restore handle and context manager (ADR-0006)."""

import pytest
from qililab.typings.enums import Parameter

from seqtante_open.controllers.parameter_restore import ParameterRestore, restoring_parameters, restoring_platform


class FakePlatform:
    """Records set/get calls against an in-memory {(alias, parameter): value} store."""

    def __init__(self, values=None):
        self.values = dict(values or {})
        self.sets = []  # (alias, parameter, value) in call order

    def get_parameter(self, *, alias, parameter):
        return self.values[alias, parameter]

    def set_parameter(self, *, alias, parameter, value):
        self.values[alias, parameter] = value
        self.sets.append((alias, parameter, value))


def test_restores_prior_value_on_exit():
    platform = FakePlatform({("flux_z_q1", Parameter.FLUX): 0.1})
    with restoring_parameters(platform) as restore:
        restore.set("flux_z_q1", Parameter.FLUX, 0.5)
        assert platform.values["flux_z_q1", Parameter.FLUX] == pytest.approx(0.5)
    assert platform.values["flux_z_q1", Parameter.FLUX] == pytest.approx(0.1)


def test_restores_even_when_body_raises():
    platform = FakePlatform({("flux_z_q1", Parameter.FLUX): 0.1})

    def run_and_raise():
        with restoring_parameters(platform) as restore:
            restore.set("flux_z_q1", Parameter.FLUX, 0.5)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_and_raise()
    assert platform.values["flux_z_q1", Parameter.FLUX] == pytest.approx(0.1)


def test_repeated_sets_restore_the_earliest_value():
    platform = FakePlatform({("drive_q1", Parameter.LO_FREQUENCY): 5e9})
    with restoring_parameters(platform) as restore:
        restore.set("drive_q1", Parameter.LO_FREQUENCY, 6e9)
        restore.set("drive_q1", Parameter.LO_FREQUENCY, 7e9)
    assert platform.values["drive_q1", Parameter.LO_FREQUENCY] == pytest.approx(5e9)


def test_untouched_parameters_are_left_alone():
    platform = FakePlatform(
        {
            ("flux_z_q1", Parameter.FLUX): 0.1,
            ("flux_z_q2", Parameter.FLUX): 0.2,
        }
    )
    with restoring_parameters(platform) as restore:
        restore.set("flux_z_q1", Parameter.FLUX, 0.5)
    assert all(alias != "flux_z_q2" for alias, _, _ in platform.sets)
    assert platform.values["flux_z_q2", Parameter.FLUX] == pytest.approx(0.2)


def test_a_failing_restore_still_restores_the_others_and_re_raises():
    platform = FakePlatform(
        {
            ("flux_z_q1", Parameter.FLUX): 0.1,
            ("flux_z_q2", Parameter.FLUX): 0.2,
        }
    )
    restore = ParameterRestore(platform)
    restore.set("flux_z_q1", Parameter.FLUX, 0.5)
    restore.set("flux_z_q2", Parameter.FLUX, 0.6)

    original_set = platform.set_parameter

    def failing_set(*, alias, parameter, value):
        if alias == "flux_z_q2":
            raise RuntimeError("instrument said no")
        original_set(alias=alias, parameter=parameter, value=value)

    platform.set_parameter = failing_set

    with pytest.raises(ExceptionGroup) as excinfo:
        restore.restore()

    # The failing restore (flux_z_q2) does not strand flux_z_q1 at its set value.
    assert platform.values["flux_z_q1", Parameter.FLUX] == pytest.approx(0.1)
    assert [type(error) for error in excinfo.value.exceptions] == [RuntimeError]


def test_prior_value_is_read_once_per_parameter():
    platform = FakePlatform({("flux_z_q1", Parameter.FLUX): 0.1})
    restore = ParameterRestore(platform)
    gets = []
    original_get = platform.get_parameter

    def counting_get(*, alias, parameter):
        gets.append((alias, parameter))
        return original_get(alias=alias, parameter=parameter)

    platform.get_parameter = counting_get
    restore.set("flux_z_q1", Parameter.FLUX, 0.2)
    restore.set("flux_z_q1", Parameter.FLUX, 0.3)
    restore.restore()

    assert gets == [("flux_z_q1", Parameter.FLUX)]
    assert platform.values["flux_z_q1", Parameter.FLUX] == pytest.approx(0.1)


def test_restoring_platform_routes_set_through_the_handle():
    # A driver that calls platform.set_parameter directly has its writes recorded and undone, so a value
    # it sets (LO, IF, power) does not leak into the next node (ADR-0006).
    platform = FakePlatform({("drive_q1", Parameter.LO_FREQUENCY): 5e9})
    with restoring_parameters(platform) as restore:
        proxy = restoring_platform(platform, restore)
        proxy.set_parameter(alias="drive_q1", parameter=Parameter.LO_FREQUENCY, value=6e9)
        assert platform.values["drive_q1", Parameter.LO_FREQUENCY] == pytest.approx(6e9)
    assert platform.values["drive_q1", Parameter.LO_FREQUENCY] == pytest.approx(5e9)


def test_restoring_platform_delegates_everything_else():
    platform = FakePlatform({("readout_q1", Parameter.IF): 1e8})
    proxy = restoring_platform(platform, ParameterRestore(platform))
    # Anything that is not set_parameter (get_parameter, attributes a StreamArray reads) goes straight
    # through to the wrapped platform.
    assert proxy.get_parameter(alias="readout_q1", parameter=Parameter.IF) == pytest.approx(1e8)
    assert proxy.values is platform.values


def test_restoring_platform_rejects_channel_scoped_set():
    platform = FakePlatform()
    proxy = restoring_platform(platform, ParameterRestore(platform))
    with pytest.raises(NotImplementedError):
        proxy.set_parameter("drive_q1", Parameter.LO_FREQUENCY, 6e9, 0)  # extra positional arg (e.g. channel id)
