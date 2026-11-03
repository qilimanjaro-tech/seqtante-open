"""Tests for the ``(target, line[, axis]) -> bus alias`` mapping (ADR-0008)."""

import pytest

from seqtante_open.calibration.bus_alias import Role, bus_alias, parse_bus_alias
from seqtante_open.targets import CouplerTarget, QubitTarget


@pytest.mark.parametrize(
    ("target", "role", "axis", "expected"),
    [
        ("q14", Role.drive, None, "drive_q14"),
        ("q14", Role.readout, None, "readout_q14"),
        ("q14", Role.flux, "z", "flux_z_q14"),
        ("q14", Role.flux, "x", "flux_x_q14"),
        ("q14", Role.flux, None, "flux_q14"),
        ("c2_3", Role.flux, "z", "flux_z_c2_3"),
        ("c1_10", Role.flux, "z", "flux_z_c1_10"),
    ],
)
def test_bus_alias(target: str, role: Role, axis: str | None, expected: str) -> None:
    assert bus_alias(target, role, axis) == expected


def test_bus_alias_accepts_a_target_object() -> None:
    assert bus_alias(QubitTarget(14), Role.readout) == "readout_q14"
    assert bus_alias(CouplerTarget.between(3, 2), Role.flux, "z") == "flux_z_c2_3"


def test_bus_alias_rejects_a_misordered_coupler() -> None:
    # Canonicalisation is targets.py's job; callers order couplers with CouplerTarget.between.
    with pytest.raises(ValueError, match="Invalid target"):
        bus_alias("c3_2", Role.flux, "z")


def test_bus_alias_rejects_an_invalid_target() -> None:
    with pytest.raises(ValueError, match="Invalid target"):
        bus_alias("not_a_target", Role.flux, "z")


def test_bus_alias_rejects_axis_on_non_flux_line() -> None:
    with pytest.raises(ValueError, match="take no axis"):
        bus_alias("q14", Role.drive, "z")


def test_bus_alias_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="unknown flux axis"):
        bus_alias("q14", Role.flux, "w")


@pytest.mark.parametrize(
    ("bus", "expected"),
    [
        ("drive_q14", ("q14", Role.drive, None)),
        ("readout_q14", ("q14", Role.readout, None)),
        ("flux_z_q14", ("q14", Role.flux, "z")),
        ("flux_q14", ("q14", Role.flux, None)),  # no axis
        ("flux_z_c2_3", ("c2_3", Role.flux, "z")),  # axis first, coupler target is the remainder
        ("flux_c2_3", ("c2_3", Role.flux, None)),
    ],
)
def test_parse_bus_alias(bus: str, expected: tuple[str, Role, str | None]) -> None:
    assert parse_bus_alias(bus) == expected


def test_parse_bus_alias_rejects_unknown_line() -> None:
    with pytest.raises(ValueError, match="'xyz'"):
        parse_bus_alias("xyz_q14")


def test_parse_bus_alias_rejects_non_alias() -> None:
    with pytest.raises(ValueError, match="not a bus alias"):
        parse_bus_alias("qdac2")


@pytest.mark.parametrize(
    ("target", "role", "axis"),
    [
        ("q14", Role.drive, None),
        ("q14", Role.readout, None),
        ("q14", Role.flux, "z"),
        ("c2_3", Role.flux, "z"),
        ("q14", Role.flux, None),
    ],
)
def test_bus_alias_round_trips(target: str, role: Role, axis: str | None) -> None:
    assert parse_bus_alias(bus_alias(target, role, axis)) == (target, role, axis)
