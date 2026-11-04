"""Tests for the pulse-name grammar (ADR-0008)."""

import pytest

from seqtante_open.calibration.pulses import parse_pulse_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("readout", ("readout", None)),
        ("pi", ("pi", None)),
        ("pi_half", ("pi_half", None)),
        ("readout_long", ("readout_long", None)),
        ("readout-reset", ("readout", "reset")),
        ("pi-reset", ("pi", "reset")),
        ("pi_half-reset", ("pi_half", "reset")),
        ("x90-reset", ("x90", "reset")),
        ("readout_long-sweet_spot", ("readout_long", "sweet_spot")),
    ],
)
def test_parse_pulse_name_splits_purpose_and_point(name: str, expected: tuple[str, str | None]) -> None:
    assert parse_pulse_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "Readout",  # uppercase
        "1pi",  # leading digit
        "-reset",  # no purpose
        "pi-",  # empty point
        "pi--reset",  # double separator
        "pi-reset-extra",  # two separators
        "pi reset",  # space
        "pi.reset",  # dot
        "",  # empty
    ],
)
def test_parse_pulse_name_rejects_malformed(name: str) -> None:
    with pytest.raises(ValueError, match="invalid pulse name"):
        parse_pulse_name(name)
