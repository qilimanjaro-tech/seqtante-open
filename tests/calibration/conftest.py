"""Shared fixtures for the calibration-file model tests."""

import copy

import pytest

#: A minimal but complete EuroQCS file: q14 with sweet_spot + reset, parked couplers,
#: readout levels at reset, and a rabi section. Keys use the wire format (``if``).
_VALID_FILE: dict = {
    "operating_points_version": 1,
    "operating_points": {
        "q14": {
            "sweet_spot": {
                "flux": {"z": 0.0, "x": 0.0},
                "drive": {"lo": 3.2e9, "if": -95e6},
                "readout": {"lo": 6.38e9, "if": 63.4e6, "attenuation": 48},
            },
            "reset": {
                "flux": {"z": 0.615, "x": 0.27},
                "parked_flux": {"c13_14": {"z": 0.02}, "c14_15": {"z": -0.01}},
                "drive": {"lo": 2.9e9, "if": -100e6},
                "readout": {
                    "lo": 6.38e9,
                    "if": 63.4e6,
                    "attenuation": 48,
                    "levels": {
                        "pulse": "readout-reset",
                        "level_0": {"nominal": 0.0, "std": 0.01},
                        "level_1": {"nominal": 1.0, "std": 0.02},
                    },
                },
            },
        }
    },
    "experiments": {
        "rabi": {"q14": {"operating_point": "reset", "pulses": {"readout": "readout-reset", "drive": "pi-reset"}}}
    },
}


@pytest.fixture
def valid_file() -> dict:
    """A fresh deep copy of the valid file, safe for a test to mutate."""
    return copy.deepcopy(_VALID_FILE)
