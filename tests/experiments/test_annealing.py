"""Tests for the compartmentalized Landau-Zener QProgram builders (QHC-1808, vendored from QHC-1822)."""

import numpy as np
import pytest
from qililab import CrosstalkMatrix, QProgram

from seqtante_open.experiments.qprogram.annealing import (
    MAX_ARBITRARY_WAVEFORM_SAMPLES,
    generate_lz_qprogram_not_idling_reset,
)

_FLUX_BUSES = ("flux_z_q0", "flux_x_q0")


def _crosstalk(buses=_FLUX_BUSES) -> CrosstalkMatrix:
    return CrosstalkMatrix.from_array(buses=list(buses), matrix_array=np.eye(len(buses)))


def _build(*, path_points: int = 100, ramp: int = 50, **overrides) -> QProgram:
    kwargs = {
        "readout_if_freq": 1e8,
        "reset_dur": 200,
        "flux_path_sequence": [[0.1, 0.1, path_points], [0.2, 0.0, path_points]],
        "flux_reset": (0.3, 0.3),
        "flux_readout": (0.0, 0.0),
        "readout_dur": 1000,
        "averages": 10,
        "ramp_reset_duration": (ramp, ramp),
        "ramp_readout_duration": (ramp, ramp),
        "flux_idle_dc": (0.0, 0.0),
        "crosstalk": _crosstalk(),
        "flux_buses": _FLUX_BUSES,
    }
    kwargs.update(overrides)
    return generate_lz_qprogram_not_idling_reset(**kwargs)


def test_builds_a_qprogram():
    assert isinstance(_build(), QProgram)


def test_sequencer_memory_guard_raises_on_an_over_long_trajectory():
    # reset ramp + path + readout ramp must fit MAX_ARBITRARY_WAVEFORM_SAMPLES per flux bus.
    with pytest.raises(ValueError, match="arbitrary samples per flux bus"):
        _build(path_points=MAX_ARBITRARY_WAVEFORM_SAMPLES)


def test_a_trajectory_at_the_budget_still_builds():
    # Two path segments of n points each, plus reset ramp (2 * ramp) and readout ramp (1 * ramp).
    ramp = 50
    budget = MAX_ARBITRARY_WAVEFORM_SAMPLES - 3 * ramp
    assert isinstance(_build(path_points=budget // 2, ramp=ramp), QProgram)


def test_crosstalk_must_cover_exactly_the_flux_buses():
    with pytest.raises(ValueError, match="exactly two flux buses"):
        _build(crosstalk=_crosstalk(("flux_z_q0", "flux_x_q1")))


def test_both_axis_ramp_requires_equal_stage_durations():
    # The readout ramp moves both axes at once, so its two stage durations must match.
    with pytest.raises(ValueError, match="both entries of ramp_duration must be equal"):
        _build(ramp_readout_duration=(50, 60))


def test_non_integer_waypoint_samples_are_rejected():
    with pytest.raises(ValueError, match="whole number of samples"):
        _build(flux_path_sequence=[[0.1, 0.1, 100.5], [0.2, 0.0, 100]])
