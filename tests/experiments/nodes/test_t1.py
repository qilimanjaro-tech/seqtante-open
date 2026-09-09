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

"""Tests for ``t1_node``.

The runcard exposes qubits ``[1, 2]``, so targets ``["q1", "q2"]`` produce one
measurement each, on ``drive_q1``/``readout_q1`` and ``drive_q2``/``readout_q2``.

T1 is a benchmark rather than a calibrated parameter, so unlike ``single_tone`` or
``two_tone`` this node writes no fitted value anywhere: it measures, fits, plots and
leaves the platform and the ``Calibration`` as it found them. That is what
``test_nothing_is_written_back`` pins. The execution function, the fit model and both
writers (``save_platform``, ``serialize_to``) are mocked so nothing is measured, fitted
or written to disk.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter

from seqtante_open.experiments.nodes.t1 import t1_node
from seqtante_open.experiments.utils.flux_buses import get_all_flux_buses

RUNCARD_PATH = Path(__file__).resolve().parents[2] / "runcards" / "test_AQPU_runcard.yml"

MODULE = "seqtante_open.experiments.nodes.t1"
FN = "t1_experiment"

MEASUREMENT_ID = 777
DATA_FOLDER = "unused-mocked-folder"
CALIBRATION_PATH = "unused-mocked.yml"

DRIVE_BUSES = ("drive_q1", "drive_q2")
READOUT_BUSES = ("readout_q1", "readout_q2")


def _identity_crosstalk(platform) -> CrosstalkMatrix:
    buses = get_all_flux_buses(platform)
    return CrosstalkMatrix.from_buses({b: {bb: (1.0 if b == bb else 0.0) for bb in buses} for b in buses})


def _calibration(platform, lo: dict[str, float] | None = None) -> Calibration:
    """A real ``Calibration`` holding a crosstalk matrix and the per-target LO table.

    Real rather than a stand-in because the node writes the ``data_folder`` into
    ``parameters`` of a copy of it.
    """
    calibration = Calibration()
    calibration.crosstalk_matrix = _identity_crosstalk(platform)
    calibration.parameters = {"LO": lo or {}}
    return calibration


def _base_parameters() -> dict:
    return {
        "targets": ["q1", "q2"],
        "calibration_path": CALIBRATION_PATH,
        "data_folder": DATA_FOLDER,
        "wait_sweep": [0, 40_000, 81],
        "averages": 4000,
        "relax_duration": 200_000,
        "drive_amplitude": 0.5,
        "drive_duration": 4000,
        "readout_amplitude": 0.075,
        "readout_duration": 2000,
        "q1": {},
        "q2": {},
    }


@pytest.fixture
def run_experiment(platform, mock_db_manager, mock_recorder):
    """Run ``t1_node`` with the execution, fit and IO boundaries mocked."""
    calibration = _calibration(platform)
    fit_model = MagicMock(name="T1Fit")
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.T1Fit", output=fit_model)
    mock_recorder.mock(f"{MODULE}.save_platform")
    mock_recorder.mock(f"{MODULE}.serialize_to")

    def run(parameters: dict):
        t1_node(platform=platform, platform_path=str(RUNCARD_PATH), parameters=parameters)
        return mock_recorder

    run.platform = platform
    run.calibration = calibration
    run.fit_model = fit_model
    return run


def test_basic_parameters(run_experiment):
    """Every global parameter reaches the execution function, sweep included."""
    platform = run_experiment.platform
    recorder = run_experiment(_base_parameters())

    calls = recorder.calls[FN]
    assert len(calls) == 2, f"Expected {FN} to be called once per qubit"
    assert all(c["kwargs"]["drive_amplitude"] == pytest.approx(0.5) for c in calls)
    assert all(c["kwargs"]["drive_step_duration"] == 4000 for c in calls)
    assert all(c["kwargs"]["readout_amplitude"] == pytest.approx(0.075) for c in calls)
    assert all(c["kwargs"]["readout_duration"] == 2000 for c in calls)
    assert all(c["kwargs"]["averages"] == 4000 for c in calls)
    assert all(c["kwargs"]["relax_duration"] == 200_000 for c in calls)
    assert all(c["kwargs"]["autocalibration"] is True for c in calls)
    assert [c["kwargs"]["target"] for c in calls] == ["q1", "q2"]
    assert [c["kwargs"]["drive_bus"] for c in calls] == list(DRIVE_BUSES)
    assert [c["kwargs"]["readout_bus"] for c in calls] == list(READOUT_BUSES)

    # The idle-time axis is the raw linspace: absolute ns, nothing added to it.
    for call in calls:
        np.testing.assert_allclose(call["kwargs"]["wait_sweep"], np.linspace(0, 40_000, 81))

    # Both IFs are read off the buses rather than declared as parameters.
    for call in calls:
        assert call["kwargs"]["drive_if"] == platform.get_parameter(call["kwargs"]["drive_bus"], Parameter.IF)
        assert call["kwargs"]["readout_if"] == platform.get_parameter(call["kwargs"]["readout_bus"], Parameter.IF)

    # Each measurement gets its own stamped copy; the shared calibration stays untouched.
    for call in calls:
        calibration = call["kwargs"]["calibration"]
        assert calibration is not run_experiment.calibration
        assert calibration.parameters["data_folder"] == DATA_FOLDER
    assert "data_folder" not in run_experiment.calibration.parameters


def test_coupler_targets_are_skipped(run_experiment):
    """A coupler has no drive line of its own, so coupler tokens are not measured."""
    parameters = _base_parameters()
    parameters["targets"] = ["q1", "c1_2"]

    recorder = run_experiment(parameters)

    assert [c["kwargs"]["target"] for c in recorder.calls[FN]] == ["q1"]


def test_per_target_overwrite_reaches_execution(run_experiment):
    """A per-target override wins for that target; other targets keep the globals."""
    parameters = _base_parameters()
    parameters["q1"] = {
        "wait_sweep": [0, 10_000, 21],
        "drive_amplitude": 0.2,
        "drive_duration": 8000,
        "averages": 1500,
    }  # override only for q1

    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert calls, f"Expected {FN} to be called"

    (q1_call,) = [c for c in calls if c["kwargs"]["target"] == "q1"]
    assert q1_call["kwargs"]["drive_amplitude"] == pytest.approx(0.2)
    assert q1_call["kwargs"]["drive_step_duration"] == 8000
    assert q1_call["kwargs"]["averages"] == 1500
    np.testing.assert_allclose(q1_call["kwargs"]["wait_sweep"], np.linspace(0, 10_000, 21))

    (q2_call,) = [c for c in calls if c["kwargs"]["target"] == "q2"]
    assert q2_call["kwargs"]["drive_amplitude"] == pytest.approx(0.5)
    assert q2_call["kwargs"]["drive_step_duration"] == 4000
    assert q2_call["kwargs"]["averages"] == 4000
    assert len(q2_call["kwargs"]["wait_sweep"]) == 81


def test_defaults_are_used_when_parameters_are_missing(run_experiment):
    """The optional drive-shaping parameters fall back to the module defaults."""
    recorder = run_experiment(_base_parameters())

    for call in recorder.calls[FN]:
        assert call["kwargs"]["drive_gain"] == 1
        assert call["kwargs"]["overlap"] == 0
        assert call["kwargs"]["drive_rise_time"] == 2_000
        assert call["kwargs"]["n_sigmas"] == 4
        assert call["kwargs"]["q_relative_amplitude"] == 0


def test_declared_optional_parameters_win_over_the_defaults(run_experiment):
    """``overlap_time`` is renamed to ``overlap`` on the way through; the rest pass straight."""
    parameters = _base_parameters()
    parameters |= {
        "drive_gain": 0.8,
        "overlap_time": 2000,
        "drive_rise_time": 500,
        "n_sigmas": 6,
        "q_relative_amplitude": 0.1,
    }

    recorder = run_experiment(parameters)

    for call in recorder.calls[FN]:
        assert call["kwargs"]["drive_gain"] == pytest.approx(0.8)
        assert call["kwargs"]["overlap"] == 2000
        assert call["kwargs"]["drive_rise_time"] == 500
        assert call["kwargs"]["n_sigmas"] == 6
        assert call["kwargs"]["q_relative_amplitude"] == pytest.approx(0.1)


def test_overwrite_does_not_mutate_shared_parameters(run_experiment):
    """Merging per-target overrides must not write back into the shared dict."""
    parameters = _base_parameters()
    parameters["q1"] = {"drive_amplitude": 0.9}

    run_experiment(parameters)

    assert parameters["drive_amplitude"] == pytest.approx(0.5)
    assert parameters["q1"] == {"drive_amplitude": 0.9}


def test_drive_lo_prefers_the_calibration_entry(platform, mock_db_manager, mock_recorder):
    """The ``Calibration``'s ``LO`` table wins; a target absent from it falls back to the bus."""
    calibration = _calibration(platform, lo={"q1": 4.7e9})
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.T1Fit", output=MagicMock(name="T1Fit"))
    mock_recorder.mock(f"{MODULE}.save_platform")
    mock_recorder.mock(f"{MODULE}.serialize_to")

    t1_node(platform=platform, platform_path=str(RUNCARD_PATH), parameters=_base_parameters())

    los = {c["kwargs"]["target"]: c["kwargs"]["drive_lo"] for c in mock_recorder.calls[FN]}
    assert los == {
        "q1": 4.7e9,
        "q2": platform.get_parameter("drive_q2", Parameter.LO_FREQUENCY),
    }


def test_fit_runs_once_per_measurement(run_experiment):
    """Each measurement is fitted and plotted, pointed at the configured data folder."""
    recorder = run_experiment(_base_parameters())

    fits = recorder.calls["T1Fit"]
    assert len(fits) == 2
    assert all(f["kwargs"]["measurement_id"] == MEASUREMENT_ID for f in fits)
    assert all(f["kwargs"]["path"] == DATA_FOLDER for f in fits)
    assert [f["kwargs"]["target"] for f in fits] == ["q1", "q2"]

    assert run_experiment.fit_model.fit.call_count == 2
    assert run_experiment.fit_model.plot.call_count == 2


def test_nothing_is_written_back(run_experiment):
    """T1 is a benchmark: the run leaves the platform and the ``Calibration`` untouched.

    The other nodes update a bus (``single_tone``, ``two_tone``) or accumulate a flux
    offset (``offset_calibration``). This one only measures, so a future change that
    starts writing a fitted value has to come here and say so.
    """
    platform = run_experiment.platform
    before = {
        bus: (platform.get_parameter(bus, Parameter.IF), platform.get_parameter(bus, Parameter.LO_FREQUENCY))
        for bus in DRIVE_BUSES + READOUT_BUSES
    }

    recorder = run_experiment(_base_parameters())

    after = {
        bus: (platform.get_parameter(bus, Parameter.IF), platform.get_parameter(bus, Parameter.LO_FREQUENCY))
        for bus in DRIVE_BUSES + READOUT_BUSES
    }
    assert after == before

    # The shared calibration is the one that was read, unchanged: only the per-measurement
    # copies carry the stamped data_folder.
    assert run_experiment.calibration.parameters == {"LO": {}}

    # Nothing is persisted either: no runcard write, no calibration write.
    assert not recorder.calls["save_platform"]
    assert not recorder.calls["serialize_to"]


def test_bias_is_zeroed_even_when_the_experiment_fails(run_experiment, mock_recorder, monkeypatch):
    """``set_bias_to_zero`` lives in a ``finally``, so a failed run still leaves the qubit parked."""
    mock_recorder.reset()
    platform = run_experiment.platform
    zeroed: list[int] = []
    monkeypatch.setattr(platform, "set_bias_to_zero", lambda: zeroed.append(1))

    mock_recorder.mock(f"{MODULE}.deserialize_from", output=_calibration(platform))
    boom = mock_recorder.mock(f"{MODULE}.{FN}")
    boom.side_effect = RuntimeError("instrument exploded")

    with pytest.raises(RuntimeError, match="instrument exploded"):
        t1_node(platform=platform, platform_path=str(RUNCARD_PATH), parameters=_base_parameters())

    assert zeroed == [1]


def test_calibration_without_crosstalk_matrix_is_rejected(run_experiment, mock_recorder):
    """A calibration lacking a ``CrosstalkMatrix`` fails before anything is executed."""
    mock_recorder.reset()
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=Calibration())
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)

    with pytest.raises(ValueError, match="CrosstalkMatrix"):
        t1_node(platform=run_experiment.platform, platform_path=str(RUNCARD_PATH), parameters=_base_parameters())

    assert FN not in mock_recorder.calls or not mock_recorder.calls[FN]
