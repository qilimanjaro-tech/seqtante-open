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

"""Tests for ``two_tone_node``.

The runcard exposes qubits ``[1, 2]``, so targets ``["q1", "q2"]`` produce one
measurement each, driven on ``drive_q1``/``drive_q2`` and read on
``readout_q1``/``readout_q2``.

The node fits each measurement and writes the fitted IF back to the *drive* bus
-- into the calibration's operating point when one is configured, onto the
platform otherwise. The execution function, the fit model and both writers
(``serialize_to`` and ``save_platform``) are mocked so nothing is measured,
fitted or written to disk.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter

from seqtante_open.experiments.nodes.two_tone import two_tone_node
from seqtante_open.experiments.utils.flux_buses import get_all_flux_buses

RUNCARD_PATH = Path(__file__).resolve().parents[2] / "runcards" / "test_AQPU_runcard.yml"

MODULE = "seqtante_open.experiments.nodes.two_tone"
FN = "two_tone_experiment"

MEASUREMENT_ID = 777
FITTED_IF = 1.234e8
DATA_FOLDER = "unused-mocked-folder"

OPERATING_POINT = "park"

QUBITS = ("q1", "q2")
DRIVE_BUSES = ("drive_q1", "drive_q2")
READOUT_BUSES = ("readout_q1", "readout_q2")


def _identity_crosstalk(platform) -> CrosstalkMatrix:
    buses = get_all_flux_buses(platform)
    return CrosstalkMatrix.from_buses({b: {bb: (1.0 if b == bb else 0.0) for bb in buses} for b in buses})


def _calibration(platform) -> Calibration:
    """A real ``Calibration`` holding nothing but an identity crosstalk matrix.

    Real rather than a stand-in because the node writes the ``data_folder`` into
    ``parameters`` of a copy of it, and saves fitted IFs into the original.
    """
    calibration = Calibration()
    calibration.crosstalk_matrix = _identity_crosstalk(platform)
    return calibration


def _fit_model() -> MagicMock:
    """Stand-in for ``FluxoniumTwoToneModel``: fits nothing, reports a fixed result.

    ``results`` mirrors the real model's shape -- one entry per rotated quadrature,
    each holding ``fitted_if``/``fit_values``/``r_squared`` -- because the node
    reads ``model.results["signal"]["fitted_if"]`` to update the drive bus.
    """
    model = MagicMock(name="FluxoniumTwoToneModel")
    model.results = {
        "signal": {"fitted_if": FITTED_IF, "fit_values": np.zeros(21), "r_squared": 0.99},
        "noise": {"fitted_if": 0.0, "fit_values": np.zeros(21), "r_squared": 0.0},
    }
    return model


def _base_parameters() -> dict:
    return {
        "targets": ["q1", "q2"],
        "calibration_path": "unused-mocked.yml",
        "data_folder": DATA_FOLDER,
        "freq_sweep": [-1.5e6, 1.5e6, 21],
        "averages": 1000,
        "relax_duration": 200_000,
        "drive_duration": 40,
        "drive_amplitude": 0.5,
        "readout_amplitude": 0.075,
        "readout_duration": 2000,
        "drive_gain": 0.8,
        "ringup_time": 24,
        "overlap_time": 12,
        "q1": {},
        "q2": {},
    }


@pytest.fixture
def run_experiment(platform, mock_db_manager, mock_recorder):
    """Run ``two_tone_node`` with the execution, fit and IO boundaries mocked."""
    calibration = _calibration(platform)
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.FluxoniumTwoToneModel", output=_fit_model())
    mock_recorder.mock(f"{MODULE}.serialize_to")
    mock_recorder.mock(f"{MODULE}.save_platform")
    mock_recorder.mock(f"{MODULE}.get_operating_point")
    mock_recorder.mock(f"{MODULE}.save_parameters_to_operating_point")

    def run(parameters: dict):
        two_tone_node(platform=platform, platform_path=str(RUNCARD_PATH), parameters=parameters)
        return mock_recorder

    run.platform = platform
    run.calibration = calibration
    # The node overwrites each drive bus IF with the fitted one, so the values the
    # sweeps are built from have to be read *before* it runs.
    run.initial_ifs = {bus: platform.get_parameter(bus, Parameter.IF) for bus in DRIVE_BUSES}
    return run


def test_basic_parameters(run_experiment):
    """Every global parameter reaches the execution function, sweeps included."""
    parameters = _base_parameters()
    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert len(calls) == 2, f"Expected {FN} to be called once per qubit"
    assert all(c["kwargs"]["r_amp"] == pytest.approx(0.075) for c in calls)
    assert all(c["kwargs"]["r_duration"] == 2000 for c in calls)
    assert all(c["kwargs"]["d_amp"] == pytest.approx(0.5) for c in calls)
    assert all(c["kwargs"]["d_duration"] == 40 for c in calls)
    assert all(c["kwargs"]["averages"] == 1000 for c in calls)
    assert all(c["kwargs"]["relax_duration"] == 200_000 for c in calls)
    assert all(c["kwargs"]["drive_gain"] == pytest.approx(0.8) for c in calls)
    assert all(c["kwargs"]["ringup_time"] == 24 for c in calls)
    assert all(c["kwargs"]["overlap_time"] == 12 for c in calls)
    assert all(c["kwargs"]["autocalibration"] is True for c in calls)
    assert [c["kwargs"]["target"] for c in calls] == list(QUBITS)
    assert [c["kwargs"]["drive_bus"] for c in calls] == list(DRIVE_BUSES)
    assert [c["kwargs"]["readout_bus"] for c in calls] == list(READOUT_BUSES)

    # Each measurement gets its own stamped copy; the shared calibration stays untouched.
    for call in calls:
        calibration = call["kwargs"]["calibration"]
        assert calibration is not run_experiment.calibration
        assert calibration.parameters["data_folder"] == DATA_FOLDER
    assert "data_folder" not in run_experiment.calibration.parameters

    # The frequency sweep is the raw linspace offset by the drive bus IF, read
    # before the fit overwrote it.
    for call in calls:
        expected = np.linspace(-1.5e6, 1.5e6, 21) + run_experiment.initial_ifs[call["kwargs"]["drive_bus"]]
        np.testing.assert_allclose(call["kwargs"]["drive_IF_sweep"], expected)

    fits = recorder.calls["FluxoniumTwoToneModel"]
    assert [f["kwargs"]["lo"] for f in fits] == [
        run_experiment.platform.get_parameter(bus, Parameter.LO_FREQUENCY) for bus in DRIVE_BUSES
    ]


def test_only_qubit_targets_are_measured(run_experiment):
    """``targets`` may carry couplers; the node drives qubits only."""
    parameters = _base_parameters()
    parameters["targets"] = ["q1", "c1_2", "q2"]
    parameters["c1_2"] = {}

    recorder = run_experiment(parameters)

    assert [c["kwargs"]["target"] for c in recorder.calls[FN]] == list(QUBITS)


def test_per_target_overwrite_reaches_execution(run_experiment):
    """A per-target override wins for that target; other targets keep the globals."""
    parameters = _base_parameters()
    parameters["q1"] = {
        "freq_sweep": [-1.5e6, 1.5e6, 41],
        "drive_amplitude": 0.25,
        "drive_duration": 80,
        "averages": 1500,
    }

    recorder = run_experiment(parameters)

    (q1_call,) = [c for c in recorder.calls[FN] if c["kwargs"]["target"] == "q1"]
    assert q1_call["kwargs"]["d_amp"] == pytest.approx(0.25)
    assert q1_call["kwargs"]["d_duration"] == 80
    assert q1_call["kwargs"]["averages"] == 1500
    np.testing.assert_allclose(
        q1_call["kwargs"]["drive_IF_sweep"], np.linspace(-1.5e6, 1.5e6, 41) + run_experiment.initial_ifs["drive_q1"]
    )

    (q2_call,) = [c for c in recorder.calls[FN] if c["kwargs"]["target"] == "q2"]
    assert q2_call["kwargs"]["d_amp"] == pytest.approx(0.5)
    assert q2_call["kwargs"]["d_duration"] == 40
    assert q2_call["kwargs"]["averages"] == 1000
    assert len(q2_call["kwargs"]["drive_IF_sweep"]) == 21


def test_defaults_are_used_when_parameters_are_missing(run_experiment):
    """Optional gain and timing parameters fall back to the module defaults."""
    parameters = _base_parameters()
    del parameters["drive_gain"]
    del parameters["ringup_time"]
    del parameters["overlap_time"]

    recorder = run_experiment(parameters)

    for call in recorder.calls[FN]:
        assert call["kwargs"]["drive_gain"] == 1
        assert call["kwargs"]["ringup_time"] == 0
        assert call["kwargs"]["overlap_time"] == 0


def test_overwrite_does_not_mutate_shared_parameters(run_experiment):
    """Merging per-target overrides must not write back into the shared dict."""
    parameters = _base_parameters()
    parameters["q1"] = {"drive_amplitude": 0.9}

    run_experiment(parameters)

    assert parameters["drive_amplitude"] == pytest.approx(0.5)
    assert parameters["q1"] == {"drive_amplitude": 0.9}


def test_fitted_if_is_written_to_the_platform(run_experiment):
    """Without an operating point each drive bus gets its fitted IF, and both writers run."""
    recorder = run_experiment(_base_parameters())

    platform = run_experiment.platform
    assert {bus: platform.get_parameter(bus, Parameter.IF) for bus in DRIVE_BUSES} == {
        "drive_q1": FITTED_IF,
        "drive_q2": FITTED_IF,
    }

    # One fit per measurement, each pointed at the configured data folder.
    fits = recorder.calls["FluxoniumTwoToneModel"]
    assert len(fits) == 2
    assert all(f["args"] == (MEASUREMENT_ID,) for f in fits)
    assert all(f["kwargs"]["path"] == DATA_FOLDER for f in fits)
    assert [f["kwargs"]["target"] for f in fits] == list(QUBITS)

    # The updated calibration and platform are each persisted once, at the end.
    assert recorder.calls["serialize_to"][0]["args"] == (run_experiment.calibration, "unused-mocked.yml")
    assert len(recorder.calls["serialize_to"]) == 1
    assert recorder.calls["save_platform"] == [
        {"args": (), "kwargs": {"path": str(RUNCARD_PATH), "platform": platform}}
    ]


def test_operating_point_is_applied_before_each_measurement(run_experiment):
    """With ``operating_point`` configured, every qubit is parked before it is measured.

    The helper itself is pinned in ``tests/experiments/utils``; all that matters
    here is that the node reaches it with the right target, name and platform.
    """
    parameters = _base_parameters()
    parameters["operating_point"] = OPERATING_POINT

    recorder = run_experiment(parameters)

    calls = recorder.calls["get_operating_point"]
    assert len(calls) == 2, "Expected one operating point per qubit"
    for qubit, applied in zip(QUBITS, calls, strict=True):
        assert applied["args"] == (run_experiment.calibration,)
        assert applied["kwargs"] == {
            "target": qubit,
            "operating_point_name": OPERATING_POINT,
            "platform": run_experiment.platform,
        }


def test_operating_point_is_left_alone_when_not_configured(run_experiment):
    """Without it the node measures the platform as it already stands."""
    recorder = run_experiment(_base_parameters())

    assert recorder.calls["get_operating_point"] == []
    assert recorder.calls["save_parameters_to_operating_point"] == []


def test_fitted_if_is_saved_to_the_operating_point(run_experiment):
    """With an operating point the fit goes to the calibration instead of the platform."""
    parameters = _base_parameters()
    parameters["operating_point"] = OPERATING_POINT

    recorder = run_experiment(parameters)

    calls = recorder.calls["save_parameters_to_operating_point"]
    assert len(calls) == 2, "Expected one save per qubit"
    for qubit, saved in zip(QUBITS, calls, strict=True):
        assert saved["args"] == ((f"drive_{qubit}", Parameter.IF, FITTED_IF),)
        assert saved["kwargs"] == {
            "calibration": run_experiment.calibration,
            "target": qubit,
            "operating_point_name": OPERATING_POINT,
        }

    platform = run_experiment.platform
    assert {bus: platform.get_parameter(bus, Parameter.IF) for bus in DRIVE_BUSES} == run_experiment.initial_ifs


def test_results_are_saved_even_when_the_experiment_fails(run_experiment, mock_recorder):
    """Both writers live in a ``finally``, so a failed run still persists."""
    mock_recorder.reset()
    calibration = _calibration(run_experiment.platform)
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.serialize_to")
    mock_recorder.mock(f"{MODULE}.save_platform")
    boom = mock_recorder.mock(f"{MODULE}.{FN}")
    boom.side_effect = RuntimeError("instrument exploded")

    with pytest.raises(RuntimeError, match="instrument exploded"):
        two_tone_node(platform=run_experiment.platform, platform_path=str(RUNCARD_PATH), parameters=_base_parameters())

    assert len(mock_recorder.calls["serialize_to"]) == 1
    assert len(mock_recorder.calls["save_platform"]) == 1


def test_calibration_without_crosstalk_matrix_is_rejected(run_experiment, mock_recorder):
    """A calibration lacking a ``CrosstalkMatrix`` fails before anything is executed."""
    mock_recorder.reset()
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=Calibration())
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.serialize_to")
    mock_recorder.mock(f"{MODULE}.save_platform")

    with pytest.raises(ValueError, match="CrosstalkMatrix"):
        two_tone_node(platform=run_experiment.platform, platform_path=str(RUNCARD_PATH), parameters=_base_parameters())

    assert not mock_recorder.calls[FN]
