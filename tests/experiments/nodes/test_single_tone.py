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

"""Tests for ``single_tone_node``.

The runcard exposes qubits ``[1, 2]``, so targets ``["q1", "q2"]`` produce one
measurement each, on ``readout_q1`` and ``readout_q2``.

Unlike the crosstalk experiments, this one runs to completion instead of raising
``InterruptCalibration``: it fits each measurement and writes the fitted IF back
into the readout bus. The execution function, the fit model and the writer
(``save_platform``) are mocked so nothing is measured, fitted or written to disk.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter

from seqtante_open.experiments.nodes.single_tone import single_tone_node
from seqtante_open.experiments.utils.flux_buses import get_all_flux_buses

RUNCARD_PATH = Path(__file__).resolve().parents[2] / "runcards" / "test_AQPU_runcard.yml"

MODULE = "seqtante_open.experiments.nodes.single_tone"
FN = "single_tone_experiment"

MEASUREMENT_ID = 777
FITTED_IF = 1.234e8
DATA_FOLDER = "unused-mocked-folder"

READOUT_BUSES = ("readout_q1", "readout_q2")


def _identity_crosstalk(platform) -> CrosstalkMatrix:
    buses = get_all_flux_buses(platform)
    return CrosstalkMatrix.from_buses({b: {bb: (1.0 if b == bb else 0.0) for bb in buses} for b in buses})


def _fit_model() -> MagicMock:
    """Stand-in for ``FluxoniumSingleToneModel``: fits nothing, reports a fixed result.

    ``results`` mirrors the real model's shape -- one entry per rotated quadrature,
    each holding ``fitted_if``/``fit_values``/``r_squared`` -- because the experiment
    reads ``model.results["signal"]["fitted_if"]`` to update the readout bus.
    """
    model = MagicMock(name="FluxoniumSingleToneModel")
    model.results = {
        "signal": {"fitted_if": FITTED_IF, "fit_values": np.zeros(21), "r_squared": 0.99},
        "noise": {"fitted_if": 0.0, "fit_values": np.zeros(21), "r_squared": 0.01},
    }
    return model


def _base_parameters() -> dict:
    return {
        "targets": ["q1", "q2"],
        "calibration_path": "unused-mocked.yml",
        "data_folder": DATA_FOLDER,
        "if_sweep": [-1.5e6, 1.5e6, 21],
        "averages": 4000,
        "relax_duration": 200_000,
        "readout_amplitude": 0.075,
        "readout_duration": 2000,
        "ringup_time": 0,
        "q1": {},
        "q2": {},
    }


@pytest.fixture
def run_experiment(platform, mock_db_manager, mock_recorder):
    """Run ``single_tone_node`` with the execution, fit and IO boundaries mocked."""
    calibration = SimpleNamespace(crosstalk_matrix=_identity_crosstalk(platform))
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.FluxoniumSingleToneModel", output=_fit_model())
    mock_recorder.mock(f"{MODULE}.save_platform")

    def run(parameters: dict):
        single_tone_node(platform=platform, platform_path=str(RUNCARD_PATH), parameters=parameters)
        return mock_recorder

    run.platform = platform
    run.calibration = calibration
    # The experiment overwrites each readout bus IF with the fitted one, so the
    # values the sweeps are built from have to be read *before* it runs.
    run.initial_ifs = {bus: platform.get_parameter(bus, Parameter.IF) for bus in READOUT_BUSES}
    return run


def test_basic_parameters(run_experiment):
    """Every global parameter reaches the execution function, sweeps included."""
    platform = run_experiment.platform
    parameters = _base_parameters()
    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert len(calls) == 2, f"Expected {FN} to be called once per qubit"
    assert all(c["kwargs"]["readout_amplitude"] == 0.075 for c in calls)
    assert all(c["kwargs"]["averages"] == 4000 for c in calls)
    assert all(c["kwargs"]["readout_duration"] == 2000 for c in calls)
    assert all(c["kwargs"]["relax_duration"] == 200_000 for c in calls)
    assert all(c["kwargs"]["calibration"] is run_experiment.calibration for c in calls)
    assert [c["kwargs"]["qubit_idx"] for c in calls] == ["q1", "q2"]
    assert [c["kwargs"]["readout_bus"] for c in calls] == list(READOUT_BUSES)

    # The IF sweep is the raw linspace offset by the readout bus IF, both read
    # before the fit overwrote the IF.
    for call in calls:
        readout_bus = call["kwargs"]["readout_bus"]
        expected = np.linspace(-1.5e6, 1.5e6, 21) + run_experiment.initial_ifs[readout_bus]
        np.testing.assert_allclose(call["kwargs"]["if_sweep"], expected)

    fits = recorder.calls["FluxoniumSingleToneModel"]
    assert [f["kwargs"]["lo"] for f in fits] == [
        platform.get_parameter(bus, Parameter.LO_FREQUENCY) for bus in READOUT_BUSES
    ]


def test_per_target_overwrite_reaches_execution(run_experiment):
    """A per-target override wins for that target; other targets keep the globals."""
    parameters = _base_parameters()
    parameters["q1"] = {
        "if_sweep": [-1.5e6, 1.5e6, 41],
        "readout_amplitude": 0.05,
        "readout_duration": 3000,
        "averages": 1500,
    }  # override only for q1

    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert calls, f"Expected {FN} to be called"

    (q1_call,) = [c for c in calls if c["kwargs"]["qubit_idx"] == "q1"]
    assert q1_call["kwargs"]["readout_amplitude"] == 0.05
    assert q1_call["kwargs"]["readout_duration"] == 3000
    assert q1_call["kwargs"]["averages"] == 1500
    np.testing.assert_allclose(
        q1_call["kwargs"]["if_sweep"], np.linspace(-1.5e6, 1.5e6, 41) + run_experiment.initial_ifs["readout_q1"]
    )

    (q2_call,) = [c for c in calls if c["kwargs"]["qubit_idx"] == "q2"]
    assert q2_call["kwargs"]["readout_amplitude"] == 0.075
    assert q2_call["kwargs"]["readout_duration"] == 2000
    assert q2_call["kwargs"]["averages"] == 4000
    assert len(q2_call["kwargs"]["if_sweep"]) == 21


def test_defaults_are_used_when_parameters_are_missing(run_experiment):
    """Optional timing parameters fall back to the module defaults."""
    parameters = _base_parameters()
    del parameters["ringup_time"]

    recorder = run_experiment(parameters)

    for call in recorder.calls[FN]:
        assert call["kwargs"]["ringup_time"] == 0


def test_overwrite_does_not_mutate_shared_parameters(run_experiment):
    """Merging per-target overrides must not write back into the shared dict."""
    parameters = _base_parameters()
    parameters["q1"] = {"readout_amplitude": 0.9}

    run_experiment(parameters)

    assert parameters["readout_amplitude"] == 0.075
    assert parameters["q1"] == {"readout_amplitude": 0.9}


def test_fitted_if_is_written_to_the_platform(run_experiment):
    """Each measured readout bus gets its fitted IF, and the platform is saved."""
    recorder = run_experiment(_base_parameters())

    platform = run_experiment.platform
    assert {bus: platform.get_parameter(bus, Parameter.IF) for bus in READOUT_BUSES} == {
        "readout_q1": FITTED_IF,
        "readout_q2": FITTED_IF,
    }

    # One fit per measurement, each pointed at the configured data folder.
    fits = recorder.calls["FluxoniumSingleToneModel"]
    assert len(fits) == 2
    assert all(f["args"] == (MEASUREMENT_ID,) for f in fits)
    assert all(f["kwargs"]["path"] == DATA_FOLDER for f in fits)
    assert [f["kwargs"]["target"] for f in fits] == ["q1", "q2"]

    # The updated platform is persisted once, at the end.
    assert len(recorder.calls["save_platform"]) == 1
    assert recorder.calls["save_platform"][0]["kwargs"] == {
        "path": str(RUNCARD_PATH),
        "platform": platform,
    }


def test_platform_is_saved_even_when_the_experiment_fails(run_experiment, mock_recorder):
    """``save_platform`` lives in a ``finally``, so a failed run still persists."""
    mock_recorder.reset()
    calibration = SimpleNamespace(crosstalk_matrix=_identity_crosstalk(run_experiment.platform))
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.save_platform")
    boom = mock_recorder.mock(f"{MODULE}.{FN}")
    boom.side_effect = RuntimeError("instrument exploded")

    with pytest.raises(RuntimeError, match="instrument exploded"):
        single_tone_node(
            platform=run_experiment.platform, platform_path=str(RUNCARD_PATH), parameters=_base_parameters()
        )

    assert len(mock_recorder.calls["save_platform"]) == 1


def test_calibration_without_crosstalk_matrix_is_rejected(run_experiment, mock_recorder):
    """A calibration lacking a ``CrosstalkMatrix`` fails before anything is executed."""
    mock_recorder.reset()
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=SimpleNamespace(crosstalk_matrix=None))
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.save_platform")

    with pytest.raises(ValueError, match="CrosstalkMatrix"):
        single_tone_node(
            platform=run_experiment.platform, platform_path=str(RUNCARD_PATH), parameters=_base_parameters()
        )

    assert FN not in mock_recorder.calls or not mock_recorder.calls[FN]
