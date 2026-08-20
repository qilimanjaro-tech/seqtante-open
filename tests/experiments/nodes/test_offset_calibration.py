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

"""Tests for ``offset_calibration``.

The runcard exposes qubits ``[1, 2]`` with ``qubit_loops=2`` and
``coupler_loops=1``, so targets ``["q1", "c1_2"]`` produce three measurements:
``flux_q1_z``, ``flux_q1_x`` and ``flux_c1_2_z``.

Unlike the crosstalk experiments, this one runs to completion instead of raising
``InterruptCalibration``: it fits each measurement and writes the fitted offset
into the calibration's crosstalk matrix. The fit model and the two writers
(``save_platform``, ``serialize_to``) are mocked so nothing is fitted or written
to disk.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter

from seqtante_open.experiments.nodes.offset_calibration import single_tone_vs_flux_fluxonium
from seqtante_open.experiments.utils.flux_buses import get_all_flux_buses

RUNCARD_PATH = Path(__file__).resolve().parents[2] / "runcards" / "test_AQPU_runcard.yml"

MODULE = "seqtante_open.experiments.nodes.offset_calibration"
FN = "single_tone_vs_flux"

MEASUREMENT_ID = 777
FITTED_OFFSET = 0.123


def _identity_crosstalk(platform) -> CrosstalkMatrix:
    buses = get_all_flux_buses(platform)
    return CrosstalkMatrix.from_buses({b: {bb: (1.0 if b == bb else 0.0) for bb in buses} for b in buses})


def _fit_model() -> MagicMock:
    """Stand-in for ``FluxoniumSingleToneFluxModel``: fits nothing, reports an offset."""
    return MagicMock(name="FluxoniumSingleToneFluxModel", offset=FITTED_OFFSET)


def _base_parameters() -> dict:
    # The experiment indexes ``parameters[target]`` directly (not ``.get``), so every
    # target needs an entry, as does the readout qubit of every coupler target.
    return {
        "targets": ["q1", "c1_2"],
        "calibration_path": "unused-mocked.yml",
        "data_folder": "unused-mocked-folder",
        "if_sweep": [-1.5e6, 1.5e6, 21],
        "flux_sweep": [-1, 1, 11],
        "readout_amp": 0.075,
        "duration": 2000,
        "averages": 1000,
        "x_loop_readout_flux": 0.3,
        "q1": {},
        "c1_2": {},
    }


@pytest.fixture
def run_experiment(platform, mock_db_manager, mock_recorder):
    """Run ``single_tone_vs_flux_fluxonium`` with the execution, fit and IO boundaries mocked."""
    calibration = SimpleNamespace(crosstalk_matrix=_identity_crosstalk(platform))
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.FluxoniumSingleToneFluxModel", output=_fit_model())
    mock_recorder.mock(f"{MODULE}.serialize_to")

    def run(parameters: dict):
        single_tone_vs_flux_fluxonium(platform=platform, platform_path="unused", parameters=parameters)
        return mock_recorder

    run.calibration = calibration
    return run


def test_basic_parameters(platform, run_experiment):
    """Every global parameter reaches the execution function, sweeps included."""
    parameters = _base_parameters()
    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert calls, f"Expected {FN} to be called"
    assert all(c["kwargs"]["r_amp"] == 0.075 for c in calls)
    assert all(c["kwargs"]["averages"] == 1000 for c in calls)
    assert all(c["kwargs"]["duration"] == 2000 for c in calls)
    assert all(c["kwargs"]["flux_parameter"] == Parameter.FLUX for c in calls)
    assert all(c["kwargs"]["calibration"] is run_experiment.calibration for c in calls)

    # The flux sweep is the raw linspace; the IF sweep is offset by the readout bus IF.
    for call in calls:
        readout_bus = call["kwargs"]["readout_bus"]
        expected_if = np.linspace(-1.5e6, 1.5e6, 21) + platform.get_parameter(readout_bus, Parameter.IF)
        np.testing.assert_allclose(call["kwargs"]["if_sweep"], expected_if)
        np.testing.assert_allclose(call["kwargs"]["flux_sweep"], np.linspace(-1, 1, 11))
        assert call["kwargs"]["lo"] == platform.get_parameter(readout_bus, Parameter.LO_FREQUENCY)


def test_loops_over_all_loops(run_experiment):
    """Both qubit loops (``qubit_loops=2``) and the single coupler loop are measured."""
    parameters = _base_parameters()
    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert calls, f"Expected {FN} to be called"

    measured = {(c["kwargs"]["qubit_idx"], c["kwargs"]["flux_bus"], c["kwargs"]["readout_bus"]) for c in calls}
    assert measured == {
        ("q1", "flux_q1_z", "readout_q1"),
        ("q1", "flux_q1_x", "readout_q1"),
        ("c1_2", "flux_c1_2_z", "readout_q1"),  # coupler read out through its lowest qubit
    }
    assert len(calls) == 3, "each flux bus should be measured exactly once"


def test_x_loop_flux_is_set_if_specified(platform, mock_db_manager, mock_recorder):
    """While the z loop is measured, the qubit's x loop sits at ``x_loop_readout_flux``.

    Asserted at call time rather than afterwards: the run ends with
    ``set_bias_to_zero``, so the final platform state says nothing about the bias
    the measurement actually saw.
    """
    calibration = SimpleNamespace(crosstalk_matrix=_identity_crosstalk(platform))
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.FluxoniumSingleToneFluxModel", output=_fit_model())
    mock_recorder.mock(f"{MODULE}.serialize_to")

    parameters = _base_parameters()
    parameters["q1"] = {"x_loop_readout_flux": 0.2}  # per-qubit value wins over the global 0.3

    x_flux_during = {}

    def _snapshot(**kwargs) -> int:
        x_flux_during[kwargs["flux_bus"]] = platform.get_parameter("flux_q1_x", Parameter.FLUX)
        return MEASUREMENT_ID

    with patch(f"{MODULE}.{FN}", autospec=True, side_effect=_snapshot):
        single_tone_vs_flux_fluxonium(platform=platform, platform_path="unused", parameters=parameters)

    assert x_flux_during["flux_q1_z"] == pytest.approx(0.2)
    assert x_flux_during["flux_c1_2_z"] == pytest.approx(0.2)
    # The x loop itself is the swept bus, so it is not pre-biased.
    assert x_flux_during["flux_q1_x"] == pytest.approx(0.0)


def test_readout_qubit_can_be_overwriten(run_experiment):
    """``coupler_readout_qubit`` redirects a coupler to another qubit's readout."""
    parameters = _base_parameters()
    parameters["coupler_readout_qubit"] = {"c1_2": "q2"}
    parameters["q2"] = {}  # the new readout qubit needs its own entry

    recorder = run_experiment(parameters)

    (coupler_call,) = [c for c in recorder.calls[FN] if c["kwargs"]["qubit_idx"] == "c1_2"]
    assert coupler_call["kwargs"]["readout_bus"] == "readout_q2"
    assert coupler_call["kwargs"]["flux_bus"] == "flux_c1_2_z"


def test_per_target_overwrite_reaches_execution(run_experiment):
    """A per-target override wins for that target; other targets keep the globals."""
    parameters = _base_parameters()
    parameters["q1"] = {
        "if_sweep": [-1.5e6, 1.5e6, 41],
        "flux_sweep": [-0.5, 0.5, 4],
        "readout_amp": 0.05,
        "duration": 3000,
        "averages": 1500,
    }  # override only for q1

    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert calls, f"Expected {FN} to be called"

    qubit_calls = [c for c in calls if c["kwargs"]["qubit_idx"] == "q1"]
    assert len(qubit_calls) == 2
    for call in qubit_calls:
        assert call["kwargs"]["r_amp"] == 0.05
        assert call["kwargs"]["duration"] == 3000
        assert call["kwargs"]["averages"] == 1500
        assert len(call["kwargs"]["if_sweep"]) == 41
        np.testing.assert_allclose(call["kwargs"]["flux_sweep"], np.linspace(-0.5, 0.5, 4))

    (coupler_call,) = [c for c in calls if c["kwargs"]["qubit_idx"] == "c1_2"]
    assert coupler_call["kwargs"]["r_amp"] == 0.075
    assert coupler_call["kwargs"]["duration"] == 2000
    assert coupler_call["kwargs"]["averages"] == 1000
    assert len(coupler_call["kwargs"]["if_sweep"]) == 21


def test_overwrite_does_not_mutate_shared_parameters(run_experiment):
    """Merging per-target overrides must not write back into the shared dict."""
    parameters = _base_parameters()
    parameters["q1"] = {"readout_amp": 0.9}

    run_experiment(parameters)

    assert parameters["readout_amp"] == 0.075
    assert parameters["q1"] == {"readout_amp": 0.9}


def test_fitted_offsets_are_written_to_the_calibration(run_experiment):
    """Each measured flux bus gets its fitted offset, and the calibration is saved."""
    recorder = run_experiment(_base_parameters())

    offsets = run_experiment.calibration.crosstalk_matrix.flux_offsets
    assert {bus: offsets[bus] for bus in ("flux_q1_z", "flux_q1_x", "flux_c1_2_z")} == {
        "flux_q1_z": FITTED_OFFSET,
        "flux_q1_x": FITTED_OFFSET,
        "flux_c1_2_z": FITTED_OFFSET,
    }

    # One fit per measurement, each pointed at the configured data folder.
    fits = recorder.calls["FluxoniumSingleToneFluxModel"]
    assert len(fits) == 3
    assert all(f["args"] == (MEASUREMENT_ID,) for f in fits)
    assert all(f["kwargs"]["path"] == "unused-mocked-folder" for f in fits)

    # The updated calibration and platform are persisted once, at the end.
    assert len(recorder.calls["serialize_to"]) == 1
    assert recorder.calls["serialize_to"][0]["args"] == (run_experiment.calibration, "unused-mocked.yml")
