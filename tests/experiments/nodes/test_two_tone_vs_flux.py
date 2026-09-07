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

"""Tests for ``two_tone_vs_flux``.

The runcard exposes qubits ``[1, 2]`` with ``qubit_loops=2`` and
``coupler_loops=1``, so targets ``["q1", "c1_2"]`` produce three measurements:
``flux_q1_z``, ``flux_q1_x`` and ``flux_c1_2_z``.

The node fits each measurement and adds the fitted offset to the calibration's
crosstalk matrix, but only for a fit that converged. The execution function, the
fit model and the writer (``serialize_to``) are mocked so nothing is measured,
fitted or written to disk.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix
from qililab.typings.enums import Parameter

from seqtante_open.experiments.nodes.two_tone_vs_flux import two_tone_frequency_vs_flux_node
from seqtante_open.experiments.utils.flux_buses import get_all_flux_buses

RUNCARD_PATH = Path(__file__).resolve().parents[2] / "runcards" / "test_AQPU_runcard.yml"

MODULE = "seqtante_open.experiments.nodes.two_tone_vs_flux"
FN = "two_tone_vs_flux_experiment"

MEASUREMENT_ID = 777
FITTED_OFFSET = 0.123
DATA_FOLDER = "unused-mocked-folder"

MEASURED_BUSES = ("flux_q1_x", "flux_q1_z", "flux_c1_2_z")
"""The flux buses of ``["q1", "c1_2"]``, in the order the node sweeps them."""


def _identity_crosstalk(platform) -> CrosstalkMatrix:
    buses = get_all_flux_buses(platform)
    return CrosstalkMatrix.from_buses({b: {bb: (1.0 if b == bb else 0.0) for bb in buses} for b in buses})


def _calibration(platform, lo: dict[str, float] | None = None) -> Calibration:
    """A real ``Calibration`` holding a crosstalk matrix and the per-bus LO table.

    Real rather than a stand-in because the node writes the per-measurement
    ``data_folder`` into ``parameters`` of a copy of it.
    """
    calibration = Calibration()
    calibration.crosstalk_matrix = _identity_crosstalk(platform)
    calibration.parameters = {"LO": lo or {}}
    return calibration


def _fit_model() -> MagicMock:
    """Stand-in for ``FluxoniumTwoToneFluxModel``: fits nothing, reports an offset."""
    return MagicMock(name="FluxoniumTwoToneFluxModel", offset=FITTED_OFFSET, fitted=True)


def _base_parameters() -> dict:
    # The node indexes ``parameters[target]`` directly (not ``.get``), so every
    # target needs an entry, as does the readout qubit of every coupler target.
    return {
        "targets": ["q1", "c1_2"],
        "calibration_path": "unused-mocked.yml",
        "data_folder": DATA_FOLDER,
        "freq_sweep": [-1.5e6, 1.5e6, 21],
        "flux_sweep": [-1, 1, 11],
        "averages": 1000,
        "relax_duration": 200_000,
        "drive_duration": 40,
        "drive_amplitude": 0.5,
        "readout_amplitude": 0.075,
        "readout_duration": 2000,
        "drive_gain": 0.8,
        "ringup_time": 24,
        "overlap_time": 12,
        "x_loop_readout_flux": 0.3,
        "q1": {},
        "c1_2": {},
    }


@pytest.fixture
def run_experiment(platform, mock_db_manager, mock_recorder):
    """Run ``two_tone_frequency_vs_flux_node`` with the execution, fit and IO boundaries mocked."""
    calibration = _calibration(platform)
    model = _fit_model()
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.FluxoniumTwoToneFluxModel", output=model)
    mock_recorder.mock(f"{MODULE}.serialize_to")

    def run(parameters: dict):
        two_tone_frequency_vs_flux_node(platform=platform, platform_path="unused", parameters=parameters)
        return mock_recorder

    run.calibration = calibration
    run.model = model
    return run


def test_basic_parameters(platform, run_experiment):
    """Every global parameter reaches the execution function, sweeps included."""
    parameters = _base_parameters()
    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert calls, f"Expected {FN} to be called"
    assert all(c["kwargs"]["r_amp"] == pytest.approx(0.075) for c in calls)
    assert all(c["kwargs"]["r_duration"] == 2000 for c in calls)
    assert all(c["kwargs"]["d_amp"] == pytest.approx(0.5) for c in calls)
    assert all(c["kwargs"]["d_duration"] == 40 for c in calls)
    assert all(c["kwargs"]["averages"] == 1000 for c in calls)
    assert all(c["kwargs"]["relax_duration"] == 200_000 for c in calls)
    assert all(c["kwargs"]["drive_gain"] == pytest.approx(0.8) for c in calls)
    assert all(c["kwargs"]["ringup_time"] == 24 for c in calls)
    assert all(c["kwargs"]["overlap_time"] == 12 for c in calls)
    assert all(c["kwargs"]["flux_parameter"] == Parameter.FLUX for c in calls)
    assert all(c["kwargs"]["autocalibration"] is True for c in calls)

    # Each measurement gets its own stamped copy; the shared calibration stays untouched.
    for call in calls:
        calibration = call["kwargs"]["calibration"]
        assert calibration is not run_experiment.calibration
        assert calibration.parameters["data_folder"] == DATA_FOLDER + call["kwargs"]["flux_bus"]
    assert "data_folder" not in run_experiment.calibration.parameters

    # The flux sweep is the raw linspace; the frequency sweep is offset by the drive bus IF.
    for call in calls:
        drive_bus = call["kwargs"]["drive_bus"]
        readout_bus = call["kwargs"]["readout_bus"]
        expected_freq = np.linspace(-1.5e6, 1.5e6, 21) + platform.get_parameter(drive_bus, Parameter.IF)
        np.testing.assert_allclose(call["kwargs"]["drive_IF_sweep"], expected_freq)
        np.testing.assert_allclose(call["kwargs"]["flux_sweep"], np.linspace(-1, 1, 11))
        assert call["kwargs"]["readout_if_freq"] == platform.get_parameter(readout_bus, Parameter.IF)
        assert call["kwargs"]["drive_LO"] == platform.get_parameter(drive_bus, Parameter.LO_FREQUENCY)


def test_loops_over_all_loops(run_experiment):
    """Both qubit loops (``qubit_loops=2``) and the single coupler loop are measured."""
    parameters = _base_parameters()
    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert calls, f"Expected {FN} to be called"

    measured = {
        (c["kwargs"]["target"], c["kwargs"]["flux_bus"], c["kwargs"]["drive_bus"], c["kwargs"]["readout_bus"])
        for c in calls
    }
    assert measured == {
        ("q1", "flux_q1_z", "drive_q1", "readout_q1"),
        ("q1", "flux_q1_x", "drive_q1", "readout_q1"),
        # A coupler has no drive line of its own, so it is driven and read through its lowest qubit.
        ("c1_2", "flux_c1_2_z", "drive_q1", "readout_q1"),
    }
    assert len(calls) == 3, "each flux bus should be measured exactly once"


def test_x_loop_flux_is_set_if_specified(platform, mock_db_manager, mock_recorder):
    """While the z loop is measured, the qubit's x loop sits at ``x_loop_readout_flux``.

    Asserted at call time rather than afterwards: the run ends with
    ``set_bias_to_zero``, so the final platform state says nothing about the bias
    the measurement actually saw.
    """
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=_calibration(platform))
    mock_recorder.mock(f"{MODULE}.FluxoniumTwoToneFluxModel", output=_fit_model())
    mock_recorder.mock(f"{MODULE}.serialize_to")

    parameters = _base_parameters()
    parameters["q1"] = {"x_loop_readout_flux": 0.2}

    x_flux_during = {}

    def _snapshot(**kwargs) -> int:
        x_flux_during[kwargs["flux_bus"]] = platform.get_parameter("flux_q1_x", Parameter.FLUX)
        return MEASUREMENT_ID

    with patch(f"{MODULE}.{FN}", autospec=True, side_effect=_snapshot):
        two_tone_frequency_vs_flux_node(platform=platform, platform_path="unused", parameters=parameters)

    assert x_flux_during["flux_q1_z"] == pytest.approx(0.2)
    assert x_flux_during["flux_c1_2_z"] == pytest.approx(0.2)
    # The x loop itself is the swept bus, so it is not pre-biased.
    assert x_flux_during["flux_q1_x"] == pytest.approx(0.0)


def test_readout_qubit_can_be_overwriten(run_experiment):
    """``coupler_readout_qubit`` redirects a coupler to another qubit's drive and readout."""
    parameters = _base_parameters()
    parameters["coupler_readout_qubit"] = {"c1_2": "q2"}
    parameters["q2"] = {}

    recorder = run_experiment(parameters)

    (coupler_call,) = [c for c in recorder.calls[FN] if c["kwargs"]["target"] == "c1_2"]
    assert coupler_call["kwargs"]["readout_bus"] == "readout_q2"
    assert coupler_call["kwargs"]["drive_bus"] == "drive_q2"
    assert coupler_call["kwargs"]["flux_bus"] == "flux_c1_2_z"


def test_per_target_overwrite_reaches_execution(platform, run_experiment):
    """A per-target override wins for that target; other targets keep the globals."""
    parameters = _base_parameters()
    parameters["q1"] = {
        "freq_sweep": [-3e6, 3e6, 41],
        "flux_sweep": [-0.5, 0.5, 4],
        "readout_amplitude": 0.05,
        "readout_duration": 3000,
        "drive_amplitude": 0.9,
        "averages": 1500,
    }

    recorder = run_experiment(parameters)

    calls = recorder.calls[FN]
    assert calls, f"Expected {FN} to be called"

    qubit_calls = [c for c in calls if c["kwargs"]["target"] == "q1"]
    assert len(qubit_calls) == 2
    drive_if = platform.get_parameter("drive_q1", Parameter.IF)
    for call in qubit_calls:
        assert call["kwargs"]["r_amp"] == pytest.approx(0.05)
        assert call["kwargs"]["r_duration"] == 3000
        assert call["kwargs"]["d_amp"] == pytest.approx(0.9)
        assert call["kwargs"]["averages"] == 1500
        np.testing.assert_allclose(call["kwargs"]["drive_IF_sweep"], np.linspace(-3e6, 3e6, 41) + drive_if)
        np.testing.assert_allclose(call["kwargs"]["flux_sweep"], np.linspace(-0.5, 0.5, 4))

    (coupler_call,) = [c for c in calls if c["kwargs"]["target"] == "c1_2"]
    assert coupler_call["kwargs"]["r_amp"] == pytest.approx(0.075)
    assert coupler_call["kwargs"]["r_duration"] == 2000
    assert coupler_call["kwargs"]["d_amp"] == pytest.approx(0.5)
    assert coupler_call["kwargs"]["averages"] == 1000
    assert len(coupler_call["kwargs"]["drive_IF_sweep"]) == 21


def test_overwrite_does_not_mutate_shared_parameters(run_experiment):
    """Merging per-target overrides must not write back into the shared dict."""
    parameters = _base_parameters()
    parameters["q1"] = {"readout_amplitude": 0.9}

    run_experiment(parameters)

    assert parameters["readout_amplitude"] == pytest.approx(0.075)
    assert parameters["q1"] == {"readout_amplitude": 0.9}


def test_defaults_are_used_when_parameters_are_missing(run_experiment):
    """The optional drive and timing parameters fall back to the module defaults."""
    parameters = _base_parameters()
    for key in ("drive_gain", "ringup_time", "overlap_time"):
        del parameters[key]

    recorder = run_experiment(parameters)

    for call in recorder.calls[FN]:
        assert call["kwargs"]["drive_gain"] == 1
        assert call["kwargs"]["ringup_time"] == 0
        assert call["kwargs"]["overlap_time"] == 0


def test_lo_from_the_calibration_is_per_target(platform, mock_db_manager, mock_recorder):
    """With no ``LO`` parameter, the calibration's per-bus table is consulted next."""
    calibration = _calibration(platform, lo={"c1_2": 4.7e9})
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.FluxoniumTwoToneFluxModel", output=_fit_model())
    mock_recorder.mock(f"{MODULE}.serialize_to")

    parameters = _base_parameters()
    parameters["coupler_readout_qubit"] = {"c1_2": "q1"}
    parameters["q1"] = {}

    two_tone_frequency_vs_flux_node(platform=platform, platform_path="unused", parameters=parameters)

    los = {c["kwargs"]["target"]: c["kwargs"]["drive_LO"] for c in mock_recorder.calls[FN]}
    assert los == {
        "q1": platform.get_parameter("drive_q1", Parameter.LO_FREQUENCY),
        "c1_2": 4.7e9,
    }


def test_fitted_offsets_are_written_to_the_calibration(run_experiment):
    """Each measured flux bus gets its fitted offset, and the calibration is saved."""
    recorder = run_experiment(_base_parameters())

    offsets = run_experiment.calibration.crosstalk_matrix.flux_offsets
    assert {bus: offsets[bus] for bus in MEASURED_BUSES} == dict.fromkeys(MEASURED_BUSES, FITTED_OFFSET)

    # One fit per measurement, each pointed at the configured data folder and its own flux bus.
    fits = recorder.calls["FluxoniumTwoToneFluxModel"]
    assert len(fits) == 3
    assert all(f["args"] == (MEASUREMENT_ID,) for f in fits)
    assert [f["kwargs"]["path"] for f in fits] == [DATA_FOLDER + bus for bus in MEASURED_BUSES]
    assert [f["kwargs"]["target"] for f in fits] == ["q1", "q1", "c1_2"]
    assert [f["kwargs"]["flux_bus"] for f in fits] == list(MEASURED_BUSES)
    assert run_experiment.model.fit.call_count == 3
    assert run_experiment.model.plot.call_count == 3

    # The updated calibration is persisted once, at the end.
    assert len(recorder.calls["serialize_to"]) == 1
    assert recorder.calls["serialize_to"][0]["args"] == (run_experiment.calibration, "unused-mocked.yml")


def test_offset_is_not_written_when_the_fit_did_not_converge(run_experiment):
    """``fitted`` false means the offset is meaningless, so the calibration keeps its own."""
    run_experiment.model.fitted = False

    run_experiment(_base_parameters())

    offsets = run_experiment.calibration.crosstalk_matrix.flux_offsets
    assert {bus: offsets[bus] for bus in MEASURED_BUSES} == dict.fromkeys(MEASURED_BUSES, 0)


def test_calibration_is_saved_even_when_the_experiment_fails(platform, mock_db_manager, mock_recorder):
    """``serialize_to`` lives in a ``finally``, so a failed run still persists."""
    calibration = _calibration(platform)
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=calibration)
    mock_recorder.mock(f"{MODULE}.FluxoniumTwoToneFluxModel", output=_fit_model())
    mock_recorder.mock(f"{MODULE}.serialize_to")

    with patch(f"{MODULE}.{FN}", autospec=True, side_effect=RuntimeError("instrument exploded")):
        with pytest.raises(RuntimeError, match="instrument exploded"):
            two_tone_frequency_vs_flux_node(platform=platform, platform_path="unused", parameters=_base_parameters())

    assert len(mock_recorder.calls["serialize_to"]) == 1


def test_calibration_without_crosstalk_matrix_is_rejected(platform, mock_db_manager, mock_recorder):
    """A calibration lacking a ``CrosstalkMatrix`` fails before anything is executed."""
    mock_recorder.mock(f"{MODULE}.deserialize_from", output=Calibration())
    mock_recorder.mock(f"{MODULE}.{FN}", output=MEASUREMENT_ID)
    mock_recorder.mock(f"{MODULE}.serialize_to")

    with pytest.raises(ValueError, match="CrosstalkMatrix"):
        two_tone_frequency_vs_flux_node(platform=platform, platform_path="unused", parameters=_base_parameters())

    assert not mock_recorder.calls[FN]
    assert not mock_recorder.calls["serialize_to"]
