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

"""Tests for :class:`~seqtante_open.experiments.nodes.TwoToneVsFluxNode` (QHC-1804).

The real three-phase ``run`` drives the node; the QProgram builder, the stream array, the fit and the
QDAC timing helpers are stubbed, so nothing is measured, fitted or written to disk. The tests assert what
the node hands ``two_tone_vs_flux_saturation``, what it sets on the platform, and what ``apply`` writes
into the operating point (ADR-0011: the measured bias, never the crosstalk offset).
"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from qililab import Parameter
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.calibration.io import read_calibration_file, write_calibration_file
from seqtante_open.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    ExperimentSection,
    Flux,
    OperatingPoint,
    ReadoutFreqs,
)
from seqtante_open.experiments.nodes import TwoToneVsFluxNode
from seqtante_open.experiments.nodes import two_tone_vs_flux as node_module
from seqtante_open.targets import ExperimentTarget
from tests.experiments.nodes.harness import ExecutingPlatform, FakeStreamArray, build_context, stub_output_and_save

Q1 = ExperimentTarget.from_str("q1")
C1_2 = ExperimentTarget.from_str("c1_2")
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1", "flux_z_c1_2")
DRIVE_BUS = "drive_q1"
READOUT_BUS = "readout_q1"

FLUX_Z = 0.6
FLUX_X = 0.27
PARKED_COUPLER = 0.02
DRIVE_LO = 2.9e9
DRIVE_IF = -0.1e9
READOUT_LO = 7e9
READOUT_IF = 1e8
READOUT_ATT = 50.0
DRIVE_POWER = -20
TIME_OF_FLIGHT = 200
QDAC_TRIGGER = 3
SETTLE_FROM_FILTERS = 5_123.4
PRE_STEP_DEFAULT = 1_400
SWEET_SPOT = 0.6031
MEAS_ID = 101

#: The windows the step hands the node, as np.linspace args of offsets from the stored values.
FLUX_WINDOW = (-0.02, 0.02, 5)
IF_WINDOW = (-10e6, 10e6, 3)
N_FLUX = FLUX_WINDOW[2]
N_IF = IF_WINDOW[2]
PARAMS = {
    "loop": "z",
    "flux_sweep": FLUX_WINDOW,
    "drive_if_sweep": IF_WINDOW,
    "averages": 100,
    "relax_duration": 20_000,
    "drive_duration": 2_000,
    "drive_amplitude": 0.1,
    "drive_power": DRIVE_POWER,
}


class FakeFluxFit:
    """Scripted stand-in for ``FluxoniumTwoToneFluxModel``: a parabola with its vertex at ``SWEET_SPOT``."""

    fitted = True
    r2 = 0.97
    events: ClassVar[list[str]] = []
    saved_loops: ClassVar[list[str]] = []
    init: ClassVar[dict] = {}

    def __init__(self, measurement_id, target=None, path=None, lo=None, flux_bus=None):
        type(self).init = {
            "measurement_id": measurement_id,
            "target": target,
            "path": path,
            "lo": lo,
            "flux_bus": flux_bus,
        }
        self.center = SWEET_SPOT

    def fit(self):
        type(self).events.append("fit")

    @property
    def r_squared(self):
        return type(self).r2

    def save_fitting(self, loop):
        type(self).saved_loops.append(loop)
        type(self).events.append("save")

    def plot(self):
        type(self).events.append("plot")
        return "plot.html"


class FakeFluxPlatform(ExecutingPlatform):
    """An :class:`ExecutingPlatform` whose execution ``n`` (from 1) returns an (IQ, flux, IF) map filled with ``n``.

    Filling each map with its execution number makes the running average over repetitions checkable.
    """

    def __init__(self) -> None:
        super().__init__({(READOUT_BUS, Parameter.TIME_OF_FLIGHT): TIME_OF_FLIGHT})

    def results(self) -> dict:
        return {READOUT_BUS: [SimpleNamespace(array=np.full((2, N_FLUX, N_IF), float(self.executions)))]}


def _calibration(flux: Flux | None = None) -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = CrosstalkMatrix.from_buses(
        {b: {bb: (1.0 if b == bb else 0.0) for bb in CROSSTALK_BUSES} for b in CROSSTALK_BUSES}
    )
    write_calibration_file(
        calibration,
        CalibrationFile(
            operating_points_version=1,
            operating_points={
                "q1": {
                    "sweet_spot": OperatingPoint(
                        flux=flux if flux is not None else Flux(z=FLUX_Z, x=FLUX_X),
                        parked_flux={"c1_2": Flux(z=PARKED_COUPLER)},
                        drive=DriveFreqs(lo=DRIVE_LO, if_=DRIVE_IF),
                        readout=ReadoutFreqs(lo=READOUT_LO, if_=READOUT_IF, attenuation=READOUT_ATT),
                    )
                }
            },
            experiments={"two_tone_vs_flux": {"q1": ExperimentSection(operating_point="sweet_spot")}},
        ),
    )
    return calibration


def _make(calibration, platform=None, targets=(Q1,), **params) -> TwoToneVsFluxNode:
    return TwoToneVsFluxNode(
        name="sweet_spot_flux",
        idx=0,
        targets=list(targets),
        parameters={target: {**PARAMS, **params} for target in targets},
        context=build_context(calibration, platform if platform is not None else FakeFluxPlatform()),
    )


def _stored_point(calibration: Calibration) -> OperatingPoint:
    return read_calibration_file(calibration).operating_points["q1"]["sweet_spot"]


@pytest.fixture
def calls(monkeypatch):
    """No hardware, no disk: stub the builder, the stream, the fit, the QDAC helpers and the output/save."""
    recorded: dict = {"build": None, "timings": None}

    def fake_build(**kwargs):
        recorded["build"] = kwargs
        return SimpleNamespace(name="qprogram")

    def fake_timings(platform, settle_override, pre_step_override):
        recorded["timings"] = (settle_override, pre_step_override)
        if settle_override is None:
            return SETTLE_FROM_FILTERS, PRE_STEP_DEFAULT
        return settle_override, pre_step_override

    monkeypatch.setattr(node_module, "two_tone_vs_flux_saturation", fake_build)
    monkeypatch.setattr(node_module, "StreamArray", FakeStreamArray)
    monkeypatch.setattr(node_module, "FluxoniumTwoToneFluxModel", FakeFluxFit)
    monkeypatch.setattr(node_module, "qdac_step_timings", fake_timings)
    monkeypatch.setattr(node_module, "get_qdac_out_trigger", lambda platform: QDAC_TRIGGER)
    stub_output_and_save(monkeypatch)
    monkeypatch.setattr(FakeFluxFit, "fitted", True)
    monkeypatch.setattr(FakeFluxFit, "r2", 0.97)
    monkeypatch.setattr(FakeFluxFit, "events", [])
    monkeypatch.setattr(FakeFluxFit, "saved_loops", [])
    monkeypatch.setattr(FakeFluxFit, "init", {})
    monkeypatch.setattr(FakeStreamArray, "measurement_id", MEAS_ID)
    monkeypatch.setattr(FakeStreamArray, "last", None)
    return recorded


def test_sweeps_the_z_loop_around_the_point_and_stores_the_measured_bias(calls):
    calibration = _calibration()
    offsets_before = dict(calibration.crosstalk_matrix.flux_offsets)
    platform = FakeFluxPlatform()
    node = _make(calibration, platform=platform, data_folder="data/sweet_spot_flux/q1")

    failed = node.run({})

    assert failed == []
    build = calls["build"]
    assert build["calibration"] is calibration
    assert build["target"] == "q1"
    assert build["operating_point"] == "sweet_spot"
    assert build["flux_axis"] == "z"
    # Both windows are offsets from the stored values; the builder gets the absolute ones.
    np.testing.assert_allclose(build["flux_sweep"], FLUX_Z + np.linspace(*FLUX_WINDOW))
    np.testing.assert_allclose(build["drive_if_sweep"], DRIVE_IF + np.linspace(*IF_WINDOW))
    # The QDAC timings come from the filters (rounded up to whole ns), the time of flight and the trigger
    # output from the platform, and the unset drive/readout shape from the node defaults.
    assert build["flux_settle_duration"] == 5_124
    assert build["flux_pre_step_duration"] == PRE_STEP_DEFAULT
    assert build["readout_time_of_flight"] == TIME_OF_FLIGHT
    assert build["qdac_trigger_channel"] == QDAC_TRIGGER
    assert (build["drive_ringup_time"], build["drive_ringdown_time"], build["drive_n_sigmas"]) == (200, 200, 4)
    assert (build["overlap_time"], build["readout_ringup_time"]) == (0, 0)
    # The node placed the point itself: the unswept x loop and the parked coupler sit at their stored
    # biases, and the line is at the point's LOs/attenuation with the drive armed.
    snap = platform.values_at_execute
    assert snap["flux_x_q1", Parameter.FLUX] == FLUX_X
    assert snap["flux_z_c1_2", Parameter.FLUX] == PARKED_COUPLER
    assert snap[DRIVE_BUS, Parameter.LO_FREQUENCY] == int(DRIVE_LO)
    assert snap[READOUT_BUS, Parameter.LO_FREQUENCY] == int(READOUT_LO)
    assert snap[READOUT_BUS, Parameter.OUT0_ATT] == int(READOUT_ATT)
    assert snap[DRIVE_BUS, Parameter.POWER] == DRIVE_POWER
    # The stored map carries the absolute axes the fit reads back, keyed apart by the swept bus.
    stream = FakeStreamArray.last
    assert stream.kwargs["shape"] == (N_FLUX, N_IF, 2)
    assert stream.kwargs["optional_identifier"] == "flux_z_q1"
    np.testing.assert_allclose(stream.kwargs["loops"]["flux"]["array"], build["flux_sweep"])
    np.testing.assert_allclose(stream.kwargs["loops"]["IF_frequency"]["array"], build["drive_if_sweep"])
    # The fit read that measurement into the step's folder, and ran fit -> save -> plot.
    assert FakeFluxFit.init == {
        "measurement_id": MEAS_ID,
        "target": "q1",
        "path": "data/sweet_spot_flux/q1",
        "lo": DRIVE_LO,
        "flux_bus": "flux_z_q1",
    }
    assert FakeFluxFit.events == ["fit", "save", "plot"]
    assert FakeFluxFit.saved_loops == ["z"]
    # apply stored the vertex as the measured z bias; x is untouched, and so is the crosstalk offset.
    point = _stored_point(calibration)
    assert point.flux.z == pytest.approx(SWEET_SPOT)
    assert point.flux.x == FLUX_X
    assert dict(calibration.crosstalk_matrix.flux_offsets) == offsets_before


def test_the_x_loop_sweeps_and_stores_the_x_bias(calls):
    calibration = _calibration()

    failed = _make(calibration, loop="x").run({})

    assert failed == []
    assert calls["build"]["flux_axis"] == "x"
    np.testing.assert_allclose(calls["build"]["flux_sweep"], FLUX_X + np.linspace(*FLUX_WINDOW))
    assert FakeStreamArray.last.kwargs["optional_identifier"] == "flux_x_q1"
    assert FakeFluxFit.saved_loops == ["x"]
    point = _stored_point(calibration)
    assert point.flux.x == pytest.approx(SWEET_SPOT)
    assert point.flux.z == FLUX_Z


def test_repetitions_stream_the_running_average(calls):
    platform = FakeFluxPlatform()

    _make(_calibration(), platform=platform, repetitions=3).run({})

    assert platform.executions == 3
    # Executions return maps of 1, 2, 3: the stream holds the mean after each one.
    # Executions return maps of 1, 2, 3: the stream holds the mean after each one.
    writes = [np.asarray(value) for _, value in FakeStreamArray.last.writes]
    assert [float(write.mean()) for write in writes] == pytest.approx([1.0, 1.5, 2.0])
    assert writes[-1].shape == (N_FLUX, N_IF, 2)


def test_qdac_timing_overrides_are_passed_through(calls):
    _make(_calibration(), flux_settle_duration=8_000, flux_pre_step_duration=2_000).run({})

    assert calls["timings"] == (8_000, 2_000)
    assert calls["build"]["flux_settle_duration"] == 8_000
    assert calls["build"]["flux_pre_step_duration"] == 2_000


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        pytest.param("fitted", False, id="no_vertex_in_the_window"),
        pytest.param("r2", 0.3, id="r2_below_minimum"),
        pytest.param("r2", float("nan"), id="r2_undefined"),
    ],
)
def test_a_rejected_fit_fails_the_target_and_writes_nothing(calls, monkeypatch, knob, value):
    monkeypatch.setattr(FakeFluxFit, knob, value)
    calibration = _calibration()

    failed = _make(calibration).run({})

    assert failed == ["q1"]
    assert FakeFluxFit.saved_loops == []  # a rejected fit never reaches the DB
    assert _stored_point(calibration).flux.z == FLUX_Z


@pytest.mark.parametrize(
    ("flux", "params", "targets"),
    [
        pytest.param(Flux(z=FLUX_Z), {"loop": "x"}, (Q1,), id="x_loop_with_no_stored_x_bias"),
        pytest.param(None, {"loop": "y"}, (Q1,), id="unknown_loop"),
        pytest.param(None, {}, (C1_2,), id="coupler_target"),
    ],
)
def test_an_unmeasurable_step_fails_the_target_before_measuring(calls, flux, params, targets):
    calibration = _calibration(flux)
    platform = FakeFluxPlatform()

    failed = _make(calibration, platform=platform, targets=targets, **params).run({})

    assert failed == [str(targets[0])]
    assert calls["build"] is None
    assert platform.executions == 0
