"""Call-assertion test for the ``landau_zener`` benchmark node (QHC-1808).

Drives the full three-phase run with the compartmentalized Landau-Zener driver and the normalising fit
scripted (a fake ``LandauZenerFit``, a stubbed driver, no hardware, no database, no disk), and asserts
the node measures with the idle DC flux and readout frequencies from the operating point, the annealing
geometry and AC crosstalk from the tree, normalises with the reset-point readout levels, and -- with no
thresholds set yet -- leaves the benchmark unscored ("Not evaluated"). A second test checks the ADR-0001
D8 refusal when the stored levels were taken with a different readout pulse.
"""

import numpy as np
from qililab import IQPair, Square
from qililab.qprogram.calibration import Calibration
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.calibration.bus_alias import Role
from seqtante_open.calibration.io import write_calibration_file
from seqtante_open.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    ExperimentSection,
    Flux,
    Measured,
    OperatingPoint,
    ReadoutFreqs,
    ReadoutLevels,
)
from seqtante_open.controllers.calibration_node import RunContext
from seqtante_open.experiments.nodes import LandauZenerNode
from seqtante_open.experiments.nodes import landau_zener as lz_module
from seqtante_open.outputs import output_controller
from seqtante_open.targets import ExperimentTarget

Q1 = ExperimentTarget.from_str("q1")
FLUX_BUSES = ["flux_z_q1", "flux_x_q1"]
READOUT_LO = 7e9
READOUT_IF = 1e8
IDLE_Z = 0.1
IDLE_X = 0.2
LEVEL_0 = 0.0
LEVEL_1 = 1.0
MEAS_ID = 100
READOUT_PULSE = "readout-reset"

PARAMS = {
    "flux_readout": [0.0, 0.0],
    "flux_reset": [0.3, 0.3],
    "reset_duration": 200,
    "annealing_paths": [[[0.1, 0.1, 100], [0.2, 0.0, 100]]],
    "annealing_times": [200],
    "readout_duration": 1000,
    "averages": 1000,
    "ramp_reset_duration": [50, 50],
    "ramp_readout_duration": [50, 50],
    "readout_amplitude": 0.4,
    "ringup_time": 0,
    "ac_crosstalk": [[1.0, 0.0], [0.0, 1.0]],
}


class FakePlatform:
    def __init__(self):
        self.values: dict = {}
        self.calls: list = []

    def set_crosstalk(self, *, crosstalk):
        self.calls.append("set_crosstalk")

    def set_flux_to_zero(self):
        self.calls.append("set_flux_to_zero")

    def set_bias_to_zero(self):
        self.calls.append("set_bias_to_zero")

    def get_parameter(self, *, alias, parameter):
        return self.values.get((alias, parameter), 0.0)

    def set_parameter(self, *, alias, parameter, value):
        self.values[alias, parameter] = value
        self.calls.append(f"set:{alias}")


class FakeLandauZenerFit:
    """Scripted stand-in for ``LandauZenerFit``: no DB load, records the readout levels it was given."""

    last: dict = {}

    def __init__(self, measurement_id=None, target=None, level_0=None, level_1=None, path=None):
        type(self).last = {"measurement_id": measurement_id, "target": target, "level_0": level_0, "level_1": level_1}

    def plot(self):
        return None


def _calibration(readout_levels_pulse: str = READOUT_PULSE) -> Calibration:
    calibration = Calibration()
    # The run needs a DC crosstalk matrix to set flux (separate from the AC matrix the node builds).
    calibration.crosstalk_matrix = CrosstalkMatrix.from_array(buses=FLUX_BUSES, matrix_array=np.eye(2))
    # The section declares the readout pulse, which must exist on its bus (ADR-0008 existence check).
    calibration.add_waveform(
        "readout_q1",
        READOUT_PULSE,
        IQPair(I=Square(amplitude=0.1, duration=1000), Q=Square(amplitude=0.0, duration=1000)),
    )
    write_calibration_file(
        calibration,
        CalibrationFile(
            operating_points_version=1,
            operating_points={
                "q1": {
                    "reset": OperatingPoint(
                        flux=Flux(z=IDLE_Z, x=IDLE_X),
                        drive=DriveFreqs(lo=2.9e9, if_=-0.1e9),
                        readout=ReadoutFreqs(
                            lo=READOUT_LO,
                            if_=READOUT_IF,
                            attenuation=50.0,
                            levels=ReadoutLevels(
                                pulse=readout_levels_pulse,
                                level_0=Measured(nominal=LEVEL_0, std=0.0),
                                level_1=Measured(nominal=LEVEL_1, std=0.0),
                            ),
                        ),
                    )
                }
            },
            experiments={
                "landau_zener": {"q1": ExperimentSection(operating_point="reset", pulses={Role.readout: READOUT_PULSE})}
            },
        ),
    )
    return calibration


def _make(calibration: Calibration) -> LandauZenerNode:
    context = RunContext(
        platform=FakePlatform(),
        platform_path="runcard.yml",
        calibration=calibration,
        calibration_path="calibration.yml",
    )
    # No thresholds: the run leaves the benchmark unscored ("Not evaluated") until QHC-1921.
    return LandauZenerNode(name="lz", idx=0, targets=[Q1], parameters={Q1: dict(PARAMS)}, context=context)


def _patch(monkeypatch) -> dict:
    """No hardware, no disk: stub the Landau-Zener driver, the fit and the output controller."""
    calls: dict = {"n": 0}

    def fake_driver(**kwargs):
        calls.update(kwargs)
        calls["n"] += 1
        return MEAS_ID, None

    monkeypatch.setattr(lz_module, "lz_transition_not_idling_reset__custom_linear_ramp", fake_driver)
    monkeypatch.setattr(lz_module, "LandauZenerFit", FakeLandauZenerFit)
    monkeypatch.setattr(output_controller, "_initialized", True)
    monkeypatch.setattr(output_controller, "_db_manager", object())
    return calls


def test_runs_the_sweep_and_leaves_the_benchmark_unscored(monkeypatch):
    calls = _patch(monkeypatch)
    node = _make(_calibration())

    failed = node.run({})

    assert failed == []
    assert calls["n"] == 1  # one Landau-Zener sweep
    # Idle DC flux and readout frequencies come from the operating point...
    assert calls["idle_point_dc"] == [IDLE_Z, IDLE_X]
    assert calls["readout_if_freq"] == int(READOUT_IF)
    assert calls["readout_LO"] == int(READOUT_LO)
    assert calls["flux_buses"] == FLUX_BUSES
    # ...the annealing geometry and AC crosstalk come from the tree.
    assert calls["flux_reset"] == PARAMS["flux_reset"]
    assert calls["flux_readout"] == PARAMS["flux_readout"]
    assert calls["annealing_paths_list"] == PARAMS["annealing_paths"]
    assert calls["landau_zener_annealing_time"] == PARAMS["annealing_times"]
    assert calls["ramp_reset_duration"] == PARAMS["ramp_reset_duration"]
    assert isinstance(calls["crosstalk"], CrosstalkMatrix)
    assert set(calls["crosstalk"].matrix) == set(FLUX_BUSES)
    # analyze normalised with the reset-point readout levels...
    assert FakeLandauZenerFit.last["level_0"] == LEVEL_0
    assert FakeLandauZenerFit.last["level_1"] == LEVEL_1
    assert FakeLandauZenerFit.last["measurement_id"] == MEAS_ID
    # ...and reported no value, so the run leaves the benchmark unscored.
    assert node.benchmark_results == ()


def test_refuses_readout_levels_taken_with_a_different_pulse(monkeypatch):
    _patch(monkeypatch)
    # Levels stored under a different pulse than the section plays -> ADR-0001 D8 refusal in analyze.
    node = _make(_calibration(readout_levels_pulse="readout-other"))

    failed = node.run({})

    assert failed == ["q1"]  # the target fails the step; no benchmark is recorded
    assert node.benchmark_results == ()
