"""Call-assertion test for the shared two-tone pulsed-saturation measurement (QHC-1807, QHC-1805).

Runs :meth:`TwoToneSaturationNode._measure_two_tone_saturation` through a minimal leaf node inside the real
restore scope, with the qili QProgram builder and the stream stubbed and a fake platform, so what the node
hands the builder and what it sets on the hardware are asserted without hardware, a database or a disk.
"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from qililab import Parameter
from qililab.qprogram.calibration import Calibration

from seqtante_open.calibration.io import write_calibration_file
from seqtante_open.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    ExperimentSection,
    Flux,
    OperatingPoint,
    ReadoutFreqs,
)
from seqtante_open.controllers import calibration_node as node_module
from seqtante_open.controllers import node_registry
from seqtante_open.controllers.calibration_node import AnalysisResult, RunContext
from seqtante_open.experiments.nodes import two_tone_saturation as spine_module
from seqtante_open.experiments.nodes.two_tone_saturation import TwoToneSaturationNode
from seqtante_open.outputs import output_controller
from seqtante_open.targets import ExperimentTarget
from tests.experiments.nodes.harness import ExecutingPlatform, FakeStreamArray, identity_crosstalk

Q1 = ExperimentTarget.from_str("q1")
EXPERIMENT = "example_two_tone_saturation"
DRIVE_LO = 2.9e9
SWEEP = np.linspace(-0.11e9, -0.09e9, 51)


@pytest.fixture
def example_node_cls(monkeypatch):
    """A minimal leaf on the spine (registered in a scratch registry) that measures once and writes nothing."""
    monkeypatch.setattr(node_registry, "NODE_REGISTRY", dict(node_registry.NODE_REGISTRY))

    class ExampleNode(TwoToneSaturationNode[None]):
        experiment: ClassVar[str] = EXPERIMENT
        analyses: ClassVar[tuple[str, ...]] = ("none",)

        def execute_node(self, targets):
            return {target: [self._measure_two_tone_saturation(target, SWEEP)] for target in targets}

        def analyze(self, target, measurements):
            return AnalysisResult(fitted=None)

        def apply(self, target, result):
            return None

    return ExampleNode


class FakeSaturationPlatform(ExecutingPlatform):
    """An :class:`ExecutingPlatform` with a Rohde & Schwarz drive whose one execution returns the whole sweep."""

    def __init__(self) -> None:
        super().__init__(instruments={"drive_q1": "ROHDE_SCHWARZ"})

    def results(self) -> dict:
        return {"readout_q1": [SimpleNamespace(array=np.zeros((2, len(SWEEP))))]}


def _calibration() -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = identity_crosstalk()
    file = CalibrationFile(
        operating_points_version=1,
        operating_points={
            "q1": {
                "sweet_spot": OperatingPoint(
                    flux=Flux(z=0.6),
                    drive=DriveFreqs(lo=DRIVE_LO, if_=-0.08e9),
                    readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=50.0),
                )
            }
        },
        experiments={EXPERIMENT: {"q1": ExperimentSection(operating_point="sweet_spot")}},
    )
    write_calibration_file(calibration, file)
    return calibration


def test_measure_builds_the_qili_sweep_and_sets_the_hardware(monkeypatch, example_node_cls):
    built: dict = {}

    def fake_two_tone_saturation(**kwargs):
        built.update(kwargs)
        return SimpleNamespace(name="qprogram")  # sentinel; the fake platform/stream ignore its content

    monkeypatch.setattr(spine_module, "two_tone_saturation", fake_two_tone_saturation)
    monkeypatch.setattr(spine_module, "StreamArray", FakeStreamArray)
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(output_controller, "_initialized", True)
    monkeypatch.setattr(output_controller, "_db_manager", object())
    calibration = _calibration()
    platform = FakeSaturationPlatform()
    context = RunContext(
        platform=platform, platform_path="runcard.yml", calibration=calibration, calibration_path="calibration.yml"
    )
    node = example_node_cls(name="step", idx=0, targets=[Q1], parameters={Q1: {"drive_power": -30}}, context=context)

    failed = node.run({})

    assert failed == []
    # The sweep is built at the target's operating point; the readout pulse and IF come from the named point.
    assert built["target"] == "q1"
    assert built["operating_point"] == "sweet_spot"
    assert built["calibration"] is calibration
    np.testing.assert_array_equal(built["drive_if_sweep"], SWEEP)
    # Unset tree parameters fall back to the defaults, mapped onto qili's argument names.
    assert built["drive_ringup_time"] == 200
    assert built["drive_ringdown_time"] == 200
    assert built["drive_n_sigmas"] == 4
    assert built["readout_ringup_time"] == 600
    # The node set the operating point's hardware itself (snapshotted before the run restored it):
    # drive/readout LOs, readout attenuation, drive power, and the R&S IQ modulation and RF output.
    snap = platform.values_at_execute
    assert snap["drive_q1", Parameter.LO_FREQUENCY] == int(DRIVE_LO)
    assert snap["readout_q1", Parameter.LO_FREQUENCY] == int(7e9)
    assert snap["readout_q1", Parameter.OUT0_ATT] == 50
    assert snap["drive_q1", Parameter.POWER] == -30
    assert snap["drive_q1", Parameter.IQ_MODULATION] is True
    assert snap["drive_q1", Parameter.RF_ON] is True
    # The run restored what it set: the RF output is back off after the node.
    assert not platform.values["drive_q1", Parameter.RF_ON]


def test_a_single_point_sweep_is_rejected(example_node_cls):
    node = example_node_cls.__new__(example_node_cls)
    with pytest.raises(ValueError, match="at least 2 points"):
        node._measure_two_tone_saturation(Q1, np.array([1e8]))


def test_a_node_without_an_experiment_cannot_measure():
    class NoExperiment(TwoToneSaturationNode[None]):  # a spine-like subclass: sets no experiment, not registered
        def execute_node(self, targets):
            return {}

        def analyze(self, target, measurements):
            return AnalysisResult(fitted=None)

        def apply(self, target, result):
            return None

    node = NoExperiment.__new__(NoExperiment)
    with pytest.raises(TypeError, match="sets no experiment"):
        node._measure_two_tone_saturation(Q1, SWEEP)
