"""Tests for the hardware setup every fixed-flux node shares (``FixedFluxNode._set_operating_point_hardware``).

A minimal leaf node (registered in a scratch registry) sets its operating point's drive/readout line
inside a real restore scope on a fake platform, so each drive-instrument branch, the required-argument
errors and the attenuation check are asserted without hardware.
"""

from types import SimpleNamespace
from typing import ClassVar

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
from seqtante_open.controllers import node_registry
from seqtante_open.controllers.calibration_node import (
    NO_DRIVE_INSTRUMENT_MSG,
    AnalysisResult,
    FixedFluxNode,
    RunContext,
)
from seqtante_open.controllers.parameter_restore import restoring_parameters
from seqtante_open.targets import ExperimentTarget

Q1 = ExperimentTarget.from_str("q1")
EXPERIMENT = "example_fixed_flux"


class InstrumentPlatform:
    """A platform whose drive bus reports ``instrument`` and that records parameter writes."""

    def __init__(self, instrument: str):
        self.instrument = instrument
        self.values: dict = {}

    def get_parameter(self, *, alias, parameter):
        return self.values.get((alias, parameter), 0.0)

    def set_parameter(self, *, alias, parameter, value):
        self.values[alias, parameter] = value

    def get_element(self, alias):
        return SimpleNamespace(instruments=[SimpleNamespace(name=SimpleNamespace(name=self.instrument))])


class _Leaf:
    """The three phases a leaf must define; the tests only call the shared hardware helper."""

    def execute_node(self, targets):
        return {}

    def analyze(self, target, measurements):
        return AnalysisResult(fitted=None)

    def apply(self, target, result):
        return None


@pytest.fixture
def leaf_cls(monkeypatch):
    monkeypatch.setattr(node_registry, "NODE_REGISTRY", dict(node_registry.NODE_REGISTRY))

    class ExampleNode(_Leaf, FixedFluxNode[None]):
        experiment: ClassVar[str] = EXPERIMENT
        analyses: ClassVar[tuple[str, ...]] = ("none",)

    return ExampleNode


def _calibration(attenuation: float = 30.0) -> Calibration:
    calibration = Calibration()
    write_calibration_file(
        calibration,
        CalibrationFile(
            operating_points_version=1,
            operating_points={
                "q1": {
                    "sweet_spot": OperatingPoint(
                        flux=Flux(z=0.6),
                        drive=DriveFreqs(lo=2.9e9, if_=-0.1e9),
                        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=attenuation),
                    )
                }
            },
            experiments={EXPERIMENT: {"q1": ExperimentSection(operating_point="sweet_spot")}},
        ),
    )
    return calibration


def _set(node_cls, platform, calibration=None, **kwargs) -> None:
    context = RunContext(
        platform=platform,
        platform_path="runcard.yml",
        calibration=calibration or _calibration(),
        calibration_path="calibration.yml",
    )
    node = node_cls(name="step", idx=0, targets=[Q1], parameters={Q1: {}}, context=context)
    with restoring_parameters(platform) as restore:
        node._restore = restore
        node._set_operating_point_hardware(Q1, **kwargs)
        platform.values_in_scope = dict(platform.values)


def test_rohde_schwarz_sets_power_iq_modulation_and_rf(leaf_cls):
    platform = InstrumentPlatform("ROHDE_SCHWARZ")

    _set(leaf_cls, platform, drive_power=-20)

    snap = platform.values_in_scope
    assert snap["drive_q1", Parameter.LO_FREQUENCY] == 2_900_000_000
    assert snap["readout_q1", Parameter.LO_FREQUENCY] == 7_000_000_000
    assert snap["readout_q1", Parameter.OUT0_ATT] == 30
    assert snap["drive_q1", Parameter.POWER] == -20
    assert snap["drive_q1", Parameter.IQ_MODULATION] is True
    assert snap["drive_q1", Parameter.RF_ON] is True
    assert not platform.values["drive_q1", Parameter.RF_ON]  # the scope restored it on exit


def test_qcm_rf_sets_gain_and_hardware_modulation(leaf_cls):
    platform = InstrumentPlatform("QCMRF")

    _set(leaf_cls, platform, drive_power=-20, drive_gain=0.4)

    snap = platform.values_in_scope
    assert snap["drive_q1", Parameter.GAIN] == pytest.approx(0.4)  # the gain, not the R&S power
    assert snap["drive_q1", Parameter.HARDWARE_MODULATION] is True
    assert ("drive_q1", Parameter.POWER) not in snap


@pytest.mark.parametrize(
    ("instrument", "kwargs", "message"),
    [
        ("ROHDE_SCHWARZ", {"drive_gain": 0.4}, "needs drive_power"),
        ("QCMRF", {"drive_power": -20}, "needs drive_gain"),
    ],
)
def test_the_drive_instrument_needs_its_own_key(leaf_cls, instrument, kwargs, message):
    platform = InstrumentPlatform(instrument)
    with pytest.raises(ValueError, match=message):
        _set(leaf_cls, platform, **kwargs)


def test_an_unknown_drive_instrument_warns_and_sets_no_power(leaf_cls):
    platform = InstrumentPlatform("SOMETHING_ELSE")

    with pytest.warns(UserWarning, match=NO_DRIVE_INSTRUMENT_MSG):
        _set(leaf_cls, platform, drive_power=-20)

    assert ("drive_q1", Parameter.POWER) not in platform.values_in_scope


def test_a_fractional_attenuation_is_rejected(leaf_cls):
    platform = InstrumentPlatform("ROHDE_SCHWARZ")
    calibration = _calibration(attenuation=31.5)
    with pytest.raises(ValueError, match="not a whole number of dB"):
        _set(leaf_cls, platform, calibration, drive_power=-20)


def test_a_node_without_an_experiment_cannot_set_hardware():
    class Spine(_Leaf, FixedFluxNode[None]):  # sets no experiment, so it never registers
        pass

    node = Spine.__new__(Spine)
    with pytest.raises(TypeError, match="sets no experiment"):
        node._set_operating_point_hardware(Q1, drive_power=-20)
