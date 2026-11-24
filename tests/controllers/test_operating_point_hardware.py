"""Tests for what every node shares on the base: step parameters, the drive source, the operating point's line.

``CalibrationNode._param`` reads a step parameter or the node's ``defaults``; ``_set_drive_source`` arms
the drive with no operating point; ``_set_operating_point_hardware`` sets the LOs and attenuation, then
the drive, inside a real restore scope. A minimal leaf on each spine (registered in a scratch registry)
exercises them: the Rohde & Schwarz branch on a fake platform, the QCM-RF branch on a real offline qililab
platform (so its gain round-trips through the restore as on hardware), and the missing-level, attenuation
and spine errors. The instrument branches themselves are tested in ``test_drive_source.py``.
"""

from pathlib import Path
from typing import Any, ClassVar

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
from seqtante_open.controllers.calibration_node import AnalysisResult, FixedFluxNode, FluxSweepNode, RunContext
from seqtante_open.controllers.parameter_restore import restoring_parameters
from seqtante_open.targets import ExperimentTarget
from tests.controllers.fakes import InstrumentPlatform

#: The offline runcard the ``platform`` fixture builds a real qililab platform from (QCM-RF ``drive_q1``).
RUNCARD_PATH = Path(__file__).resolve().parents[1] / "runcards" / "test_AQPU_runcard.yml"

Q1 = ExperimentTarget.from_str("q1")
EXPERIMENT = "example_operating_point"
DRIVE_LO = 2.9e9
READOUT_LO = 7e9


class _Leaf:
    """The three phases a leaf must define; the tests only call the shared helpers."""

    def execute_node(self, targets):
        return {}

    def analyze(self, target, measurements):
        return AnalysisResult(fitted=None)

    def apply(self, target, result):
        return None


@pytest.fixture(params=[FixedFluxNode, FluxSweepNode], ids=["fixed_flux", "flux_sweep"])
def leaf_cls(request, monkeypatch):
    """A leaf on each spine: the helpers are on the base, so both shapes of node get them."""
    monkeypatch.setattr(node_registry, "NODE_REGISTRY", dict(node_registry.NODE_REGISTRY))

    class ExampleNode(_Leaf, request.param[None]):
        experiment: ClassVar[str] = EXPERIMENT
        analyses: ClassVar[tuple[str, ...]] = ("none",)
        defaults: ClassVar[dict[str, Any]] = {"averages": 1000}

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
                        drive=DriveFreqs(lo=DRIVE_LO, if_=-0.1e9),
                        readout=ReadoutFreqs(lo=READOUT_LO, if_=1e8, attenuation=attenuation),
                    )
                }
            },
            experiments={EXPERIMENT: {"q1": ExperimentSection(operating_point="sweet_spot")}},
        ),
    )
    return calibration


def _node(node_cls, platform, parameters: dict, calibration=None):
    context = RunContext(
        platform=platform,
        platform_path="runcard.yml",
        calibration=calibration or _calibration(),
        calibration_path="calibration.yml",
    )
    return node_cls(name="step", idx=0, targets=[Q1], parameters={Q1: parameters}, context=context)


def _snapshot(platform, keys) -> dict:
    return {(alias, parameter): platform.get_parameter(alias=alias, parameter=parameter) for alias, parameter in keys}


def _set_in_scope(node, platform, read_back=None, *, operating_point=True) -> dict:
    """Set the hardware inside a restore scope; return the values set (or ``read_back`` read in scope)."""
    with restoring_parameters(platform) as restore:
        node._restore = restore
        if operating_point:
            node._set_operating_point_hardware(Q1)
        else:
            node._set_drive_source(Q1)
        return _snapshot(platform, read_back) if read_back is not None else dict(platform.values)


def test_a_step_parameter_overrides_the_node_default(leaf_cls):
    node = _node(leaf_cls, InstrumentPlatform("ROHDE_SCHWARZ"), {"averages": 50})

    assert node._param(Q1, "averages") == 50


def test_a_parameter_with_no_default_is_required(leaf_cls):
    node = _node(leaf_cls, InstrumentPlatform("ROHDE_SCHWARZ"), {})

    with pytest.raises(KeyError, match="step 'step': q1 sets no 'idle_duration_sweep'"):
        node._param(Q1, "idle_duration_sweep")


def test_rohde_schwarz_sets_power_iq_modulation_and_rf(leaf_cls):
    platform = InstrumentPlatform("ROHDE_SCHWARZ")
    node = _node(leaf_cls, platform, {"drive_power": -20})

    snap = _set_in_scope(node, platform)

    assert snap["drive_q1", Parameter.LO_FREQUENCY] == 2_900_000_000
    assert snap["readout_q1", Parameter.LO_FREQUENCY] == 7_000_000_000
    assert snap["readout_q1", Parameter.OUT0_ATT] == 30
    assert snap["drive_q1", Parameter.POWER] == -20
    assert snap["drive_q1", Parameter.IQ_MODULATION] is True
    assert snap["drive_q1", Parameter.RF_ON] is True
    assert not platform.values["drive_q1", Parameter.RF_ON]  # the scope restored it on exit


def test_the_drive_power_falls_back_to_the_node_default(leaf_cls):
    class WithPowerDefault(leaf_cls):  # inherits the experiment, so it does not register again
        defaults: ClassVar[dict[str, Any]] = {"drive_power": -35}

    platform = InstrumentPlatform("ROHDE_SCHWARZ")
    node = _node(WithPowerDefault, platform, {})

    snap = _set_in_scope(node, platform)

    assert snap["drive_q1", Parameter.POWER] == -35


def test_qcm_rf_sets_the_gain_and_restores_it_on_a_real_platform(leaf_cls, platform):
    # drive_q1 is a QCM-RF in the runcard; its GAIN reads back as (gain_i, gain_q) (QHC-1803 review).
    keys = [
        ("drive_q1", Parameter.GAIN),
        ("drive_q1", Parameter.HARDWARE_MODULATION),
        ("drive_q1", Parameter.LO_FREQUENCY),
        ("readout_q1", Parameter.LO_FREQUENCY),
        ("readout_q1", Parameter.OUT0_ATT),
    ]
    before = _snapshot(platform, keys)
    node = _node(leaf_cls, platform, {"drive_gain": 0.4})

    in_scope = _set_in_scope(node, platform, read_back=keys)

    assert in_scope["drive_q1", Parameter.GAIN] == pytest.approx((0.4, 0.4))  # the gain, not the R&S power
    assert in_scope["drive_q1", Parameter.HARDWARE_MODULATION] is True
    assert in_scope["drive_q1", Parameter.LO_FREQUENCY] == pytest.approx(DRIVE_LO)
    assert in_scope["readout_q1", Parameter.LO_FREQUENCY] == pytest.approx(READOUT_LO)
    assert _snapshot(platform, keys) == before  # every write, the gain included, restored on exit


@pytest.mark.parametrize(
    ("instrument", "parameters", "message"),
    [
        ("ROHDE_SCHWARZ", {"drive_gain": 0.4}, "needs drive_power"),
        ("QCMRF", {"drive_power": -20}, "needs drive_gain"),
    ],
)
def test_a_missing_drive_level_fails_before_any_write(leaf_cls, instrument, parameters, message):
    platform = InstrumentPlatform(instrument)
    node = _node(leaf_cls, platform, parameters)

    with pytest.raises(ValueError, match=message):
        _set_in_scope(node, platform)

    assert platform.values == {}


def test_the_drive_source_needs_no_operating_point(leaf_cls):
    # The experiment has no section in this calibration file, so no operating point: the drive still arms.
    platform = InstrumentPlatform("ROHDE_SCHWARZ")
    node = _node(leaf_cls, platform, {"drive_power": -20}, calibration=Calibration())

    snap = _set_in_scope(node, platform, operating_point=False)

    assert snap["drive_q1", Parameter.POWER] == -20
    assert ("drive_q1", Parameter.LO_FREQUENCY) not in snap


def test_a_fractional_attenuation_is_rejected(leaf_cls):
    platform = InstrumentPlatform("ROHDE_SCHWARZ")
    node = _node(leaf_cls, platform, {"drive_power": -20}, calibration=_calibration(attenuation=31.5))

    with pytest.raises(ValueError, match="not a whole number of dB"):
        _set_in_scope(node, platform)


def test_a_node_without_an_experiment_cannot_set_hardware():
    class Spine(_Leaf, FixedFluxNode[None]):  # sets no experiment, so it never registers
        pass

    node = Spine.__new__(Spine)
    with pytest.raises(TypeError, match="sets no experiment"):
        node._set_operating_point_hardware(Q1)
