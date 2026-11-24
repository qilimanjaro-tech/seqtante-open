"""Call-assertion tests for the ``rabi`` node's two analyses (QHC-1806).

One test per analysis drives the full three-phase run with the measurement and the fit scripted (a fake
:class:`RabiAmplitudeModel`, a stubbed ``execute_node``), so each analysis's write is asserted without
hardware, a database or a disk write: ``readout_levels`` writes the |0>/|1> levels onto the operating
point, ``pi_amplitude`` writes the fitted amplitude into the ``pi-<operating_point>`` DRAG pulse in
``Calibration.waveforms``.
"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from qililab import IQDrag, Parameter, Square
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
from seqtante_open.controllers import calibration_node as node_module
from seqtante_open.controllers.calibration_node import RunContext
from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.nodes import RabiNode
from seqtante_open.experiments.nodes import rabi as rabi_module
from seqtante_open.outputs import output_controller
from seqtante_open.targets import ExperimentTarget

Q1 = ExperimentTarget.from_str("q1")
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1")

# Scripted fit outputs the fake model returns.
FAKE_PI = 0.42
FAKE_L0, FAKE_L0_STD = -1.5, 0.03
FAKE_L1, FAKE_L1_STD = 1.5, 0.04


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


class FakeRabiPlatform(FakePlatform):
    """A :class:`FakePlatform` that also answers the execution calls the inlined node makes.

    ``get_element`` reports a Rohde & Schwarz drive instrument (so the IQ-modulation/POWER/RF branch
    runs) and ``execute_qprogram`` returns a readout result. On execute it snapshots the parameters set
    so far into :attr:`values_at_execute`, before the node's run restores them.
    """

    def __init__(self):
        super().__init__()
        self.values_at_execute: dict = {}

    def get_element(self, alias):
        self.calls.append(f"get_element:{alias}")
        return SimpleNamespace(instruments=[SimpleNamespace(name=SimpleNamespace(name="ROHDE_SCHWARZ"))])

    def execute_qprogram(self, qprogram, **kwargs):
        self.calls.append("execute_qprogram")
        self.values_at_execute = dict(self.values)
        return SimpleNamespace(results={"readout_q1": [SimpleNamespace(array=np.zeros((2, 2)))]})


class FakeStreamArray:
    """Stand-in for ``StreamArray``: a context manager that swallows the write and carries a measurement id."""

    def __init__(self, **kwargs):
        self.measurement = SimpleNamespace(measurement_id=777)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __setitem__(self, key, value):
        return None


class FakeRabiModel:
    """A scripted stand-in for ``RabiAmplitudeModel``: no DB load, no fit, fixed outputs.

    ``events`` records what the node did with the fit, in order (fit, save, plot), so a test can assert
    the T1 order and that a rejected fit is never saved or plotted.
    """

    r_squared = 0.99
    angular_frequency = 1.0
    events: ClassVar[list[str]] = []

    def __init__(self, measurement_id, target=None, path=None):
        self.id = measurement_id
        self.target = target
        self.path = path
        self.results = {"angular_frequency": type(self).angular_frequency}

    def fit(self):
        type(self).events.append("fit")
        return self.results

    def plot(self):
        type(self).events.append("plot")

    def save_fitting(self):
        type(self).events.append("save")

    @property
    def pi_amplitude(self):
        return FAKE_PI

    @property
    def readout_levels(self):
        return (FAKE_L0, FAKE_L0_STD, FAKE_L1, FAKE_L1_STD)


def _crosstalk() -> CrosstalkMatrix:
    return CrosstalkMatrix.from_buses(
        {b: {bb: (1.0 if b == bb else 0.0) for bb in CROSSTALK_BUSES} for b in CROSSTALK_BUSES}
    )


def _operating_point() -> OperatingPoint:
    return OperatingPoint(
        flux=Flux(z=0.1),
        drive=DriveFreqs(lo=4e9, if_=1e8),
        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=10.0),
    )


def _rabi_file(point_name: str, pulses: dict | None = None) -> CalibrationFile:
    """A calibration file with one ``rabi`` section for q1 at ``point_name``, naming ``pulses`` if given."""
    return CalibrationFile(
        operating_points_version=1,
        operating_points={"q1": {point_name: _operating_point()}},
        experiments={"rabi": {"q1": ExperimentSection(operating_point=point_name, pulses=pulses or {})}},
    )


def _add_played_pulses(calibration: Calibration, point_name: str, *, pi: bool = True) -> None:
    """Store the ``readout-<point>`` and (unless ``pi`` is False) ``pi-<point>`` pulses rabi_amplitude plays."""
    calibration.add_waveform(
        bus="readout_q1", name=f"readout-{point_name}", waveform=Square(amplitude=0.1, duration=2000)
    )
    if pi:
        calibration.add_waveform(
            bus="drive_q1",
            name=f"pi-{point_name}",
            waveform=IQDrag(amplitude=1.0, duration=40, num_sigmas=4, drag_coefficient=0.1),
        )


def _calibration(file: CalibrationFile) -> Calibration:
    calibration = Calibration()
    calibration.crosstalk_matrix = _crosstalk()
    write_calibration_file(calibration, file)
    return calibration


def _make_rabi(*, analysis: str, calibration: Calibration, parameters: dict | None = None) -> RabiNode:
    context = RunContext(
        platform=FakePlatform(),
        platform_path="runcard.yml",
        calibration=calibration,
        calibration_path="calibration.yml",
    )
    node = RabiNode(
        name="step",
        idx=0,
        targets=[Q1],
        parameters={Q1: parameters or {}},
        context=context,
        analysis=analysis,
    )
    node.execute_node = lambda targets: {target: [777] for target in targets}  # no hardware
    return node


def test_readout_levels_analysis_writes_levels_onto_the_point(monkeypatch):
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("reset"))

    node = _make_rabi(analysis="readout_levels", calibration=calibration)
    failed = node.run({})

    assert failed == []
    levels = read_calibration_file(calibration).operating_points["q1"]["reset"].readout.levels
    assert levels is not None
    assert levels.pulse == "readout-reset"  # the readout pulse rabi_amplitude plays at the point
    assert levels.level_0.nominal == pytest.approx(FAKE_L0)
    assert levels.level_0.std == pytest.approx(FAKE_L0_STD)
    assert levels.level_1.nominal == pytest.approx(FAKE_L1)
    assert levels.level_1.std == pytest.approx(FAKE_L1_STD)


def test_pi_amplitude_analysis_writes_into_the_named_pulse(monkeypatch):
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("sweet_spot"))
    # The pi amplitude is written into the pi-sweet_spot DRAG pulse, keeping its other shape parameters.
    _add_played_pulses(calibration, "sweet_spot")

    node = _make_rabi(analysis="pi_amplitude", calibration=calibration)
    failed = node.run({})

    assert failed == []
    pulse = calibration.get_waveform(bus="drive_q1", name="pi-sweet_spot")
    assert pulse.amplitude == pytest.approx(FAKE_PI)  # amplitude updated to the fitted pi amplitude
    assert pulse.duration == 40  # shape preserved
    assert pulse.num_sigmas == 4
    assert pulse.drag_coefficient == pytest.approx(0.1)
    # The operating point's readout levels are untouched by the pi-amplitude analysis.
    assert read_calibration_file(calibration).operating_points["q1"]["sweet_spot"].readout.levels is None


def test_readout_levels_fit_below_min_r2_fails_the_step(monkeypatch):
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)

    class LowR2Model(FakeRabiModel):
        r_squared = 0.5  # below the default 0.9 minimum

    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", LowR2Model)
    calibration = _calibration(_rabi_file("reset"))

    node = _make_rabi(analysis="readout_levels", calibration=calibration)
    failed = node.run({})

    assert failed == ["q1"]  # a poor fit fails the target
    assert read_calibration_file(calibration).operating_points["q1"]["reset"].readout.levels is None


def test_execute_builds_the_qili_sweep_and_sets_the_hardware(monkeypatch):
    """The real ``execute_node`` builds the qili QProgram at the operating point and sets the HW from it."""
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    monkeypatch.setattr(output_controller, "_initialized", True)
    monkeypatch.setattr(output_controller, "_db_manager", object())
    built: dict = {}

    def fake_rabi_amplitude(**kwargs):
        built.update(kwargs)
        return SimpleNamespace(name="qprogram")  # sentinel; the fake platform/stream ignore its content

    monkeypatch.setattr(rabi_module, "rabi_amplitude", fake_rabi_amplitude)
    monkeypatch.setattr(rabi_module, "StreamArray", FakeStreamArray)

    calibration = _calibration(_rabi_file("sweet_spot"))
    _add_played_pulses(calibration, "sweet_spot")
    platform = FakeRabiPlatform()
    context = RunContext(
        platform=platform, platform_path="runcard.yml", calibration=calibration, calibration_path="calibration.yml"
    )
    params = {"drive_amplitude_sweep": (0.0, 1.0, 21), "averages": 1000, "relax_duration": 50_000, "drive_power": -10}
    node = RabiNode(
        name="step", idx=0, targets=[Q1], parameters={Q1: params}, context=context, analysis="readout_levels"
    )

    failed = node.run({})

    assert failed == []
    # The sweep was built at the target's operating point; the pulses/IFs come from the named point.
    assert built["target"] == "q1"
    assert built["operating_point"] == "sweet_spot"
    assert built["calibration"] is calibration
    np.testing.assert_array_equal(built["drive_amplitude_sweep"], np.linspace(0.0, 1.0, 21))
    # The node set the operating point's hardware itself (snapshotted before the run restored it):
    # drive/readout LOs, readout attenuation, drive power, and the R&S IQ modulation and RF output.
    snap = platform.values_at_execute
    assert snap["drive_q1", Parameter.LO_FREQUENCY] == int(4e9)
    assert snap["readout_q1", Parameter.LO_FREQUENCY] == int(7e9)
    assert snap["readout_q1", Parameter.OUT0_ATT] == 10
    assert snap["drive_q1", Parameter.POWER] == -10
    assert snap["drive_q1", Parameter.IQ_MODULATION] is True
    assert snap["drive_q1", Parameter.RF_ON] is True


def test_analyze_raises_analysis_error_below_min_r2(monkeypatch):
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("reset"))
    node = _make_rabi(analysis="readout_levels", calibration=calibration, parameters={"min_r2": 0.999})

    with pytest.raises(AnalysisError, match="below the minimum"):
        node.analyze(Q1, [777])


def test_analyze_fits_checks_saves_then_plots(monkeypatch):
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    monkeypatch.setattr(FakeRabiModel, "events", [])
    node = _make_rabi(analysis="readout_levels", calibration=_calibration(_rabi_file("reset")))

    node.analyze(Q1, [777])

    assert FakeRabiModel.events == ["fit", "save", "plot"]


@pytest.mark.parametrize(
    ("analysis", "knob", "value"),
    [
        ("readout_levels", "r_squared", float("nan")),  # a flat trace: no variance to explain
        ("readout_levels", "r_squared", 0.5),  # below the default minimum
        ("pi_amplitude", "angular_frequency", 0.0),  # no oscillation: there is no pi amplitude
    ],
)
def test_a_rejected_fit_is_never_saved_or_plotted(monkeypatch, analysis, knob, value):
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    monkeypatch.setattr(FakeRabiModel, "events", [])
    monkeypatch.setattr(FakeRabiModel, knob, value)
    node = _make_rabi(analysis=analysis, calibration=_calibration(_rabi_file("reset")))

    with pytest.raises(AnalysisError):
        node.analyze(Q1, [777])

    assert FakeRabiModel.events == ["fit"]


def _executing_rabi(calibration: Calibration, monkeypatch) -> tuple[RabiNode, FakeRabiPlatform]:
    """A rabi node whose real ``execute_node`` runs against a fake platform, with the QProgram builder stubbed."""
    monkeypatch.setattr(rabi_module, "rabi_amplitude", lambda **kwargs: SimpleNamespace(name="qprogram"))
    monkeypatch.setattr(rabi_module, "StreamArray", FakeStreamArray)
    monkeypatch.setattr(output_controller, "_initialized", True)
    monkeypatch.setattr(output_controller, "_db_manager", object())
    platform = FakeRabiPlatform()
    context = RunContext(
        platform=platform, platform_path="runcard.yml", calibration=calibration, calibration_path="calibration.yml"
    )
    params = {"drive_amplitude_sweep": (0.0, 1.0, 21), "averages": 1000, "relax_duration": 50_000, "drive_power": -10}
    node = RabiNode(name="step", idx=0, targets=[Q1], parameters={Q1: params}, context=context, analysis="pi_amplitude")
    return node, platform


def test_a_missing_pi_pulse_fails_before_any_hardware_is_set(monkeypatch):
    """rabi seeds no pulse: without ``pi-<point>`` it raises, naming the pulse, before touching the platform."""
    calibration = _calibration(_rabi_file("sweet_spot"))
    _add_played_pulses(calibration, "sweet_spot", pi=False)
    node, platform = _executing_rabi(calibration, monkeypatch)

    with pytest.raises(ValueError, match="'pi-sweet_spot' pulse on drive_q1"):
        node.execute_node([Q1])

    assert platform.calls == []


def test_a_section_naming_another_pulse_is_rejected(monkeypatch):
    """rabi_amplitude only plays ``pi-<point>``, so a section naming another drive pulse is a config error."""
    calibration = _calibration(_rabi_file("sweet_spot", pulses={"drive": "pi-reset"}))
    _add_played_pulses(calibration, "sweet_spot")
    _add_played_pulses(calibration, "reset")  # the named pulse exists, so only rabi's own check can reject it
    node, platform = _executing_rabi(calibration, monkeypatch)

    with pytest.raises(ValueError, match="names drive pulse 'pi-reset', but rabi_amplitude plays 'pi-sweet_spot'"):
        node.execute_node([Q1])

    assert platform.calls == []


def test_pi_amplitude_refuses_a_pi_pulse_that_is_not_a_drag(monkeypatch):
    """The fitted amplitude only fits an ``IQDrag``; any other waveform under ``pi-<point>`` fails the step."""
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("sweet_spot"))
    calibration.add_waveform(bus="drive_q1", name="pi-sweet_spot", waveform=Square(amplitude=1.0, duration=40))
    node = _make_rabi(analysis="pi_amplitude", calibration=calibration)

    with pytest.raises(AnalysisError, match="is a Square, not an IQDrag"):
        node.apply(Q1, node.analyze(Q1, [777]))
