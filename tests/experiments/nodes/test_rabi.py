"""Call-assertion tests for the ``rabi`` node's two analyses (QHC-1806).

One test per analysis drives the full three-phase run with the measurement and the fit scripted (a fake
:class:`RabiAmplitudeModel`, a stubbed ``execute_node``), so each analysis's write is asserted without
hardware, a database or a disk write: ``readout_levels`` writes the |0>/|1> levels onto the operating
point, ``pi_amplitude`` writes the fitted amplitude into the ``pi-<operating_point>`` pulse in
``Calibration.waveforms``.
"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from qili_qprograms.calibration.io import read_calibration_file, write_calibration_file
from qili_qprograms.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    ExperimentSection,
    Flux,
    OperatingPoint,
    ReadoutFreqs,
)
from qili_qprograms.targets import ExperimentTarget
from qililab import IQDrag, Parameter, Square
from qililab.qprogram.calibration import Calibration

from seqtante_open.controllers.verdicts import AnalysisError
from seqtante_open.experiments.nodes import RabiNode
from seqtante_open.experiments.nodes import rabi as rabi_module
from tests.experiments.nodes.harness import (
    ExecutingPlatform,
    build_context,
    identity_crosstalk,
    stub_output_and_save,
)

Q1 = ExperimentTarget.from_str("q1")

# Scripted fit outputs the fake model returns.
FAKE_PI = 0.42
FAKE_L0, FAKE_L0_STD = -1.5, 0.03
FAKE_L1, FAKE_L1_STD = 1.5, 0.04


class FakeRabiPlatform(ExecutingPlatform):
    """An :class:`ExecutingPlatform` (a Rohde & Schwarz drive by default) whose one execution returns the whole sweep."""

    def __init__(self, drive_instrument: str = "ROHDE_SCHWARZ") -> None:
        super().__init__(instruments={"drive_q1": drive_instrument})

    def results(self) -> dict:
        return {"readout_q1": [SimpleNamespace(array=np.zeros((2, 2)))]}


class FakeRabiModel:
    """A scripted stand-in for ``RabiAmplitudeModel``: no DB load, no fit, fixed outputs.

    ``events`` records what the node did with the fit, in order (fit, save, plot), so a test can assert
    the T1 order and that a rejected fit is never saved or plotted.
    """

    r_squared = 0.99
    pi_amplitude = FAKE_PI
    pi_amplitude_std = 0.01
    level_std = FAKE_L0_STD
    amplitudes = np.linspace(0.0, 0.8, 21)  # the swept window; FAKE_PI sits inside it
    events: ClassVar[list[str]] = []

    def __init__(self, measurement_id, target=None, path=None):
        self.id = measurement_id
        self.target = target
        self.path = path
        self.results = {"pi_amplitude": type(self).pi_amplitude, "pi_amplitude_std": type(self).pi_amplitude_std}

    def fit(self):
        type(self).events.append("fit")
        return self.results

    def plot(self):
        type(self).events.append("plot")

    def save_fitting(self):
        type(self).events.append("save")

    @property
    def readout_levels(self):
        return (FAKE_L0, type(self).level_std, FAKE_L1, FAKE_L1_STD)


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


def _two_point_file() -> CalibrationFile:
    """q1 has a reset and a sweet_spot point, and its one ``rabi`` section names sweet_spot, as in the example tree."""
    reset = OperatingPoint(
        flux=Flux(z=0.4),
        drive=DriveFreqs(lo=3e9, if_=1e8),
        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=20.0),
    )
    return CalibrationFile(
        operating_points_version=1,
        operating_points={"q1": {"sweet_spot": _operating_point(), "reset": reset}},
        experiments={"rabi": {"q1": ExperimentSection(operating_point="sweet_spot")}},
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
    calibration.crosstalk_matrix = identity_crosstalk()
    write_calibration_file(calibration, file)
    return calibration


def _make_rabi(*, analysis: str, calibration: Calibration, parameters: dict | None = None) -> RabiNode:
    node = RabiNode(
        name="step",
        idx=0,
        targets=[Q1],
        parameters={Q1: parameters or {}},
        context=build_context(calibration),
        analysis=analysis,
    )
    node.execute_node = lambda targets: {target: [777] for target in targets}  # no hardware
    return node


def test_readout_levels_analysis_writes_levels_onto_the_point(monkeypatch):
    stub_output_and_save(monkeypatch)
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
    stub_output_and_save(monkeypatch)
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("sweet_spot"))
    # The pi amplitude is written into the pi-sweet_spot pulse, keeping its other shape parameters.
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


def _executing_rabi(
    calibration: Calibration,
    monkeypatch,
    *,
    analysis: str = "pi_amplitude",
    built: dict | None = None,
    platform: FakeRabiPlatform | None = None,
    drive_level: dict | None = None,
) -> tuple[RabiNode, FakeRabiPlatform]:
    """A rabi node whose real ``execute_node`` runs against a fake platform, with the QProgram builder stubbed.

    The stub records the builder's keyword arguments into ``built`` when given. ``drive_level`` replaces
    the step's ``drive_power`` for the Rohde & Schwarz default ``platform``.
    """

    def fake_rabi_amplitude(**kwargs):
        if built is not None:
            built.update(kwargs)
        return SimpleNamespace(name="qprogram")  # sentinel; the fake platform/stream ignore its content

    monkeypatch.setattr(rabi_module, "rabi_amplitude", fake_rabi_amplitude)
    stub_output_and_save(monkeypatch)
    platform = platform or FakeRabiPlatform()
    params = {"drive_amplitude_sweep": (0.0, 1.0, 21), "averages": 1000, "relax_duration": 50_000}
    params |= {"drive_power": -10} if drive_level is None else drive_level
    node = RabiNode(
        name="step",
        idx=0,
        targets=[Q1],
        parameters={Q1: params},
        context=build_context(calibration, platform),
        analysis=analysis,
    )
    return node, platform


def test_execute_builds_the_qili_sweep_and_sets_the_hardware(monkeypatch):
    """The real ``execute_node`` builds the qili QProgram at the operating point and sets the HW from it."""
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("sweet_spot"))
    _add_played_pulses(calibration, "sweet_spot")
    built: dict = {}
    node, platform = _executing_rabi(calibration, monkeypatch, analysis="readout_levels", built=built)

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


def test_analyze_fits_checks_saves_then_plots(monkeypatch):
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    monkeypatch.setattr(FakeRabiModel, "events", [])
    node = _make_rabi(analysis="readout_levels", calibration=_calibration(_rabi_file("reset")))

    node.analyze(Q1, [777])

    assert FakeRabiModel.events == ["fit", "save", "plot"]


@pytest.mark.parametrize(
    ("analysis", "knob", "value"),
    [
        ("readout_levels", "r_squared", 0.5),  # below the node's default minimum
        ("pi_amplitude", "pi_amplitude", float("nan")),  # no oscillation: there is no pi amplitude
        ("pi_amplitude", "pi_amplitude", 0.9),  # the sweep stops short of the pi amplitude
        ("readout_levels", "pi_amplitude", 0.9),  # ... so |1> was never measured, only extrapolated
        ("readout_levels", "level_std", float("nan")),  # no covariance: the levels have no uncertainty
        ("pi_amplitude", "pi_amplitude_std", float("nan")),  # no covariance: the pi amplitude has none
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


def test_a_missing_pi_pulse_fails_before_any_hardware_is_set(monkeypatch):
    """rabi seeds no pulse: without ``pi-<point>`` it raises, naming the pulse, before touching the platform."""
    calibration = _calibration(_rabi_file("sweet_spot"))
    _add_played_pulses(calibration, "sweet_spot", pi=False)
    node, platform = _executing_rabi(calibration, monkeypatch)

    with pytest.raises(ValueError, match="'pi-sweet_spot' pulse on drive_q1"):
        node.execute_node([Q1])

    assert platform.calls == []


@pytest.mark.parametrize(
    ("stored", "written"),
    [
        pytest.param(IQDrag(amplitude=0.3, duration=40, num_sigmas=4, drag_coefficient=0.1), FAKE_PI, id="drag"),
        pytest.param(
            IQDrag(amplitude=-0.3, duration=40, num_sigmas=4, drag_coefficient=0.1), -FAKE_PI, id="negative-drag"
        ),
        pytest.param(Square(amplitude=-0.5, duration=40), -FAKE_PI, id="negative-square"),
    ],
)
def test_pi_amplitude_writes_the_played_pulse_at_the_fitted_gain(monkeypatch, stored, written):
    """rabi_amplitude plays the stored pulse at unit amplitude, so any shape works and the sign is kept."""
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("sweet_spot"))
    calibration.add_waveform(bus="drive_q1", name="pi-sweet_spot", waveform=stored)
    node = _make_rabi(analysis="pi_amplitude", calibration=calibration)

    node.apply(Q1, node.analyze(Q1, [777]))

    pulse = calibration.get_waveform(bus="drive_q1", name="pi-sweet_spot")
    assert type(pulse) is type(stored)
    assert pulse.amplitude == pytest.approx(written)
    assert pulse.duration == 40


def test_a_qcm_rf_drive_runs_at_unit_gain_by_default(monkeypatch):
    """rabi sweeps the drive gain itself, so it arms a QCM-RF drive at static gain 1."""
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_rabi_file("sweet_spot"))
    _add_played_pulses(calibration, "sweet_spot")
    node, platform = _executing_rabi(
        calibration, monkeypatch, platform=FakeRabiPlatform(drive_instrument="QCMRF"), drive_level={}
    )

    assert node.run({}) == []
    assert platform.values_at_execute["drive_q1", Parameter.GAIN] == 1
    # The drive instrument is looked up once per target, so an unknown one is warned about once.
    assert platform.calls.count("get_element:drive_q1") == 1


def test_a_qcm_rf_drive_gain_other_than_1_fails_before_any_hardware(monkeypatch):
    calibration = _calibration(_rabi_file("sweet_spot"))
    _add_played_pulses(calibration, "sweet_spot")
    node, platform = _executing_rabi(
        calibration, monkeypatch, platform=FakeRabiPlatform(drive_instrument="QCMRF"), drive_level={"drive_gain": 0.5}
    )

    with pytest.raises(ValueError, match=r"drive_gain 0\.5 is not 1; rabi sweeps the drive gain itself"):
        node.execute_node([Q1])

    assert platform.calls == ["get_element:drive_q1"]  # it read which instrument drives q1, and set nothing


# ------------- rabi_reset and rabi_sweet_spot share one section (QHC-1936) -------------


def test_rabi_reset_measures_at_and_writes_the_reset_point_not_the_section_s(monkeypatch):
    """A step naming ``reset`` builds, sets the hardware and writes ``pi-reset`` there; ``pi-sweet_spot`` is untouched."""
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_two_point_file())
    _add_played_pulses(calibration, "reset")
    _add_played_pulses(calibration, "sweet_spot")
    built: dict = {}
    node, platform = _executing_rabi(calibration, monkeypatch, built=built)
    node.parameters[Q1]["operating_point"] = "reset"

    assert node.run({}) == []

    assert built["operating_point"] == "reset"
    assert platform.values_at_execute["drive_q1", Parameter.LO_FREQUENCY] == int(3e9)
    assert platform.values_at_execute["readout_q1", Parameter.OUT0_ATT] == 20
    assert calibration.get_waveform(bus="drive_q1", name="pi-reset").amplitude == pytest.approx(FAKE_PI)
    assert calibration.get_waveform(bus="drive_q1", name="pi-sweet_spot").amplitude == pytest.approx(1.0)


def test_rabi_reset_writes_the_readout_levels_onto_the_reset_point(monkeypatch):
    stub_output_and_save(monkeypatch)
    monkeypatch.setattr(rabi_module, "RabiAmplitudeModel", FakeRabiModel)
    calibration = _calibration(_two_point_file())

    node = _make_rabi(analysis="readout_levels", calibration=calibration, parameters={"operating_point": "reset"})

    assert node.run({}) == []
    points = read_calibration_file(calibration).operating_points["q1"]
    assert points["reset"].readout.levels.pulse == "readout-reset"
    assert points["reset"].readout.levels.level_0.nominal == pytest.approx(FAKE_L0)
    assert points["sweet_spot"].readout.levels is None
