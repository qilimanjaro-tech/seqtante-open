"""Tests for the helpers every node shares on ``CalibrationNode`` (QHC-1932).

Sweeps, fit acceptance, pulse names and the stream: each was written once per node before, with small
differences, and now lives on the base.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from qili_qprograms.bus_alias import Role
from qili_qprograms.calibration.io import write_calibration_file
from qili_qprograms.calibration.model import (
    CalibrationFile,
    DriveFreqs,
    ExperimentSection,
    Flux,
    OperatingPoint,
    ReadoutFreqs,
)
from qili_qprograms.targets import ExperimentTarget
from qililab import IQDrag, Square
from qililab.qprogram.calibration import Calibration

from seqtante_open.controllers.calibration_node import (
    AnalysisResult,
    FixedFluxNode,
    write_per_point,
    write_whole,
)
from seqtante_open.controllers.verdicts import AnalysisError
from tests.experiments.nodes.harness import ExecutingPlatform, FakeStreamArray, build_context, stub_output_and_save

Q1 = ExperimentTarget.from_str("q1")


class _Leaf(FixedFluxNode):
    experiment = "helpers_example"
    analyses = ("first", "second")
    defaults = {"min_r2": 0.8, "max_rel_uncertainty": 0.5}

    def execute_node(self, targets):  # pragma: no cover - the helpers are called directly
        return {}

    def analyze(self, target, measurements):  # pragma: no cover
        return AnalysisResult()

    def apply(self, target, result):  # pragma: no cover
        pass


class _Platform(ExecutingPlatform):
    """Returns ``points`` acquisitions of a ``(2, 3)`` array whose values count the executions."""

    def __init__(self, points: int) -> None:
        super().__init__(instruments={})
        self.points = points

    def results(self) -> dict:
        return {"readout_q1": [SimpleNamespace(array=np.full((2, 3), self.executions)) for _ in range(self.points)]}


def _calibration(pulses: dict | None = None) -> Calibration:
    calibration = Calibration()
    write_calibration_file(
        calibration,
        CalibrationFile(
            operating_points_version=1,
            operating_points={
                "q1": {
                    "sweet_spot": OperatingPoint(
                        flux=Flux(z=0.1),
                        drive=DriveFreqs(lo=4e9, if_=1e8),
                        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=10.0),
                    ),
                    "reset": OperatingPoint(
                        flux=Flux(z=0.4),
                        drive=DriveFreqs(lo=3e9, if_=1e8),
                        readout=ReadoutFreqs(lo=7e9, if_=1e8, attenuation=10.0),
                    ),
                }
            },
            experiments={
                "helpers_example": {"q1": ExperimentSection(operating_point="sweet_spot", pulses=pulses or {})}
            },
        ),
    )
    return calibration


def _node(params: dict | None = None, *, calibration: Calibration | None = None, platform=None, **kwargs) -> _Leaf:
    context = build_context(calibration if calibration is not None else _calibration(), platform)
    return _Leaf(name="step", idx=0, targets=[Q1], parameters={Q1: params or {}}, context=context, **kwargs)


# --------------------------------- analysis ---------------------------------


def test_the_analysis_defaults_to_the_experiment_first_one():
    assert _node().analysis == "first"
    assert _node(analysis="second").analysis == "second"


# ---------------------------------- sweeps ----------------------------------


def test_a_sweep_is_the_step_linspace_and_an_absolute_sweep_is_offset_onto_the_stored_value():
    node = _node({"window": [-1.0, 1.0, 3]})

    np.testing.assert_array_equal(node._sweep(Q1, "window"), [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(node._absolute_sweep(Q1, "window", 10.0), [9.0, 10.0, 11.0])


def test_a_sweep_with_fewer_than_two_points_is_rejected():
    node = _node({"window": [0.0, 1.0, 1]})
    with pytest.raises(ValueError, match="window must have at least 2 points, got 1"):
        node._sweep(Q1, "window")


def test_a_sweep_that_starts_and_stops_at_the_same_value_is_rejected():
    node = _node({"window": [0.0, 0.0, 11]})
    with pytest.raises(ValueError, match="window starts and stops at 0, so it sweeps nothing"):
        node._sweep(Q1, "window")


def test_a_spine_has_no_section_to_read():
    class Spine(FixedFluxNode):
        def execute_node(self, targets):  # pragma: no cover
            return {}

        def analyze(self, target, measurements):  # pragma: no cover
            return AnalysisResult()

        def apply(self, target, result):  # pragma: no cover
            pass

    spine = Spine(name="step", idx=0, targets=[Q1], parameters={Q1: {}}, context=build_context(_calibration()))
    with pytest.raises(TypeError, match="Spine sets no experiment"):
        spine._section(Q1)


# ------------------------------ fit acceptance ------------------------------


@pytest.mark.parametrize("r_squared", [0.5, float("nan")])
def test_a_low_or_undefined_r_squared_fails_the_target(r_squared):
    node = _node()
    with pytest.raises(AnalysisError, match=r"below the minimum 0\.8"):
        node._require_r2(Q1, r_squared)


def test_the_step_min_r2_overrides_the_default():
    _node({"min_r2": 0.5})._require_r2(Q1, 0.6)
    strict = _node({"min_r2": 0.9})
    with pytest.raises(AnalysisError):
        strict._require_r2(Q1, 0.85)


@pytest.mark.parametrize("relative_uncertainty", [0.6, float("inf"), float("nan")])
def test_a_large_or_non_finite_relative_uncertainty_fails_the_target(relative_uncertainty):
    node = _node()
    with pytest.raises(AnalysisError, match=r"exceeds the maximum 0\.5"):
        node._require_rel_uncertainty(Q1, relative_uncertainty)


def test_an_accepted_fit_passes_both_checks():
    node = _node()
    node._require_r2(Q1, 0.95)
    node._require_rel_uncertainty(Q1, 0.1)


# ---------------------------------- pulses ----------------------------------


def test_a_pulse_is_named_by_purpose_and_operating_point():
    node = _node()

    assert node._pulse(Q1, Role.drive, "pi") == "pi-sweet_spot"
    assert node._pulse(Q1, Role.readout, "readout") == "readout-sweet_spot"


def test_a_section_naming_another_pulse_is_rejected():
    calibration = _calibration({"drive": "pi-reset"})
    # The named pulse exists, so only the node's own check can reject it.
    calibration.add_waveform(bus="drive_q1", name="pi-reset", waveform=Square(amplitude=1.0, duration=40))
    node = _node(calibration=calibration)

    with pytest.raises(
        ValueError,
        match=r"drive pulse 'pi-reset', from the helpers_example section, is not the 'pi-sweet_spot' its QProgram plays",
    ):
        node._pulse(Q1, Role.drive, "pi")


def test_a_required_pulse_must_be_stored():
    calibration = _calibration()
    node = _node(calibration=calibration)
    with pytest.raises(ValueError, match=r"needs the 'pi-sweet_spot' pulse on drive_q1"):
        node._require_pulse(Q1, Role.drive, "pi")

    calibration.add_waveform(
        bus="drive_q1",
        name="pi-sweet_spot",
        waveform=IQDrag(amplitude=1.0, duration=40, num_sigmas=4, drag_coefficient=0),
    )
    assert node._require_pulse(Q1, Role.drive, "pi") == "pi-sweet_spot"


# ------------------------------ operating point ------------------------------


def test_a_step_naming_no_point_runs_at_its_section_s():
    node = _node()

    assert node._operating_point_name(Q1) == "sweet_spot"
    assert node._operating_point(Q1).flux.z == pytest.approx(0.1)


def test_two_steps_of_one_experiment_run_each_at_the_point_it_names():
    """``rabi_reset`` and ``rabi_sweet_spot`` share one section; each runs and names pulses at its own point."""
    reset, sweet_spot = _node({"operating_point": "reset"}), _node({"operating_point": "sweet_spot"})

    assert reset._operating_point(Q1).flux.z == pytest.approx(0.4)
    assert reset._pulse(Q1, Role.drive, "pi") == "pi-reset"
    assert sweet_spot._operating_point(Q1).flux.z == pytest.approx(0.1)
    assert sweet_spot._pulse(Q1, Role.drive, "pi") == "pi-sweet_spot"


def test_a_step_is_placed_at_the_point_it_names():
    node = _node({"operating_point": "reset"})
    writes: list = []
    node._restore = SimpleNamespace(set=lambda alias, parameter, value: writes.append((alias, value)))

    node._place_operating_point(Q1)

    assert writes == [("flux_z_q1", pytest.approx(0.4))]


def test_a_step_naming_a_point_the_target_lacks_is_rejected():
    node = _node({"operating_point": "idle"})

    with pytest.raises(
        ValueError, match=r"operating_point 'idle' is not one of its operating points \(reset, sweet_spot\)"
    ):
        node._operating_point_name(Q1)


def test_a_step_naming_another_pulse_is_rejected():
    node = _node({"operating_point": "reset", "pulses": {"drive": "pi-sweet_spot"}})

    # The pulses may come from defaults or an override, so the error names the step's resolved pulses.
    with pytest.raises(
        ValueError,
        match=r"drive pulse 'pi-sweet_spot', from the pulses of step 'step' \(defaults, the step or an override\), "
        r"is not the 'pi-reset' its QProgram plays at operating point 'reset'",
    ):
        node._pulse(Q1, Role.drive, "pi")


def test_the_section_s_pulse_is_checked_when_the_step_names_one_at_its_point():
    """A step naming the played pulse does not hide a section that names another one at the same point."""
    calibration = _calibration({"drive": "drag"})
    calibration.add_waveform(bus="drive_q1", name="drag", waveform=Square(amplitude=1.0, duration=40))
    node = _node({"pulses": {"drive": "pi-sweet_spot"}}, calibration=calibration)

    with pytest.raises(
        ValueError, match=r"drive pulse 'drag', from the helpers_example section, is not the 'pi-sweet_spot'"
    ):
        node._pulse(Q1, Role.drive, "pi")


def test_the_section_s_pulses_do_not_apply_at_another_point():
    """The section names its pulses for its own point; a step at another point plays that point's pulse."""
    calibration = _calibration({"drive": "pi-sweet_spot"})
    calibration.add_waveform(bus="drive_q1", name="pi-sweet_spot", waveform=Square(amplitude=1.0, duration=40))
    node = _node({"operating_point": "reset"}, calibration=calibration)

    assert node._pulse(Q1, Role.drive, "pi") == "pi-reset"


# ---------------------------------- stream ----------------------------------


def _streamed(monkeypatch, points: int, **kwargs) -> tuple[int, FakeStreamArray, _Platform]:
    stub_output_and_save(monkeypatch)
    monkeypatch.setattr(FakeStreamArray, "measurement_id", 7)
    platform = _Platform(points)
    measurement = _node(platform=platform)._stream(
        Q1, SimpleNamespace(name="qprogram"), experiment_name="helpers", loops={}, shape=(points, 2), **kwargs
    )
    assert FakeStreamArray.last is not None
    return measurement, FakeStreamArray.last, platform


def test_the_stream_writes_one_acquisition_per_point(monkeypatch):
    measurement, stream, platform = _streamed(monkeypatch, 3, write=write_per_point)

    assert measurement == 7
    assert platform.executions == 1
    assert [key for key, _ in stream.writes] == [(0,), (1,), (2,)]
    assert stream.kwargs["experiment_name"] == "helpers"
    assert stream.kwargs["qubit_idx"] == "q1"


def test_the_stream_writes_the_running_average_once_per_repetition(monkeypatch):
    _, stream, platform = _streamed(monkeypatch, 1, write=write_whole, repetitions=3)

    assert platform.executions == 3
    assert [key for key, _ in stream.writes] == [(), (), ()]
    # Executions return 1, 2, 3: the stream holds the average so far after each one.
    for (_, written), average in zip(stream.writes, (1.0, 1.5, 2.0), strict=True):
        np.testing.assert_array_equal(written, np.full((3, 2), average))


@pytest.mark.parametrize("repetitions", [0, -1])
def test_a_stream_with_no_repetition_is_rejected(monkeypatch, repetitions):
    stub_output_and_save(monkeypatch)
    node = _node(platform=_Platform(1))
    qprogram = SimpleNamespace(name="qprogram")

    with pytest.raises(ValueError, match="repetitions must be at least 1"):
        node._stream(Q1, qprogram, experiment_name="helpers", loops={}, shape=(1, 2), repetitions=repetitions)
