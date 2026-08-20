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

"""Tests for the ``mock_recorder`` fixture itself, in particular its signature
validation: a stub must reject calls the real callable would reject, so a test
can't keep asserting on a parameter production code has renamed or dropped.

The stand-in targets live in this module, so patching them needs no experiment
imports and touches nothing else.
"""

import pytest

MODULE = __name__


def execute(platform, *, r_amp, averages=100):
    """Stand-in for an execution function; must never run once mocked."""
    raise AssertionError("real function must never be called")


def load(file, cls=None):
    """Stand-in for an IO boundary; must never run once mocked."""
    raise AssertionError("real function must never be called")


def test_valid_call_is_recorded_and_returns_output(mock_recorder):
    mock_recorder.mock(f"{MODULE}.execute", output=4242)

    assert execute("platform", r_amp=0.9) == 4242

    (call,) = mock_recorder.calls["execute"]
    assert call["args"] == ("platform",)
    assert call["kwargs"] == {"r_amp": 0.9}


def test_unknown_kwarg_raises_and_records_nothing(mock_recorder):
    """A renamed/typo'd keyword fails loudly instead of being silently recorded."""
    mock_recorder.mock(f"{MODULE}.execute", output=4242)

    with pytest.raises(TypeError):
        execute("platform", r_amp=0.9, readout_amplitude=0.9)

    assert mock_recorder.calls["execute"] == []


def test_missing_required_argument_raises(mock_recorder):
    mock_recorder.mock(f"{MODULE}.execute", output=4242)

    with pytest.raises(TypeError):
        execute("platform")


def test_too_many_positional_arguments_raise(mock_recorder):
    mock_recorder.mock(f"{MODULE}.load", output="calibration")

    with pytest.raises(TypeError):
        load("a.yml", dict, "extra")


def test_opt_out_accepts_any_arguments(mock_recorder):
    mock_recorder.mock(f"{MODULE}.execute", output=1, validate_signature=False)

    assert execute(anything=1) == 1
    assert mock_recorder.calls["execute"] == [{"args": (), "kwargs": {"anything": 1}}]


def test_returned_stub_exposes_call_assertions(mock_recorder):
    stub = mock_recorder.mock(f"{MODULE}.load", output="calibration")

    assert load("a.yml") == "calibration"

    stub.assert_called_once_with("a.yml")
    assert stub.call_count == 1


def test_callable_target_and_custom_name(mock_recorder):
    mock_recorder.mock(load, output="calibration", name="loader")

    assert load("a.yml") == "calibration"
    assert mock_recorder.calls["loader"] == [{"args": ("a.yml",), "kwargs": {}}]


def test_reset_restores_the_real_callable(mock_recorder):
    mock_recorder.mock(f"{MODULE}.load", output="calibration")
    assert load("a.yml") == "calibration"

    mock_recorder.reset()

    assert mock_recorder.calls == {}
    with pytest.raises(AssertionError):
        load("a.yml")
