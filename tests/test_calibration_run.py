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

"""Tests for the calibration run."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from seqtante_open import calibration_run
from seqtante_open.controllers import TreeParseError


def test_a_tree_that_fails_to_parse_writes_nothing(monkeypatch: pytest.MonkeyPatch):
    controller = MagicMock(name="output_controller")
    platform = MagicMock(name="platform", analog_compilation_settings=SimpleNamespace(qubits=[1], topology=[]))
    monkeypatch.setattr(calibration_run, "output_controller", controller)
    monkeypatch.setattr(calibration_run, "build_platform", lambda runcard: platform)
    monkeypatch.setattr(calibration_run, "load_config", lambda path: {"sample": "s", "targets": ["q01"]})

    with pytest.raises(TreeParseError, match="Invalid target 'q01'"):
        calibration_run.CalibrationRun(platform_path="runcard.yml", config_path="tree.yml")

    controller.reset.assert_not_called()
    controller.start_run.assert_not_called()
    platform.connect.assert_not_called()
